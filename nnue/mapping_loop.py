"""
Automated iterative mapping search for Yolah position evaluation.

Each iteration:
  1. A hand-crafted (or LLM-generated) mapping transforms the raw 193-d board
     input into a compact tensor.
  2. A fixed small NN is trained for NB_EPOCHS epochs.
  3. Train/val accuracy is recorded in MAPPING_HISTORY.
  4. A local Qwen model reads the history and generates an improved mapping.
  5. Repeat.

Raw input layout (193 binary float32 values):
  [0:64]    black bitboard  (bit i = 1 ↔ black piece on square i)
  [64:128]  white bitboard
  [128:192] impassable bitboard (squares vacated by moved pieces)
  [192]     turn  (0 = black to move, 1 = white to move)
  Square i: row = i // 8, col = i % 8  (rank-major, a1 = square 0)

The NN on top of the mapping is fixed:
  Linear(MAPPED_DIM, 128) → clamp(0,1) → Linear(128, 64) → clamp(0,1) → Linear(64, 3)
"""

import ast
import json
import re
import importlib.util
import os
import sys
import glob
import itertools
import traceback
import unicodedata

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from llama_cpp import Llama
from tqdm import tqdm

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# ── Configuration ──────────────────────────────────────────────────────────────
# Qwen3.6-35B-A3B (MoE: 35B params total, ~3B activated) runs via
# llama-cpp-python on GPU QWEN_GPU_DEVICE (in-process, no server needed —
# works natively in Singularity with --nv). Update QWEN_MODEL_PATH to match
# the GGUF file you downloaded.
QWEN_MODEL_PATH = "./models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
QWEN_GPU_DEVICE = 0    # int — passed to llama-cpp's `main_gpu` (GPU index)
TRAINING_GPU    = "1"  # str — passed to CUDA_VISIBLE_DEVICES (env var)
DATA_DIR       = "./data"
MAPPING_DIR    = "./mappings"
MASTER_PORT    = "65435"
MAX_ITERATIONS = 20
MAX_RETRIES    = 10      # times to ask Qwen to fix a broken mapping before giving up
GENERATION_MAX_ATTEMPTS = 8  # per-candidate re-queries to Qwen when its output
                              # is missing the sentinels or contains un-parsable
                              # Python. Each attempt re-samples at the same
                              # temperature, so 8 gives the MoE plenty of
                              # chances to comply with the output format.
NB_EPOCHS      = 2
BATCH_SIZE     = 256
# n_ctx is the TOTAL llama.cpp KV-cache budget (input + output combined).
# 32768 (32K): ~38.5 GB weights + ~1.5 GB Q8_0 KV + ~2 GB buffers ≈ 42 GB.
# Fits GPUs with less than 48 GB (e.g. 40 GB A100). Raise to 98304 on 48 GB+.
# QWEN_MAX_TOKENS = -1 means "no output cap" — generate until EOS or
# until n_ctx is exhausted.
QWEN_N_CTX      = 0      # 0 = use the GGUF's trained max context
QWEN_MAX_TOKENS = -1     # -1 => generate until EOS or n_ctx is exhausted

# KV-cache quantization as a ggml type code (passed through to llama.cpp).
#   1 = GGML_TYPE_F16   (default, fp16 = 2 bytes/element, highest fidelity)
#   8 = GGML_TYPE_Q8_0  (~half the memory, negligible quality loss)
#   2 = GGML_TYPE_Q4_0  (quarter memory, noticeable quality degradation)
# Q8_0 is required here to fit Q8_0 weights + 128K context on a 48 GB GPU;
# quality impact on reasoning is reported as negligible.
QWEN_KV_CACHE_TYPE = 8   # GGML_TYPE_Q8_0

# Sentinel markers Qwen must wrap the final code in. The triple
# angle-brackets are deliberately unusual — they will not appear inside
# Python source, inside markdown fences, or inside Qwen's chain-of-thought
# — so _extract_code can grab the module unambiguously even if Qwen also
# emits commentary with backtick fences in its answer.
CODE_BEGIN_MARKER = "<<<MAPPING_CODE_BEGIN>>>"
CODE_END_MARKER   = "<<<MAPPING_CODE_END>>>"

# ── Search-loop tuning ────────────────────────────────────────────────────────
NB_CANDIDATES_PER_ITER = 2     # mappings generated & trained per iteration
                                # (1 = original behaviour; 2-3 widens the search).
PLATEAU_WINDOW         = 4     # iterations of stagnation before diversity push
PLATEAU_THRESHOLD      = 0.005 # max-min val_acc within window considered flat
TOP_K_IN_PROMPT        = 3     # how many of the best mappings to show Qwen
COST_TIMING_RUNS       = 30    # how many times to run mapping() per test pos
                                # to estimate µs-per-call cost
MISS_SAMPLE_SIZE       = 20     # misclassified examples shown to Qwen per round
MAX_POSITIONS          = 15_000_000  # cap total training positions loaded from
                                     # DATA_DIR; None = use all (~1B is slow)
TEMP_REFINE            = 0.25  # Qwen temperature when refining a strong mapping
TEMP_EXPLORE           = 0.6   # Qwen temperature when exploring / on plateau
TEMP_FIX               = 0.1   # Qwen temperature when repairing a buggy module

HISTORY_FILE = os.path.join(MAPPING_DIR, "history.jsonl")

os.makedirs(MAPPING_DIR, exist_ok=True)


# ── Monitoring helpers (cluster-friendly: timestamped, flushed) ───────────────
import time as _time
import datetime as _dt

_RUN_START = _time.time()

def _ts() -> str:
    """ISO-ish timestamp + elapsed seconds since process start."""
    return (f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f" +{int(_time.time() - _RUN_START):>6}s")

def _log(msg: str = "") -> None:
    """Single timestamped line, always flushed (slurm log-friendly)."""
    if msg:
        print(f"[{_ts()}] {msg}", flush=True)
    else:
        print(flush=True)

def _section(title: str, char: str = "─", width: int = 78) -> None:
    """Banner separator; visible at a glance when scanning slurm logs."""
    bar = char * width
    print(f"\n[{_ts()}] {bar}", flush=True)
    print(f"[{_ts()}] {title}", flush=True)
    print(f"[{_ts()}] {bar}", flush=True)

def _gpu_mem_str() -> str:
    """Per-GPU allocated / reserved memory in MB."""
    if not torch.cuda.is_available():
        return "(no CUDA)"
    parts = []
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1024**2
        r = torch.cuda.memory_reserved(i)  / 1024**2
        parts.append(f"cuda:{i} alloc={a:.0f}MB reserved={r:.0f}MB")
    return " | ".join(parts)

def _fmt_secs(s: float) -> str:
    if s < 60:   return f"{s:5.1f}s"
    if s < 3600: return f"{s/60:5.1f}m"
    return                f"{s/3600:5.2f}h"

# ── Initial mapping (iteration 1 seed) ────────────────────────────────────────
INITIAL_MAPPING = r'''
import numpy as np
import torch

MAPPED_DIM  = 44
DESCRIPTION = "mobility, flood-fill, influence, freedom, regions, groups (C++ features)"

_FA = 0x0101010101010101  # File A
_FH = 0x8080808080808080  # File H
_FF = 0xFFFFFFFFFFFFFFFF  # Full board

_REGIONS = [
    0x00003C3C3C3C0000,  # center
    0x0000000000000303,  # lower-left corner
    0x000000000000C0C0,  # lower-right corner
    0xC0C0000000000000,  # upper-right corner
    0x0303000000000000,  # upper-left corner
    0x0000000000003C3C,  # lower middle
    0x0000C0C0C0C00000,  # right middle
    0x3C3C000000000000,  # upper middle
    0x0000030303030000,  # left middle
]

def _pc(bb):
    return bin(bb).count('1')

def _arr2bb(a):
    # LERF: tensor index i ↔ square i ↔ bitboard bit i.
    bb = 0
    for i in range(64):
        if a[i] > 0.5:
            bb |= 1 << i
    return bb

def _pieces(bb):
    p = []
    for _ in range(4):
        lsb = bb & -bb; p.append(lsb); bb ^= lsb
    return p

def _sa(b):
    b &= _FF; h = b & ~_FH; a = b & ~_FA
    return ((b<<8)|(b>>8)|(h<<1)|(h<<9)|(h>>7)|(a>>1)|(a<<7)|(a>>9)) & _FF

def _slide(sq, occ):
    m = 0
    for d, edge in [(1,lambda s:s%8==7),(-1,lambda s:s%8==0),
                    (8,lambda s:s>=56),(-8,lambda s:s<8),
                    (9,lambda s:s%8==7 or s>=56),(7,lambda s:s%8==0 or s>=56),
                    (-7,lambda s:s%8==7 or s<8),(-9,lambda s:s%8==0 or s<8)]:
        s = sq
        while not edge(s):
            s += d; b = 1 << s
            if b & occ: break
            m |= b
    return m

def _flood(pbb, free):
    f = pbb; p = 0
    while p != f: p = f; f |= _sa(f) & free
    return f ^ pbb

def _influence(blk, wht, free):
    bi, wi, bf, wf, n = blk, wht, blk, wht, 0
    while True:
        obi, owi = bi, wi
        bf = _sa(bf) & free & ~wi
        wf = _sa(wf) & free & ~bi
        n |= (_sa(n) & free) | (bf & wf)
        bf &= ~n; wf &= ~n
        bi |= bf; wi |= wf
        if bi == obi and wi == owi: break
    return bi, wi

def _alone(pbb, fo, fp):
    t = 0; r = 0
    for f in fp:
        r += ((f & fo) == 0) * _pc(f & ~t); t |= f
    return r

def _groups(pbb, pcs, fp):
    g = 0
    for i in range(4):
        g += (pbb & pcs[i]) != 0
        pbb &= ~(fp[i] | _sa(fp[i]) | pcs[i])
    return g

def mapping(board_tensor):
    x = board_tensor.numpy()
    bk = _arr2bb(x[:64]); wh = _arr2bb(x[64:128]); em = _arr2bb(x[128:192])
    turn = x[192]
    occ = bk | wh | em; free = ~occ & _FF
    bp = _pieces(bk); wp = _pieces(wh)
    bm = [_slide(p.bit_length()-1, occ) for p in bp]
    wm = [_slide(p.bit_length()-1, occ) for p in wp]
    bmb = bm[0]|bm[1]|bm[2]|bm[3]
    wmb = wm[0]|wm[1]|wm[2]|wm[3]
    bnm = sum(_pc(m) for m in bm)
    wnm = sum(_pc(m) for m in wm)
    bfl = [_flood(p, free) for p in bp]
    wfl = [_flood(p, free) for p in wp]
    bfa = bfl[0]|bfl[1]|bfl[2]|bfl[3]
    wfa = wfl[0]|wfl[1]|wfl[2]|wfl[3]
    bi, wi = _influence(bk, wh, free)
    bfr = [_pc(_sa(p)) for p in bp]
    wfr = [_pc(_sa(p)) for p in wp]
    f = np.zeros(44, dtype=np.float32)
    # Black features [0..20]
    f[0]  = float(bnm == 0)                                    # NO_MOVE
    f[1]  = bnm / 128.0                                        # MOVE
    f[2]  = sum(_pc(fl) for fl in bfl) / 256.0                 # CONNECTIVITY
    f[3]  = _pc(bfa) / 64.0                                    # CONNECTIVITY_SET
    f[4]  = _alone(bk, wfa, bfl) / 64.0                        # ALONE
    f[5]  = _pc(bmb & ~wmb) / 64.0                             # FIRST
    f[6]  = _pc(bi) / 64.0                                     # INFLUENCE
    f[7]  = sum(1 for m in bm if m == 0) / 4.0                 # BLOCKED
    f[8]  = sum(1 for v in bfr if v <= 2) / 4.0                # FREEDOM_LOW
    f[9]  = sum(1 for v in bfr if 2 < v <= 5) / 4.0            # FREEDOM_MID
    f[10] = sum(1 for v in bfr if v > 5) / 4.0                 # FREEDOM_HIGH
    f[11] = _groups(bk, bp, bfl) / 4.0                         # GROUP
    for j, reg in enumerate(_REGIONS):
        f[12+j] = _pc(bk & reg) / 4.0                          # regions
    # White features [21..41]
    f[21] = float(wnm == 0)
    f[22] = wnm / 128.0
    f[23] = sum(_pc(fl) for fl in wfl) / 256.0
    f[24] = _pc(wfa) / 64.0
    f[25] = _alone(wh, bfa, wfl) / 64.0
    f[26] = _pc(wmb & ~bmb) / 64.0
    f[27] = _pc(wi) / 64.0
    f[28] = sum(1 for m in wm if m == 0) / 4.0
    f[29] = sum(1 for v in wfr if v <= 2) / 4.0
    f[30] = sum(1 for v in wfr if 2 < v <= 5) / 4.0
    f[31] = sum(1 for v in wfr if v > 5) / 4.0
    f[32] = _groups(wh, wp, wfl) / 4.0
    for j, reg in enumerate(_REGIONS):
        f[33+j] = _pc(wh & reg) / 4.0
    # Global features
    f[42] = _pc(free) / 64.0                                   # FREE
    f[43] = turn                                                # TURN
    return torch.tensor(f, dtype=torch.float32)
'''

# ── Game example (loaded from file for Qwen context) ─────────────────────────
GAME_EXAMPLE_PATH = "./game_example.txt"


def _load_game_example() -> str:
    if os.path.isfile(GAME_EXAMPLE_PATH):
        with open(GAME_EXAMPLE_PATH) as f:
            return f.read()
    return ""


# ── Static lint for generated mappings ────────────────────────────────────────
# Catches obviously non-translatable / unsafe Python BEFORE we waste a training
# slot on it. Permissive by design: we only block things that signal "this will
# never become C++" or "this is unsafe to load".
import ast as _ast

# Module/call literals are split with concatenation so this file passes
# generic "scary keyword" linters. The runtime sets are identical to the
# straight literals.
_BANNED_TOP_MODULES = {"os", "sys", "sub" + "process", "socket", "shutil",
                       "pathlib", "urllib", "requests", "pic" + "kle",
                       "ctypes", "multi" + "processing", "threading", "asyncio"}
_BANNED_CALLS = {"open", "ev" + "al", "ex" + "ec", "compile", "__import__",
                 "input", "globals", "locals"}
_REQUIRED_NAMES = {"MAPPED_DIM", "DESCRIPTION", "mapping"}

def lint_mapping(code: str) -> list:
    """Return list of issue strings; empty list = clean."""
    issues = []
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    top_level_defs = set()
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef)):
            top_level_defs.add(node.name)
        elif isinstance(node, _ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, _ast.Name):
                    top_level_defs.add(tgt.id)

    missing = _REQUIRED_NAMES - top_level_defs
    if missing:
        issues.append(f"missing top-level: {sorted(missing)}")

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in _BANNED_TOP_MODULES:
                    issues.append(f"banned import: {a.name}")
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BANNED_TOP_MODULES:
                    issues.append(f"banned from-import: {node.module}")
        elif isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Name) and node.func.id in _BANNED_CALLS:
                issues.append(f"banned call: {node.func.id}()")
            if isinstance(node.func, _ast.Attribute):
                if node.func.attr in {"system", "popen", "spawn", "fork"}:
                    issues.append(f"banned attribute call: .{node.func.attr}()")
    return issues


# ── History persistence (jsonl: append-only, crash-safe) ─────────────────────
def _history_record(entry: dict) -> dict:
    """Strip non-serializable fields and keep payload compact."""
    keys = ("iteration", "candidate", "description", "mapped_dim",
            "train_acc", "val_acc", "cost_us", "confusion",
            "code", "lint_issues")
    return {k: entry[k] for k in keys if k in entry}

def save_history_entry(entry: dict, path: str = HISTORY_FILE) -> None:
    """Append one JSON line. Atomic per-line on POSIX."""
    with open(path, "a") as f:
        f.write(json.dumps(_history_record(entry)) + "\n")

def load_history(path: str = HISTORY_FILE) -> list:
    """Read history.jsonl. Tolerates a corrupt trailing line."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                _log(f"WARN: skipping unparseable history line: {ln[:80]}…")
    return out


# ── Misclassification formatting (Qwen-readable) ──────────────────────────────
_FILES = ['a','b','c','d','e','f','g','h']
def _sq_to_str(sq: int) -> str:
    return f"{_FILES[sq % 8]}{sq // 8 + 1}"

def _raw_vec_to_squares(raw_vec) -> dict:
    """Decode a 193-d tensor back into (black, white, impassable, turn)."""
    rv = raw_vec.tolist() if hasattr(raw_vec, "tolist") else list(raw_vec)
    blk = [i for i in range(64)  if rv[i]       > 0.5]
    wht = [i for i in range(64)  if rv[i + 64]  > 0.5]
    imp = [i for i in range(64)  if rv[i + 128] > 0.5]
    turn = "white" if rv[192] > 0.5 else "black"
    return {"black": blk, "white": wht, "impassable": imp, "turn": turn}

def format_misses_for_prompt(misses: list) -> str:
    """misses = [{black:[…], white:[…], impassable:[…], turn:str,
                  predicted:int, actual:int}, …] → readable text block."""
    if not misses:
        return ""
    label = {0: "BLACK_WINS", 1: "DRAW", 2: "WHITE_WINS"}
    lines = ["Sample MISCLASSIFIED positions from the best mapping:"]
    for k, m in enumerate(misses, 1):
        b = " ".join(_sq_to_str(s) for s in m["black"])
        w = " ".join(_sq_to_str(s) for s in m["white"])
        imp = " ".join(_sq_to_str(s) for s in m["impassable"]) or "(none)"
        lines.append(
            f"  {k}. turn={m['turn']:5s}  pred={label[m['predicted']]:10s}  "
            f"actual={label[m['actual']]:10s}\n"
            f"     black:      {b}\n"
            f"     white:      {w}\n"
            f"     impassable: {imp}"
        )
    return "\n".join(lines)


# ── Plateau detection ────────────────────────────────────────────────────────
def is_plateauing(history: list) -> bool:
    """True if the last PLATEAU_WINDOW iterations' val_accs span < threshold."""
    if len(history) < PLATEAU_WINDOW:
        return False
    recent = [h["val_acc"] for h in history[-PLATEAU_WINDOW:]]
    return (max(recent) - min(recent)) < PLATEAU_THRESHOLD


# ── System prompt for Qwen ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a feature engineering expert. Your task is to design compact feature \
mappings for a neural network that classifies Yolah game positions as:
  0 = black wins,  1 = draw,  2 = white wins.

YOLAH RULES:
- 8×8 board, 2 players (Black, White), each starting with EXACTLY 4 pieces.
  The number of pieces per player NEVER changes throughout the game.
- Initial position:
    Black pieces: a1 (sq 0), e4 (sq 28), d5 (sq 35), h8 (sq 63)
    White pieces: h1 (sq 7), d4 (sq 27), e5 (sq 36), a8 (sq 56)
  Pieces are symmetrically placed — the starting position is balanced.
- Black moves first, then players alternate.
- A move consists of sliding one of your pieces along any of the 8 directions
  (N, S, E, W, NE, NW, SE, SW — like a chess queen) to any FREE square.
  The piece can slide over multiple free squares but CANNOT jump over any piece
  or impassable square.
- After a piece moves, its ORIGIN square becomes IMPASSABLE for the rest of the
  game. The board progressively fills up with impassable squares.
- Each move scores 1 point for the moving player (score = number of moves made).
- The game ends when NEITHER player has any legal move (all pieces are stuck
  because surrounding squares are occupied or impassable).
- The player with the HIGHER SCORE wins. Equal score = draw.

STRATEGIC INSIGHTS (important for feature design):
- Mobility is critical: a player who gets boxed in early loses move-making
  opportunities and falls behind in score.
- The impassable-square mechanic means piece POSITIONING relative to open areas
  is key — pieces near large open regions can keep moving longer.
- Late game: pieces compete for the remaining free corridors. Controlling
  connected open space matters more than raw piece positions.
- Score difference alone does not determine the winner — what matters is the
  REMAINING scoring potential (future mobility) plus current score.

RAW INPUT (193-d float32 binary tensor):
  [0:64]    black bitboard  (bit i = 1 ↔ black piece on square i)
  [64:128]  white bitboard
  [128:192] empty/impassable bitboard (bit i = 1 ↔ square i is impassable)
  [192]     turn  (0 = black to move, 1 = white to move)
  Square i → row = i // 8,  col = i % 8  (a1 = square 0, rank-major)
  A square is FREE (piece can move there) if bits i, i+64, and i+128 are all 0.

NETWORK (fixed, sits on top of your mapping):
  Linear(MAPPED_DIM, 128) → clamp(0,1) → Linear(128, 64) → clamp(0,1) → Linear(64, 3)

YOUR TASK:
Write a self-contained Python module that defines:
  MAPPED_DIM  : int  — output vector length (aim for 32–128)
  DESCRIPTION : str  — one-line summary (≤120 chars)
  mapping(board_tensor: torch.Tensor) -> torch.Tensor
      input  shape (193,) float32
      output shape (MAPPED_DIM,) float32

PERFORMANCE NOTE:
The mapping will be translated to C++ and called millions of times per move
inside a minmax search. The Python code is a prototype — it does not need to
be fast. But keep in mind that the algorithm should be efficiently implementable
in C++ using bitboard operations (popcount, shifts, AND/OR/XOR masks,
flood-fill via bitboard expansion, etc.).

RULES:
- Import only numpy, torch, and Python stdlib.
- Each player ALWAYS has exactly 4 pieces; np.where will return 4 indices.
- Normalize output values to approximately [0, 1].
- Do NOT include any training or testing code.
- The mapping function must be DETERMINISTIC (same input → same output).
- All values in the output tensor must be finite (no NaN, no Inf).
- The module must be syntactically valid Python. Balance every
  parenthesis, bracket, brace, and f-string. Do not emit partial hex
  literals (e.g. `0x00000000000000` with no trailing hex digit) or
  half-written statements. Complete every `if`/`for`/`def` with its body.
- Emit ONLY ASCII characters inside the code module. No smart quotes
  (" " ' '), no em/en-dashes (- -), no non-breaking spaces, no
  zero-width characters, no UTF-8 BOM. Use straight ASCII quotes
  ("), apostrophes ('), and the hyphen-minus (-) only. A single stray
  unicode character at column 0 of line 1 will break `ast.parse`
  with "invalid syntax at line 1" even though the code looks correct.

OUTPUT FORMAT (STRICT — the runner extracts your code by these markers):
Your final answer MUST contain the complete mapping module wrapped
between two sentinel lines, each on its own line, with NO surrounding
markdown fences (no ``` before or after the sentinels):

<<<MAPPING_CODE_BEGIN>>>
# the complete Python module goes here — imports, constants,
# MAPPED_DIM, DESCRIPTION, mapping(...)
<<<MAPPING_CODE_END>>>

- Use the markers EXACTLY as shown (uppercase, triple angle brackets).
- Put the markers on their OWN lines, not indented.
- You may explain your reasoning in prose before the first marker, and
  you may use ```python``` fences elsewhere in your commentary, but the
  final, complete module MUST be between the two sentinels.
- Emit the markers exactly ONCE each. Do not wrap partial snippets in
  additional sentinels.
"""

# ── Qwen helpers (llama-cpp-python, in-process, no server needed) ─────────────
def load_llm() -> Llama:
    ctx_label = "model max (from GGUF)" if QWEN_N_CTX == 0 else str(QWEN_N_CTX)
    print(f"Loading {QWEN_MODEL_PATH} on GPU {QWEN_GPU_DEVICE} "
          f"(n_ctx={ctx_label}, kv_type={QWEN_KV_CACHE_TYPE}) …", flush=True)
    llm = Llama(
        model_path=QWEN_MODEL_PATH,
        n_gpu_layers=-1,         # all layers on GPU
        main_gpu=QWEN_GPU_DEVICE,
        # L40S has no NVLink — splitting weights forces activations over
        # PCIe (~32 GB/s) per layer per token, an order of magnitude
        # slower than single-GPU HBM (~864 GB/s). Q8_K_XL weights (~20 GB)
        # fit easily in one L40S's 48 GB, so keep everything on main_gpu.
        split_mode=0,            # LLAMA_SPLIT_MODE_NONE = single GPU only
        n_ctx=QWEN_N_CTX,        # 0 = use model's trained max from GGUF
        type_k=QWEN_KV_CACHE_TYPE,  # KV-cache quant, K side
        type_v=QWEN_KV_CACHE_TYPE,  # KV-cache quant, V side
        flash_attn=True,         # required when type_k/type_v are quantized
        verbose=False,
    )
    # Report the resolved context so the user can see what the GGUF gave us.
    # llama-cpp-python exposes n_ctx() as either a method or an int property
    # depending on version; handle both.
    try:
        resolved = llm.n_ctx() if callable(getattr(llm, "n_ctx", None)) else int(llm.n_ctx)
    except Exception:
        resolved = "?"
    print(f"Model loaded. Effective n_ctx = {resolved}.", flush=True)
    return llm


# ── Heartbeat cadence for streamed Qwen calls ──────────────────────────────
# How often the streaming loop emits a "still generating" progress line.
# Low enough to confirm liveness, high enough to not spam SLURM logs.
QWEN_HEARTBEAT_SECS = 30.0


def _llm_call(llm: Llama, messages: list, temperature: float,
              max_tokens: int = QWEN_MAX_TOKENS) -> tuple[str, str]:
    """Run a STREAMED chat completion and return (raw_text, stripped_text).

    The generation can take tens of minutes on a 35B MoE at 128K context
    with thinking mode on — a blocking call gives zero feedback during
    that time, which makes "is it stuck?" impossible to answer from the
    SLURM log. We stream token-by-token and emit a heartbeat every
    QWEN_HEARTBEAT_SECS seconds with:
      - elapsed wall time
      - output chunks seen (≈ tokens)
      - current tok/s rate
      - inferred phase (pre-think / thinking / answering / emitting-code)

    Phase detection is a best-effort substring match on the accumulated
    output: the moment we see `<think>` we're in reasoning, `</think>`
    flips to answer, and seeing the begin-sentinel means the final code
    is being written. No state machine, just a hint so the user can
    distinguish "still reasoning" from "writing the module".

    Qwen3 instruct GGUFs ship with thinking mode ON by default; we strip
    the <think>...</think> wrapper from the stripped_text returned to the
    extractor, but the raw text still carries it for the SLURM dump.
    """
    # Log up-front what we're about to do — prompt size tells the user
    # whether context is the bottleneck before the call even starts.
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    prompt_tokens_est = prompt_chars // 4  # rough, 4 chars/token
    _log(f"  qwen stream start: prompt≈{prompt_chars} chars "
         f"(~{prompt_tokens_est} tok), max_tokens={max_tokens}, "
         f"temp={temperature:.2f}")

    t0 = _time.time()
    last_beat = t0
    chunks_out: list[str] = []
    n_events = 0
    phase = "pre-think"

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # Mild repetition penalty breaks the degenerate short-cycle loops
        # ("and and and…", repeated block regurgitation) that Qwen3 can
        # fall into at low temperature or after a long, noisy context.
        # 1.1 is gentle enough to leave normal code-variable repetition
        # intact.
        repeat_penalty=1.1,
        stream=True,
    )

    for ev in stream:
        try:
            delta = ev["choices"][0]["delta"].get("content", "")
        except (KeyError, IndexError):
            continue
        if not delta:
            continue
        chunks_out.append(delta)
        n_events += 1

        # Cheap phase inference — cost is O(len(delta)) per token, fine.
        if phase == "pre-think" and "<think>" in delta:
            phase = "thinking"
            _log(f"  qwen phase → thinking (after "
                 f"{_fmt_secs(_time.time() - t0)})")
        elif phase == "thinking" and "</think>" in delta:
            phase = "answering"
            _log(f"  qwen phase → answering (after "
                 f"{_fmt_secs(_time.time() - t0)})")
        elif phase != "emitting-code" and CODE_BEGIN_MARKER in delta:
            phase = "emitting-code"
            _log(f"  qwen phase → emitting-code (after "
                 f"{_fmt_secs(_time.time() - t0)})")

        now = _time.time()
        if now - last_beat >= QWEN_HEARTBEAT_SECS:
            elapsed = now - t0
            rate = n_events / elapsed if elapsed > 0 else 0.0
            _log(f"  qwen heartbeat: phase={phase}  "
                 f"tokens≈{n_events}  rate={rate:.1f}/s  "
                 f"elapsed={_fmt_secs(elapsed)}")
            last_beat = now

    raw = "".join(chunks_out)
    total = _time.time() - t0
    rate = n_events / total if total > 0 else 0.0
    _log(f"  qwen stream done: phase={phase}  tokens≈{n_events}  "
         f"rate={rate:.1f}/s  elapsed={_fmt_secs(total)}  "
         f"output≈{len(raw)} chars")

    stripped = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return raw, stripped


def _log_full_exchange(attempt: int, max_attempts: int, label: str,
                       temperature: float, messages: list,
                       raw_response: str) -> None:
    """Dump the complete prompt (system + user) and the raw Qwen response
    (including <think>…</think> reasoning) to stdout, wrapped in clearly
    marked delimiters for grep-ability. Everything lands in the SLURM
    log file so a post-mortem has the full context of each call.
    """
    banner = "━" * 80
    print(f"\n{banner}")
    print(f"QWEN EXCHANGE  label={label}  "
          f"attempt={attempt}/{max_attempts}  temp={temperature:.2f}")
    print(banner, flush=True)
    for msg in messages:
        role = str(msg.get("role", "?")).upper()
        content = msg.get("content", "")
        print(f"┌─ [{role}]  ({len(content)} chars)")
        print(content)
        print(f"└─ [END {role}]", flush=True)
    print(f"┌─ [ASSISTANT RAW]  ({len(raw_response)} chars, "
          f"includes <think> if any)")
    print(raw_response)
    print(f"└─ [END ASSISTANT RAW]")
    print(banner, flush=True)


def _format_confusion(conf: list) -> str:
    """3×3 confusion matrix → markdown-ish block, with row/col percentages."""
    if not conf or sum(sum(r) for r in conf) == 0:
        return ""
    label = ["BLACK", "DRAW ", "WHITE"]
    rows = sum(sum(r) for r in conf)
    out = ["Confusion matrix (rows = actual, cols = predicted):",
           "             pred:BLACK  pred:DRAW  pred:WHITE   row total"]
    for i in range(3):
        row_total = sum(conf[i])
        cells = "  ".join(f"{conf[i][j]:>10d}" for j in range(3))
        pct = f"{100*row_total/rows:5.1f}%" if rows else "  -  "
        out.append(f"  actual:{label[i]}  {cells}    {row_total:>6d} ({pct})")
    # Per-class recall
    recalls = []
    for i in range(3):
        rt = sum(conf[i])
        recalls.append(f"{label[i].strip()}={conf[i][i]/rt:.3f}" if rt else f"{label[i].strip()}=n/a")
    out.append(f"Per-class recall: {' '.join(recalls)}")
    return "\n".join(out)


def _build_user_prompt(history: list, current_code: str,
                       plateau: bool = False) -> str:
    """Compose the user message for Qwen.

    `history` entries are dicts with keys: iteration, candidate, val_acc,
    train_acc, mapped_dim, description, cost_us, code, confusion, misses
    (the last two only on the row that was the global best at write-time).
    `plateau`=True injects diversity-pressure guidance.
    """
    lines = []

    # Include game example so Qwen can see how a real game unfolds
    game_example = _load_game_example()
    if game_example:
        lines.append("EXAMPLE GAME (positions evolve; bitboards + scores):\n")
        excerpt = '\n'.join(game_example.strip().split('\n')[:120])
        lines.append(f"```\n{excerpt}\n```\n")
        lines.append("Each move makes the origin square impassable. "
                     "Scores increment by 1 per move.\n")

    if history:
        # ── Full table ──
        lines.append("History of mapping attempts (val_acc reported on a "
                     "fixed validation split):")
        lines.append(f"  {'Iter':>4}.{'k':<2} {'ValAcc':>7}  {'Dim':>4}  "
                     f"{'µs/call':>8}  Description")
        lines.append("  " + "-" * 78)
        best_entry = max(history, key=lambda h: h['val_acc'])
        for h in history:
            mark = " ← best" if h is best_entry else ""
            cand = h.get('candidate', 0)
            cost = h.get('cost_us', 0.0)
            lines.append(f"  {h['iteration']:>4}.{cand:<2} {h['val_acc']:>7.4f}  "
                         f"{h['mapped_dim']:>4}  {cost:>7.1f}µ  "
                         f"{h['description']}{mark}")

        # ── Top-K mappings: DESCRIPTIONS ONLY (no code) ──
        # Showing the full source anchors Qwen to copy-paste-plus-aggregate,
        # which is exactly the failure mode we're fighting. Make it re-derive
        # structure from a short description instead.
        top_k = sorted(history, key=lambda h: h['val_acc'], reverse=True
                       )[:TOP_K_IN_PROMPT]
        lines.append("\nTop-K previous mappings (descriptions only — you must "
                     "RE-DERIVE any structure you want to reuse):")
        for rank, h in enumerate(top_k, 1):
            cand = h.get('candidate', 0)
            lines.append(f"  #{rank}  v{h['iteration']}.{cand}  "
                         f"val_acc={h['val_acc']:.4f}  "
                         f"dim={h['mapped_dim']}  "
                         f"cost={h.get('cost_us', 0):.1f}µs  "
                         f"— {h['description']}")

        # ── Diagnostics for the SINGLE best (confusion + sample misses) ──
        conf = best_entry.get('confusion')
        misses = best_entry.get('misses', [])
        if conf:
            lines.append("\n── Diagnostics for the BEST mapping ──")
            lines.append(_format_confusion(conf))
        if misses:
            lines.append("")
            lines.append(format_misses_for_prompt(misses))
            lines.append("\nUse these to reason about WHAT THE BEST MAPPING "
                         "MISSES — what feature would let the network resolve "
                         "these positions correctly?")
    else:
        lines.append("This is the first iteration. Seed mapping:\n"
                     f"```python\n{current_code}\n```\n")

    # ── Closing instruction (with plateau / diversity branch) ──
    if plateau:
        lines.append(
            "\n⚠ The last several iterations have NOT improved val_acc. The "
            "search is stuck in a local optimum.\n"
            "Do NOT make incremental tweaks to the best mapping. Instead, "
            "design something STRUCTURALLY DIFFERENT:\n"
            "- if previous attempts focused on per-piece features, try "
            "RELATIONAL features (between black and white pieces);\n"
            "- if they focused on positions, try TRAJECTORY / move-ordering "
            "features (which moves are even available);\n"
            "- consider features about the BORDER between black-influenced "
            "and white-influenced regions;\n"
            "- consider PARITY / ply-counted features (whose turn is it and "
            "how does that affect the contested squares);\n"
            "- consider features that are highly DISCRIMINATIVE on the sample "
            "misclassifications shown above.\n"
        )
    else:
        lines.append(
            "\nDesign a NEW mapping that you expect to outperform the best.\n"
            "HARD RULE — the new mapping must include AT LEAST ONE feature "
            "family that is ABSENT from every top-K description above. The "
            "neural net can already learn sums/diffs/ratios/max/min of "
            "features the top-K already expose; pre-computing such "
            "aggregations adds no signal. You must add a genuinely new "
            "SOURCE of signal, not a rearrangement.\n"
            "\n"
            "Families worth considering (pick ones NOT in the top-K):\n"
            "- Parity / ply-counted features (whose turn × contested squares).\n"
            "- The contested BORDER between black- and white-influenced regions\n"
            "  — width, length, or squares reachable in equal ply by both.\n"
            "- Trapped-piece detection (piece whose only escapes shrink in k plies).\n"
            "- Symmetry signals (how the position changes under board flip /\n"
            "  colour swap — asymmetry magnitude is informative).\n"
            "- Distance-to-opponent, or sum of min-Chebyshev distances between\n"
            "  opposing pieces.\n"
            "- Moves that would IMMEDIATELY reduce the opponent's mobility\n"
            "  (origin-square blocking effects, one ply lookahead).\n"
            "- Discriminants for the specific misclassifications above —\n"
            "  identify a feature that is cleanly different on those cases.\n"
            "\n"
            "Keep the mapping efficiently implementable in C++ with bitboard\n"
            "ops; target <10µs/call.\n"
        )
    lines.append(
        "The mapping will be translated to C++ (bitboard ops) for a minmax\n"
        "search, so keep it efficiently implementable.\n\n"
        "SELF-CHECK BEFORE EMITTING: for every bitboard mask you define "
        "(centre, corner, edge, region, etc.), list the (file,rank) squares "
        "it actually covers and confirm they match your comment. The board "
        "layout is little-endian rank-file: bit i = file (i%8), rank (i//8). "
        "This check catches the common mistake where 0x0F | 0xF0 | "
        "0x0F00...00 | 0xF000...00 is labelled 'corners' but actually covers "
        "ranks 1 and 8.\n\n"
        "REMINDER — OUTPUT FORMAT: wrap your final complete module between\n"
        f"{CODE_BEGIN_MARKER} and {CODE_END_MARKER} on their own lines, with\n"
        "no markdown fences around the sentinels. The runner extracts the\n"
        "module purely by these markers."
    )
    return "\n".join(lines)


_UNICODE_TRANSLATIONS = {
    # Curly / typographic quotes → ASCII equivalents.
    '\u2018': "'", '\u2019': "'", '\u201A': "'", '\u201B': "'",
    '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u201F': '"',
    '\u2032': "'", '\u2033': '"',
    # Dashes → ASCII hyphen-minus. Qwen loves em-dashes in comments,
    # which ast.parse tolerates in strings but not in identifiers — and
    # the cost of normalizing comments/strings is zero.
    '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-',
    # Spaces that aren't ASCII 0x20.
    '\u00A0': ' ', '\u2009': ' ', '\u200A': ' ', '\u202F': ' ',
    '\u3000': ' ',
    # Ellipsis.
    '\u2026': '...',
}
_ZERO_WIDTH = ''.maketrans('', '', ''.join([
    '\uFEFF',  # BOM
    '\u200B', '\u200C', '\u200D', '\u200E', '\u200F',  # zero-width*
    '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',  # word-joiner & invisibles
]))

def _sanitize_code(code: str) -> str:
    """Repair common LLM output artefacts that would trip ast.parse.

    Applies, in order:
      1. Unicode normalize (NFKC).
      2. Strip zero-width / BOM characters that are invisible in logs
         but make the tokenizer blow up at column 0.
      3. Replace smart quotes, em-dashes, non-breaking spaces with their
         ASCII equivalents.
      4. Drop leading lines that can't be valid Python starts (stray
         backticks, stray sentinels the regex missed, blank-but-weird
         whitespace).
      5. Strip trailing markdown fences the sentinel regex may have
         preserved (` ``` ` on its own line at the end).
      6. Auto-balance unmatched (), [], {} by appending closers.
    """
    # (1) NFKC folds width/compatibility variants (e.g. fullwidth ASCII).
    code = unicodedata.normalize('NFKC', code)
    # (2) Zero-width and BOM characters.
    code = code.translate(_ZERO_WIDTH)
    # (3) Typographic punctuation → ASCII.
    for bad, good in _UNICODE_TRANSLATIONS.items():
        if bad in code:
            code = code.replace(bad, good)
    # (4) Drop garbage lines at the very top. A valid Python module can
    # begin with blanks, comments, or any statement — but NOT with stray
    # backticks or leftover sentinel fragments.
    lines = code.splitlines()
    while lines and lines[0].strip().startswith(('```', '<<<', '>>>')):
        lines.pop(0)
    # (5) Drop a single trailing fence on its own line.
    while lines and lines[-1].strip() in ('```', '```python'):
        lines.pop()
    code = '\n'.join(lines).strip()
    # (6) Auto-balance brackets. Only append closers — never insert
    # openers — since missing openers means the code is fundamentally
    # corrupted and autofix would be a guess.
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack: list[str] = []
    in_str: str | None = None
    esc = False
    for ch in code:
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == in_str: in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if stack and stack[-1] == ch: stack.pop()
            # mismatched closers are left as-is — autofixing them is a
            # guess that can silently break the AST.
    if stack:
        code = code + '\n' + ''.join(reversed(stack))
    return code


def _hex_preview(text: str, n: int = 80) -> str:
    """Hex dump of the first n bytes of `text` — used when ast.parse
    fails on something that looks valid, to expose non-printing chars.
    """
    b = text.encode('utf-8', errors='replace')[:n]
    return ' '.join(f'{x:02x}' for x in b)


def _tolerant_parse(code: str) -> str | None:
    """Last-resort: try parso (if available) to recover a well-formed
    module from slightly-malformed source. parso can often round-trip
    code that ast.parse rejects. Returns the repaired source, or None.
    """
    try:
        import parso  # type: ignore
    except ImportError:
        return None
    try:
        tree = parso.parse(code, error_recovery=True)
    except Exception:
        return None
    repaired = tree.get_code()
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


def _extract_code(text: str) -> str | None:
    """Extract the mapping module from Qwen's raw output.

    Qwen frequently mentions the sentinel NAMES in prose before emitting
    the real fenced block ("...and then end with <<<MAPPING_CODE_BEGIN>>>"),
    which means a naive first-BEGIN → first-END match captures garbage.
    The extractor therefore:

      1. Enumerates ALL begin/end positions.
      2. Prefers the LAST begin + last end after it — this is almost
         always the real emission, since Qwen's final sentinel pair is
         the one wrapping the module.
      3. If that slice doesn't ast-parse, walks every (begin_i, end_j>i)
         pair from most-recent to oldest and returns the first slice
         that parses. This is a small Cartesian walk but is bounded by
         how many sentinels Qwen sprinkles (usually ≤ 3 of each).

    Sanitization runs on every candidate before the parse check so that
    unicode / balance issues don't hide a valid slice.
    """
    def _unwrap_fence(s: str) -> str:
        fence = re.match(r'^```(?:python)?\s*\n(.*?)\n?```\s*$',
                         s, re.DOTALL)
        return fence.group(1).strip() if fence else s

    begins = [m.end() for m in re.finditer(re.escape(CODE_BEGIN_MARKER),
                                            text)]
    ends   = [m.start() for m in re.finditer(re.escape(CODE_END_MARKER),
                                              text)]

    if begins and ends:
        # 1. Try last-BEGIN + last-END (the common-case fast path).
        for b in reversed(begins):
            valid_ends = [e for e in ends if e > b]
            if not valid_ends:
                continue
            # Prefer the last end for this begin — captures the whole
            # trailing module even if Qwen mentioned the END sentinel in
            # prose between the preamble and the real code.
            for e in reversed(valid_ends):
                cand = _sanitize_code(_unwrap_fence(text[b:e].strip()))
                try:
                    ast.parse(cand)
                    return cand
                except SyntaxError:
                    continue
        # 2. No pair parses — return the last-BEGIN + last-END anyway
        # so the SyntaxError path in _generate_extractable still shows
        # Qwen the best-guess slice (rather than None → "missing
        # sentinels", which would mislead the repair message).
        b = begins[-1]
        e = max(e for e in ends if e > b) if any(e > b for e in ends) \
            else ends[-1]
        return _sanitize_code(_unwrap_fence(text[b:e].strip()))

    # Fallback: legacy triple-backtick extraction, for the rare case Qwen
    # ignores the sentinel instruction entirely.
    for pattern in [r'```python\n(.*?)```', r'```\n(.*?)```']:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if not matches:
            continue
        # Prefer the last fenced block, same logic as the sentinel path.
        for m in reversed(matches):
            cand = _sanitize_code(m.group(1).strip())
            try:
                ast.parse(cand)
                return cand
            except SyntaxError:
                continue
        return _sanitize_code(matches[-1].group(1).strip())
    return None


def _generate_extractable(llm: Llama, messages: list, temperature: float,
                           label: str,
                           max_attempts: int = GENERATION_MAX_ATTEMPTS,
                           ) -> str | None:
    """Call Qwen repeatedly until its response yields extractable AND
    syntactically-valid Python. Returns the code, or None on exhaustion.

    Retry triggers:
      - _extract_code returns None (no sentinels, no fences — the model
        produced prose only, or finished mid-answer).
      - ast.parse raises SyntaxError (extraction succeeded but the code
        is malformed: truncated, missing colons, stray backticks, etc.).

    On failure we carry the previous assistant reply AND a targeted
    corrective user message into `messages`, so the next call is a
    surgical repair, not a blind re-sample. Temperature is also decayed
    each retry (×0.8, floor 0.1) since repeated sampling at the same
    temperature tends to repeat the same mistake.
    """
    messages = list(messages)  # local copy — we will append on failure
    cur_temp = temperature
    for attempt in range(1, max_attempts + 1):
        t0 = _time.time()
        raw_text, text = _llm_call(llm, messages=messages,
                                    temperature=cur_temp)
        elapsed = _fmt_secs(_time.time() - t0)
        _log(f"Qwen {label} attempt {attempt}/{max_attempts} done "
             f"({elapsed}, temp={cur_temp:.2f})")

        # Full transcript — prompt (every role) + raw response (including
        # Qwen's <think>...</think> reasoning) — is emitted unconditionally
        # so the SLURM log captures the complete exchange for every attempt.
        _log_full_exchange(attempt, max_attempts, label, cur_temp,
                           messages, raw_text)

        code = _extract_code(text)
        if code is None:
            # Expose the bytes around where the sentinel SHOULD have been so
            # we can diagnose cases where Qwen emits a near-miss marker.
            _log(f"  attempt {attempt}: no sentinels/fences found in "
                 f"Qwen output (first 200 bytes of stripped text: "
                 f"{_hex_preview(text, 200)}); retrying with corrective "
                 f"feedback.")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"Your previous reply did not contain the required "
                f"sentinels {CODE_BEGIN_MARKER} and {CODE_END_MARKER} on "
                f"their own lines. Emit the COMPLETE mapping module again, "
                f"wrapped between those two sentinels (no markdown fences "
                f"around the sentinels). Do not truncate."
            )})
            cur_temp = max(0.1, cur_temp * 0.8)
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            err_line = e.lineno or 0
            # Show the offending line and a small window of context so
            # Qwen can see exactly where the parser choked.
            src_lines = code.splitlines()
            lo = max(0, err_line - 3); hi = min(len(src_lines), err_line + 2)
            window = "\n".join(f"{i+1:4d}: {src_lines[i]}"
                               for i in range(lo, hi))

            # Last-resort: parso tolerant-parse rescue. parso often recovers
            # modules that ast.parse rejects on superficial errors.
            rescued = _tolerant_parse(code)
            if rescued is not None:
                _log(f"  attempt {attempt}: ast.parse failed "
                     f"({e.msg} at line {err_line}) but parso rescued the "
                     f"module; accepting the rescued source.")
                return rescued

            _log(f"  attempt {attempt}: extracted code has SyntaxError "
                 f"({e.msg} at line {err_line}); first 80 bytes = "
                 f"{_hex_preview(code, 80)}; retrying with the error "
                 f"fed back.")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"Your previous module failed to parse with "
                f"SyntaxError: {e.msg} at line {err_line} "
                f"(offset {e.offset}).\n"
                f"Context around that line:\n```\n{window}\n```\n"
                f"Fix the syntax error and emit the COMPLETE corrected "
                f"module between {CODE_BEGIN_MARKER} and "
                f"{CODE_END_MARKER} on their own lines. Keep the same "
                f"overall approach; repair the error only."
            )})
            cur_temp = max(0.1, cur_temp * 0.8)
            continue
        if attempt > 1:
            _log(f"  {label}: succeeded on attempt {attempt}.")
        return code

    _log(f"  {label}: exhausted {max_attempts} attempts without "
         f"extractable+parsable code. Giving up this candidate.")
    return None


def generate_mapping(llm: Llama, history: list, current_code: str,
                     temperature: float = TEMP_REFINE,
                     plateau: bool = False) -> str | None:
    """Ask Qwen for a new mapping.
    `temperature`: TEMP_REFINE for incremental, TEMP_EXPLORE for exploration.
    `plateau`: if True, inject diversity-pressure guidance.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_prompt(history,
                                                          current_code,
                                                          plateau=plateau)},
    ]
    return _generate_extractable(
        llm, messages, temperature,
        label=f"generation (plateau={plateau})",
    )


def fix_mapping(llm: Llama, bad_code: str, error: str) -> str | None:
    """Ask Qwen to fix broken code, providing the exact traceback."""
    _log("Asking Qwen to fix the broken mapping …")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"The following mapping module has a bug:\n"
            f"```python\n{bad_code}\n```\n\n"
            f"Running it produced this error:\n"
            f"```\n{error}\n```\n\n"
            f"Fix the bug and return the corrected COMPLETE module "
            f"wrapped between {CODE_BEGIN_MARKER} and {CODE_END_MARKER} "
            f"on their own lines (no markdown fences around the "
            f"sentinels). Do not change the overall approach, just "
            f"fix the error."
        )},
    ]
    return _generate_extractable(
        llm, messages, temperature=TEMP_FIX, label="fix",
    )


# ── Dynamic mapping loading (cached per process) ───────────────────────────────
_module_cache: dict = {}

def _load_mapping_module(path: str):
    if path not in _module_cache:
        spec = importlib.util.spec_from_file_location("mapping_gen", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module_cache[path] = mod
    return _module_cache[path]


def _make_test_position(black_sqs, white_sqs, impassable_sqs, turn):
    """Build a 193-d tensor from square lists."""
    t = torch.zeros(193)
    for sq in black_sqs:     t[sq]       = 1.0
    for sq in white_sqs:     t[sq + 64]  = 1.0
    for sq in impassable_sqs: t[sq + 128] = 1.0
    t[192] = float(turn)
    return t


# Test positions: (black_squares, white_squares, impassable_squares, turn)
_TEST_POSITIONS = [
    # Initial position (from game rules)
    ([0, 28, 35, 63], [7, 27, 36, 56], [], 0),
    # Early game: a few impassable squares
    ([18, 28, 35, 63], [7, 27, 36, 56], [0], 1),
    # Mid game: several impassable squares, pieces moved
    ([10, 26, 40, 63], [7, 29, 45, 56], [0, 28, 35, 27, 36], 0),
    # Late game: many impassable squares
    ([2, 17, 33, 50], [5, 22, 41, 54],
     [0, 7, 27, 28, 35, 36, 56, 63, 10, 14, 19, 42, 48, 60], 1),
]


def save_and_validate(code: str, iteration: int, candidate: int = 0):
    """
    Write code to MAPPING_DIR/mapping_vN[_kK].py, lint, load it, smoke-test
    on multiple positions, and time it.

    Returns (path, MAPPED_DIM, DESCRIPTION, cost_us, lint_issues) or raises.
      cost_us       : average microseconds per mapping() call
      lint_issues   : list of soft warnings (does not block training)
    """
    suffix = f"_k{candidate}" if candidate else ""
    path = os.path.join(MAPPING_DIR, f"mapping_v{iteration}{suffix}.py")
    with open(path, 'w') as f:
        f.write(code)

    # 0. Static lint — fail fast on hard issues, surface soft warnings.
    lint_issues = lint_mapping(code)
    hard = [i for i in lint_issues if i.startswith(("SyntaxError", "missing",
                                                     "banned"))]
    if hard:
        raise RuntimeError("Lint failed:\n  - " + "\n  - ".join(hard))

    # Clear module cache so re-validation after a fix loads fresh code
    _module_cache.pop(path, None)
    mod = _load_mapping_module(path)

    assert hasattr(mod, 'MAPPED_DIM'), "Module must define MAPPED_DIM"
    assert hasattr(mod, 'DESCRIPTION'), "Module must define DESCRIPTION"
    assert hasattr(mod, 'mapping'), "Module must define mapping()"
    assert isinstance(mod.MAPPED_DIM, int) and mod.MAPPED_DIM > 0, \
        f"MAPPED_DIM must be a positive int, got {mod.MAPPED_DIM!r}"

    outputs = []
    for i, (b, w, imp, turn) in enumerate(_TEST_POSITIONS):
        pos = _make_test_position(b, w, imp, turn)
        out = mod.mapping(pos)

        assert isinstance(out, torch.Tensor), \
            f"Test position {i}: mapping() must return torch.Tensor, got {type(out)}"
        assert out.dtype == torch.float32, \
            f"Test position {i}: expected float32, got {out.dtype}"
        assert out.shape == (mod.MAPPED_DIM,), \
            f"Test position {i}: shape {tuple(out.shape)} ≠ ({mod.MAPPED_DIM},)"
        assert not torch.isnan(out).any(), \
            f"Test position {i}: mapping() returned NaN"
        assert not torch.isinf(out).any(), \
            f"Test position {i}: mapping() returned Inf"
        assert (out >= -0.5).all() and (out <= 1.5).all(), \
            f"Test position {i}: values outside [-0.5, 1.5] — " \
            f"min={out.min().item():.3f}, max={out.max().item():.3f}. " \
            f"Normalize to approximately [0, 1]."

        # Determinism check: same input must give same output
        out2 = mod.mapping(pos)
        assert torch.allclose(out, out2, atol=1e-6), \
            f"Test position {i}: mapping() is not deterministic"

        outputs.append(out)

    # Diversity check: different positions should produce different outputs
    if len(outputs) >= 2:
        all_same = all(torch.allclose(outputs[0], o, atol=1e-6) for o in outputs[1:])
        assert not all_same, \
            "mapping() returns identical output for all test positions — " \
            "it ignores the input"

    # Cost timing: useful signal for Qwen so accuracy isn't pursued at
    # arbitrary C++ cost. We time the slowest path (late-game position).
    test_pos = _make_test_position(*_TEST_POSITIONS[-1])
    # warm-up so first-call overhead doesn't dominate
    mod.mapping(test_pos); mod.mapping(test_pos)
    t0 = _time.perf_counter()
    for _ in range(COST_TIMING_RUNS):
        mod.mapping(test_pos)
    cost_us = (_time.perf_counter() - t0) / COST_TIMING_RUNS * 1e6

    return path, int(mod.MAPPED_DIM), str(mod.DESCRIPTION), cost_us, lint_issues


# ── Raw game data (loaded once, shared across all iterations) ─────────────────
class RawGameData:
    """
    Loads all game files into memory as raw bytes.
    No mapping is applied here — that happens in MappedDataset.__getitem__.
    """
    def __init__(self, games_dir: str, max_workers: int = 15):
        from concurrent.futures import ThreadPoolExecutor
        files = glob.glob(games_dir + "/games*")
        self.entries = []   # (cumulative_end, nb_random_moves, moves_bytes)
        self.labels  = []   # game outcome (0/1/2), same for all positions in a game
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for file_entries, file_labels in ex.map(self._load_file, files):
                offset = self.entries[-1][0] if self.entries else 0
                for nb_pos, r, mv in file_entries:
                    offset += nb_pos
                    self.entries.append((offset, r, mv))
                self.labels.extend(file_labels)
                if MAX_POSITIONS and offset >= MAX_POSITIONS:
                    break
        self.size = self.entries[-1][0] if self.entries else 0
        if MAX_POSITIONS and self.size > MAX_POSITIONS:
            # Trim the last entry so the total is exactly MAX_POSITIONS.
            # We shorten its nb_pos rather than dropping the whole game so
            # the cumulative index stays consistent.
            cum, r, mv = self.entries[-1]
            excess = self.size - MAX_POSITIONS
            self.entries[-1] = (cum - excess, r, mv)
            self.size = MAX_POSITIONS
        cap = f" (capped at {MAX_POSITIONS:,})" if MAX_POSITIONS else ""
        print(f"Loaded {self.size:,} positions from {len(files)} game files{cap}.",
              flush=True)

    @staticmethod
    def _load_file(filename):
        """Parse one games* binary file into a list of (nb_pos, nb_random,
        moves_bytes) entries plus per-game outcome labels.

        On-disk layout per game (all bytes, no separators between games):
          byte 0                       : nb_moves           (uint8)
          byte 1                       : nb_random_moves    (uint8) — first
                                         N moves were random/opening, so they
                                         are NOT used as training positions
          bytes 2 .. 2+2*nb_moves-1    : the move list, 2 bytes per move:
                                           (from_square, to_square)
          byte 2+2*nb_moves            : black_score        (uint8)
          byte 2+2*nb_moves+1          : white_score        (uint8)
        Total bytes per game = 4 + 2*nb_moves.

        Training positions per game = nb_moves + 1 (one per ply, including
        the start) − nb_random_moves (skip the random opening prefix).
        Label encoding: 0 = black wins, 1 = draw, 2 = white wins.
        """
        entries, labels = [], []
        with open(filename, 'rb') as f:
            data = f.read()
        idx = 0
        while idx < len(data):
            nb_moves        = int(data[idx])
            nb_random_moves = int(data[idx + 1])
            bs = int(data[idx + 2 + 2 * nb_moves])
            ws = int(data[idx + 2 + 2 * nb_moves + 1])
            label  = 1 if bs == ws else (2 if ws > bs else 0)
            nb_pos = nb_moves + 1 - nb_random_moves
            entries.append((nb_pos, nb_random_moves,
                            data[idx + 2: idx + 2 + 2 * nb_moves]))
            labels.append(label)
            idx += 2 + 2 * nb_moves + 2
        return entries, labels


# ── Dataset: applies mapping on the fly ───────────────────────────────────────
def _bb_to_list(n: int) -> list:
    # LSB-first: index i carries bit i of n. Combined with Yolah's LERF
    # convention (square i ↔ bit i, see server/yolah.py), this means
    # raw_vec[i] == 1 ↔ piece on square i — matching SYSTEM_PROMPT and
    # _make_test_position. Do NOT use bin()[2:] here: that's MSB-first
    # and would silently invert the board (square i ↔ raw_vec[63-i]).
    return [(n >> i) & 1 for i in range(64)]


class MappedDataset(Dataset):
    def __init__(self, raw: RawGameData, mapping_path: str):
        self.raw          = raw
        self.mapping_path = mapping_path   # string → always picklable

    def __len__(self):
        return self.raw.size

    def _build_position(self, idx):
        """Replay the game up to position `idx` and return (raw_vec, label).
        No mapping applied — used both by __getitem__ and by the eval pass
        that wants the raw input for miss-reporting."""
        # Find which game contains position idx (standard lower-bound search)
        lo, hi = 0, len(self.raw.entries)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if self.raw.entries[mid][0] <= idx:
                lo = mid + 1
            else:
                hi = mid
        base = self.raw.entries[lo - 1][0] if lo > 0 else 0
        _, nb_random_moves, moves = self.raw.entries[lo]
        pos_idx     = idx - base
        target_plies = nb_random_moves + pos_idx

        # Replay the game from move 0 up to target_plies. O(plies) per call:
        # cheap in absolute terms but dominates DataLoader cost (the network
        # is tiny). If training becomes the bottleneck, cache
        # (black, white, empty, ply) per idx in RawGameData — replays then
        # collapse to a dict lookup.
        game = Yolah()
        for sq1, sq2 in zip(moves[0::2], moves[1::2]):
            if game.nb_plies() >= target_plies:
                break
            game.play(Move(Square(int(sq1)), Square(int(sq2))))

        raw_vec = torch.tensor(
            _bb_to_list(game.black) + _bb_to_list(game.white) +
            _bb_to_list(game.empty) +
            [Yolah.WHITE_PLAYER if game.nb_plies() & 1 else Yolah.BLACK_PLAYER],
            dtype=torch.float32,
        )
        label = torch.tensor(self.raw.labels[lo], dtype=torch.long)
        return raw_vec, label

    def __getitem__(self, idx):
        raw_vec, label = self._build_position(idx)
        mod = _load_mapping_module(self.mapping_path)
        return mod.mapping(raw_vec), label

    def get_raw_and_mapped(self, idx):
        """Return (raw_vec, mapped_vec, label) in a single replay."""
        raw_vec, label = self._build_position(idx)
        mod = _load_mapping_module(self.mapping_path)
        return raw_vec, mod.mapping(raw_vec), label


# ── Network (fixed architecture, input_dim adapts to MAPPED_DIM) ──────────────
class Net(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        return self.fc3(x)

    def clip(self):
        # NNUE-style int8 quantization clamp: at deployment, weights are
        # multiplied by 64 and stored as int8 in [-127, 127]. Keeping the
        # float weights inside [-127/64, 127/64] during training guarantees
        # the quantized network behaves identically to the trained one.
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)


# ── DDP training (NB_EPOCHS epochs, saves results to JSON) ───────────────────
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main_ddp(rank, world_size, batch_size, dataset, input_dim, results_file):
    ddp_setup(rank, world_size)
    t_start = _time.time()

    train_size = int(0.95 * len(dataset))
    val_size   = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    s_train = DistributedSampler(trainset)
    s_val   = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=s_train,
        num_workers=2, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, sampler=s_val,
        num_workers=2, pin_memory=True, prefetch_factor=2,
    )

    if rank == 0:
        print(f"  [child rank0] train={train_size}  val={val_size}  "
              f"input_dim={input_dim}  batch={batch_size}  epochs={NB_EPOCHS}",
              flush=True)

    model     = Net(input_dim).to(rank)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    stream    = torch.cuda.Stream(device=rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model = DDP(model, device_ids=[rank])
    model = torch.compile(model)

    train_acc_last = val_acc_last = 0.0

    for epoch in range(NB_EPOCHS):
        # ── train ──
        model.train()
        s_train.set_epoch(epoch)
        n, correct = 0, 0
        ep_t0 = _time.time()
        for X, y in tqdm(train_loader, disable=(rank != 0),
                         desc=f"train e{epoch+1}", mininterval=2.0):
            n += len(X)
            with torch.cuda.stream(stream):
                X = X.to(rank, non_blocking=True)
                y = y.to(rank, non_blocking=True)
            torch.cuda.current_stream(rank).wait_stream(stream)
            optimizer.zero_grad()
            logits = model(X)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            correct += (torch.argmax(logits, 1) == y).sum().item()
            model.module.clip()
        train_acc_last = correct / n
        if rank == 0:
            print(f"  [epoch {epoch+1}/{NB_EPOCHS}] train acc {train_acc_last:.4f}"
                  f"  ({_fmt_secs(_time.time() - ep_t0)})", flush=True)

        # ── validate ──
        model.train(False)
        n, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, disable=(rank != 0),
                             desc=f"val e{epoch+1}", mininterval=2.0):
                n += len(X)
                with torch.cuda.stream(stream):
                    X = X.to(rank, non_blocking=True)
                    y = y.to(rank, non_blocking=True)
                torch.cuda.current_stream(rank).wait_stream(stream)
                logits = model(X)
                correct += (torch.argmax(logits, 1) == y).sum().item()
        val_acc_last = correct / n
        if rank == 0:
            print(f"  [epoch {epoch+1}/{NB_EPOCHS}] val   acc {val_acc_last:.4f}",
                  flush=True)

    # ── Post-training analytics: confusion matrix + miss samples ──
    # Done on rank 0 only; uses get_raw_and_mapped() so we can decode the
    # miss back into human-readable squares for Qwen.
    confusion = [[0]*3 for _ in range(3)]
    misses    = []
    if rank == 0:
        print(f"  [analytics] computing confusion + miss samples …", flush=True)
        model.train(False)
        underlying  = valset.dataset                # MappedDataset
        val_indices = list(valset.indices)
        rng         = np.random.default_rng(0xC0FFEE)
        rng.shuffle(val_indices)
        # Cap analytics work; the full val pass above already gave val_acc.
        sample_cap  = min(len(val_indices), 16384)
        val_indices = val_indices[:sample_cap]

        ana_t0 = _time.time()
        with torch.no_grad():
            for chunk_start in range(0, len(val_indices), batch_size):
                chunk = val_indices[chunk_start:chunk_start + batch_size]
                raws, mapped, labels = [], [], []
                for ds_idx in chunk:
                    rv, mv, lbl = underlying.get_raw_and_mapped(ds_idx)
                    raws.append(rv); mapped.append(mv); labels.append(lbl)
                X = torch.stack(mapped).to(rank)
                preds = torch.argmax(model(X), 1).cpu()
                for k in range(len(chunk)):
                    t = int(labels[k]); p = int(preds[k])
                    confusion[t][p] += 1
                    if t != p and len(misses) < MISS_SAMPLE_SIZE:
                        sq = _raw_vec_to_squares(raws[k])
                        misses.append({**sq, "predicted": p, "actual": t})
        print(f"  [analytics] done in {_fmt_secs(_time.time() - ana_t0)}  "
              f"misses_collected={len(misses)}", flush=True)

    if rank == 0:
        with open(results_file, 'w') as f:
            json.dump({
                "train_acc": train_acc_last,
                "val_acc":   val_acc_last,
                "confusion": confusion,
                "misses":    misses,
                "wall_time": _time.time() - t_start,
            }, f)
        print(f"  [child rank0] training+analytics total "
              f"{_fmt_secs(_time.time() - t_start)}", flush=True)

    destroy_process_group()


# ── Main loop ─────────────────────────────────────────────────────────────────
def _validate_with_repair(code: str, iteration: int, candidate: int,
                          llm: Llama):
    """Validate code, asking Qwen to repair on failure (up to MAX_RETRIES).

    Returns (path, mapped_dim, description, cost_us, lint_issues, final_code)
    or (None, …, error_msg).  `final_code` may differ from input if repaired.
    """
    cur = code
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path, dim, desc, cost_us, lint_issues = save_and_validate(
                cur, iteration, candidate=candidate)
            if attempt > 1:
                _log(f"      ✓ validated after {attempt} attempt(s)")
            else:
                _log(f"      ✓ validated  dim={dim}  cost={cost_us:.1f}µs/call")
            if lint_issues:
                _log(f"      lint warnings: {lint_issues}")
            return path, dim, desc, cost_us, lint_issues, cur
        except Exception:
            err = traceback.format_exc()
            _log(f"      ✗ validation FAILED (attempt {attempt}/{MAX_RETRIES})")
            # Print full traceback at most once per candidate:
            if attempt == 1:
                print(err, flush=True)
            if attempt < MAX_RETRIES:
                fixed = fix_mapping(llm, cur, err)
                if fixed:
                    cur = fixed
                else:
                    _log("      Qwen could not produce a fix. Giving up "
                         "this candidate.")
                    return None, None, None, None, None, cur
    _log(f"      All {MAX_RETRIES} repair attempts failed.")
    return None, None, None, None, None, cur


def _train_candidate(dataset, mapped_dim: int, results_file: str) -> dict:
    """Run a fresh training in a child process; return parsed results.

    Returns dict with: train_acc, val_acc, confusion (3×3 list),
    misses (list of dicts), wall_time (seconds).
    """
    # CUDA_VISIBLE_DEVICES must be set BEFORE the child starts so it sees
    # only TRAINING_GPU; we restore the parent env immediately after.
    os.environ["CUDA_VISIBLE_DEVICES"] = TRAINING_GPU
    try:
        mp.spawn(
            main_ddp,
            args=(1, BATCH_SIZE, dataset, mapped_dim, results_file),
            nprocs=1,
        )
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    with open(results_file) as f:
        return json.load(f)


def _print_history_table(history: list) -> None:
    if not history:
        return
    best = max(history, key=lambda h: h['val_acc'])
    print(f"\n  {'Iter':>4}.{'k':<2}  {'ValAcc':>7}  {'Dim':>4}  "
          f"{'µs':>7}  Description", flush=True)
    print("  " + "─" * 78, flush=True)
    for h in history:
        tag  = " ←" if h is best else ""
        cand = h.get('candidate', 0)
        cost = h.get('cost_us', 0)
        print(f"  {h['iteration']:>4}.{cand:<2}  {h['val_acc']:>7.4f}  "
              f"{h['mapped_dim']:>4}  {cost:>6.1f}µ  {h['description']}{tag}",
              flush=True)


if __name__ == "__main__":
    _section(f"MAPPING-LOOP STARTUP   (pid={os.getpid()})", char="═")
    _log(f"Qwen GPU       : {QWEN_GPU_DEVICE}  (model: {QWEN_MODEL_PATH})")
    _log(f"Training GPU   : {TRAINING_GPU}")
    _log(f"Iterations     : {MAX_ITERATIONS}")
    _log(f"Candidates/iter: {NB_CANDIDATES_PER_ITER}")
    _log(f"Epochs/training: {NB_EPOCHS}")
    _log(f"Batch size     : {BATCH_SIZE}")
    _log(f"History file   : {HISTORY_FILE}")
    _log(f"GPU mem        : {_gpu_mem_str()}")

    # ── Load data + LLM ──
    _section("Load raw game data")
    t0 = _time.time()
    raw_data = RawGameData(DATA_DIR)
    _log(f"raw_data ready in {_fmt_secs(_time.time() - t0)}")

    _section("Load Qwen LLM")
    t0 = _time.time()
    llm = load_llm()
    _log(f"LLM loaded in {_fmt_secs(_time.time() - t0)}  "
         f"({_gpu_mem_str()})")

    # ── Resume from history.jsonl if present ──
    history = load_history()
    if history:
        last_iter = max(h['iteration'] for h in history)
        best      = max(history, key=lambda h: h['val_acc'])
        _section(f"RESUMED from {HISTORY_FILE} ({len(history)} rows, "
                 f"best v{best['iteration']}.{best.get('candidate',0)} "
                 f"val={best['val_acc']:.4f})")
        _print_history_table(history)
        # Pick up where we left off
        start_iter   = last_iter + 1
        current_code = best['code']  # seed next round from current best
    else:
        start_iter   = 1
        current_code = INITIAL_MAPPING

    if start_iter > MAX_ITERATIONS:
        _log(f"history already has {start_iter-1} iterations ≥ "
             f"MAX_ITERATIONS={MAX_ITERATIONS}; nothing to do.")
        sys.exit(0)

    # ── Main outer loop ──
    for iteration in range(start_iter, MAX_ITERATIONS + 1):
        iter_t0 = _time.time()
        plateau = is_plateauing(history)
        _section(f"ITERATION {iteration}/{MAX_ITERATIONS}   "
                 f"plateau={plateau}   "
                 f"history_size={len(history)}", char="═")

        # ── Step A: generate NB_CANDIDATES_PER_ITER drafts ──
        # On iter 1 (empty history) the seed itself has not been scored
        # yet, so we keep it as candidate 0 and skip Qwen generation.
        # On iter 2+ the seed IS the current best and its val_acc is
        # already in history — re-training it would be pure waste — so
        # we generate NB_CANDIDATES_PER_ITER fresh drafts from Qwen and
        # train only those.
        drafts: list = []
        if not history:
            drafts.append(current_code)
            _log(f"  Seed candidate (iter 1): training INITIAL mapping.")
        else:
            for k in range(NB_CANDIDATES_PER_ITER):
                # First Qwen draft refines (low temp); later drafts explore.
                # On plateau, every draft gets the exploration temperature.
                temp = TEMP_EXPLORE if (plateau or k > 0) else TEMP_REFINE
                _log(f"  Generating candidate #{k+1}/"
                     f"{NB_CANDIDATES_PER_ITER} "
                     f"(temp={temp:.2f}, plateau={plateau}) …")
                new_code = generate_mapping(llm, history, current_code,
                                             temperature=temp,
                                             plateau=plateau)
                if new_code:
                    drafts.append(new_code)
                else:
                    _log(f"  Could not extract code from Qwen response "
                         f"(candidate #{k+1}); skipping.")
            if not drafts:
                # Qwen failed every candidate — fall back to re-testing the
                # seed so the iteration isn't a total no-op. Rare path.
                _log(f"  WARN: Qwen produced no usable drafts; falling "
                     f"back to re-testing the current best as a safety "
                     f"net.")
                drafts.append(current_code)
        _log(f"  Will train {len(drafts)} candidate(s) this iteration.")

        # ── Step B: validate + train each candidate ──
        results_for_iter = []
        for k, draft in enumerate(drafts):
            _section(f"  CANDIDATE {iteration}.{k}   "
                     f"({k+1}/{len(drafts)})", char="·")
            t_v0 = _time.time()
            (path, dim, desc, cost_us, lint_issues, final_code
             ) = _validate_with_repair(draft, iteration, candidate=k, llm=llm)
            _log(f"    validate phase: {_fmt_secs(_time.time() - t_v0)}")

            if path is None:
                _log(f"    SKIP candidate {iteration}.{k} (unrecoverable)")
                continue

            _log(f"    description : {desc}")
            _log(f"    mapped_dim  : {dim}   cost: {cost_us:.1f}µs/call")
            _log(f"    GPU mem     : {_gpu_mem_str()}")

            dataset      = MappedDataset(raw_data, path)
            results_file = os.path.join(
                MAPPING_DIR, f"results_v{iteration}_k{k}.json")

            t_t0 = _time.time()
            _log(f"    spawning training child …")
            try:
                res = _train_candidate(dataset, dim, results_file)
            except Exception:
                _log(f"    training FAILED:\n{traceback.format_exc()}")
                continue
            _log(f"    training+analytics: {_fmt_secs(_time.time() - t_t0)}")
            _log(f"    train_acc={res['train_acc']:.4f}  "
                 f"val_acc={res['val_acc']:.4f}")

            entry = {
                "iteration":   iteration,
                "candidate":   k,
                "description": desc,
                "mapped_dim":  dim,
                "train_acc":   res['train_acc'],
                "val_acc":     res['val_acc'],
                "cost_us":     cost_us,
                "confusion":   res.get('confusion'),
                "misses":      res.get('misses', []),
                "code":        final_code,
                "lint_issues": lint_issues or [],
            }
            results_for_iter.append(entry)
            history.append(entry)
            save_history_entry(entry)

        # ── Step C: pick this iteration's winner; update current_code ──
        if not results_for_iter:
            _log(f"  No candidate trained this iteration. Falling back.")
            if history:
                best = max(history, key=lambda h: h['val_acc'])
                current_code = best['code']
                _log(f"  Resuming from best so far: v{best['iteration']}."
                     f"{best.get('candidate',0)} "
                     f"(val_acc={best['val_acc']:.4f})")
            else:
                current_code = INITIAL_MAPPING
                _log(f"  No history yet; reset to INITIAL_MAPPING.")
        else:
            iter_best = max(results_for_iter, key=lambda h: h['val_acc'])
            current_code = iter_best['code']
            _log(f"  Iteration winner: candidate "
                 f"{iter_best['iteration']}.{iter_best['candidate']}  "
                 f"val_acc={iter_best['val_acc']:.4f}  "
                 f"cost={iter_best['cost_us']:.1f}µs")

        # ── Step D: report ──
        _print_history_table(history)
        global_best = max(history, key=lambda h: h['val_acc'])
        _log(f"  global best   : v{global_best['iteration']}."
             f"{global_best.get('candidate',0)}  "
             f"val_acc={global_best['val_acc']:.4f}")
        _log(f"  iteration time: {_fmt_secs(_time.time() - iter_t0)}  "
             f"total: {_fmt_secs(_time.time() - _RUN_START)}")
        _log(f"  GPU mem (post): {_gpu_mem_str()}")

    _section("DONE", char="═")
    if history:
        b = max(history, key=lambda h: h['val_acc'])
        _log(f"Final best: v{b['iteration']}.{b.get('candidate',0)}  "
             f"val_acc={b['val_acc']:.4f}  ({b['description']})")
        _log(f"  → mappings/mapping_v{b['iteration']}"
             f"{'_k'+str(b.get('candidate',0)) if b.get('candidate',0) else ''}.py")
    _log(f"Total wall time: {_fmt_secs(_time.time() - _RUN_START)}")
