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
  Linear(MAPPED_DIM, 128) → clamp(0,1) → Linear(64, 32) → clamp(0,1) → Linear(32, 3)
"""

import json
import re
import importlib.util
import os
import sys
import glob
import itertools
import traceback

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
# Qwen3-Coder runs via llama-cpp-python on GPU QWEN_GPU_DEVICE (in-process,
# no server needed — works natively in Singularity with --nv).
# Download the GGUF model once with:
#   huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF \
#       --include "qwen3-coder-30b-a3b-instruct-q4_k_m.gguf" --local-dir ./models
QWEN_MODEL_PATH = "./models/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf"
QWEN_GPU_DEVICE = 0    # GPU index for Qwen (0 = first GPU allocated by SLURM)
TRAINING_GPU    = "1"  # passed to CUDA_VISIBLE_DEVICES for training
DATA_DIR       = "./data"
MAPPING_DIR    = "./mappings"
MASTER_PORT    = "65435"
MAX_ITERATIONS = 20
MAX_RETRIES    = 10      # times to ask Qwen to fix a broken mapping before giving up
NB_EPOCHS      = 2
BATCH_SIZE     = 512 * 2

os.makedirs(MAPPING_DIR, exist_ok=True)

# ── Initial mapping (iteration 1 seed) ────────────────────────────────────────
INITIAL_MAPPING = r'''
import numpy as np
import torch

_REGION_HEX = [
    0x00003C3C3C3C0000, 0x0000000000000303, 0x000000000000C0C0,
    0xC0C0000000000000, 0x0303000000000000, 0x0000030303030000,
    0x0000C0C0C0C00000, 0x0000000000003C3C, 0x3C3C000000000000,
]
REGION_MASKS = np.array(
    [[(v >> i) & 1 for i in range(64)] for v in _REGION_HEX], dtype=np.float32
)
_TRIU = np.triu_indices(4, k=1)

MAPPED_DIM  = 63
DESCRIPTION = ("sorted piece coords (16) + within-color pairwise sq-dist (12)"
               " + cross-color sq-dist (16) + regional counts (18) + turn (1)")

def mapping(board_tensor):
    x    = board_tensor.numpy()
    blk  = x[:64];  wht = x[64:128];  turn = x[192:193]
    b_idx = np.where(blk > 0.5)[0]
    w_idx = np.where(wht > 0.5)[0]
    b_rc  = np.stack([b_idx // 8, b_idx % 8], 1).astype(np.float32) / 7.0
    w_rc  = np.stack([w_idx // 8, w_idx % 8], 1).astype(np.float32) / 7.0
    b_d2  = np.sum((b_rc[:,None]-b_rc[None,:])**2, 2)[_TRIU] / 2.0
    w_d2  = np.sum((w_rc[:,None]-w_rc[None,:])**2, 2)[_TRIU] / 2.0
    c_d2  = np.sum((b_rc[:,None]-w_rc[None,:])**2, 2).flatten() / 2.0
    b_reg = REGION_MASKS @ blk / 4.0
    w_reg = REGION_MASKS @ wht / 4.0
    return torch.tensor(
        np.concatenate([b_rc.flatten(), w_rc.flatten(),
                        b_d2, w_d2, c_d2, b_reg, w_reg, turn]),
        dtype=torch.float32)
'''

# ── Game example (loaded from file for Qwen context) ─────────────────────────
GAME_EXAMPLE_PATH = "./game_example.txt"


def _load_game_example() -> str:
    if os.path.isfile(GAME_EXAMPLE_PATH):
        with open(GAME_EXAMPLE_PATH) as f:
            return f.read()
    return ""


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
- Return the complete module as a SINGLE ```python ... ``` code block.
- Do NOT include any training or testing code.
- The mapping function must be DETERMINISTIC (same input → same output).
- All values in the output tensor must be finite (no NaN, no Inf).
"""

# ── Qwen helpers (llama-cpp-python, in-process, no server needed) ─────────────
def load_llm() -> Llama:
    print(f"Loading {QWEN_MODEL_PATH} on GPU {QWEN_GPU_DEVICE} …", flush=True)
    return Llama(
        model_path=QWEN_MODEL_PATH,
        n_gpu_layers=-1,      # all layers on GPU
        main_gpu=QWEN_GPU_DEVICE,
        n_ctx=8192,
        verbose=False,
    )


def _llm_call(llm: Llama, messages: list, temperature: float,
              max_tokens: int = 2048) -> str:
    """
    Run a chat completion and return the response text.
    Strips Qwen3 <think>...</think> blocks (thinking mode is on by default
    for Qwen3-Coder instruct GGUF models).
    """
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response["choices"][0]["message"]["content"]
    # Remove thinking block if present (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def _build_user_prompt(history: list, current_code: str) -> str:
    lines = []

    # Include game example so Qwen can see how a real game unfolds
    game_example = _load_game_example()
    if game_example:
        lines.append("EXAMPLE GAME (showing how positions evolve, with bitboards and scores):\n")
        # Include a meaningful excerpt (first ~30 moves) to keep prompt manageable
        example_lines = game_example.strip().split('\n')
        # Show about 20 moves worth of data (~ first 120 lines)
        excerpt = '\n'.join(example_lines[:120])
        lines.append(f"```\n{excerpt}\n```\n")
        lines.append(
            "Each move makes the origin square impassable. Scores increment by 1 per move.\n"
            "Notice how the impassable bitboard grows and constrains piece movement.\n"
        )

    if history:
        lines.append("History of mapping attempts (sorted by iteration):\n")
        lines.append(f"{'Iter':>4}  {'ValAcc':>7}  {'Dim':>5}  Description")
        lines.append("-" * 78)
        for h in history:
            marker = " ← best" if h == max(history, key=lambda x: x['val_acc']) else ""
            lines.append(f"{h['iteration']:>4}  {h['val_acc']:>7.4f}"
                         f"  {h['mapped_dim']:>5}  {h['description']}{marker}")
        best = max(history, key=lambda h: h['val_acc'])
        lines.append(f"\nCode of best mapping (v{best['iteration']}, "
                     f"val_acc={best['val_acc']:.4f}):\n```python\n{best['code']}\n```\n")
    else:
        lines.append("This is the first iteration. Seed mapping:\n"
                     f"```python\n{current_code}\n```\n")

    lines.append(
        "Design a NEW mapping that you expect to outperform the best so far.\n"
        "Think about what geometric, mobility, or structural features distinguish "
        "winning from losing positions in Yolah. Consider:\n"
        "- Mobility: how many free squares each piece can reach (sliding in 8 dirs)\n"
        "- Territory: how much connected free space each player's pieces control\n"
        "- Score difference and remaining free squares (future scoring potential)\n"
        "- Piece clustering vs. spread (boxed-in pieces lose)\n\n"
        "The mapping will be translated to C++ (bitboard ops) for a minmax\n"
        "search, so keep the algorithm efficiently implementable.\n"
        "Return the complete module in a single ```python``` block."
    )
    return "\n".join(lines)


def _extract_code(text: str) -> str | None:
    for pattern in [r'```python\n(.*?)```', r'```\n(.*?)```']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def generate_mapping(llm: Llama, history: list, current_code: str) -> str | None:
    text = _llm_call(
        llm,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(history, current_code)},
        ],
        temperature=0.7,
    )
    print("\n── Qwen response (first 600 chars) ──")
    print(text[:600])
    print("─────────────────────────────────────\n", flush=True)
    return _extract_code(text)


def fix_mapping(llm: Llama, bad_code: str, error: str) -> str | None:
    """Ask Qwen to fix broken code, providing the exact traceback."""
    print("Asking Qwen to fix the broken mapping …", flush=True)
    text = _llm_call(
        llm,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The following mapping module has a bug:\n"
                f"```python\n{bad_code}\n```\n\n"
                f"Running it produced this error:\n"
                f"```\n{error}\n```\n\n"
                f"Fix the bug and return the corrected complete module "
                f"in a single ```python``` code block. "
                f"Do not change the overall approach, just fix the error."
            )},
        ],
        temperature=0.2,
    )
    return _extract_code(text)


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


def save_and_validate(code: str, iteration: int):
    """
    Write code to MAPPING_DIR/mapping_vN.py, load it, smoke-test it on
    multiple realistic positions.
    Returns (path, MAPPED_DIM, DESCRIPTION) or raises on failure.
    """
    path = os.path.join(MAPPING_DIR, f"mapping_v{iteration}.py")
    with open(path, 'w') as f:
        f.write(code)

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

    return path, int(mod.MAPPED_DIM), str(mod.DESCRIPTION)


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
        self.size = self.entries[-1][0] if self.entries else 0
        print(f"Loaded {self.size} positions from {len(files)} game files.", flush=True)

    @staticmethod
    def _load_file(filename):
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
    b = [int(d) for d in bin(n)[2:]]
    return [0] * (64 - len(b)) + b


class MappedDataset(Dataset):
    def __init__(self, raw: RawGameData, mapping_path: str):
        self.raw          = raw
        self.mapping_path = mapping_path   # string → always picklable

    def __len__(self):
        return self.raw.size

    def __getitem__(self, idx):
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

        # Replay moves up to target_plies
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
        mod = _load_mapping_module(self.mapping_path)
        return mod.mapping(raw_vec), torch.tensor(self.raw.labels[lo], dtype=torch.long)


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
        num_workers=0, pin_memory=True,
    )

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
        for X, y in tqdm(train_loader, disable=(rank != 0)):
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
            print(f"  epoch {epoch+1}/{NB_EPOCHS}  train acc: {train_acc_last:.4f}",
                  flush=True)

        # ── validate ──
        model.train(False)
        n, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, disable=(rank != 0)):
                n += len(X)
                with torch.cuda.stream(stream):
                    X = X.to(rank, non_blocking=True)
                    y = y.to(rank, non_blocking=True)
                torch.cuda.current_stream(rank).wait_stream(stream)
                logits = model(X)
                correct += (torch.argmax(logits, 1) == y).sum().item()
        val_acc_last = correct / n
        if rank == 0:
            print(f"  epoch {epoch+1}/{NB_EPOCHS}  val   acc: {val_acc_last:.4f}",
                  flush=True)

    if rank == 0:
        with open(results_file, 'w') as f:
            json.dump({"train_acc": train_acc_last, "val_acc": val_acc_last}, f)

    destroy_process_group()


# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # GPU layout: QWEN_GPU runs the ollama server (started externally),
    # TRAINING_GPU is used exclusively for training.
    print(f"Qwen GPU: {QWEN_GPU_DEVICE}  Training GPU: {TRAINING_GPU}", flush=True)

    raw_data = RawGameData(DATA_DIR)
    llm      = load_llm()

    history      = []
    current_code = INITIAL_MAPPING  # Qwen's working draft — always updated,
                                    # even when broken, so Qwen retains context

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"  Iteration {iteration}/{MAX_ITERATIONS}", flush=True)
        print(f"{'='*60}", flush=True)

        # 1. Validate current_code (with Qwen-assisted repair on failure).
        #    training_code is what actually gets trained — falls back to best
        #    on unrecoverable failure, but current_code (Qwen's draft) is
        #    always kept so Qwen can continue refining its own direction.
        candidate_code = current_code
        mapping_path = mapped_dim = description = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                mapping_path, mapped_dim, description = \
                    save_and_validate(candidate_code, iteration)
                current_code = candidate_code   # adopt repaired version if fixed
                break
            except Exception:
                error_msg = traceback.format_exc()
                print(f"Mapping validation FAILED (attempt {attempt}/{MAX_RETRIES}):",
                      flush=True)
                print(error_msg, flush=True)
                if attempt < MAX_RETRIES:
                    fixed = fix_mapping(llm, candidate_code, error_msg)
                    if fixed:
                        candidate_code = fixed
                    else:
                        print("Qwen could not produce a fix. Giving up.", flush=True)
                        break
                else:
                    print(f"All {MAX_RETRIES} attempts failed.", flush=True)

        if mapping_path is None:
            print("Skipping training this iteration. "
                  "Qwen will retry from its current draft next iteration.", flush=True)
            continue

        print(f"  Mapping dim  : {mapped_dim}", flush=True)
        print(f"  Description  : {description}", flush=True)
        print(f"\n── Mapping code (v{iteration}) ──────────────────────────────")
        print(current_code)
        print("─────────────────────────────────────────────────────────\n",
              flush=True)

        # 2. Build dataset with the validated mapping
        dataset      = MappedDataset(raw_data, mapping_path)
        results_file = os.path.join(MAPPING_DIR, f"results_v{iteration}.json")

        # 3. Training on TRAINING_GPU only (QWEN_GPU is reserved for ollama)
        os.environ["CUDA_VISIBLE_DEVICES"] = TRAINING_GPU
        mp.spawn(
            main_ddp,
            args=(1, BATCH_SIZE, dataset, mapped_dim, results_file),
            nprocs=1,
        )
        del os.environ["CUDA_VISIBLE_DEVICES"]

        # 4. Read results written by rank 0
        with open(results_file) as f:
            res = json.load(f)
        train_acc, val_acc = res["train_acc"], res["val_acc"]

        # 5. Record in history
        history.append({
            "iteration":   iteration,
            "description": description,
            "mapped_dim":  mapped_dim,
            "train_acc":   train_acc,
            "val_acc":     val_acc,
            "code":        current_code,
        })

        # 6. Print history table
        print(f"\n{'─'*60}")
        print(f"  {'Iter':>4}  {'ValAcc':>7}  {'Dim':>5}  Description")
        print(f"  {'─'*56}")
        for h in history:
            tag = " ←" if h == max(history, key=lambda x: x['val_acc']) else ""
            print(f"  {h['iteration']:>4}  {h['val_acc']:>7.4f}"
                  f"  {h['mapped_dim']:>5}  {h['description']}{tag}")
        print(f"{'─'*60}", flush=True)

        # 7. Ask Qwen to design a better mapping
        print("\nGenerating new mapping with Qwen …", flush=True)
        new_code = generate_mapping(llm, history, current_code)

        if new_code:
            current_code = new_code
            print("New mapping code received.", flush=True)
        else:
            print("Could not extract code from Qwen response. "
                  "Keeping current mapping.", flush=True)
