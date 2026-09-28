"""
alphazero_learn.py — AlphaZero / KataGo-style reinforcement learning for the
value/policy ResNet of the AlphaZero MCTS player.

One process drives the whole loop on one node (see alphazero_learn.sh):

  ┌──────────────┐ samples  ┌──────────────────────┐ export  ┌────────────────┐
  │ self-play    │────────▶ │ trainer (this file)  │───────▶ │ models/az_*.ts │
  │ C++, 1/GPU   │  .bin    │ replay window, SGD   │         │ latest.json    │
  └──────────────┘          └──────────────────────┘         └───────┬────────┘
        ▲  hot-swaps the newest network (no gating, like KataGo)     │
        └────────────────────────────────────────────────────────────┘
                              every --eval-hours (or --eval-every exports):
                              `alphazero_learn match` new vs old (10 games,
                              2 s per move, colours alternated) → evals.csv

Everything lives in the work directory and survives a job that is killed:

  <work>/checkpoint.pt      trainer state (weights, EMA, optimizer, counters)
  <work>/state.json         human-readable counters + evaluation state (Elo)
  <work>/latest.json        {"step": s, "ts": "models/az_<s>.ts"} → self-play
  <work>/models/            az_<step>.pt (plain state_dict, resnet_export.py
                            can read it) and az_<step>.ts (TorchScript)
  <work>/selfplay/*.bin     training samples (format: player/alphazero_selfplay.h)
  <work>/evals.csv          one line per evaluation match
  <work>/evals/*.json       full match records (every move of every game)
  <work>/train_log.csv      losses, throughput
  <work>/logs/              self-play / match output

Resubmitting the job with the same work directory continues where it stopped.

What comes from KataGo (github.com/lightvector/KataGo, python/train.py,
python/shuffle.py and the paper "Accelerating Self-Play Learning in Go"):
  • playout cap randomization, forced playouts + policy target pruning and
    graph search on the self-play side (C++, alphazero_mcts.cpp);
  • only the full searches become training rows;
  • the replay window grows sub-linearly with the data generated,
        window = N0 · (1 + β·((N/N0)^α − 1)/α),  α = 0.75, β = 0.4
    (shuffle.py), so early, weak data ages out quickly;
  • a fixed ratio of training samples per generated row (--reuse);
  • the exported network is an average of recent weights (KataGo uses SWA
    snapshots; here an exponential moving average, BatchNorm statistics
    included);
  • no gating: self-play always uses the newest network. The evaluation
    matches only monitor progress.
  • the 8 symmetries of the board as data augmentation.
  • optionally (--aux, alphazero_aux.py), KataGo's auxiliary heads, adapted
    to Yolah: future ownership of each square and the rest of the score
    margin. They exist only in training; the exported network is the plain
    two-headed one, so the C++ side is unchanged. Without --aux, nothing of
    this runs and the files keep their original format.
Not ported: the opponent-policy head and KataGo's short-term value targets.

Usage (from nnue/, inside the .sif or locally):

  python3 alphazero_learn.py --work /work --init-model cnn_resnet_256x30_value_policy.pt \
          --bin ../build/alphazero_learn
"""
import argparse
import copy
import csv
import glob
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../server"))
sys.path.append(HERE)
from cnn_resnet_value_policy_chunked import Net  # noqa: E402

# ── sample format: MUST match az::TrainingSample (player/alphazero_selfplay.h) ──
MAX_NB_MOVES = 75
SAMPLE_DTYPE = np.dtype([
    ("black", "<u8"), ("white", "<u8"), ("empty", "<u8"),
    ("turn", "u1"), ("z", "i1"), ("nb_moves", "u1"), ("flags", "u1"),
    ("root_q", "<f4"), ("model_step", "<u4"),
    ("action", "<u2", (MAX_NB_MOVES,)), ("prob", "<u2", (MAX_NB_MOVES,)),
])
assert SAMPLE_DTYPE.itemsize == 336
# Version-2 rows (--aux, az::TrainingSampleAux): the same fields, then the
# future ownership map, 2 bits per square (see alphazero_aux.py).
SAMPLE_DTYPE_AUX = np.dtype(SAMPLE_DTYPE.descr + [("own", "u1", (16,))])
assert SAMPLE_DTYPE_AUX.itemsize == 352
ROW_DTYPES = {1: SAMPLE_DTYPE, 2: SAMPLE_DTYPE_AUX}      # file version → row
HEADER_DTYPE = np.dtype([("magic", "S8"), ("version", "<u4"), ("sample_size", "<u4"),
                         ("nb_samples", "<u4"), ("nb_games", "<u4")])
assert HEADER_DTYPE.itemsize == 24


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def read_header(path):
    """(rows, games, version) of a self-play file; version 1 or 2 (--aux rows)."""
    h = np.fromfile(path, dtype=HEADER_DTYPE, count=1)
    if len(h) != 1 or h[0]["magic"] != b"YOLAHSP1":
        raise ValueError(f"{path}: not a self-play sample file")
    version = int(h[0]["version"])
    if version not in ROW_DTYPES or h[0]["sample_size"] != ROW_DTYPES[version].itemsize:
        raise ValueError(f"{path}: unknown row format (version {version})")
    return int(h[0]["nb_samples"]), int(h[0]["nb_games"]), version


def read_samples(path, dtype=SAMPLE_DTYPE):
    """
    The rows of a file, converted to `dtype`. Both formats mix freely:
      • version-2 rows read as SAMPLE_DTYPE drop their ownership map;
      • version-1 rows read as SAMPLE_DTYPE_AUX get a map of 0xFF bytes, i.e.
        every square OWN_PAST: no auxiliary target, masked in the loss.
    """
    n, _, version = read_header(path)
    rows = np.fromfile(path, dtype=ROW_DTYPES[version], count=n, offset=HEADER_DTYPE.itemsize)
    if rows.dtype == dtype:
        return rows
    out = np.empty(n, dtype=dtype)
    for name in SAMPLE_DTYPE.names:
        out[name] = rows[name]
    if "own" in dtype.names:
        out["own"] = 0xFF
    return out


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ── symmetries ───────────────────────────────────────────────────────────────
def symmetry_tables():
    """
    The 8 symmetries of the board (same group as distill_common.symmetries).

    Returns
      src (8, 64): new plane cell j takes the value of old cell src[s, j]
      act (8, 4096): policy index of the image of move index a under symmetry s
    Plane cell c holds square 63 - c (see nn_evaluator.h), so a square maps
    through cells: sq' = 63 - inv[63 - sq]. Index 0 (a1a1) is the pass and
    stays the pass.
    """
    cells = torch.arange(64).view(8, 8)
    srcs = []
    for k in range(4):
        r = torch.rot90(cells, k, dims=(0, 1))
        srcs.append(r.flatten())
        srcs.append(torch.flip(r, dims=(1,)).flatten())
    src = torch.stack(srcs)
    inv = torch.argsort(src, dim=1)
    sq = 63 - inv[:, 63 - torch.arange(64)]
    act = (sq[:, :, None] * 64 + sq[:, None, :]).reshape(8, 4096).clone()
    act[:, 0] = 0
    return src, act


# ── replay window ────────────────────────────────────────────────────────────
class ReplayWindow:
    """
    Ring buffer of the most recent rows, filled from <work>/selfplay/*.bin in
    file-name (= creation time) order. Sampling is uniform over the last
    window_size(total) rows.
    """
    def __init__(self, directory, capacity, min_rows, alpha=0.75, beta=0.4, dtype=SAMPLE_DTYPE):
        self.dir = directory
        self.dtype = dtype
        self.capacity = capacity
        self.min_rows = min_rows
        self.alpha, self.beta = alpha, beta
        self.buf = np.zeros(capacity, dtype=dtype)
        self.head = 0          # next write position
        self.filled = 0
        self.total = 0         # rows generated since the beginning (all files)
        self.games = 0
        self.known = set()

    def window_size(self):
        """KataGo's W(N) (shuffle.py), bounded by what the buffer holds."""
        n0, n = self.min_rows, self.total
        if n <= n0:
            w = n
        else:
            w = n0 * (1 + self.beta * ((n / n0) ** self.alpha - 1) / self.alpha)
        return int(min(w, self.filled, self.capacity))

    def _append(self, rows):
        """Copy rows in at the head, wrapping around (overwriting the oldest)."""
        rows = rows[-self.capacity:]
        n = len(rows)
        first = min(n, self.capacity - self.head)
        self.buf[self.head:self.head + first] = rows[:first]
        if n > first:
            self.buf[:n - first] = rows[first:]
        self.head = (self.head + n) % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def scan(self):
        """Pick up new files; returns the number of new rows."""
        files = sorted(f for f in glob.glob(os.path.join(self.dir, "*.bin"))
                       if os.path.basename(f) not in self.known)
        if not files:
            return 0
        headers = []
        for f in files:
            try:
                headers.append((f, *read_header(f)))
            except (ValueError, OSError) as e:
                log(f"skipping {f}: {e}")
                self.known.add(os.path.basename(f))
        # Only the rows that can still be in the buffer need to be read (on a
        # restart that is the tail of a possibly huge history).
        need, start = 0, len(headers)
        while start > 0 and need < self.capacity:
            start -= 1
            need += headers[start][1]
        new_rows = 0
        for i, (f, n, g, _version) in enumerate(headers):
            self.known.add(os.path.basename(f))
            self.total += n
            self.games += g
            new_rows += n
            if i >= start:
                self._append(read_samples(f, self.dtype))
        return new_rows

    def sample(self, batch_size, rng):
        w = self.window_size()
        back = rng.integers(1, w + 1, size=batch_size)
        return self.buf[(self.head - back) % self.capacity]


# ── batches ──────────────────────────────────────────────────────────────────
class BatchMaker:
    def __init__(self, device, aux=False):
        self.device = device
        self.aux = aux                   # also return the ownership targets
        src, act = symmetry_tables()
        self.src = src.to(device)
        self.act = act.to(device)
        self.shifts = (63 - torch.arange(64, device=device)).view(1, 1, 64)

    def __call__(self, rows, augment=True):
        """
        rows (numpy, SAMPLE_DTYPE) → (x, z, actions, probs) on the device:
          x       (B, 4, 8, 8) float, channels-last — the network input
          z       (B,)  float — game result for the side to move
          actions (B, 75) long — policy indices of the legal moves (0-padded)
          probs   (B, 75) float — π over those moves (0 on the padding)
        and with aux=True, a fifth element:
          own     (B, 64) long — future ownership codes, in plane cell order
        The planes are decoded from the bitboards on the GPU (only 25 bytes
        per row cross the bus instead of 1 KB of floats).
        """
        B = len(rows)
        dev = self.device
        # uint64 bitboards viewed as int64: `>>` then sign-extends, but `& 1`
        # keeps only the bit asked for, so the sign never matters.
        bbs = np.stack([rows["black"], rows["white"], rows["empty"]], axis=1).view(np.int64)
        bbs = torch.from_numpy(bbs).to(dev, non_blocking=True)                # (B, 3)
        # Cell i of a plane is bit 63 - i (see nn_evaluator.h): shifts = 63 - i.
        planes = ((bbs[:, :, None] >> self.shifts) & 1).float()               # (B, 3, 64)
        turn = torch.from_numpy(rows["turn"].astype(np.float32)).to(dev)
        planes = torch.cat([planes, turn.view(B, 1, 1).expand(B, 1, 64)], dim=1)
        actions = torch.from_numpy(rows["action"].astype(np.int64)).to(dev)   # (B, M)
        # π was quantised to 16 bits per move: renormalise so it sums to 1.
        probs = torch.from_numpy(rows["prob"].astype(np.float32)).to(dev)
        probs = probs / probs.sum(1, keepdim=True).clamp_min(1.0)
        z = torch.from_numpy(rows["z"].astype(np.float32)).to(dev)
        own = None
        if self.aux:
            from alphazero_aux import decode_ownership
            own = decode_ownership(torch.from_numpy(rows["own"]).to(dev))       # (B, 64)
        if augment:
            # One random symmetry per row: permute the 64 cells of every plane
            # (one gather) and map every move index through the same symmetry
            # (one table lookup). z is invariant. The ownership map is a plane
            # like the others: the same gather; the margin it sums to is invariant.
            sym = torch.randint(0, 8, (B,), device=dev)
            planes = planes.gather(2, self.src[sym].unsqueeze(1).expand(B, 4, 64))
            actions = self.act[sym.unsqueeze(1), actions]
            if own is not None:
                own = own.gather(1, self.src[sym])
        x = planes.view(B, 4, 8, 8).contiguous(memory_format=torch.channels_last)
        if self.aux:
            return x, z, actions, probs, own
        return x, z, actions, probs


# ── model files ──────────────────────────────────────────────────────────────
def load_state_dict(path):
    """A state_dict from a plain .pt, a DDP one ("module." prefix) or our checkpoint.pt."""
    with open(path, "rb") as f:
        if f.read(100).startswith(b"version https://git-lfs"):
            sys.exit(f"{path} is a Git LFS pointer, not a network: copy the real file "
                     f"(the repository stores it with Git LFS)")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and "optimizer" in sd:
        sd = sd["model"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def make_net(sd):
    """
    Build a Net whose sizes (channels, blocks, heads) are read off the
    state_dict. The parameters of the auxiliary heads (--aux runs) are not
    part of a Net and are ignored.
    """
    from alphazero_aux import base_state_dict
    sd = base_state_dict(sd)
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    net = Net(channels=C, nb_blocks=B, value_fc_size=sd["value_fc1.weight"].shape[0],
              num_actions=sd["policy_fc.weight"].shape[0])
    net.load_state_dict(sd)
    return net


def model_paths(work, step):
    """(<work>/models/az_<step>.pt, .ts) — step 0 is the initial network."""
    base = os.path.join(work, "models", f"az_{step:08d}")
    return base + ".pt", base + ".ts"


def export_model(net, work, step):
    """
    Write az_<step>.pt (state_dict) + az_<step>.ts (TorchScript) atomically.
    An AuxNet is exported as the plain Net it contains (value + policy only):
    the files have the same format with or without --aux, and the C++
    backends never see the auxiliary heads.
    """
    if hasattr(net, "forward_all"):
        net = make_net(net.state_dict())
    pt, ts = model_paths(work, step)
    cpu = copy.deepcopy(net).float().cpu().eval()
    torch.save(cpu.state_dict(), pt + ".tmp")
    os.replace(pt + ".tmp", pt)
    with torch.no_grad():
        traced = torch.jit.trace(cpu, torch.zeros(1, 4, 8, 8))
    traced.save(ts + ".tmp")
    os.replace(ts + ".tmp", ts)
    return pt, ts


def publish_latest(work, step, aux=False):
    """
    Point self-play at az_<step>.ts (relative path: the work dir may be
    bind-mounted elsewhere). With aux, also ask it to record the auxiliary
    targets ("aux": true); without, the file is exactly as before.
    """
    _, ts = model_paths(work, step)
    latest = {"step": step, "ts": os.path.relpath(ts, work)}
    if aux:
        latest["aux"] = True
    atomic_write_text(os.path.join(work, "latest.json"), json.dumps(latest) + "\n")


# ── state shared with the evaluator thread ───────────────────────────────────
class State:
    """Persistent counters + evaluation bookkeeping (state.json)."""
    def __init__(self, work):
        self.path = os.path.join(work, "state.json")
        self.lock = threading.Lock()
        self.d = {"step": 0, "samples_trained": 0, "exports": [0], "pending_evals": [],
                  "evaluated": [0], "elo": {"0": 0.0}}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.d.update(json.load(f))

    def save(self):
        with self.lock:
            atomic_write_text(self.path, json.dumps(self.d, indent=1) + "\n")


def elo_from_score(w, d, l):
    """Elo difference from a match result, with a +1/2 prior so 10-0 stays finite."""
    n = w + d + l
    s = (w + 0.5 * d + 0.5) / (n + 1)
    return 400.0 * math.log10(s / (1.0 - s))


# ── subprocesses ─────────────────────────────────────────────────────────────
class SelfPlayPool:
    """One `alphazero_learn selfplay` per GPU, restarted if it dies."""
    def __init__(self, args, gpus):
        self.args = args
        self.gpus = gpus
        self.procs = {}
        self.logs = {}
        self.next_start = {}

    def _start(self, gpu):
        a = self.args
        # Unique across jobs: two jobs on one node both see their GPU as "0".
        job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        tag = f"{socket.gethostname()}_{job}_gpu{gpu}"
        cmd = [a.bin, "selfplay", "--config", a.selfplay_config, "--work", a.work,
               "--games", str(a.selfplay_games), "--tag", tag]
        for kv in a.selfplay_set:
            cmd += ["--set", kv]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        logf = open(os.path.join(a.work, "logs", f"selfplay_gpu{gpu}.log"), "a")
        log(f"starting self-play on GPU {gpu}: {' '.join(cmd)}")
        self.procs[gpu] = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        self.logs[gpu] = logf

    def start(self):
        for g in self.gpus:
            self._start(g)

    def check(self):
        now = time.time()
        for g, p in list(self.procs.items()):
            if p.poll() is None:
                continue
            if g not in self.next_start:
                log(f"self-play on GPU {g} exited with code {p.returncode}; restarting in 60 s "
                    f"(see logs/selfplay_gpu{g}.log)")
                self.next_start[g] = now + 60
            elif now >= self.next_start.pop(g):
                self.logs[g].close()
                self._start(g)

    def stop(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for g, p in self.procs.items():
            try:
                p.wait(timeout=120)     # finishes its current searches and writes finished games
            except subprocess.TimeoutExpired:
                p.kill()
            self.logs[g].close()


class Evaluator(threading.Thread):
    """Plays the queued evaluation matches one after the other."""
    CSV_FIELDS = ["time", "model_step", "opponent_step", "games", "wins", "draws", "losses",
                  "score", "elo_diff", "model_elo", "samples_trained", "rows_generated", "hours"]

    def __init__(self, args, state, gpu, stop_event, progress):
        super().__init__(daemon=True)
        self.args, self.state, self.gpu = args, state, gpu
        self.stop_event = stop_event
        self.progress = progress         # callable → (samples_trained, rows_generated, hours)
        self.proc = None
        self.wake = threading.Event()

    def opponents(self, step):
        with self.state.lock:
            evaluated = [s for s in self.state.d["evaluated"] if s < step]
        res = []
        for o in self.args.eval_opponents:
            if o == "previous" and evaluated:
                res.append(evaluated[-1])
            elif o == "initial":
                res.append(0)
            elif o.startswith("lag:") and evaluated:
                res.append(evaluated[max(0, len(evaluated) - int(o[4:]))])
        return sorted(set(res), reverse=True)

    def ensure_ts(self, step):
        """The TorchScript file of `step`, re-traced from its .pt if it was pruned."""
        pt, ts = model_paths(self.args.work, step)
        if not os.path.exists(ts) and os.path.exists(pt):
            export_model(make_net(load_state_dict(pt)), self.args.work, step)
        return ts

    def run_match(self, step, opp):
        a = self.args
        ts_a = self.ensure_ts(step)
        ts_b = self.ensure_ts(opp)
        out = os.path.join(a.work, "evals", f"az_{step:08d}_vs_az_{opp:08d}.json")
        cmd = [a.bin, "match", "--config", a.eval_config, "--a", ts_a, "--b", ts_b,
               "--games", str(a.eval_games), "--time", str(int(a.eval_seconds * 1e6)),
               "--opening-plies", str(a.eval_opening_plies), "--seed", str(step + 1),
               "--out", out]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu)
        log(f"evaluation: az_{step} vs az_{opp} ({a.eval_games} games, {a.eval_seconds} s/move)")
        with open(os.path.join(a.work, "logs", "eval.log"), "a") as logf:
            self.proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            rc = self.proc.wait()
        self.proc = None
        if rc != 0 or not os.path.exists(out):
            return None
        with open(out) as f:
            return json.load(f)

    def evaluate(self, step):
        results = []
        for opp in self.opponents(step):
            r = self.run_match(step, opp)
            if self.stop_event.is_set():
                return False            # interrupted: stays pending
            if r is None:
                log(f"evaluation az_{step} vs az_{opp} FAILED (see logs/eval.log)")
                continue
            results.append((opp, r))
        with self.state.lock:
            elo = self.state.d["elo"]
            for opp, r in results:
                diff = elo_from_score(r["wins"], r["draws"], r["losses"])
                if str(step) not in elo and str(opp) in elo:
                    elo[str(step)] = elo[str(opp)] + diff   # Elo chained through the first opponent
                trained, rows, hours = self.progress()
                row = {"time": time.strftime("%F %T"), "model_step": step, "opponent_step": opp,
                       "games": r["games"], "wins": r["wins"], "draws": r["draws"], "losses": r["losses"],
                       "score": f"{r['score']:.3f}", "elo_diff": f"{diff:+.0f}",
                       "model_elo": f"{elo.get(str(step), float('nan')):+.0f}",
                       "samples_trained": trained, "rows_generated": rows, "hours": f"{hours:.2f}"}
                path = os.path.join(self.args.work, "evals.csv")
                new = not os.path.exists(path)
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                log(f"EVAL az_{step} vs az_{opp}: {r['wins']}W {r['draws']}D {r['losses']}L "
                    f"score {r['score']:.2f}  Elo {diff:+.0f}  (az_{step} ≈ {row['model_elo']} vs initial)")
            if results:
                self.state.d["evaluated"].append(step)
            self.state.d["pending_evals"].remove(step)
        self.state.save()
        return True

    def run(self):
        while not self.stop_event.is_set():
            with self.state.lock:
                pending = list(self.state.d["pending_evals"])
            if not pending:
                self.wake.wait(30)
                self.wake.clear()
                continue
            self.evaluate(pending[0])

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


# ── trainer ──────────────────────────────────────────────────────────────────
def main():
    """
    Setup (arguments, signals, GPUs), resume or start (checkpoint, initial
    export), start the self-play processes and the evaluator thread, then the
    training loop until a signal or --max-hours, then a clean shutdown.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="work directory (everything is kept there)")
    ap.add_argument("--init-model", default=os.path.join(HERE, "cnn_resnet_256x30_value_policy.pt"),
                    help="starting network (used only when the work directory is new)")
    ap.add_argument("--bin", default=os.path.join(HERE, "../build/alphazero_learn"))
    ap.add_argument("--selfplay-config", default=os.path.join(HERE, "../config/alphazero_mcts_selfplay_player.cfg"))
    ap.add_argument("--eval-config", default=os.path.join(HERE, "../config/alphazero_mcts_eval_player.cfg"))
    ap.add_argument("--selfplay-games", type=int, default=128, help="concurrent games per self-play process")
    ap.add_argument("--selfplay-set", action="append", default=[], help="'key=value' override for self-play")
    ap.add_argument("--no-selfplay", action="store_true", help="train only (self-play runs elsewhere)")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--ema-decay", type=float, default=0.995, help="0 = export the raw weights")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--reuse", type=float, default=4.0, help="training samples per generated row")
    # data
    ap.add_argument("--min-rows", type=int, default=100_000, help="rows before training starts (window N0)")
    ap.add_argument("--max-window", type=int, default=4_000_000)
    # cadence
    ap.add_argument("--export-every", type=int, default=131_072, help="training samples between exports")
    ap.add_argument("--eval-hours", type=float, default=4.0,
                    help="evaluate the newest network when this long has passed since the last one")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="evaluate every N exports instead (0 = use --eval-hours)")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-seconds", type=float, default=2.0)
    ap.add_argument("--eval-opening-plies", type=int, default=4)
    ap.add_argument("--eval-opponents", default="previous",
                    help="comma separated: previous (last evaluated model), initial, lag:K")
    ap.add_argument("--keep-all", action="store_true", help="keep every exported model")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long (0 = never)")
    ap.add_argument("--train-gpu", type=int, default=0, help="index among the visible GPUs")
    ap.add_argument("--eval-gpu", type=int, default=-1,
                    help="index among the visible GPUs (-1 = the last one: training uses the first)")
    ap.add_argument("--allow-cpu", action="store_true", help="run even without CUDA (tests only)")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16"], default="auto",
                    help="mixed precision of the trainer (auto: bf16 on Ampere+, fp16 before)")
    # KataGo's auxiliary heads (alphazero_aux.py). Off: everything as before.
    ap.add_argument("--aux", action="store_true",
                    help="train the ownership and score-margin heads (self-play records their targets)")
    ap.add_argument("--own-weight", type=float, default=1.5,
                    help="weight of the ownership loss (KataGo: 1.5/b² on the sum over the board)")
    ap.add_argument("--score-weight", type=float, default=0.02, help="weight of the margin cross-entropy")
    ap.add_argument("--score-cdf-weight", type=float, default=0.02, help="weight of the margin CDF loss")
    ap.add_argument("--aux-ramp-steps", type=int, default=2000,
                    help="the auxiliary weights grow linearly from 0 over this many steps")
    args = ap.parse_args()
    args.work = os.path.abspath(args.work)
    args.eval_opponents = [o.strip() for o in args.eval_opponents.split(",") if o.strip()]
    for d in ("models", "selfplay", "evals", "logs"):
        os.makedirs(os.path.join(args.work, d), exist_ok=True)

    t_start = time.time()
    stop_event = threading.Event()

    def on_signal(signum, _frame):
        log(f"signal {signum}: stopping cleanly")
        stop_event.set()
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, on_signal)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpus = [g.strip() for g in cvd.split(",") if g.strip()] if cvd else \
           [str(i) for i in range(max(1, torch.cuda.device_count()))]
    # No GPU is almost always a container problem (driver not bound): the
    # self-play backend would silently fall back to the CPU and run ~50x
    # slower for two weeks. Refuse instead.
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("alphazero_learn: CUDA is not available (use --allow-cpu to run anyway)")
    device = torch.device(f"cuda:{args.train_gpu}" if torch.cuda.is_available() else "cpu")
    log(f"trainer on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ── model / optimizer / checkpoint ──
    state = State(args.work)
    ckpt_path = os.path.join(args.work, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
    if ckpt is None:
        log(f"new run: initial network {args.init_model}")
        src_sd = load_state_dict(args.init_model)
    else:
        src_sd = ckpt["model"]
    # Variant: with --aux the network carries the auxiliary heads. A run can
    # switch between the variants from one job to the next: the shared
    # parameters are kept, the auxiliary heads start from scratch (or are
    # dropped), and the optimizer — whose state is per parameter — restarts.
    ckpt_aux = bool(ckpt.get("aux", False)) if ckpt is not None else False
    switched = ckpt is not None and ckpt_aux != args.aux
    if args.aux:
        from alphazero_aux import make_aux_net, aux_losses
        net, fresh_heads = make_aux_net(src_sd)
        log("auxiliary heads (ownership, score margin): " + ("new" if fresh_heads else "resumed"))
    else:
        net = make_net(src_sd)
    net = net.to(device).to(memory_format=torch.channels_last)
    ema = None
    if args.ema_decay > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Mixed precision: bf16 on Ampere and newer (A40), fp16 + gradient scaler
    # on Turing (RTX 8000), whose bf16 is emulated and slow. The cluster has
    # both, so consecutive jobs of one run may use either.
    if args.amp == "auto":
        cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
        amp_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    log(f"mixed precision: {amp_dtype}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    if ckpt is not None and switched:
        # Different parameter set: optimizer, EMA and scaler start afresh
        # (the EMA from the current weights), with a new learning-rate warm-up.
        state.d["opt_start_step"] = ckpt["step"]
        if args.aux:
            state.d["aux_start_step"] = ckpt["step"]
        log(f"variant switch: {'with' if args.aux else 'without'} auxiliary heads from step {ckpt['step']} "
            f"(optimizer and weight average restart)")
    if ckpt is not None:
        if not switched:
            if ema is not None and ckpt.get("ema") is not None:
                ema.load_state_dict(ckpt["ema"])
            opt.load_state_dict(ckpt["optimizer"])
        # A bf16 job saves the state of a *disabled* scaler, which is empty
        # and which an enabled scaler refuses to load: an fp16 job resuming
        # it (A40 then RTX 8000) starts with a fresh scale — harmless, the
        # scale adapts within a few steps.
        if scaler.is_enabled() and ckpt.get("scaler") and not switched:
            scaler.load_state_dict(ckpt["scaler"])
        # The checkpoint is authoritative: state.json may have been saved by
        # the evaluator after it, with counters the lost steps never kept.
        state.d["step"] = ckpt["step"]
        state.d["samples_trained"] = ckpt["samples_trained"]
        log(f"resumed at step {state.d['step']}, {state.d['samples_trained']:,} samples trained")
    else:
        pt0, _ = model_paths(args.work, 0)
        if not os.path.exists(pt0):
            export_model(net, args.work, 0)
        publish_latest(args.work, 0, args.aux)
        state.save()
    # Always re-published at start: a run that switches variant must tell
    # self-play at once whether to record the auxiliary targets.
    publish_latest(args.work, state.d["exports"][-1], args.aux)

    def save_checkpoint():
        torch.save({"model": net.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": state.d["step"], "samples_trained": state.d["samples_trained"],
                    "aux": args.aux},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        state.save()

    # ── data ──
    window = ReplayWindow(os.path.join(args.work, "selfplay"), args.max_window, args.min_rows,
                          dtype=SAMPLE_DTYPE_AUX if args.aux else SAMPLE_DTYPE)
    window.scan()
    log(f"replay: {window.total:,} rows from {window.games:,} games on disk, {window.filled:,} in memory")
    make_batch = BatchMaker(device, aux=args.aux)
    rng = np.random.default_rng()

    # ── subprocesses ──
    pool = None
    if not args.no_selfplay:
        pool = SelfPlayPool(args, gpus)
        pool.start()
    evaluator = Evaluator(args, state, gpus[min(args.eval_gpu, len(gpus) - 1)] if args.eval_gpu >= 0 else gpus[-1],
                          stop_event,
                          lambda: (state.d["samples_trained"], window.total, (time.time() - t_start) / 3600))
    evaluator.start()

    def cleanup_models():
        """
        Disk budget (a 256x30 network is 141 MB per file, and a month makes
        hundreds of exports): an export that was never evaluated disappears
        at the next export; an evaluated one keeps its .pt (the history of
        the run) but loses its .ts unless it may still be needed as an
        opponent — Evaluator.ensure_ts re-traces it from the .pt if so.
        """
        if args.keep_all:
            return
        with state.lock:
            evaluated = list(state.d["evaluated"])
            live = {0, state.d["exports"][-1], *state.d["pending_evals"]}
            keep_pt = live | set(evaluated)
            keep_ts = live | set(evaluated[-1:])
            exports = list(state.d["exports"])
        for s in exports:
            pt, ts = model_paths(args.work, s)
            if s not in keep_pt and os.path.exists(pt):
                os.remove(pt)
            if s not in keep_ts and os.path.exists(ts):
                os.remove(ts)

    log_path = os.path.join(args.work, "train_log.csv")
    new_log = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_csv = csv.writer(log_file)
    if new_log:
        log_csv.writerow(["time", "hours", "step", "samples_trained", "rows_generated", "games",
                          "window", "value_loss", "policy_loss", "policy_kl", "lr", "samples_per_s"])

    # ── main loop ──
    net.train()
    acc = {"v": 0.0, "p": 0.0, "kl": 0.0, "n": 0}
    aux_acc = {"own": 0.0, "pdf": 0.0, "cdf": 0.0, "acc": 0.0, "n": 0, "n_acc": 0}
    last_scan = last_report = time.time()
    samples_at_report = state.d["samples_trained"]
    since_export = state.d["samples_trained"] - state.d.get("samples_at_last_export", 0)
    waiting_logged = False
    last_cap_log = 0.0
    while not stop_event.is_set():
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            log("--max-hours reached")
            break
        now = time.time()
        if now - last_scan > 20:
            window.scan()
            if pool is not None:
                pool.check()
            last_scan = now

        # Keep the trainer in step with self-play: at most `reuse` training
        # samples per generated row (samples_trained <= r·N), and nothing
        # before N0 rows exist. Ahead of the data → sleep and rescan.
        allowed = args.reuse * window.total
        if window.total < args.min_rows or state.d["samples_trained"] + args.batch_size > allowed:
            # Two different waits: before training starts (not enough rows
            # yet), and during training when the trainer has caught up with
            # the reuse cap — the normal steady state, logged only now and then.
            if window.total < args.min_rows:
                if not waiting_logged:
                    log(f"waiting for self-play data: {window.total:,} rows "
                        f"(training starts at {args.min_rows:,})")
                    waiting_logged = True
            elif time.time() - last_cap_log > 1800:
                log(f"trainer at the reuse cap ({args.reuse:g} samples per row): "
                    f"it trains as fast as self-play produces rows")
                last_cap_log = time.time()
            stop_event.wait(10)
            last_scan = 0
            continue
        waiting_logged = False

        # ── one SGD step ──
        # Linear warm-up: Adam's moment estimates start at zero after a fresh
        # start, so the first steps would otherwise be too large.
        step = state.d["step"]
        lr = args.lr * min(1.0, (step - state.d.get("opt_start_step", 0) + 1) / max(1, args.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = make_batch(window.sample(args.batch_size, rng))
        x, z, actions, probs = batch[:4]
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            if args.aux:
                v, logits, own_logits, score_logits = net.forward_all(x)
            else:
                v, logits = net(x)
        v = v.float()
        # Policy: cross-entropy with the soft target π, -Σ_a π(a) log p(a|s).
        # The softmax runs over the 4096 logits; π lives on the legal moves
        # only (sparse: `actions` are their indices, the padding has π = 0).
        logp = F.log_softmax(logits.float(), dim=1).gather(1, actions)
        p_loss = -(probs * logp).sum(1).mean()
        # Value: squared error against the game result z ∈ {-1, 0, +1}.
        v_loss = F.mse_loss(v, z)
        loss = args.value_weight * v_loss + args.policy_weight * p_loss
        if args.aux:
            # Auxiliary heads: ownership (per-square cross-entropy) and score
            # margin (distribution + CDF), their weights ramped up from 0 so
            # that the random new heads do not shake a trunk that already
            # plays well. Rows without targets (version 1) are masked.
            l_own, l_pdf, l_cdf, own_acc = aux_losses(own_logits, score_logits, batch[4])
            ramp = min(1.0, (step - state.d.get("aux_start_step", 0) + 1) / max(1, args.aux_ramp_steps))
            loss = loss + ramp * (args.own_weight * l_own + args.score_weight * l_pdf
                                  + args.score_cdf_weight * l_cdf)
            aux_acc["own"] += l_own.item(); aux_acc["pdf"] += l_pdf.item()
            aux_acc["cdf"] += l_cdf.item(); aux_acc["n"] += 1
            if own_acc == own_acc:                       # not NaN (batch had targets)
                aux_acc["acc"] += own_acc; aux_acc["n_acc"] += 1
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        # θ̄ ← μ θ̄ + (1-μ) θ: the averaged weights are the ones exported.
        if ema is not None:
            ema.update_parameters(net)
        state.d["step"] = step + 1
        state.d["samples_trained"] += args.batch_size
        since_export += args.batch_size

        # KL(π‖p) = CE(π, p) − H(π): the part of the policy loss the network
        # can still reduce (0 = it already predicts what the search finds).
        with torch.no_grad():
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(1).mean()
        acc["v"] += v_loss.item(); acc["p"] += p_loss.item(); acc["kl"] += (p_loss - entropy).item(); acc["n"] += 1

        if time.time() - last_report > 120:
            n = max(1, acc["n"])
            dt = time.time() - last_report
            sps = (state.d["samples_trained"] - samples_at_report) / dt
            hours = (time.time() - t_start) / 3600
            log(f"step {state.d['step']}  trained {state.d['samples_trained']:,}  rows {window.total:,} "
                f"({window.games:,} games)  window {window.window_size():,}  "
                f"v {acc['v']/n:.4f}  p {acc['p']/n:.4f}  kl {acc['kl']/n:.4f}  lr {lr:.2e}  {sps:.0f} samples/s")
            log_csv.writerow([time.strftime("%F %T"), f"{hours:.3f}", state.d["step"], state.d["samples_trained"],
                              window.total, window.games, window.window_size(), f"{acc['v']/n:.5f}",
                              f"{acc['p']/n:.5f}", f"{acc['kl']/n:.5f}", f"{lr:.3e}", f"{sps:.0f}"])
            log_file.flush()
            if args.aux and aux_acc["n"]:
                # A separate file, so that train_log.csv keeps one format for
                # every run whichever the variant.
                m = aux_acc["n"]
                own_accuracy = aux_acc["acc"] / aux_acc["n_acc"] if aux_acc["n_acc"] else float("nan")
                log(f"  aux: ownership {aux_acc['own']/m:.4f} (accuracy {own_accuracy:.3f})  "
                    f"margin pdf {aux_acc['pdf']/m:.4f}  cdf {aux_acc['cdf']/m:.4f}  weight ramp {ramp:.2f}")
                aux_path = os.path.join(args.work, "train_log_aux.csv")
                new_aux = not os.path.exists(aux_path)
                with open(aux_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new_aux:
                        w.writerow(["time", "step", "ownership_loss", "ownership_accuracy",
                                    "margin_pdf_loss", "margin_cdf_loss", "ramp"])
                    w.writerow([time.strftime("%F %T"), state.d["step"], f"{aux_acc['own']/m:.5f}",
                                f"{own_accuracy:.4f}", f"{aux_acc['pdf']/m:.5f}", f"{aux_acc['cdf']/m:.5f}",
                                f"{ramp:.3f}"])
            aux_acc = {"own": 0.0, "pdf": 0.0, "cdf": 0.0, "acc": 0.0, "n": 0, "n_acc": 0}
            acc = {"v": 0.0, "p": 0.0, "kl": 0.0, "n": 0}
            last_report = time.time()
            samples_at_report = state.d["samples_trained"]

        # ── export: a new network for self-play (and, every eval_every, a match) ──
        if since_export >= args.export_every:
            since_export = 0
            s = state.d["step"]
            export_model(ema.module if ema is not None else net, args.work, s)
            publish_latest(args.work, s, args.aux)
            with state.lock:
                state.d["exports"].append(s)
                state.d["samples_at_last_export"] = state.d["samples_trained"]
                # Evaluation schedule: every eval_every exports, or else the
                # first export at least eval_hours (wall clock, across jobs)
                # after the previously scheduled one.
                if args.eval_every > 0:
                    due = (len(state.d["exports"]) - 1) % args.eval_every == 0
                else:
                    due = time.time() - state.d.get("last_eval_time", 0) >= args.eval_hours * 3600
                if due and args.eval_games > 0:
                    state.d["pending_evals"].append(s)
                    state.d["last_eval_time"] = time.time()
            save_checkpoint()
            cleanup_models()
            evaluator.wake.set()
            log(f"exported az_{s:08d} (self-play switches to it)")

    # ── shutdown ──
    stop_event.set()
    log("saving checkpoint")
    save_checkpoint()
    evaluator.stop()
    if pool is not None:
        pool.stop()
    evaluator.join(timeout=60)
    log_file.close()
    log("stopped; resubmit with the same --work to continue")


if __name__ == "__main__":
    main()
