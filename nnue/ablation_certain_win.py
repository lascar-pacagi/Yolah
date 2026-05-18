"""
Local ablation runner for the certain-win feature.

Trains two small ResNets back-to-back on identical data:
  * Model X — 6-plane input (with certain-win planes)
  * Model Y — 4-plane input (no certain-win planes)

Same architecture, same data, same seed, same train/val split: the *only*
thing that differs is the number of input channels. Reports per-epoch
validation metrics (value MSE, value sign accuracy, policy cross-entropy,
policy top-1) and a side-by-side comparison at the end.

Designed to fit on a single small GPU (~8 GB VRAM) and finish in a few
minutes — not a strength claim against the production model, just a clean
relative-effect measurement.
"""
import os
import sys
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# ── Configuration ──────────────────────────────────────────────────────────
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAME_DIR           = "./data"
NUM_FILES          = 12                 # non-symmetry game files to load
MAX_GAMES_PER_FILE = 2500               # cap games per file
CHANNELS           = 96                 # small trunk width
NB_BLOCKS          = 4                  # small trunk depth
VALUE_FC_SIZE      = 128
NUM_ACTIONS        = 64 * 64
POLICY_IGNORE_IDX  = -1
BATCH_SIZE         = 512
NB_EPOCHS          = 12
LEARNING_RATE      = 1e-3
LR_MIN             = 1e-5               # cosine annealing floor
WEIGHT_DECAY       = 1e-4
VAL_FRACTION       = 0.05
SEED               = 42


# ── Bitboard → plane (vectorized; ~50× faster than the Python loop) ────────
def _bitboard_to_plane(n: int) -> np.ndarray:
    arr = np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))
    return arr.astype(np.float32).reshape(8, 8)


def encode_6plane(yolah) -> np.ndarray:
    """(6, 8, 8) numpy encoding — the with-certain-win variant."""
    black = _bitboard_to_plane(yolah.black)
    white = _bitboard_to_plane(yolah.white)
    empty = _bitboard_to_plane(yolah.empty)
    black_to_move = (yolah.nb_plies() & 1) == 0
    turn  = np.full((8, 8), 0.0 if black_to_move else 1.0, dtype=np.float32)

    delta = yolah.black_score - yolah.white_score
    eff   = delta - (0 if black_to_move else 1)
    bw    = np.full((8, 8), 1.0 if eff >=  1 else 0.0, dtype=np.float32)
    ww    = np.full((8, 8), 1.0 if eff <= -1 else 0.0, dtype=np.float32)

    return np.stack([black, white, empty, turn, bw, ww])


# ── Game-file parsing & per-position encoding ──────────────────────────────
def parse_file(path, max_games):
    """Return a list of (moves bytes, nb_moves, r, z_black) tuples."""
    with open(path, "rb") as f:
        data = f.read()
    games = []
    idx = 0
    while idx < len(data):
        nb_moves  = data[idx]
        nb_random = data[idx + 1]
        if nb_random == nb_moves:
            idx += 2 + 2 * nb_moves + 2
            continue
        moves = bytes(data[idx + 2: idx + 2 + 2 * nb_moves])
        bs    = data[idx + 2 + 2 * nb_moves]
        ws    = data[idx + 2 + 2 * nb_moves + 1]
        z_black = 1 if bs > ws else (-1 if ws > bs else 0)
        games.append((moves, nb_moves, nb_random, z_black))
        idx += 2 + 2 * nb_moves + 2
        if len(games) >= max_games:
            break
    return games


def encode_game(game):
    """Yield per-position (state_6plane, value_target, policy_target)."""
    moves, nb_moves, r, z_black = game
    states, values, policies = [], [], []
    y = Yolah()

    # Fast-forward the random opening
    for ply in range(r):
        s1, s2 = moves[2 * ply], moves[2 * ply + 1]
        y.play(Move(Square(s1), Square(s2)))

    # Plies r .. nb_moves are training positions
    for ply in range(r, nb_moves + 1):
        states.append(encode_6plane(y))

        current_is_black = (ply & 1) == 0
        z_cur = z_black if current_is_black else -z_black
        values.append(float(z_cur))

        if ply < nb_moves:
            s1, s2 = moves[2 * ply], moves[2 * ply + 1]
            policies.append(s1 * 64 + s2)
            y.play(Move(Square(s1), Square(s2)))
        else:
            policies.append(POLICY_IGNORE_IDX)

    return states, values, policies


def build_dataset():
    files = sorted(glob.glob(os.path.join(GAME_DIR, "games_*.txt")))
    files = [f for f in files if not f.endswith(".symmetries.txt")][:NUM_FILES]
    print(f"Loading {len(files)} game files (cap {MAX_GAMES_PER_FILE}/file):")
    for f in files:
        print(f"  {f}")

    all_games = []
    for f in files:
        all_games.extend(parse_file(f, max_games=MAX_GAMES_PER_FILE))
    print(f"  {len(all_games):,} games kept")

    print("Encoding positions...")
    t0 = time.time()
    states_list, values_list, policies_list = [], [], []
    game_bounds = [0]
    for g in all_games:
        s, v, p = encode_game(g)
        states_list.extend(s)
        values_list.extend(v)
        policies_list.extend(p)
        game_bounds.append(len(states_list))
    print(f"  {len(states_list):,} positions encoded in {time.time()-t0:.1f}s")

    states   = np.stack(states_list).astype(np.float32)
    values   = np.array(values_list, dtype=np.float32)
    policies = np.array(policies_list, dtype=np.int64)
    return states, values, policies, game_bounds


# ── Model ──────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        r = x
        x = self.conv1(torch.relu(self.bn1(x)))
        x = self.conv2(torch.relu(self.bn2(x)))
        return x + r


class Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS), nn.ReLU(inplace=True))
        self.trunk  = nn.Sequential(*[ResBlock(CHANNELS) for _ in range(NB_BLOCKS)])
        self.out_bn = nn.BatchNorm2d(CHANNELS)
        # value head
        self.v_conv = nn.Conv2d(CHANNELS, 1, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(64, VALUE_FC_SIZE)
        self.v_fc2  = nn.Linear(VALUE_FC_SIZE, 1)
        # policy head
        self.p_conv = nn.Conv2d(CHANNELS, 2, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * 64, NUM_ACTIONS)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.trunk(x)
        x = torch.relu(self.out_bn(x))
        v = torch.relu(self.v_bn(self.v_conv(x))).flatten(1)
        v = torch.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        p = torch.relu(self.p_bn(self.p_conv(x))).flatten(1)
        p = self.p_fc(p)
        return v, p


class _SliceDataset(Dataset):
    """Returns the first `in_channels` planes of the stored 6-plane state."""
    def __init__(self, states, values, policies, indices, in_channels):
        self.states     = states
        self.values     = values
        self.policies   = policies
        self.indices    = indices
        self.in_channels = in_channels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        s = self.states[idx, :self.in_channels]   # (C, 8, 8)
        return torch.from_numpy(s.copy()), self.values[idx], self.policies[idx]


# ── Train + validate one model ─────────────────────────────────────────────
def train_one(states, values, policies, train_idx, val_idx,
              in_channels, label):
    # Re-seed so each model gets identical weight init + batch order
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"\n====== {label}: {in_channels} input planes ======")

    train_ds = _SliceDataset(states, values, policies, train_idx, in_channels)
    val_ds   = _SliceDataset(states, values, policies, val_idx,   in_channels)
    gen = torch.Generator(); gen.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=gen, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    net = Net(in_channels=in_channels).to(DEVICE)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {nparams:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NB_EPOCHS, eta_min=LR_MIN)
    v_loss_fn = nn.MSELoss()
    p_loss_fn = nn.CrossEntropyLoss(ignore_index=POLICY_IGNORE_IDX)

    history = []
    for epoch in range(NB_EPOCHS):
        # ── Train pass ──────────────────────────────────────────────────
        net.train(True)
        t0 = time.time()
        n_train = 0
        v_run, p_run = 0.0, 0.0
        for sb, vb, pb in train_loader:
            sb = sb.to(DEVICE, non_blocking=True)
            vb = vb.to(DEVICE, non_blocking=True)
            pb = pb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            v_pred, p_logits = net(sb)
            v_loss = v_loss_fn(v_pred, vb)
            p_loss = p_loss_fn(p_logits, pb)
            (v_loss + p_loss).backward()
            optimizer.step()
            bs = len(sb)
            n_train += bs
            v_run += v_loss.item() * bs
            p_run += p_loss.item() * bs

        scheduler.step()

        # ── Validation pass (BN in inference mode) ──────────────────────
        net.train(False)
        n_val = 0
        vv, vp = 0.0, 0.0
        sign_corr = 0
        p_corr = 0; p_total = 0
        with torch.no_grad():
            for sb, vb, pb in val_loader:
                sb = sb.to(DEVICE)
                vb = vb.to(DEVICE)
                pb = pb.to(DEVICE)
                v_pred, p_logits = net(sb)
                vv += v_loss_fn(v_pred, vb).item() * len(sb)
                vp += p_loss_fn(p_logits, pb).item() * len(sb)
                n_val += len(sb)
                sign_corr += (torch.sign(v_pred) == torch.sign(vb)).sum().item()
                mask = pb != POLICY_IGNORE_IDX
                if mask.any():
                    preds = p_logits.argmax(dim=1)
                    p_corr  += ((preds == pb) & mask).sum().item()
                    p_total += int(mask.sum().item())

        v_mse  = vv / n_val
        p_ce   = vp / n_val
        v_sign = sign_corr / n_val
        p_top1 = (p_corr / p_total) if p_total > 0 else 0.0
        dt = time.time() - t0
        print(f"  epoch {epoch+1}/{NB_EPOCHS}  "
              f"train v_mse={v_run/n_train:.4f} p_ce={p_run/n_train:.4f}   "
              f"val v_mse={v_mse:.4f} v_sign={v_sign:.4f} "
              f"p_ce={p_ce:.4f} p_top1={p_top1:.4f}   ({dt:.1f}s)")
        history.append((v_mse, v_sign, p_ce, p_top1))

    return history


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    states, values, policies, game_bounds = build_dataset()

    # Game-level train/val split — avoids leaking same-game positions
    n_games = len(game_bounds) - 1
    game_order = list(range(n_games))
    random.Random(SEED).shuffle(game_order)
    n_val_games = max(1, int(VAL_FRACTION * n_games))
    val_games = set(game_order[:n_val_games])
    train_idx, val_idx = [], []
    for gi in range(n_games):
        rng = list(range(game_bounds[gi], game_bounds[gi + 1]))
        (val_idx if gi in val_games else train_idx).extend(rng)
    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx   = np.array(val_idx,   dtype=np.int64)
    print(f"Game-level split: train {len(train_idx):,} positions "
          f"({n_games - n_val_games:,} games)   "
          f"val {len(val_idx):,} positions ({n_val_games:,} games)")

    hist6 = train_one(states, values, policies, train_idx, val_idx,
                      in_channels=6, label="WITH certain-win")
    hist4 = train_one(states, values, policies, train_idx, val_idx,
                      in_channels=4, label="WITHOUT certain-win")

    final6, final4 = hist6[-1], hist4[-1]
    print("\n================== COMPARISON (final epoch) ==================")
    print(f"{'Metric':<22}  {'WITH (6p)':>12}  {'WITHOUT (4p)':>14}  {'delta (4-6)':>12}")
    print("-" * 68)
    names = ["val value MSE", "val value sign acc", "val policy CE", "val policy top-1"]
    for name, a, b in zip(names, final6, final4):
        print(f"{name:<22}  {a:>12.4f}  {b:>14.4f}  {b - a:>+12.4f}")
    print("-" * 68)
    print("Lower is better for MSE/CE; higher is better for sign-acc/top-1.")


if __name__ == "__main__":
    main()
