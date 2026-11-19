"""
nnue_policy_head.py — experiment 1 (2026-11-19): can a small "move" head on
the NNUE's accumulator order the moves of the minimax search better?

The value network stays FROZEN (the dev's net, ref2u_az20 by default): its
first layer (the accumulator the C++ already updates incrementally) is
computed exactly as the C++ does (quantization-aware path), and small heads
are trained on top of it to predict the move played in data/games (the dev
player's choice: the policies.i16 of policies_from_nnue_cache.py).

Every head works in the point of view of the side to move (the net's "us"
view: white's squares mirrored a <-> h), on cells c (cell c = square 63 - c,
mirrored c ^ 7 for white), and gives one logit per (from, to):

    z      = clamp(W1 · h1 + b1, 0, 1)                    hidden H (H = 0: no hidden layer)
    logit  = table[f, t] + from(z)[f] + to(z)[t] + q(z)[f] · k(z)[t]     (rank R)

h1 = the us accumulator (1024) or [us, them] (2048). Variants trained side by
side on the same batches (the frozen features are computed once):

    table   the static table only (a learned prior per (from, to), no position)
    u32r0   us,       H 32, R 0       ≈ 33 k multiply-adds per node
    u32r4   us,       H 32, R 4       ≈ 49 k
    ut32r4  us+them,  H 32, R 4       ≈ 82 k
    ut64r8  us+them,  H 64, R 8       ≈ 230 k (the upper end)

(value() of the dev net is ≈ 850 ns; fc2 alone is 1 M multiply-adds.)

The loss is the cross-entropy over the LEGAL moves only (the search only
orders legal moves; legality from the queen rays of the position). Metrics on
the validation set (last 5 %, as all trainings): masked cross-entropy, top-1,
top-3, mean rank of the move played among the legal moves, also per phase
(free squares 57-41, 40-23, 22-0), and the chance level (1 / nb legal moves).

    python3 nnue_policy_head.py train --cache /cache --net X.pt --out /work [--samples 300e6]
    python3 nnue_policy_head.py selftest --net X.pt              (legality and frames, on random games)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "../server"))
import nnue_persp as NP                                  # noqa: E402
from transformer_value_policy import CacheLoader          # noqa: E402

VARIANTS = {            # name: (them, hidden, rank, table only)
    "table":  (False, 0, 0, True),
    "u32r0":  (False, 32, 0, False),
    "u32r4":  (False, 32, 4, False),
    "ut32r4": (True, 32, 4, False),
    "ut64r8": (True, 64, 8, False),
    "ut64r0": (True, 64, 0, False),
    "ut128r0": (True, 128, 0, False),
}
# First local run (2026-11-19, 30 M samples of a 10.5 M-position cache, top-1 among the legal moves,
# all phases): table 0.231, u32r0 0.345, u32r4 0.344 (the rank-4 term adds nothing), ut32r4 0.373,
# ut64r8 0.395; middle game (23-40 free) 0.095 / 0.219 / 0.219 / 0.253 / 0.277, chance 0.050.


# ── Frozen NNUE features (what the C++ accumulator holds) ───────────────────

@torch.no_grad()
def nnue_features(net, x):
    """x: (B, 193) cache rows → h1 us (B, l1), h1 them (B, l1), us view (B, 192), white to move (B,)."""
    us, them, _ = net.perspective(x)
    w1, b1 = net.merged_fc1()
    if net.qat:
        w1, b1 = NP.q_weight(w1), NP.q_bias(b1, NP.Q)
    h1, h1t = F.linear(us, w1, b1), F.linear(them, w1, b1)
    if net.fc1_buckets > 1:
        ib = NP.bucket_of(x[:, 128:192].sum(dim=1).round().long(), net.fc1_buckets).view(-1, 1, 1).expand(-1, 1, net.l1_size)
        h1 = h1.view(-1, net.fc1_buckets, net.l1_size).gather(1, ib).squeeze(1)
        h1t = h1t.view(-1, net.fc1_buckets, net.l1_size).gather(1, ib).squeeze(1)
    return torch.clamp(h1, 0, 1), torch.clamp(h1t, 0, 1), us, x[:, 192] > 0.5


# ── Legal moves in the us view (queen moves through free cells) ─────────────

def ray_table():
    """(64, 8, 7) cell indices along the 8 directions, 64 = off board (padding)."""
    rays = torch.full((64, 8, 7), 64, dtype=torch.long)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for c in range(64):
        r, f = divmod(c, 8)
        for d, (dr, df) in enumerate(dirs):
            for k in range(7):
                rr, ff = r + dr * (k + 1), f + df * (k + 1)
                if 0 <= rr < 8 and 0 <= ff < 8:
                    rays[c, d, k] = rr * 8 + ff
    return rays


def legal_mask(view, rays):
    """view: (B, 192) us view (mine, his, holes) → (B, 64 from, 64 to) bool."""
    B = view.shape[0]
    mine = view[:, 0:64] > 0.5
    occ = (view[:, 0:64] + view[:, 64:128] + view[:, 128:192]) > 0.5
    occ_p = torch.cat([occ, torch.ones(B, 1, dtype=torch.bool, device=view.device)], dim=1)   # cell 64: blocked
    blocked = occ_p[:, rays.view(-1)].view(B, 64, 8, 7)
    reach = torch.cumprod((~blocked).to(torch.uint8), dim=3).bool()                          # free up to the first obstacle
    legal = torch.zeros(B, 64, 65, dtype=torch.bool, device=view.device)
    legal.scatter_(2, rays.view(1, 64, 56).expand(B, -1, -1), reach.view(B, 64, 56))
    return legal[:, :, :64] & mine[:, :, None]


def target_in_view(policy, white):
    """policy (square index from*64+to, -1) → index in the us view's cells (-1 kept)."""
    sf, st = policy // 64, policy % 64
    cf, ct = 63 - sf, 63 - st
    cf = torch.where(white, cf ^ 7, cf)
    ct = torch.where(white, ct ^ 7, ct)
    return torch.where(policy >= 0, cf * 64 + ct, torch.full_like(policy, -1))


# ── Heads ───────────────────────────────────────────────────────────────────

class MoveHead(nn.Module):
    def __init__(self, l1, them, hidden, rank, table_only):
        super().__init__()
        self.them, self.hidden, self.rank, self.table_only = them, hidden, rank, table_only
        self.config = {"them": them, "hidden": hidden, "rank": rank, "table_only": table_only, "l1": l1}
        self.table = nn.Parameter(torch.zeros(64, 64))
        if not table_only:
            inp = l1 * (2 if them else 1)
            self.fc = nn.Linear(inp, hidden)
            self.out = nn.Linear(hidden, 128 + 128 * rank)        # from 64, to 64, q 64·R, k 64·R
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, h1, h1t):
        B = h1.shape[0]
        logits = self.table.unsqueeze(0).expand(B, -1, -1)
        if self.table_only:
            return logits
        h = torch.cat([h1, h1t], 1) if self.them else h1
        # quantization-aware, as the C++ (NNUE_PolicyHead): int8 weights at scale 64, int16 bias at
        # scale 64², the >> 6 of the int32 sums (floor), clamp to [0, 1] (0..64 in C++)
        z = torch.clamp(NP.q_floor(F.linear(h, NP.q_weight(self.fc.weight), NP.q_bias(self.fc.bias, NP.Q * NP.Q))), 0, 1)
        o = self.out(z)
        logits = logits + o[:, 0:64].unsqueeze(2) + o[:, 64:128].unsqueeze(1)      # from[f] + to[t]
        if self.rank:
            q = o[:, 128:128 + 64 * self.rank].view(B, 64, self.rank)
            k = o[:, 128 + 64 * self.rank:].view(B, 64, self.rank)
            logits = logits + q @ k.transpose(1, 2)
        return logits


class Stats:
    PHASES = ((41, 57), (23, 40), (0, 22))      # free squares

    def __init__(self):
        self.d = {}

    def add(self, name, logits, legal, tgt, free):
        lg = logits.masked_fill(~legal, -1e9).view(len(tgt), -1)
        ce = F.cross_entropy(lg, tgt, reduction='none')
        tl = lg.gather(1, tgt[:, None])
        rank = (lg > tl).sum(1)                                    # 0 = first
        nleg = legal.view(len(tgt), -1).sum(1)
        for ph, (lo, hi) in enumerate(((0, 57),) + self.PHASES):
            m = (free >= lo) & (free <= hi)
            s = self.d.setdefault((name, ph), [0, 0., 0, 0, 0., 0.])
            s[0] += int(m.sum()); s[1] += float(ce[m].sum()); s[2] += int((rank[m] == 0).sum())
            s[3] += int((rank[m] < 3).sum()); s[4] += float(rank[m].float().sum()); s[5] += float((1.0 / nleg[m].float()).sum())

    def table(self):
        lines = [f"{'head':8} {'phase':>8} {'n':>10} {'ce':>7} {'top1':>7} {'top3':>7} {'rank':>6} {'chance':>7}"]
        for (name, ph), s in sorted(self.d.items(), key=lambda kv: (kv[0][1], list(VARIANTS).index(kv[0][0]))):
            if s[0] == 0:
                continue
            lab = "all" if ph == 0 else f"{self.PHASES[ph - 1][0]}-{self.PHASES[ph - 1][1]}"
            lines.append(f"{name:8} {lab:>8} {s[0]:>10,} {s[1] / s[0]:7.4f} {s[2] / s[0]:7.3f} {s[3] / s[0]:7.3f} "
                         f"{s[4] / s[0]:6.2f} {s[5] / s[0]:7.3f}")
        return "\n".join(lines)


def batch_to_rows(X):
    """Loader planes (B, 4, 8, 8) → cache rows (B, 193)."""
    return torch.cat([X[:, :3].reshape(-1, 192), X[:, 3, 0, 0, None]], dim=1)


def prepare(net, X, p, rays):
    x = batch_to_rows(X)
    h1, h1t, view, white = nnue_features(net, x)
    legal = legal_mask(view, rays)
    tgt = target_in_view(p, white)
    ok = tgt >= 0
    ok &= legal.view(len(tgt), -1).gather(1, tgt.clamp(min=0)[:, None]).squeeze(1)   # passes / ends out
    free = 64 - 8 - x[:, 128:192].sum(1).round().long()
    return h1[ok], h1t[ok], legal[ok], tgt[ok], free[ok], int((~ok).sum()), int(((tgt >= 0) & ~ok).sum())


def cmd_train(args):
    device = torch.device("cuda")
    net = NP.load_persp(args.net).to(device).eval()
    rays = ray_table().to(device)
    heads = nn.ModuleDict({n: MoveHead(net.l1_size, *VARIANTS[n]) for n in args.variants.split(",")}).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)
    with open(os.path.join(args.cache, "meta.json")) as f:
        n_total = int(json.load(f)["n_positions"])
    n_train = int(0.95 * n_total)
    train = CacheLoader(args.cache, 0, n_train, args.batch, 0, 1, chunk_size=args.chunk, shuffle=True)
    val = CacheLoader(args.cache, n_train, n_total, args.batch, 0, 1, chunk_size=args.chunk, shuffle=False)
    steps = int(args.samples) // args.batch
    print(f"net {args.net}: l1 {net.l1_size}, fc1 sets {net.fc1_buckets}, qat {net.qat}; heads "
          + ", ".join(f"{n} {sum(p.numel() for p in h.parameters()):,} params" for n, h in heads.items())
          + f"; {steps:,} steps of {args.batch}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    step, t0, epoch, bad = 0, time.time(), 0, 0
    while step < steps:
        train.set_epoch(epoch)
        for X, _, p in train:
            X, p = X.to(device, non_blocking=True), p.to(device, non_blocking=True)
            h1, h1t, legal, tgt, free, _, b = prepare(net, X, p, rays)
            bad += b
            lr = args.lr * (min(1.0, (step + 1) / 500)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = 0.0
            for n, h in heads.items():
                lg = h(h1, h1t).masked_fill(~legal, -1e9).view(len(tgt), -1)
                loss = loss + F.cross_entropy(lg, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for h in heads.values():
                    if not h.table_only:
                        h.fc.weight.clamp_(-NP.WEIGHT_MAX, NP.WEIGHT_MAX)
            step += 1
            if step % 2000 == 0:
                print(f"step {step:,}/{steps:,} loss/head {float(loss) / len(heads):.4f} lr {lr:.2e} "
                      f"{2000 * args.batch / (time.time() - t0):,.0f} pos/s; played move not legal so far: {bad}", flush=True)
                t0 = time.time()
            if step >= steps:
                break
        epoch += 1
    torch.save({"heads": heads.state_dict(), "configs": {n: h.config for n, h in heads.items()}, "net": args.net},
               os.path.join(args.out, "policy_heads.pt"))
    st = Stats()
    with torch.no_grad():
        for i, (X, _, p) in enumerate(val):
            if args.val_chunks > 0 and i >= args.val_chunks * val.batches_per_chunk:
                break
            X, p = X.to(device), p.to(device)
            h1, h1t, legal, tgt, free, _, _ = prepare(net, X, p, rays)
            for n, h in heads.items():
                st.add(n, h(h1, h1t), legal, tgt, free)
    report = st.table()
    print(report, flush=True)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(report + "\n")


def export_head(head, path):
    """Text file for the C++ NNUE_PolicyHead (nnue/nnue_policy_head.h):
        POLICYHEAD <them 0/1> <l1> <hidden>
        hidden × inputs int8 (W1 · 64), hidden int16 (b1 · 4096),
        128 × hidden floats (W2 / 64: from 0-63, to 64-127, on the 0..64 integer z), 128 floats (b2),
        4096 floats (the table, index from_cell · 64 + to_cell, cells of the us view)."""
    assert not head.table_only and head.rank == 0 and head.hidden % 8 == 0
    w1 = torch.clamp(torch.round(head.fc.weight.detach().float() * NP.Q), -127, 127).long()
    b1 = torch.clamp(torch.round(head.fc.bias.detach().float() * NP.Q * NP.Q), -32768, 32767).long()
    w2 = head.out.weight.detach().float()[:128] / NP.Q
    b2 = head.out.bias.detach().float()[:128]
    with open(path, "w") as f:
        f.write(f"POLICYHEAD {int(head.them)} {head.config['l1']} {head.hidden}\n")
        for row in w1.tolist():
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, b1.tolist())) + "\n")
        for row in w2.tolist():
            f.write(" ".join(f"{v:.9g}" for v in row) + "\n")
        f.write(" ".join(f"{v:.9g}" for v in b2.tolist()) + "\n")
        f.write(" ".join(f"{v:.9g}" for v in head.table.detach().float().view(-1).tolist()) + "\n")
    print(f"wrote {path}")


def cmd_export(args):
    ck = torch.load(args.heads, map_location="cpu")
    cfg = ck["configs"][args.variant]
    head = MoveHead(cfg["l1"], cfg["them"], cfg["hidden"], cfg["rank"], cfg["table_only"])
    head.load_state_dict({k[len(args.variant) + 1:]: v for k, v in ck["heads"].items() if k.startswith(args.variant + ".")})
    export_head(head, args.out)
    if args.check:      # reference logits of the bench positions, for test/policy_head_check
        net = NP.load_persp(args.net or ck["net"]).eval()
        from yolah import Yolah
        rows = []
        for line in open(args.check):
            line = line.strip()
            if not line:
                continue
            y = Yolah()
            y.from_json(line)
            r = np.zeros(193, dtype=np.float32)
            for i in range(64):
                r[i] = (y.black >> (63 - i)) & 1
                r[64 + i] = (y.white >> (63 - i)) & 1
                r[128 + i] = (y.empty >> (63 - i)) & 1
            r[192] = y.nb_plies() & 1
            rows.append(r)
        x = torch.tensor(np.array(rows))
        h1, h1t, view, white = nnue_features(net, x)
        with torch.no_grad():
            lg = head(h1, h1t)
        legal = legal_mask(view, ray_table())
        with open(args.out + ".check.txt", "w") as f:
            for i in range(len(rows)):
                idx = legal[i].view(-1).nonzero().view(-1).tolist()
                f.write(" ".join(f"{j}:{float(lg[i].view(-1)[j]):.5f}" for j in idx) + "\n")
        print(f"wrote {args.out}.check.txt ({len(rows)} positions, logits of the legal moves, us-view cells)")


def cmd_selftest(args):
    """Legal mask = Yolah's move generator; targets of real games are legal; heads run."""
    from yolah import Yolah, Move, Square
    rng = np.random.default_rng(3)
    rays = ray_table()
    rows, pols, legal_ref = [], [], []
    for g in range(60):
        y = Yolah()
        while not y.game_over():
            row = np.zeros(193, dtype=np.float32)
            for i in range(64):
                row[i] = (y.black >> (63 - i)) & 1
                row[64 + i] = (y.white >> (63 - i)) & 1
                row[128 + i] = (y.empty >> (63 - i)) & 1
            row[192] = y.nb_plies() & 1
            ms = y.moves()
            m = ms[int(rng.integers(len(ms)))]
            rows.append(row)
            pols.append(-1 if m == Move.none() else int(m.from_sq.value) * 64 + int(m.to_sq.value))
            legal_ref.append(set() if ms[0] == Move.none() else {int(x.from_sq.value) * 64 + int(x.to_sq.value) for x in ms})
            y.play(m)
    x = torch.tensor(np.array(rows))
    p = torch.tensor(pols)
    net = NP.NetPersp(k=8, them=True, fc1_buckets=4)
    if args.net:
        net = NP.load_persp(args.net)
    net.eval()
    h1, h1t, view, white = nnue_features(net, x)
    legal = legal_mask(view, rays)
    tgt = target_in_view(p, white)
    bad = 0
    for i in range(len(rows)):
        got = set()
        for idx in legal[i].view(-1).nonzero().view(-1).tolist():
            cf, ct = divmod(idx, 64)
            if white[i]:
                cf, ct = cf ^ 7, ct ^ 7
            got.add((63 - cf) * 64 + (63 - ct))
        bad += got != legal_ref[i]
        if pols[i] >= 0:
            assert legal[i].view(-1)[tgt[i]], i
    print(f"{len(rows)} positions: legal mask != move generator in {bad}")
    assert bad == 0
    for n, v in VARIANTS.items():
        h = MoveHead(net.l1_size, *v)
        assert h(h1, h1t).shape == (len(rows), 64, 64)
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "selftest", "export"])
    ap.add_argument("--heads", help="export: policy_heads.pt")
    ap.add_argument("--variant", default="ut64r0", help="export: which head")
    ap.add_argument("--check", help="export: positions (JSON lines) → <out>.check.txt reference logits")
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--net", default=None)
    ap.add_argument("--out", default="/work")
    ap.add_argument("--variants", default="table,u32r0,ut32r4,ut64r0,ut64r8,ut128r0")
    ap.add_argument("--samples", type=float, default=300e6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-chunks", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1 << 22, help="positions per chunk of the loader")
    args = ap.parse_args()
    {"train": cmd_train, "selftest": cmd_selftest, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
