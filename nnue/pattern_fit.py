"""
pattern_fit.py — piece-square tables and pattern tables (n-tuples, as in the
Othello programs Logistello and Edax) per game phase, against the teacher:
cheap evaluations, updated move by move almost for free. How close do they
get to the NNUE? (continues phase_fit.py)

    python3 pattern_fit.py CACHE_DIR PHASEFIT_DIR

Squares from the side to move's view: 0 free, 1 mine, 2 his, 3 hole.
    pst       one weight per (square, state), one table per phase, least squares
    lines     + one weight per configuration of each row, column and the two long
              diagonals (4^8 = 65536 configurations; rows/columns equivalent by
              symmetry share a table: edges, 2nd, 3rd, 4th lines, diagonals),
              one set of tables per phase, Adam on tanh(sum) vs the teacher
Phases: 8 ranges of holes (as the NNUE heads, holes · 8 // 57). Fit on the
first 85 % of the cache, measured on the last 15 %; the NNUE's error is read
from PHASEFIT_DIR (phase_fit_dump, stride 3) on the same last 15 %.
"""
import os
import sys
import numpy as np
import torch

K = 8


def phase(holes):
    return np.minimum(K - 1, holes * K // 57)


def squares(rows):
    """(N, 193) cache rows → (N, 64) states from the side to move's view (0 free, 1 mine, 2 his, 3 hole)."""
    black, white, empty = rows[:, :64], rows[:, 64:128], rows[:, 128:192]
    turn = rows[:, 192:193]
    mine = np.where(turn == 0, black, white)
    his = np.where(turn == 0, white, black)
    return (mine * 1 + his * 2 + empty * 3).astype(np.int64)


def line_indices():
    """The 18 lines (as lists of 8 squares in the cache's order) and the table each one uses."""
    lines, table = [], []
    for r in range(8):
        t = min(r, 7 - r)                     # 0 edge, 1, 2, 3 centre lines
        lines.append([8 * r + c for c in range(8)]); table.append(t)
        lines.append([8 * c + r for c in range(8)]); table.append(t)
    lines.append([9 * i for i in range(8)]); table.append(4)
    lines.append([7 * (i + 1) for i in range(8)]); table.append(4)
    return np.array(lines), np.array(table)


def main():
    cache, pf = sys.argv[1], sys.argv[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = np.fromfile(os.path.join(cache, "inputs.u8"), dtype=np.uint8).reshape(-1, 193)
    n = len(rows)
    teacher = np.fromfile(os.path.join(cache, "teacher_values.f16"), dtype=np.float16).astype(np.float32)
    sq = squares(rows)
    holes = (sq == 3).sum(1)
    ph = phase(holes)
    split = int(0.85 * n)
    tr, va = np.arange(n) < split, np.arange(n) >= split

    # NNUE error on the measured part (positions of phase_fit_dump: every 3rd)
    idx = np.fromfile(os.path.join(pf, "index.i64"), dtype=np.int64)
    nnue = np.fromfile(os.path.join(pf, "nnue.f32"), dtype=np.float32)
    nnue_err = {}
    for b in range(K):
        m = (idx >= split) & (ph[idx] == b)
        nnue_err[b] = np.mean((nnue[m] - teacher[idx[m]]) ** 2)

    # 1. Piece-square tables: 64 squares × 3 non-free states (one-hot) + constant, per phase.
    pst_err = {}
    for b in range(K):
        a, v = tr & (ph == b), va & (ph == b)
        def X(s):
            x = np.zeros((len(s), 192), np.float32)
            for st in (1, 2, 3):
                x[:, (st - 1) * 64:st * 64] = s == st
            return np.hstack([x, np.ones((len(s), 1), np.float32)]).astype(np.float64)
        Xa, Xv = X(sq[a]), X(sq[v])
        G = Xa.T @ Xa + 1e-3 * np.eye(193) * len(Xa)
        w = np.linalg.solve(G, Xa.T @ np.arctanh(np.clip(teacher[a], -0.98, 0.98)))
        pst_err[b] = np.mean((np.tanh(Xv @ w) - teacher[v]) ** 2)

    # 2. Line patterns + piece-square tables, per phase, trained with Adam.
    lines, table = line_indices()
    pow4 = 4 ** np.arange(7, -1, -1)
    conf = (sq[:, lines] * pow4).sum(-1)                     # (N, 18) configuration of each line
    gidx = torch.tensor(table * 65536, dtype=torch.long)     # offset of each line's table
    conf_t = torch.tensor(conf, dtype=torch.long)
    sq_t = torch.tensor(sq, dtype=torch.long)
    ph_t = torch.tensor(ph, dtype=torch.long)
    y_t = torch.tensor(teacher)
    lines_w = torch.zeros(K, 5 * 65536, device=dev, requires_grad=True)
    pst_w = torch.zeros(K, 64 * 4, device=dev, requires_grad=True)
    bias = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.Adam([lines_w, pst_w, bias], lr=3e-3)
    sq_off = torch.arange(64) * 4

    def predict(sel):
        p = ph_t[sel].to(dev)
        c = (conf_t[sel] + gidx).to(dev)
        s = (sq_t[sel] + sq_off).to(dev)
        z = lines_w[p[:, None], c].sum(1) + pst_w[p[:, None], s].sum(1) + bias[p]
        return torch.tanh(z)

    tr_idx = torch.tensor(np.nonzero(tr)[0])
    va_idx = torch.tensor(np.nonzero(va)[0])
    batch = 4096
    for epoch in range(6):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm), batch):
            sel = perm[i:i + batch]
            loss = ((predict(sel) - y_t[sel].to(dev)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = torch.cat([predict(va_idx[i:i + 65536]) for i in range(0, len(va_idx), 65536)]).cpu().numpy()
        print(f"  epoch {epoch + 1}: lines val mse {np.mean((pv - teacher[va]) ** 2):.4f}", flush=True)
    lines_err = {b: np.mean((pv[ph[va] == b] - teacher[va][ph[va] == b]) ** 2) for b in range(K)}

    print(f"\n{n} positions ({tr.sum()} fit, {va.sum()} measured). MSE against the teacher:")
    print(f"{'phase':>5} {'holes':>7} {'NNUE':>7} {'pst':>7} {'lines':>7}   lines / NNUE")
    for b in range(K):
        hs = holes[ph == b]
        print(f"{b:>5} {hs.min():>3}–{hs.max():<3} {nnue_err[b]:7.4f} {pst_err[b]:7.4f} {lines_err[b]:7.4f}   ×{lines_err[b] / nnue_err[b]:.2f}")
    w = np.array([(va & (ph == b)).sum() for b in range(K)], np.float64)
    tot = lambda e: sum(e[b] * w[b] for b in range(K)) / w.sum()
    print(f"{'all':>13} {tot(nnue_err):7.4f} {tot(pst_err):7.4f} {tot(lines_err):7.4f}")


if __name__ == "__main__":
    main()
