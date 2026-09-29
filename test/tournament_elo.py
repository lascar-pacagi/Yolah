"""
tournament_elo.py — Elo ratings from the games of a tournament (Yolah --tournament).

    python3 tournament_elo.py tournament_games.csv [--anchor random_player] [--bootstrap 200]
                              [--matrix pairs.csv]

Model (Bradley–Terry with Elo scaling): player i scores against player j,
in expectation,
        E_ij = 1 / (1 + 10^((r_j − r_i) / 400)),
a draw counting as half a point. The ratings r maximise the likelihood of all
the games at once,
        L(r) = Σ_games  s·log E_bw + (1 − s)·log(1 − E_bw),   s ∈ {1, ½, 0},
so a player's rating uses every opponent — not only the ones it met most. The
ratings are defined up to a constant: the anchor player is set to 0 (by
default the mean is 0).

Prior: every player also has --prior (2) virtual draws against a player rated
at the mean, as in BayesElo. Without it, a player that lost (or won) all its
games would be rated at −∞ (+∞) and would drag every other rating along; with
the hundreds of games per pair of a real tournament, the prior is negligible.

Uncertainty: the games are resampled with replacement and the ratings fitted
again (bootstrap); the table gives the 2.5 % and 97.5 % quantiles — a 95 %
interval for each rating RELATIVE TO THE ANCHOR.

Colours: the tournament plays every opening once with each colour, so the
first player's advantage cancels within each pair and is not modelled here.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np


def load(path):
    games = []
    with open(path) as f:
        for r in csv.DictReader(f):
            games.append((r["black"], r["white"], float(r["result"])))
    return games


def fit(players, games_idx, weights=None, iters=200, prior=2.0):
    """Maximum a posteriori ratings (Newton; `prior` virtual draws per player vs rating 0)."""
    n = len(players)
    b = np.array([g[0] for g in games_idx]); w = np.array([g[1] for g in games_idx])
    s = np.array([g[2] for g in games_idx], dtype=float)
    k = np.ones(len(s)) if weights is None else weights
    c = math.log(10) / 400
    r = np.zeros(n)
    for _ in range(iters):
        e = 1.0 / (1.0 + np.exp(-c * (r[b] - r[w])))
        g = k * c * (s - e)                               # d L / d(r_b − r_w)
        grad = np.bincount(b, g, n) - np.bincount(w, g, n)
        h = k * c * c * e * (1 - e)
        H = np.zeros((n, n))
        np.add.at(H, (b, b), -h); np.add.at(H, (w, w), -h)
        np.add.at(H, (b, w), h); np.add.at(H, (w, b), h)
        # prior: `prior` draws of each player against a virtual player at 0
        ep = 1.0 / (1.0 + np.exp(-c * r))
        grad += prior * c * (0.5 - ep)
        H -= np.diag(prior * c * c * ep * (1 - ep))
        H -= 1e-9 * np.eye(n)
        step = np.linalg.solve(H, -grad)
        r += step
        if np.abs(step).max() < 1e-6:
            break
    return r - r.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games")
    ap.add_argument("--anchor", default=None, help="player rated 0 (default: the mean is 0)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--matrix", default=None, help="write the pairwise scores to this CSV")
    ap.add_argument("--prior", type=float, default=2.0, help="virtual draws per player against the mean")
    args = ap.parse_args()

    games = load(args.games)
    if not games:
        sys.exit("no games")
    players = sorted({p for g in games for p in g[:2]})
    idx = {p: i for i, p in enumerate(players)}
    gi = [(idx[b], idx[w], s) for b, w, s in games]
    n = len(players)

    def anchored(r):
        return r - r[idx[args.anchor]] if args.anchor else r

    r = anchored(fit(players, gi, prior=args.prior))
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(args.bootstrap):
        counts = np.bincount(rng.integers(0, len(gi), len(gi)), minlength=len(gi)).astype(float)
        boots.append(anchored(fit(players, gi, weights=counts, iters=50, prior=args.prior)))
    boots = np.array(boots) if boots else np.zeros((1, n))
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    pts = defaultdict(float); cnt = defaultdict(int); draws = defaultdict(int)
    pair = defaultdict(lambda: [0.0, 0])
    for b, w, s in games:
        pts[b] += s; pts[w] += 1 - s; cnt[b] += 1; cnt[w] += 1
        if s == 0.5: draws[b] += 1; draws[w] += 1
        pair[(b, w)][0] += s; pair[(b, w)][1] += 1
        pair[(w, b)][0] += 1 - s; pair[(w, b)][1] += 1

    order = np.argsort(-r)
    print(f"{len(games)} games, {n} players" + (f", {args.anchor} = 0" if args.anchor else ", mean = 0"))
    print(f"{'rank':>4}  {'player':42s} {'Elo':>6}  {'95% interval':>15}  {'games':>6}  {'score':>6}  {'draws':>6}")
    for k, i in enumerate(order, 1):
        p = players[i]
        print(f"{k:4d}  {p:42s} {r[i]:+6.0f}  [{lo[i]:+6.0f},{hi[i]:+6.0f}]  {cnt[p]:6d}  "
              f"{100 * pts[p] / cnt[p]:5.1f}%  {100 * draws[p] / cnt[p]:5.1f}%")

    if args.matrix:
        with open(args.matrix, "w", newline="") as f:
            wr = csv.writer(f)
            names = [players[i] for i in order]
            wr.writerow(["player"] + names)
            for a in names:
                wr.writerow([a] + ["" if a == b or pair[(a, b)][1] == 0
                                   else f"{100 * pair[(a, b)][0] / pair[(a, b)][1]:.1f}%" for b in names])
        print(f"pairwise scores (row player's %) in {args.matrix}")


if __name__ == "__main__":
    main()
