"""
make_book_fig.py — fig-book.png of the "book of balanced openings" section of
../learning.qmd: (left) the minimax value of 200 000 random 6-ply openings and
the window the book is taken from; (right) Black's actual score against that
value, from games of the minimax player against itself (the calibration of
`opening_book build --target`).

    ../../build/opening_book check --book all.txt --games 6000 --depth 10 \
        --random-plies 0 --csv book_calibration.csv       # all.txt: build --max-value 30000
    python3 make_book_fig.py ../../config/openings/balanced.txt book_calibration.csv
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def book_header(path):
    """The depth-6 histogram, the target and the window of a book's header."""
    bins, target, half = [], 0, 0
    for line in open(path):
        if not line.startswith("#"):
            break
        m = re.match(r"#\s+\[\s*([+-]?\d+),\s*([+-]?\d+)\)\s+(\d+)", line)
        if m:
            bins.append((int(m[1]), int(m[2]), int(m[3])))
        m = re.search(r"all within ([+-]?\d+) ± (\d+)", line)
        if m:
            target, half = int(m[1]), int(m[2])
    return bins, target, half


def main():
    book = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "config", "openings", "balanced.txt")
    calib = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "book_calibration.csv")
    bins, target, half = book_header(book)
    d = np.loadtxt(calib, delimiter=",", skiprows=1)
    v, r = d[:, 0], d[:, 1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    lo = np.array([x[0] for x in bins]); n = np.array([x[2] for x in bins])
    a.bar(lo / 30000, n / n.sum() * 100, width=1500 / 30000, align="edge", color="#8aa", edgecolor="white")
    a.axvspan((target - half) / 30000, (target + half) / 30000, color="#d62", alpha=0.35, label="the book")
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("value for Black at depth 6 (tanh scale)")
    a.set_ylabel("% of the random openings")
    a.set_title("200 000 random 6-ply openings")
    a.legend(frameon=False)

    edges = np.arange(-6000, 14001, 1000)
    xs, ys, es = [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (v >= e0) & (v < e1)
        if m.sum() >= 20:
            xs.append((e0 + e1) / 2 / 30000); ys.append(r[m].mean()); es.append(1.96 * r[m].std() / np.sqrt(m.sum()))
    b.errorbar(xs, ys, yerr=es, fmt="o", ms=4, color="#246", capsize=2, label="games (bins of 1000)")
    sel = np.abs(v - 3000) < 6000
    c1, c0 = np.polyfit(v[sel], r[sel], 1)
    xx = np.linspace(-3000, 9000, 50)
    b.plot(xx / 30000, c0 + c1 * xx, color="#d62", label="linear fit")
    b.axhline(0.5, color="k", lw=0.6); b.axvline(0, color="k", lw=0.6)
    b.axvline(target / 30000, color="#d62", ls="--", lw=1, label=f"target {target / 30000:+.2f}")
    b.set_xlabel("value for Black at depth 12 (tanh scale)")
    b.set_ylabel("Black's score")
    b.set_title("minimax vs itself, depth 10, 6000 games")
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-book.png"), dpi=200)


if __name__ == "__main__":
    main()
