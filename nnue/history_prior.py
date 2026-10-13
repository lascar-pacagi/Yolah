"""
history_prior.py — the history prior of the dev player (step T): what the
history learns during a search, learnt beforehand from the cutoffs of many
searches.

    ./search_bench cutstats --player CFG --cache DIR --depth 8 --out cs_0.bin [--stride S --offset K]
    python3 history_prior.py OUT.prior.bin cs_0.bin [cs_1.bin ...] [--scale 1500] [--smooth 50]

For each phase (free squares · 8 // 57), colour, from and to: how often the
move was tried in a node's main loop and how often it cut. The prior is the
move's cut rate against the phase's mean cut rate, as a difference of
log-odds:

    p     = (cut + smooth · mean) / (tried + smooth)     (a move seldom tried stays near the mean)
    prior = scale · (logit(p) − logit(mean)),  clipped to ±8192

written as int16, in the order of MinMaxNNUE_DevPlayer::prior_of
([phase][colour][from][to]). The player scales it by "history prior weight".
"""
import sys
import numpy as np

PHASES, SQ = 8, 64


def main():
    args = sys.argv[1:]
    opts = {"--scale": 1500.0, "--smooth": 50.0}
    files = []
    i = 0
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = float(args[i + 1])
            i += 2
        else:
            files.append(args[i])
            i += 1
    out, inputs = files[0], files[1:]
    shape = (PHASES, 2, SQ, SQ)
    tried = np.zeros(shape, np.float64)
    cut = np.zeros(shape, np.float64)
    for f in inputs:
        a = np.fromfile(f, dtype=np.uint32)
        tried += a[:a.size // 2].reshape(shape)
        cut += a[a.size // 2:].reshape(shape)
    logit = lambda p: np.log(p / (1 - p))
    prior = np.zeros(shape, np.float64)
    print(f"{'phase':>5} {'free':>7} {'tried':>12} {'cut rate':>9} {'moves seen':>11} {'≥100 tries':>11}"
          f" {'prior sd':>9} {'min':>7} {'max':>7}")
    for ph in range(PHASES):
        t, c = tried[ph], cut[ph]
        mean = c.sum() / max(t.sum(), 1)
        if t.sum() == 0:
            continue
        p = (c + opts["--smooth"] * mean) / (t + opts["--smooth"])
        prior[ph] = np.clip(opts["--scale"] * (logit(p) - logit(mean)), -8192, 8192)
        seen = t > 0
        free_lo = -(-ph * 57 // PHASES)
        free_hi = -(-(ph + 1) * 57 // PHASES) - 1
        print(f"{ph:>5} {free_lo:>3}–{free_hi:<3} {int(t.sum()):>12} {mean:9.3f} {int(seen.sum()):>11}"
              f" {int((t >= 100).sum()):>11} {prior[ph][seen].std():9.0f} {prior[ph].min():7.0f} {prior[ph].max():7.0f}")
    prior.round().astype(np.int16).tofile(out)
    print(f"prior written to {out} ({PHASES}×2×64×64 int16)")


if __name__ == "__main__":
    main()
