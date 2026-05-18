"""
Count how often the certain-win shortcut would fire at MCTS leaves.

We don't need the NN — the certain-win predicate is purely a function of
(black_score, white_score, ply). For every position reachable in the recorded
games we replay scores from the move bytes and check:

    effective_delta = (black_score - white_score) - (0 if black-to-move else 1)
    certain_win     = |effective_delta| >= 1

We report the fraction over:
    • all non-terminal positions (what MCTS leaves resemble)
    • broken down by game phase (early / mid / late / very-late)
    • duplicated-as-in-dataset count (matches __getitem__ semantics)

Skips the *.symmetries.txt files: those are D4 reflections of the originals
and have identical scores, so they'd 8× inflate the totals without changing
the ratio.
"""

import glob
import os
import sys
from collections import Counter

GAME_DIR = "./data"


def analyze_file(path):
    """Return per-bucket (positions, certain_wins) counts for one game file."""
    with open(path, "rb") as f:
        data = f.read()

    # buckets indexed by game-phase label
    counts = Counter()      # total positions per bucket
    cwins  = Counter()      # certain-win positions per bucket
    # also track dataset-style duplicated count
    ds_positions = 0
    ds_cwins     = 0

    idx = 0
    n_games = 0
    while idx < len(data):
        nb_moves        = data[idx]
        nb_random_moves = data[idx + 1]
        if nb_random_moves == nb_moves:
            idx += 2 + 2 * nb_moves + 2
            continue

        moves       = data[idx + 2 : idx + 2 + 2 * nb_moves]
        black_score = data[idx + 2 + 2 * nb_moves]
        white_score = data[idx + 2 + 2 * nb_moves + 1]
        n_games += 1

        # ── Replay scores from ply 0; check certain-win at each ply ───────
        bs, ws = 0, 0
        for ply in range(nb_moves + 1):
            # Check certain-win at this position.
            black_to_move = (ply & 1) == 0
            eff_delta = (bs - ws) - (0 if black_to_move else 1)
            is_cwin = eff_delta >= 1 or eff_delta <= -1

            # Bucket by game phase. Plies 0..63 roughly map to early/mid/late.
            if   ply < 16:  bucket = "0–15  (early)"
            elif ply < 32:  bucket = "16–31 (mid)"
            elif ply < 48:  bucket = "32–47 (late)"
            else:           bucket = "48+   (very late)"

            counts[bucket] += 1
            if is_cwin:
                cwins[bucket] += 1

            # Dataset replicates terminal position nb_random_moves extra times
            # (see GameDataset.__getitem__ semantics).
            if ply <= nb_moves:
                ds_positions += 1
                if is_cwin:
                    ds_cwins += 1

            # Apply move ply → ply+1
            if ply < nb_moves:
                sq1 = moves[2 * ply]
                sq2 = moves[2 * ply + 1]
                # Move.none() is (SQ_A1, SQ_A1) = (0, 0) and doesn't score.
                is_real = not (sq1 == 0 and sq2 == 0)
                if is_real:
                    if black_to_move:
                        bs += 1
                    else:
                        ws += 1

        # Sanity check final scores match the file footer.
        # (Disabled by default; uncomment to verify parser.)
        # assert bs == black_score and ws == white_score, (bs, ws, black_score, white_score)

        # Dataset adds `r` duplicated samples of the terminal position
        # (slices past the end of `moves` are no-ops).
        terminal_ply  = nb_moves
        b2m_terminal  = (terminal_ply & 1) == 0
        eff_terminal  = (bs - ws) - (0 if b2m_terminal else 1)
        cwin_terminal = eff_terminal >= 1 or eff_terminal <= -1
        ds_positions += nb_random_moves
        if cwin_terminal:
            ds_cwins += nb_random_moves

        idx += 2 + 2 * nb_moves + 2

    return counts, cwins, ds_positions, ds_cwins, n_games


def main(file_limit=None):
    files = sorted(glob.glob(os.path.join(GAME_DIR, "games_*.txt")))
    # Skip the augmented copies — they have identical scores per position.
    files = [f for f in files if not f.endswith(".symmetries.txt")]
    if file_limit:
        files = files[:file_limit]

    if not files:
        print(f"No files matched in {GAME_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Analyzing {len(files)} game files…", flush=True)

    total_counts = Counter()
    total_cwins  = Counter()
    ds_pos       = 0
    ds_cwins     = 0
    n_games      = 0

    for i, path in enumerate(files):
        c, w, dp, dw, ng = analyze_file(path)
        total_counts.update(c)
        total_cwins.update(w)
        ds_pos   += dp
        ds_cwins += dw
        n_games  += ng
        if (i + 1) % 100 == 0:
            print(f"  …{i + 1}/{len(files)} files", flush=True)

    print()
    print(f"Games processed: {n_games:,}")
    print()
    print("── Certain-win frequency by game phase ─────────────────────────")
    print(f"{'phase':<18}  {'positions':>14}  {'certain-win':>14}  {'fraction':>10}")
    print("-" * 64)
    bucket_order = ["0–15  (early)", "16–31 (mid)", "32–47 (late)", "48+   (very late)"]
    for bucket in bucket_order:
        n  = total_counts.get(bucket, 0)
        cw = total_cwins.get(bucket, 0)
        frac = cw / n if n else 0.0
        print(f"{bucket:<18}  {n:>14,}  {cw:>14,}  {frac:>10.4%}")
    n_all  = sum(total_counts.values())
    cw_all = sum(total_cwins.values())
    print("-" * 64)
    print(f"{'ALL':<18}  {n_all:>14,}  {cw_all:>14,}  {cw_all/n_all:>10.4%}")
    print()
    print("── Dataset-sampling frequency (matches __getitem__) ────────────")
    print(f"  total samples (with terminal duplicates) : {ds_pos:,}")
    print(f"  certain-win samples                       : {ds_cwins:,}")
    print(f"  fraction                                  : {ds_cwins/ds_pos:.4%}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(file_limit=limit)
