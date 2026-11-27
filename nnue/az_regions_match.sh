#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# az_regions_match.sh — do the settled positions help the AlphaZero MCTS?
#
# Retrograde analysis project, step D for the MCTS: with "regions" (SearchParams::
# regions), a leaf where no region of free squares is contested any more is a
# finished game for the search: each side will make exactly its private reserve
# of moves (player/solo_region.h, solved exactly), so its exact value replaces
# the network's. A: with it, B: without; both with the app's score utility
# (0.05, scale 10). Otherwise as below (the match of az_score_match.sh).
#
# (The comment of az_score_match.sh, which this script copies:)
#
# The search values a finished game ±1 (win/loss) or 0 (draw): once the game
# is decided every move looks the same and it plays loose moves. With a score
# utility u (SearchParams::score_utility, "score utility" in the config) a
# finished game is worth (1 − u)·sign(d) + u·tanh(d / scale), d the final
# margin: a win still beats a draw, but a bigger win beats a smaller one.
# The network predicts no score, so only the finished games of the search
# carry it (the endgames).
#
# A match of one network against itself: A with the score utility, B without
# (config/alphazero_mcts_eval_player.cfg otherwise), balanced openings, both
# colours, split over an array of GPUs (alphazero_match.sh, 16 h each).
#
#     bash ~/Yolah/nnue/az_regions_match.sh            # submit (login node)
#     U=0.1 bash ~/Yolah/nnue/az_regions_match.sh      # another utility
#     bash ~/Yolah/nnue/az_regions_match.sh status     # sum of the tasks so far
#     U=0.1 bash ~/Yolah/nnue/az_regions_match.sh status
#
# status (also while the tasks run: it reads their logs): each task's games so
# far, A's W/D/L, score, Elo with its 95 % interval, and the mean final margin
# of A's wins and of B's wins (A should win by more if the utility does what it
# is meant to). JOB=<array job id> to keep one submission only.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/AlphaZeroLearn/work}"   # the AlphaZero 256 run
STEP="${STEP:-271360}"           # the network of the app's AlphaZero (models/az_00271360)
U="${U:-0.05}"                   # score utility
SCALE="${SCALE:-10}"             # tanh scale of the margin, in points
GAMES="${GAMES:-2000}"
TASKS="${TASKS:-5}"
MOVE_SECONDS="${MOVE_SECONDS:-1}"
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22,crn23}"
TAG="regions"
NAME=$(printf "az_%08d" "${STEP}")
mkdir -p "${WORK_DIR}/logs"

if [[ "${1:-submit}" == "status" ]]; then
    # Read from the tasks' logs: running tasks count too (every finished game prints a line).
    # JOB=<array job id> keeps one submission only (default: every submission of this U).
    python3 - "${WORK_DIR}/logs" "${TAG}" "${JOB:-}" <<'EOF'
import glob, math, re, sys
d, tag, job = sys.argv[1:4]
logs = sorted(glob.glob(f"{d}/az_regions_{tag}_{job or '*'}_*.out"))
if not logs:
    sys.exit(f"no log yet ({d}/az_regions_{tag}_*.out)")
pat = re.compile(r"^game (\d+)/(\d+): A (black|white) score (\d+)-(\d+) \(\d+ plies\) -> A (win|draw|loss)")
w = dr = l = 0; win_a = []; win_b = []
print(f"score utility {tag[2:]} (A) vs none (B)")
for f in logs:
    done = total = 0
    for line in open(f, errors="replace"):
        m = pat.match(line)
        if not m: continue
        done, total = int(m.group(1)), int(m.group(2))
        margin = abs(int(m.group(4)) - int(m.group(5)))
        res = m.group(6)
        if res == "win": w += 1; win_a.append(margin)
        elif res == "loss": l += 1; win_b.append(margin)
        else: dr += 1
    print(f"  {f.rsplit('/', 1)[1]}: {done}/{total or '?'} games")
n = w + dr + l
if n == 0:
    sys.exit("no finished game yet")
s = (w + dr / 2) / n
se = math.sqrt(max((w + dr / 4) / n - s * s, 0) / n)
elo = lambda x: 400 * math.log10(min(max(x, 1e-3), 1 - 1e-3) / (1 - min(max(x, 1e-3), 1 - 1e-3)))
print(f"{n} games: A {w}W {dr}D {l}L, score {s:.3f} ± {1.96 * se:.3f}  →  Elo {elo(s):+.0f}  [{elo(s - 1.96 * se):+.0f}, {elo(s + 1.96 * se):+.0f}]")
mean = lambda v: sum(v) / len(v) if v else float("nan")
print(f"mean margin of A's wins {mean(win_a):.1f} ({len(win_a)} games), of B's wins {mean(win_b):.1f} ({len(win_b)} games)")
EOF
    exit 0
fi

[[ -s "${WORK_DIR}/models/${NAME}.ts" || -s "${WORK_DIR}/models/${NAME}.pt" ]] || { echo "missing: ${WORK_DIR}/models/${NAME}.ts (or .pt); STEP=... for another network" >&2; exit 1; }
(( GAMES % (2 * TASKS) == 0 )) || { echo "GAMES must be a multiple of 2 × TASKS" >&2; exit 1; }
cd "${WORK_DIR}/logs"
JID=$(SET_A="regions=true;score utility=${U};score scale=${SCALE}" SET_B="score utility=${U};score scale=${SCALE}" TAG="${TAG}" WORK_DIR="${WORK_DIR}" \
    sbatch --parsable --export=ALL,A="${STEP}",B="${STEP}",GAMES=$(( GAMES / TASKS )),MOVE_SECONDS="${MOVE_SECONDS}" \
    --array=0-$(( TASKS - 1 )) --time=16:00:00 --exclude="${GPU_EXCLUDE}" \
    --job-name="az_regions" --output="${WORK_DIR}/logs/az_regions_${TAG}_%A_%a.out" --error="${WORK_DIR}/logs/az_regions_${TAG}_%A_%a.out" \
    "${YOLAH_DIR}/nnue/alphazero_match.sh")
echo "az_regions ${JID}: ${NAME} with regions vs without (both score utility ${U}, scale ${SCALE}), ${GAMES} games on ${TASKS} GPUs, ${MOVE_SECONDS} s/move"
echo "logs: ${WORK_DIR}/logs/az_regions_${TAG}_${JID}_*.out   —   result: bash ${YOLAH_DIR}/nnue/az_regions_match.sh status"
echo "$(date '+%F %T') az_regions ${JID} U=${U} SCALE=${SCALE} STEP=${STEP} GAMES=${GAMES}" >> "${WORK_DIR}/az_regions_jobs.txt"
