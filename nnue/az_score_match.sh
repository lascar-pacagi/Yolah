#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# az_score_match.sh — does a score utility help the AlphaZero MCTS?
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
#     bash ~/Yolah/nnue/az_score_match.sh            # submit (login node)
#     U=0.1 bash ~/Yolah/nnue/az_score_match.sh      # another utility
#     bash ~/Yolah/nnue/az_score_match.sh status     # sum of the tasks so far
#     U=0.1 bash ~/Yolah/nnue/az_score_match.sh status
#
# status: A's W/D/L, score, Elo with its 95 % interval, and the mean final
# margin of A's wins and of B's wins (A should win by more if the utility
# does what it is meant to).
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
TAG="su${U}"
NAME=$(printf "az_%08d" "${STEP}")
mkdir -p "${WORK_DIR}/logs"

if [[ "${1:-submit}" == "status" ]]; then
    python3 - "${WORK_DIR}/evals" "${NAME}+${TAG}" "${NAME}" <<'EOF'
import glob, json, math, sys
d, a, b = sys.argv[1:4]
files = sorted(glob.glob(f"{d}/match_{a}_vs_{b}_*.json"))
if not files:
    sys.exit(f"no finished task yet ({d}/match_{a}_vs_{b}_*.json)")
w = dr = l = 0; win_a = []; win_b = []
for f in files:
    r = json.load(open(f))
    w += r["wins"]; dr += r["draws"]; l += r["losses"]
    for g in r["records"]:
        m = abs(g["black_score"] - g["white_score"])
        if g["result_for_a"] == "win": win_a.append(m)
        elif g["result_for_a"] == "loss": win_b.append(m)
n = w + dr + l; s = (w + dr / 2) / n
se = math.sqrt(max((w + dr / 4) / n - s * s, 0) / n)
elo = lambda x: 400 * math.log10(min(max(x, 1e-3), 1 - 1e-3) / (1 - min(max(x, 1e-3), 1 - 1e-3)))
print(f"{a} vs {b}: {len(files)} task(s) finished, {n} games")
print(f"A {w}W {dr}D {l}L, score {s:.3f} ± {1.96 * se:.3f}  →  Elo {elo(s):+.0f}  [{elo(s - 1.96 * se):+.0f}, {elo(s + 1.96 * se):+.0f}]")
mean = lambda v: sum(v) / len(v) if v else float("nan")
print(f"mean margin of A's wins {mean(win_a):.1f} ({len(win_a)} games), of B's wins {mean(win_b):.1f} ({len(win_b)} games)")
EOF
    exit 0
fi

[[ -s "${WORK_DIR}/models/${NAME}.ts" || -s "${WORK_DIR}/models/${NAME}.pt" ]] || { echo "missing: ${WORK_DIR}/models/${NAME}.ts (or .pt); STEP=... for another network" >&2; exit 1; }
(( GAMES % (2 * TASKS) == 0 )) || { echo "GAMES must be a multiple of 2 × TASKS" >&2; exit 1; }
cd "${WORK_DIR}/logs"
JID=$(SET_A="score utility=${U};score scale=${SCALE}" TAG="${TAG}" WORK_DIR="${WORK_DIR}" \
    sbatch --parsable --export=ALL,A="${STEP}",B="${STEP}",GAMES=$(( GAMES / TASKS )),MOVE_SECONDS="${MOVE_SECONDS}" \
    --array=0-$(( TASKS - 1 )) --time=16:00:00 --exclude="${GPU_EXCLUDE}" \
    --job-name="az_score" --output="${WORK_DIR}/logs/az_score_${TAG}_%A_%a.out" --error="${WORK_DIR}/logs/az_score_${TAG}_%A_%a.out" \
    "${YOLAH_DIR}/nnue/alphazero_match.sh")
echo "az_score ${JID}: ${NAME} with score utility ${U} (scale ${SCALE}) vs without, ${GAMES} games on ${TASKS} GPUs, ${MOVE_SECONDS} s/move"
echo "logs: ${WORK_DIR}/logs/az_score_${TAG}_${JID}_*.out   —   result: bash ${YOLAH_DIR}/nnue/az_score_match.sh status"
echo "$(date '+%F %T') az_score ${JID} U=${U} SCALE=${SCALE} STEP=${STEP} GAMES=${GAMES}" >> "${WORK_DIR}/az_score_jobs.txt"
