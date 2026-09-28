#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# alphazero_match.sh — a long match between two networks of a learning run,
# on its own GPU, to measure progress more precisely than the 20-game
# evaluations of alphazero_learn.sh (±78 Elo each).
#
#     sbatch alphazero_match.sh                          # latest vs initial, 200 games, 0.5 s/move
#     sbatch --export=ALL,A=7168,B=4096 alphazero_match.sh
#     sbatch --export=ALL,GAMES=400,MOVE_SECONDS=1 alphazero_match.sh
#
# A and B are training steps (the numbers of models/az_<step>), or "latest"
# (the network self-play is using) or "initial" (step 0). A network whose .ts
# was pruned by the run is re-traced from its .pt.
#
# Both players use config/alphazero_mcts_eval_player.cfg (no noise, no
# sampling). Games come in pairs: a random opening of OPENING_PLIES plies,
# played once with each colour. The result — A's wins/draws/losses, score and
# Elo difference with a 95 % confidence interval — is printed and appended to
# WORK_DIR/matches.csv; every game is in WORK_DIR/evals/match_*.json.
#
# 200 games × ~55 plies × 0.5 s ≈ 1.5 h;  ±25 Elo (95 %: about ±50).
# It can run while alphazero_learn.sh is running (another GPU, same WORK_DIR).
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=alphazero_match
#SBATCH --output=alphazero_match_%j.out
#SBATCH --error=alphazero_match_%j.out
#SBATCH --partition=insa-gpu
#SBATCH -x crn23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"                       # SIF, WORK_DIR, run(), ...

A="${A:-latest}"                  # step, "latest" or "initial"
B="${B:-initial}"
GAMES="${GAMES:-200}"             # even: games come in pairs (one opening, both colours)
# (not SECONDS: that is bash's own counter of seconds since the shell started)
SECONDS_PER_MOVE="${MOVE_SECONDS:-0.5}"
OPENING_PLIES="${OPENING_PLIES:-4}"
SEED="${SEED:-${SLURM_JOB_ID:-$$}}"
JOB="${SLURM_JOB_ID:-$$}"

(( GAMES % 2 == 0 )) || { echo "ERROR: GAMES must be even (pairs of games)"; exit 1; }
[[ -f "${WORK_DIR}/latest.json" ]] || { echo "ERROR: no run in ${WORK_DIR} (latest.json missing)"; exit 1; }

# "latest" / "initial" / a step → a step number.
resolve() {
    case "$1" in
        latest)  grep -o '"step": *[0-9]*' "${WORK_DIR}/latest.json" | grep -o '[0-9]*$' ;;
        initial) echo 0 ;;
        *)       echo "$1" ;;
    esac
}
STEP_A=$(resolve "${A}")
STEP_B=$(resolve "${B}")
NAME_A=$(printf "az_%08d" "${STEP_A}")
NAME_B=$(printf "az_%08d" "${STEP_B}")

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${JOB} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Work dir   : ${WORK_DIR}"
echo "  Match      : A = ${NAME_A} (${A})  vs  B = ${NAME_B} (${B})"
echo "  Games      : ${GAMES} (${OPENING_PLIES}-ply random openings, both colours), ${SECONDS_PER_MOVE} s/move"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

build_alphazero_learn

# Private copies of the two networks: the running learning job prunes old
# .ts files at every export and must not remove one in the middle of the match.
MDIR="/work/evals/match_${JOB}"
mkdir -p "${WORK_DIR}/evals/match_${JOB}"
run bash -c "set -e; cd /Yolah/nnue
    for S in ${STEP_A} ${STEP_B}; do
        NAME=\$(printf 'az_%08d' \$S)
        if [[ -f /work/models/\$NAME.ts ]]; then
            cp /work/models/\$NAME.ts ${MDIR}/\$NAME.ts
        elif [[ -f /work/models/\$NAME.pt ]]; then
            echo \"re-tracing \$NAME.ts from its .pt\"
            # export_model writes <dir>/models/az_<step>.{pt,ts}
            mkdir -p ${MDIR}/models
            python3 -c \"
from alphazero_learn import export_model, make_net, load_state_dict
export_model(make_net(load_state_dict('/work/models/\$NAME.pt')), '${MDIR}', \$S)
\"
            mv ${MDIR}/models/\$NAME.ts ${MDIR}/\$NAME.ts
            rm -rf ${MDIR}/models
        else
            echo \"ERROR: neither /work/models/\$NAME.ts nor .pt exists\"; exit 1
        fi
    done"

OUT="/work/evals/match_${NAME_A}_vs_${NAME_B}_${JOB}.json"
TIME_US=$(awk -v s="${SECONDS_PER_MOVE}" 'BEGIN { printf "%d", s * 1000000 }')
echo "[$(date '+%F %T')] === Match ==="
start_in_container ${BUILD_DIR}/alphazero_learn match \
    --config /Yolah/config/alphazero_mcts_eval_player.cfg \
    --a "${MDIR}/${NAME_A}.ts" --b "${MDIR}/${NAME_B}.ts" \
    --games "${GAMES}" --time "${TIME_US}" --opening-plies "${OPENING_PLIES}" \
    --seed "${SEED}" --out "${OUT}"
wait_forwarding_signals
rm -rf "${WORK_DIR}/evals/match_${JOB}"
[[ "${RC}" == "0" ]] || { echo "match failed (exit code ${RC})"; exit "${RC}"; }

# ── Result: score, Elo difference and its 95 % confidence interval ──────────
# Per-game results x ∈ {1, ½, 0}: mean s, variance E[x²] − s², standard error
# √(var/n). The interval of s maps to Elo through Δ = 400·log10(s/(1−s)).
RES="${WORK_DIR}/evals/match_${NAME_A}_vs_${NAME_B}_${JOB}.json"
# By name: the JSON keys are written in alphabetical order, not W/D/L.
count() { grep -oE "\"$1\": *[0-9]+" "${RES}" | head -1 | grep -oE '[0-9]+$'; }
W=$(count wins); D=$(count draws); L=$(count losses)
awk -v w="$W" -v d="$D" -v l="$L" -v a="${NAME_A}" -v b="${NAME_B}" -v t="${SECONDS_PER_MOVE}" \
    -v csv="${WORK_DIR}/matches.csv" -v date="$(date '+%F %T')" 'BEGIN {
    n = w + d + l; s = (w + d / 2) / n
    var = (w + d / 4) / n - s * s; se = sqrt(var / n)
    lo = s - 1.96 * se; hi = s + 1.96 * se
    if (lo < 0.001) lo = 0.001; if (hi > 0.999) hi = 0.999
    sc = s; if (sc < 0.001) sc = 0.001; if (sc > 0.999) sc = 0.999
    elo = 400 * log(sc / (1 - sc)) / log(10)
    elo_lo = 400 * log(lo / (1 - lo)) / log(10); elo_hi = 400 * log(hi / (1 - hi)) / log(10)
    printf "\n%s vs %s: %dW %dD %dL over %d games (%s s/move)\n", a, b, w, d, l, n, t
    printf "score %.3f ± %.3f  →  Elo %+.0f  (95%% interval [%+.0f, %+.0f])\n", s, 1.96 * se, elo, elo_lo, elo_hi
    new = (system("test -f " csv) != 0)
    if (new) print "time,a,b,games,seconds_per_move,wins,draws,losses,score,elo,elo_low95,elo_high95" >> csv
    printf "%s,%s,%s,%d,%s,%d,%d,%d,%.3f,%+.0f,%+.0f,%+.0f\n", date, a, b, n, t, w, d, l, s, elo, elo_lo, elo_hi >> csv
}'
echo "(appended to ${WORK_DIR}/matches.csv; games in ${RES})"
