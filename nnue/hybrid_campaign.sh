#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# hybrid_campaign.sh — campaign H1: MCTS ResNet at the start of the game, then
# the minimax NNUE (player/hybrid_player.h), against dev_u9_spsa.
#
#     bash ~/Yolah/nnue/hybrid_campaign.sh            # submit the jobs (login node)
#     bash ~/Yolah/nnue/hybrid_campaign.sh status     # Elo of each hybrid against dev
#
# The user's idea (2026-11-22): the big network's judgement where the minimax
# is shallow (many free squares), the minimax's speed and exact endgame later.
# One job per switch ply, each a 2-player tournament (config/hybrid/players_K.txt):
#   hybrid_K vs dev, K = 12, 18, 24, 30 (MCTS for plies < K; the balanced book
#   plays plies 0-5, so K = 12 gives the MCTS 3 moves of each side), and
#   resnet vs dev (the hybrid that never switches).
# dev itself is K = 6 (switch at once): 0 Elo by definition. So the five jobs
# give the Elo of the hybrid against dev as a function of K.
# Settings of the app's top tournament (top3_tournament.sh, where the MCTS
# ResNet was 60 Elo below dev_u8): 1 s per move, 4 threads for the minimax,
# the MCTS on the GPU with 16 threads × batch 16; balanced openings, both colours.
# About a minute per game (fast endgames with the solver): ROUNDS 500 = 1000
# games per job in about 15-20 h.
#
#     ONLY=18 bash ~/Yolah/nnue/hybrid_campaign.sh    # (re)submit one job; a resubmitted
#                                                    # job resumes its games.csv
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
BASE="${BASE:-${HOME}/HybridH1}"
ROUNDS="${ROUNDS:-500}"
KS="${KS:-12 18 24 30 resnet}"
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22,crn23}"     # crn22: no containers; crn23: excluded by tournament.sh
# CPUs and memory per job: every game has the GPU player (no CPU queue), one game
# at a time: the MCTS's 16 threads (mostly waiting for the GPU) + dev's 4, in turn.
# tournament.sh's defaults (32 CPUs, 64 GB) let only one job per GPU node (crn20-24:
# 48 logical CPUs, shared with the trainings): the other jobs stayed pending.
HYB_CPUS="${HYB_CPUS:-16}"
HYB_MEM="${HYB_MEM:-16G}"

if [[ "${1:-submit}" == "status" ]]; then
    for k in ${KS}; do
        f="${BASE}/${k}/games.csv"
        [[ -f "${f}" ]] || { echo "${k}: no games yet"; continue; }
        n=$(( $(wc -l < "${f}") - 1 ))
        # Elo of the first player (hybrid_K or resnet) against dev, from this pair's games only.
        python3 - "${f}" "${k}" "${n}" <<'EOF'
import csv, math, sys
f, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
w = d = l = 0
for r in csv.DictReader(open(f)):
    s = float(r["result"]) if r["black"] != "dev" else 1 - float(r["result"])   # score of the non-dev player
    w += s == 1; d += s == 0.5; l += s == 0
if n == 0: print(f"{k}: no games yet"); sys.exit()
p = (w + d / 2) / n
var = (w * (1 - p) ** 2 + d * (0.5 - p) ** 2 + l * p ** 2) / n
elo = lambda x: -400 * math.log10(1 / min(max(x, 1e-6), 1 - 1e-6) - 1)
se = math.sqrt(var / n)
print(f"{k:>7}: {n:5d} games  +{w} ={d} -{l}  Elo vs dev {elo(p):+6.1f} [{elo(p - 1.96 * se):+6.1f}, {elo(p + 1.96 * se):+6.1f}]")
EOF
    done
    exit 0
fi

for k in ${KS}; do
    [[ -n "${ONLY:-}" && "${ONLY}" != "${k}" ]] && continue
    dir="${BASE}/${k}"
    mkdir -p "${dir}"
    jid=$(cd "${dir}" && sbatch --parsable --job-name="hyb_${k}" --time=2-00:00:00 --exclude="${GPU_EXCLUDE}" --cpus-per-task="${HYB_CPUS}" --mem="${HYB_MEM}" \
        --output="${dir}/hybrid_%j.out" --error="${dir}/hybrid_%j.out" \
        --export=ALL,TOURNAMENT_DIR="${dir}",PLAYERS=/Yolah/config/hybrid/players_${k}.txt,THREADS=4,TIME_US=1000000,BOOK=/Yolah/config/openings/balanced.txt,SEED=1,ROUNDS="${ROUNDS}" \
        "${YOLAH_DIR}/nnue/tournament.sh")
    echo "H1 ${k}: job ${jid} → ${dir}/games.csv"
done
echo "results: bash ${YOLAH_DIR}/nnue/hybrid_campaign.sh status"
