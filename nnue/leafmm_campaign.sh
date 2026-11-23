#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# leafmm_campaign.sh — campaign H2: the MCTS ResNet of the app whose late
# leaves are valued by a short minimax NNUE search (player/leaf_minimax.h),
# against dev_u9_spsa.
#
#     bash ~/Yolah/nnue/leafmm_campaign.sh            # submit the jobs (login node)
#     bash ~/Yolah/nnue/leafmm_campaign.sh status     # Elo of each variant against dev
#
# The user's idea (2026-11-23), after H1 (switching players for the whole
# move: flat, ResNet alone ≈ dev): keep the MCTS and the ResNet's policy, but
# from leaf ply P on, the value of a new leaf is a dev_u9_spsa search of N
# nodes under it (with its exact endgame when it reaches it) instead of the
# value head. One job per (P, N), a 2-player tournament against dev
# (config/leafmm/players_P_N.txt, made by config/leafmm/make_configs.py):
#   P = 24, 32, 40 (the balanced book plays plies 0-5; games end around ply 52-58)
#   N = 1000, 4000, 16000 nodes (measured locally, RTX 3080: 0.7, 2.0 and 7 ms per leaf)
# The CPU budget: at most 4 leaf searches at a time ("concurrency"), as many
# threads as the opponent's minimax; the GPU as in H1. The reference with no
# leaf minimax is H1's "resnet" job (~/HybridH1/resnet, same settings).
# 1 s per move, balanced openings, both colours. ROUNDS 250 = 500 games per
# job (about 8-10 h); a resubmitted job resumes its games.csv, so the
# promising ones can be extended: ROUNDS=500 ONLY=32_4000 bash ... .
# Every game prints one line "[leaf minimax] ..." in the job's log: new
# leaves per second before / after, nodes and ms per leaf searched, % proven.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
BASE="${BASE:-${HOME}/LeafMM_H2}"
ROUNDS="${ROUNDS:-250}"
KS="${KS:-24_1000 24_4000 24_16000 32_1000 32_4000 32_16000 40_1000 40_4000 40_16000}"
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22,crn23}"     # crn22: no containers; crn23: excluded by tournament.sh
# CPUs and memory per job: the MCTS's 16 threads (4 of them at most in a leaf
# search) + dev's 4, one game at a time; 16 minimax tables of 16 MB.
HYB_CPUS="${HYB_CPUS:-16}"
HYB_MEM="${HYB_MEM:-16G}"

if [[ "${1:-submit}" == "status" ]]; then
    for k in ${KS}; do
        f="${BASE}/${k}/games.csv"
        [[ -f "${f}" ]] || { echo "${k}: no games yet"; continue; }
        n=$(( $(wc -l < "${f}") - 1 ))
        # Elo of the first player (mm_P_N) against dev, from this pair's games only.
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
    echo "reference (no leaf minimax): bash ${YOLAH_DIR}/nnue/hybrid_campaign.sh status (resnet)"
    exit 0
fi

for k in ${KS}; do
    [[ -n "${ONLY:-}" && "${ONLY}" != "${k}" ]] && continue
    dir="${BASE}/${k}"
    mkdir -p "${dir}"
    jid=$(cd "${dir}" && sbatch --parsable --job-name="lmm_${k}" --time=2-00:00:00 --exclude="${GPU_EXCLUDE}" --cpus-per-task="${HYB_CPUS}" --mem="${HYB_MEM}" \
        --output="${dir}/leafmm_%j.out" --error="${dir}/leafmm_%j.out" \
        --export=ALL,TOURNAMENT_DIR="${dir}",PLAYERS=/Yolah/config/leafmm/players_${k}.txt,THREADS=4,TIME_US=1000000,BOOK=/Yolah/config/openings/balanced.txt,SEED=1,ROUNDS="${ROUNDS}" \
        "${YOLAH_DIR}/nnue/tournament.sh")
    echo "H2 ${k}: job ${jid} → ${dir}/games.csv"
done
echo "results: bash ${YOLAH_DIR}/nnue/leafmm_campaign.sh status"
