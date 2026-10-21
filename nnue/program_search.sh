#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# program_search.sh — search cheap bitboard programs (features like the
# influence) that add to what the NNUE predicts (nnue/program_search.py).
#
#     sbatch -p insa-cpu --exclude=crn07,crn14,crn15 nnue/program_search.sh
#
# 1. The positions (once): every STRIDE-th position of data/games with the
#    value of the base network → WORK_DIR/program_search/positions_<base>.bin
# 2. The search: enumeration then evolution for MINUTES minutes, with
#    CPUS workers. Follow it live:
#        tail -f yolah_progsearch_<job>.out                 (new best and new Pareto programs)
#        cat ~/SearchStudy/program_search/<RUN>/leaderboard.txt   (best 50 + Pareto front, every minute)
#
# Environment: BASE (base network, default the repo's k8_them), BASE_FEATURES
# (known features added to the base: infl,qfirst,mob; default none), STRIDE (20),
# N (positions used, 1000000), MINUTES (1380 = 23 h), MAX_SIZE (6), BODY_SIZE (5),
# MAX_NODES (15), WEIGHTS ("1,10" = points, result), RUN (output dir name), SEED.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=yolah_progsearch
#SBATCH --output=yolah_progsearch_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/SearchStudy}"
OUT="${WORK_DIR}/program_search"
mkdir -p "${OUT}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, run(), build

BASE="${BASE:-models/nnue_persp_k8_them_distill.quantized.txt}"
BASE_TAG=$(basename "${BASE}" .quantized.txt)
STRIDE="${STRIDE:-20}"
POS="positions_${BASE_TAG}_s${STRIDE}.bin"
RUN="${RUN:-run_$(date +%m%d_%H%M)${BASE_FEATURES:+_${BASE_FEATURES//,/_}}}"
CPUS="${SLURM_CPUS_PER_TASK:-32}"
echo "[$(date '+%F %T')] $(hostname), Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?'), base ${BASE}, run ${RUN}"

build_alphazero_learn program_search_dump
if [[ ! -f "${OUT}/${POS}" ]]; then
    start_in_container bash -c "exec ${BUILD_DIR}/program_search_dump /Yolah/data/games /Yolah/${BASE} \
        /work/program_search/${POS} --stride ${STRIDE} --threads ${CPUS}"
    wait_forwarding_signals
fi

start_in_container bash -c "cd /Yolah/nnue && exec python3 -u program_search.py /work/program_search/${POS} \
    /work/program_search/${RUN} --n ${N:-1000000} --minutes ${MINUTES:-1380} --workers ${CPUS} \
    --max-size ${MAX_SIZE:-6} --body-size ${BODY_SIZE:-5} --max-nodes ${MAX_NODES:-15} \
    --weights ${WEIGHTS:-1,10} --seed ${SEED:-0} ${BASE_FEATURES:+--base ${BASE_FEATURES}}"
wait_forwarding_signals
echo "[$(date '+%F %T')] done (exit code ${RC})"
