#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# mcts_bench.sh — submit from ~/Yolah with sbatch (see the example below)
#
# Throughput of the AlphaZero MCTS player per threads × batch (alphazero_check
# --mcts-bench): the search keeps up to threads × batch descents in flight,
# each with a virtual loss; the tournaments ran 16 × 64 = 1024 of them, the
# evaluation matches of the learning loop 4 × 32 = 128. For each setting and
# each network: simulations/s, network evaluations/s, collisions (descents
# undone because another one was already expanding the leaf), and the share
# of positions where the best move is the one of a long 4 × 32 search.
# Choose the smallest number in flight that keeps ~90 % of the best
# throughput, then confirm it with a match.
#
#     sbatch -p insa-gpu -w crn25 --gres=gpu:1 --cpus-per-task=16 nnue/mcts_bench.sh
#     (TOURNAMENT_DIR: where az256.ts is and the results go, default ~/TournamentAZ,
#      whose build directory is reused; TIME_US per position, default 1 s;
#      SETTINGS, LIMIT positions, REF_TIME_US: see below)
#
# Results: TOURNAMENT_DIR/mcts_bench_<job>.txt
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=mcts_bench
#SBATCH --output=mcts_bench_%j.out
#SBATCH --partition=insa-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${TOURNAMENT_DIR:-${HOME}/TournamentAZ}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, run(), build

SETTINGS="${SETTINGS:-1x32,2x32,4x16,4x32,4x64,8x16,8x32,16x8,16x16,16x32,16x64}"
TIME_US="${TIME_US:-1000000}"
REF_TIME_US="${REF_TIME_US:-4000000}"   # 0 = no reference search
LIMIT="${LIMIT:-40}"                    # positions of test/bench_positions.txt
OUT="${WORK_DIR}/mcts_bench_${SLURM_JOB_ID:-local}.txt"

[[ -f "${WORK_DIR}/az256.ts" ]] || { echo "no az256.ts in ${WORK_DIR}" >&2; exit 1; }
echo "  Job : ${SLURM_JOB_ID:-(local)} on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee "${OUT}"
echo "  AZ  : $(cat "${WORK_DIR}/az256.name" 2>/dev/null || echo az256.ts)" | tee -a "${OUT}"

build_alphazero_learn alphazero_check

for weights in /work/az256.ts /Yolah/nnue/cnn_resnet_256x30_value_policy.ts; do
    echo "[$(date '+%F %T')] === ${weights} ===" | tee -a "${OUT}"
    run bash -c "cd /Yolah/test && ${BUILD_DIR}/alphazero_check --backend torch --weights ${weights} \
        --mcts-bench ${SETTINGS} --positions bench_positions.txt --limit ${LIMIT} \
        --time ${TIME_US} --ref-time ${REF_TIME_US}" 2>&1 | tee -a "${OUT}"
done
echo "[$(date '+%F %T')] === Done: ${OUT} ==="
