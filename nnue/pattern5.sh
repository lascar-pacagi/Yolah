#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# pattern5.sh — the 5×5 patterns around a piece that go with won games and
# with lost ones (test/pattern5_main.cpp, beam search), on all the games of
# data/games. One SLURM array task per (phase, target):
#
#     sbatch -p insa-cpu --exclude=crn07,crn14,crn15 --array=1-6 nnue/pattern5.sh
#
#     task  phase (free squares)   target
#       1   40–56  (opening)       result    (the owner's result beyond colour,
#       2   20–39  (middle game)   result     side to move, free squares)
#       3    0–19  (endgame)       result
#       4   40–56                  mobility  (also beyond the free squares
#       5   20–39                  mobility   next to the piece)
#       6    0–19                  mobility
#
# Results: WORK_DIR/patterns5/phase<P>_<target>.txt (drawings of the best
# patterns per number of fixed squares) and .tsv (every kept pattern).
# Environment: STRIDE (every STRIDE-th position, default 2), BEAM (100),
# MAX_FIXED (12), MIN_SHARE (0.0005), MIN_STEP (0.005).
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=yolah_pattern5
#SBATCH --output=yolah_pattern5_%A_%a.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/SearchStudy}"   # its own builds: never touches the AlphaZero run's
mkdir -p "${WORK_DIR}/patterns5"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, run(), build

TASK=${SLURM_ARRAY_TASK_ID:-1}
PHASE=$(( (TASK - 1) % 3 ))
TARGET=$([[ ${TASK} -le 3 ]] && echo result || echo mobility)
NAME="phase${PHASE}_${TARGET}"
echo "[$(date '+%F %T')] task ${TASK} on $(hostname), CPU:$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), phase ${PHASE}, target ${TARGET}, Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"

build_alphazero_learn pattern5

start_in_container bash -c "exec ${BUILD_DIR}/pattern5 /Yolah/data/games /work/patterns5/${NAME} \
    --phase ${PHASE} --target ${TARGET} --stride ${STRIDE:-2} --threads ${SLURM_CPUS_PER_TASK:-32} \
    --beam ${BEAM:-100} --max-fixed ${MAX_FIXED:-12} --min-share ${MIN_SHARE:-0.0005} --min-step ${MIN_STEP:-0.005}"
wait_forwarding_signals
echo "[$(date '+%F %T')] done (exit code ${RC})"
