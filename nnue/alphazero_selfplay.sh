#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# alphazero_selfplay.sh — an extra self-play GPU for a running alphazero_learn
# job, on any GPU node.
#
#     sbatch alphazero_selfplay.sh         # same WORK_DIR as alphazero_learn.sh
#
# The learning loop only communicates through files in WORK_DIR (on /data,
# visible from every node): this job reads latest.json, hot-swaps each new
# network, and writes its games to WORK_DIR/selfplay/, where the trainer picks
# them up like those of its own GPUs. Use it when a node with 3 free GPUs is
# long to get: `sbatch --gres=gpu:1 alphazero_learn.sh` plus two of these
# (3 GPUs in total, the per-user maximum).
#
# It waits for the main job to have published a network, and stops by itself
# when no new network has appeared for STALE_HOURS (the main job is over), at
# its own time limit, or on scancel — finished games are always written.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=alphazero_selfplay
#SBATCH --output=alphazero_selfplay_%j.out
#SBATCH --error=alphazero_selfplay_%j.out
#SBATCH --partition=insa-gpu
#SBATCH -x crn23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"                       # SIF, WORK_DIR, run(), ...

SELFPLAY_GAMES="${SELFPLAY_GAMES:-256}"       # concurrent games on this GPU
SELFPLAY_NN_CACHE="${SELFPLAY_NN_CACHE:-32}"  # MB of network cache per game
STALE_HOURS="${STALE_HOURS:-6}"               # stop after this long without a new network

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Work dir   : ${WORK_DIR}"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# The main job creates latest.json when it starts a new run.
until [[ -f "${WORK_DIR}/latest.json" ]]; do
    echo "[$(date '+%F %T')] waiting for ${WORK_DIR}/latest.json (is alphazero_learn.sh running?)"
    sleep 60
done

build_alphazero_learn

echo "[$(date '+%F %T')] === Self-play ==="
mkdir -p "${WORK_DIR}/logs"
start_in_container ${BUILD_DIR}/alphazero_learn selfplay \
    --config /Yolah/config/alphazero_mcts_selfplay_player.cfg --work /work \
    --games "${SELFPLAY_GAMES}" --set "nn cache=${SELFPLAY_NN_CACHE}" \
    --max-stale-hours "${STALE_HOURS}" --tag "$(hostname)_${SLURM_JOB_ID:-$$}_extra"
wait_forwarding_signals
echo "[$(date '+%F %T')] === Stopped (exit code ${RC}) ==="
