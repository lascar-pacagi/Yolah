#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# cnn_resnet_value_policy.sh — submit with `sbatch cnn_resnet_value_policy.sh`
#
# Pipeline:
#   1. Build (or skip if already present) the encoded position cache via
#      preprocess.py — runs CPU-only with many workers.
#   2. Train cnn_resnet_value_policy_chunked.py via DDP over all visible GPUs.
#      That script uses the double-buffered chunked-shuffle loader: it reads
#      the cache in large sequential chunks, so it is HDD-tolerant and needs
#      neither a page-cache pre-warm nor a large --mem.
#
# To run the DataLoader variant instead, change the script name in Phase 2 to
# cnn_resnet_value_policy.py and enable the pre-warm block below (that variant
# does random reads and needs the cache resident in RAM).
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=value_policy
#SBATCH --output=value_policy_%j.out
#SBATCH --error=value_policy_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-${SLURM_SUBMIT_DIR:-$PWD}/cnn_resnet_value_policy.sif}"
CACHE_DIR="${CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/cache}"
MODEL_DIR="${MODEL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/models}"

# Preprocessor uses the same CPU count SLURM allocated.
YOLAH_PREPROC_NPROC="${YOLAH_PREPROC_NPROC:-${SLURM_CPUS_PER_TASK:-16}}"

mkdir -p "${CACHE_DIR}" "${MODEL_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  Preproc np : ${YOLAH_PREPROC_NPROC}"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 1: preprocess (skip if cache already exists) ──────────────────────
if [[ ! -f "${CACHE_DIR}/meta.json" ]]; then
    echo "[$(date '+%F %T')] === Building position cache ==="
    singularity exec \
        --bind "${CACHE_DIR}:/cache" \
        --env "YOLAH_PREPROC_NPROC=${YOLAH_PREPROC_NPROC}" \
        "${SIF}" \
        bash -c "cd /nnue && python3 preprocess.py /cache"
else
    echo "[$(date '+%F %T')] === Cache exists, skipping preprocessing ==="
    python3 -c "
import json
m = json.load(open('${CACHE_DIR}/meta.json'))
print(f'  {m[\"n_positions\"]:,} positions across {m[\"n_games\"]:,} games')
print(f'  encoding planes : {m[\"encoding_planes\"]}')
"
fi

# ── (optional) pre-warm the OS page cache ───────────────────────────────────
# The chunked loader reads the cache sequentially and is HDD-tolerant, so this
# is NOT needed for cnn_resnet_value_policy_chunked.py. It IS needed for the
# DataLoader variant (cnn_resnet_value_policy.py), whose random reads thrash an
# HDD-backed cache. Enabling it costs ~17 min up front and needs the whole
# ~256 GB cache to fit in RAM → raise #SBATCH --mem to ~340G (node RAM ≥ 376G).
# To use it, uncomment the four lines below:
#
# echo "[$(date '+%F %T')] pre-warming page cache..."
# cat "${CACHE_DIR}"/positions.u8 "${CACHE_DIR}"/values.i8 "${CACHE_DIR}"/policies.i16 > /dev/null
# echo "[$(date '+%F %T')] cache warm:"
# free -g

# ── Phase 2: train ──────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Training: cnn_resnet_value_policy_chunked.py ==="
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_CACHE_DIR=/cache" \
    "${SIF}" \
    bash -c "cd /nnue && python3 cnn_resnet_value_policy_chunked.py"

echo "[$(date '+%F %T')] === Done ==="
