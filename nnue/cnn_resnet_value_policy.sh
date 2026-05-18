#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# cnn_resnet_value_policy.sh — submit with `sbatch cnn_resnet_value_policy.sh`
#
# Pipeline:
#   1. Build (or skip if already present) the encoded position cache via
#      preprocess.py — runs CPU-only with many workers.
#   2. Train the chosen variant via DDP over all visible GPUs.
#
# Override defaults at submit time, e.g.
#   sbatch --export=ALL,TRAIN_SCRIPT=cnn_resnet_value_policy.py \
#          cnn_resnet_value_policy.sh
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=value_policy
#SBATCH --output=value_policy_%j.out
#SBATCH --error=value_policy_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-${SLURM_SUBMIT_DIR:-$PWD}/cnn_resnet_value_policy.sif}"
CACHE_DIR="${CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/cache}"
MODEL_DIR="${MODEL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/models}"

# Pick which training script to run.  Default = without certain-win.
# For the ablation set:  TRAIN_SCRIPT=cnn_resnet_value_policy.py
TRAIN_SCRIPT="${TRAIN_SCRIPT:-cnn_resnet_value_policy.py}"

# Preprocessor uses the same CPU count SLURM allocated.
YOLAH_PREPROC_NPROC="${YOLAH_PREPROC_NPROC:-${SLURM_CPUS_PER_TASK:-16}}"
# DataLoader workers per DDP rank (8 is plenty for memmap reads).
YOLAH_DATALOADER_WORKERS="${YOLAH_DATALOADER_WORKERS:-8}"

mkdir -p "${CACHE_DIR}" "${MODEL_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  Train      : ${TRAIN_SCRIPT}"
echo "  Preproc np : ${YOLAH_PREPROC_NPROC}"
echo "  DL workers : ${YOLAH_DATALOADER_WORKERS}"
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

# ── Phase 2: train ──────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Training: ${TRAIN_SCRIPT} ==="
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_CACHE_DIR=/cache" \
    --env "YOLAH_DATALOADER_WORKERS=${YOLAH_DATALOADER_WORKERS}" \
    "${SIF}" \
    bash -c "cd /nnue && python3 ${TRAIN_SCRIPT}"

echo "[$(date '+%F %T')] === Done ==="
