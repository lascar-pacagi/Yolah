#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# features_net119x256x64x1.sh — submit with `sbatch features_net119x256x64x1.sh`
#
# Trains the 119-input features network (value head, MSE loss) via DDP
# across all visible GPUs. The .features.txt records are already in the
# format the dataset memmaps — no preprocess phase needed.
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=features119
#SBATCH --output=features119_%j.out
#SBATCH --error=features119_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-${SLURM_SUBMIT_DIR:-$PWD}/features_net119x256x64x1.sif}"
# Directory holding *.features.txt files (bind-mounted at /cache).
FEATURES_DIR="${FEATURES_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/features}"
MODEL_DIR="${MODEL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/models}"

# DataLoader workers per DDP rank (8 is plenty for memmap reads).
YOLAH_DATALOADER_WORKERS="${YOLAH_DATALOADER_WORKERS:-8}"

mkdir -p "${MODEL_DIR}"
if [[ ! -d "${FEATURES_DIR}" ]]; then
    echo "ERROR: features dir '${FEATURES_DIR}' does not exist" >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Features   : ${FEATURES_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  DL workers : ${YOLAH_DATALOADER_WORKERS}"
echo "════════════════════════════════════════════════════════════════"

echo "[$(date '+%F %T')] === Training: features_net119x256x64x1.py ==="
singularity exec --nv \
    --bind "${FEATURES_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_FEATURES_DIR=/cache" \
    --env "YOLAH_DATALOADER_WORKERS=${YOLAH_DATALOADER_WORKERS}" \
    "${SIF}" \
    bash -c "cd /nnue && python3 features_net119x256x64x1.py"

echo "[$(date '+%F %T')] === Done ==="
