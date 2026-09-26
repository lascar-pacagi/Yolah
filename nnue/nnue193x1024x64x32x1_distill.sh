#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# nnue193x1024x64x32x1_distill.sh — submit with `sbatch nnue193x1024x64x32x1_distill.sh`
#
# Distils the value/policy ResNet (cnn_resnet_256x30_value_policy.pt) into the
# 193-input NNUE. Pipeline:
#   1. Build (or reuse) the encoded position cache via preprocess_nnue.py.
#   2. Label the cache with the teacher (distill_teacher_labels.py, GPU).
#      Resumable: an interrupted job continues where it stopped, so a job
#      that hits the time limit is simply resubmitted.
#   3. Train nnue193x1024x64x32x1_distill.py via DDP over all visible GPUs,
#      starting from the last non-distilled weights.
#
# Reuses the cache of nnue193x1024x64x32x1.sh (CACHE_DIR) — the teacher labels
# are added next to it. The teacher checkpoint must be in MODEL_DIR (bind-
# mounted at /mnt); it falls back to the copy baked into the .sif.
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=nnue193_distill
#SBATCH --output=nnue193_distill_%j.out
#SBATCH --error=nnue193_distill_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-/home/pgarcia/NNUE/DistillationNNUE/nnue193x1024x64x32x1_distill.sif}"
CACHE_DIR="${CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache}"
MODEL_DIR="${MODEL_DIR:-/home/pgarcia/NNUE/DistillationNNUE/models}"

YOLAH_PREPROC_NPROC="${YOLAH_PREPROC_NPROC:-${SLURM_CPUS_PER_TASK:-16}}"
YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-20}"
# Weight of the teacher term in the loss (1 = pure distillation, 0 = outcome only).
YOLAH_DISTILL_ALPHA="${YOLAH_DISTILL_ALPHA:-0.8}"
YOLAH_LR="${YOLAH_LR:-3e-4}"
# Teacher checkpoint inside the container (/mnt = MODEL_DIR).
TEACHER_MODEL="${TEACHER_MODEL:-/mnt/cnn_resnet_256x30_value_policy.pt}"
# Set DISTILL_TTA=1 to average the teacher over the 8 board symmetries (8× slower labelling).
DISTILL_TTA="${DISTILL_TTA:-0}"
# Label only the first N positions (0 = all). Training uses the labelled part only.
LABEL_END="${LABEL_END:-0}"

mkdir -p "${CACHE_DIR}" "${MODEL_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  Teacher    : ${TEACHER_MODEL} (tta=${DISTILL_TTA})"
echo "  Epochs     : ${YOLAH_NB_EPOCHS}  alpha=${YOLAH_DISTILL_ALPHA}  lr=${YOLAH_LR}"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 1: preprocess (skip if cache already exists) ──────────────────────
if [[ ! -f "${CACHE_DIR}/meta.json" ]]; then
    echo "[$(date '+%F %T')] === Building position cache ==="
    singularity exec \
        --bind "${CACHE_DIR}:/cache" \
        --env "YOLAH_PREPROC_NPROC=${YOLAH_PREPROC_NPROC}" \
        "${SIF}" \
        bash -c "cd /nnue && python3 preprocess_nnue.py /cache"
else
    echo "[$(date '+%F %T')] === Cache exists, skipping preprocessing ==="
fi

# ── Phase 2: teacher labels (resumable) ─────────────────────────────────────
echo "[$(date '+%F %T')] === Labelling with the teacher ==="
LABEL_ARGS=""
[[ "${DISTILL_TTA}" == "1" ]] && LABEL_ARGS="${LABEL_ARGS} --tta"
[[ "${LABEL_END}" != "0" ]] && LABEL_ARGS="${LABEL_ARGS} --end ${LABEL_END}"
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    "${SIF}" \
    bash -c "cd /nnue && MODEL='${TEACHER_MODEL}'; [[ -f \"\$MODEL\" ]] || MODEL=/nnue/cnn_resnet_256x30_value_policy.pt; \
             python3 distill_teacher_labels.py /cache --model \"\$MODEL\" ${LABEL_ARGS}"

# ── Phase 3: train ──────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Training: nnue193x1024x64x32x1_distill.py ==="
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_CACHE_DIR=/cache" \
    --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS}" \
    --env "YOLAH_DISTILL_ALPHA=${YOLAH_DISTILL_ALPHA}" \
    --env "YOLAH_LR=${YOLAH_LR}" \
    --env "TORCH_NCCL_BLOCKING_WAIT=1" \
    "${SIF}" \
    bash -c "cd /nnue && python3 nnue193x1024x64x32x1_distill.py"

echo "[$(date '+%F %T')] === Done ==="
