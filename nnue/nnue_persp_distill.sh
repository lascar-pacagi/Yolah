#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# nnue_persp_distill.sh — submit from nnue/ with `sbatch nnue_persp_distill.sh`
#
# Step 1 of doc/nnue_roadmap.qmd: distils the ResNet into the NNUE seen from
# the side to move with K phase heads (NetPersp, nnue_persp.py), starting from
# the distilled 193 network. Reuses, unchanged, the cache and the teacher
# labels of nnue193x1024x64x32x1_distill.sh (no preprocessing, no labelling).
#
# The image of nnue193x1024x64x32x1_distill.sh is reused as is (python, torch);
# the new scripts are NOT in it: this directory (the submit directory) is
# bind-mounted at /work and run from there.
#
# Variants (sbatch --export=ALL,YOLAH_PERSP_K=…,YOLAH_PERSP_THEM=…):
#   YOLAH_PERSP_K=1                    point of view only
#   YOLAH_PERSP_K=8                    point of view + 8 phase heads (default)
#   YOLAH_PERSP_K=8 YOLAH_PERSP_THEM=1 + fc2 reads both views [us, them]
#   CONTROL=1                          the 193 network trained the same number of
#                                      epochs more (nnue193x1024x64x32x1_distill.py),
#                                      the fair reference for the matches
# Each variant writes to its own directory under MODEL_ROOT (the outputs of
# a variant never overwrite those of another, nor the original models).
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=nnue_persp
#SBATCH --output=nnue_persp_%j.out
#SBATCH --error=nnue_persp_%j.err
#SBATCH --partition=insa-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

SIF="${SIF:-/home/pgarcia/NNUE/DistillationNNUE/nnue193x1024x64x32x1_distill.sif}"
CACHE_DIR="${CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache}"
# The distilled 193 network, starting point of every variant.
INIT_MODEL="${INIT_MODEL:-/home/pgarcia/NNUE/DistillationNNUE/models/nnue_193x1024x64x32x1_distill.pt}"
MODEL_ROOT="${MODEL_ROOT:-/home/pgarcia/NNUE/DistillationNNUE/models_persp}"
SRC_DIR="${SRC_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

YOLAH_PERSP_K="${YOLAH_PERSP_K:-8}"
YOLAH_PERSP_THEM="${YOLAH_PERSP_THEM:-0}"
CONTROL="${CONTROL:-0}"
YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-20}"
YOLAH_DISTILL_ALPHA="${YOLAH_DISTILL_ALPHA:-0.8}"
YOLAH_LR="${YOLAH_LR:-3e-4}"

if [[ "${CONTROL}" == "1" ]]; then
    VARIANT="control193"
    SCRIPT="nnue193x1024x64x32x1_distill.py"
else
    VARIANT="k${YOLAH_PERSP_K}$([[ "${YOLAH_PERSP_THEM}" == "1" ]] && echo _them)"
    SCRIPT="nnue_persp_distill.py"
fi
MODEL_DIR="${MODEL_ROOT}/${VARIANT}"
# The rendezvous port of torch.distributed, one per job: several jobs can share a
# node (with one fixed port, the second one fails with EADDRINUSE).
YOLAH_DDP_PORT="${YOLAH_DDP_PORT:-$(( 20000 + ${SLURM_JOB_ID:-$$} % 40000 ))}"
mkdir -p "${MODEL_DIR}"

for f in "${SIF}" "${CACHE_DIR}/meta.json" "${CACHE_DIR}/teacher_meta.json" "${INIT_MODEL}" \
         "${SRC_DIR}/${SCRIPT}" "${SRC_DIR}/nnue_persp.py" "${SRC_DIR}/distill_common.py"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
# The starting weights, copied into the variant's directory (/mnt in the container).
[[ -e "${MODEL_DIR}/init_193_distill.pt" ]] || cp "${INIT_MODEL}" "${MODEL_DIR}/init_193_distill.pt"

echo "════════════════════════════════════════════════════════════════"
echo "  Job      : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  Variant  : ${VARIANT} (${SCRIPT})"
echo "  Cache    : ${CACHE_DIR}"
echo "  Models   : ${MODEL_DIR}"
echo "  Init     : ${INIT_MODEL}"
echo "  Epochs   : ${YOLAH_NB_EPOCHS}  alpha=${YOLAH_DISTILL_ALPHA}  lr=${YOLAH_LR}  ddp port=${YOLAH_DDP_PORT}"
echo "════════════════════════════════════════════════════════════════"

singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --bind "${SRC_DIR}:/work" \
    --env "YOLAH_CACHE_DIR=/cache" \
    --env "YOLAH_MODEL_DIR=/mnt/" \
    --env "YOLAH_INIT_MODEL=/mnt/init_193_distill.pt" \
    --env "YOLAH_PERSP_K=${YOLAH_PERSP_K}" \
    --env "YOLAH_PERSP_THEM=${YOLAH_PERSP_THEM}" \
    --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS}" \
    --env "YOLAH_DISTILL_ALPHA=${YOLAH_DISTILL_ALPHA}" \
    --env "YOLAH_LR=${YOLAH_LR}" \
    --env "YOLAH_DDP_PORT=${YOLAH_DDP_PORT}" \
    --env "TORCH_NCCL_BLOCKING_WAIT=1" \
    "${SIF}" \
    bash -c "cd /work && python3 ${SCRIPT}"

echo "[$(date '+%F %T')] === Done: ${MODEL_DIR} ==="
