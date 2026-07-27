#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# combined_net311x1024x64x32x1.sh — submit with `sbatch combined_net311x1024x64x32x1.sh`
#
# Pipeline:
#   1. Build (or skip if already present) the ALIGNED combined cache via
#      preprocess_combined.py. Phase A runs the baked-in `encode_features` C++
#      binary to produce per-source .features.txt (119 feature bytes + label).
#      Phase B replays each game in Python to produce the 193-input board
#      encoding, joins both into ONE row-aligned cache (inputs.u8 + features.u8 +
#      values.i8), and verifies per-row TURN/VALUE alignment. CPU-only.
#   2. Train combined_net311x1024x64x32x1.py via DDP across all visible GPUs.
#      The chunked-shuffle loader reads BOTH memmaps with large sequential reads
#      (HDD-tolerant) and a single producer thread per rank, building the 311-d
#      input as concat([nnue_bits, features/255]).
#
# NOTE: the existing per-network caches (cache_nnue193, cache_features119) are
# NOT row-aligned and CANNOT be reused here — this builds its own aligned cache.
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=combined311_h1024
#SBATCH --output=combined311_h1024_%j.out
#SBATCH --error=combined311_h1024_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-${SLURM_SUBMIT_DIR:-$PWD}/combined_net311x1024x64x32x1.sif}"
# Cache dir holds preprocess_combined.py output (inputs.u8 / features.u8 /
# values.i8 / meta.json) and the intermediate feature_txt/ from the C++ encoder.
CACHE_DIR="${CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/cache_combined311}"
MODEL_DIR="${MODEL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/models}"

# Preprocessor's Phase-B replay Pool uses the SLURM-allocated CPU count.
YOLAH_PREPROC_NPROC="${YOLAH_PREPROC_NPROC:-${SLURM_CPUS_PER_TASK:-16}}"
# Number of training epochs; the trainer reads YOLAH_NB_EPOCHS at runtime so you
# can override with `sbatch --export=...,YOLAH_NB_EPOCHS=50` or by editing this
# default — no SIF rebuild needed.
YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-100}"

mkdir -p "${CACHE_DIR}" "${MODEL_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  Preproc np : ${YOLAH_PREPROC_NPROC}"
echo "  Epochs     : ${YOLAH_NB_EPOCHS}"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 1: preprocess (skip if cache already exists) ──────────────────────
if [[ ! -f "${CACHE_DIR}/meta.json" ]]; then
    echo "[$(date '+%F %T')] === Building ALIGNED combined cache ==="
    singularity exec \
        --bind "${CACHE_DIR}:/cache" \
        --env "YOLAH_PREPROC_NPROC=${YOLAH_PREPROC_NPROC}" \
        "${SIF}" \
        bash -c "cd /nnue && python3 preprocess_combined.py /cache"
else
    echo "[$(date '+%F %T')] === Cache exists, skipping preprocessing ==="
    python3 -c "
import json
m = json.load(open('${CACHE_DIR}/meta.json'))
print(f'  {m[\"n_positions\"]:,} positions')
print(f'  input_size_nnue  : {m[\"input_size_nnue\"]}')
print(f'  n_features       : {m[\"n_features\"]}')
print(f'  model_input_size : {m[\"model_input_size\"]}')
"
fi

# ── Phase 2: train ──────────────────────────────────────────────────────────
# The chunked loader reads the cache in large sequential reads (HDD-tolerant)
# and uses a background producer thread — no DataLoader workers, no random
# reads. TORCH_NCCL_BLOCKING_WAIT=1 surfaces NCCL stalls as a clean timeout
# instead of letting them hang forever; NCCL_DEBUG=INFO prints what NCCL is
# actually doing on startup so a future hang at DDP wrap is diagnosable.
echo "[$(date '+%F %T')] === Training: combined_net311x1024x64x32x1.py ==="
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_COMBINED_DIR=/cache" \
    --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS}" \
    --env "TORCH_NCCL_BLOCKING_WAIT=1" \
    --env "NCCL_DEBUG=INFO" \
    --env "NCCL_DEBUG_SUBSYS=INIT,COLL" \
    "${SIF}" \
    bash -c "cd /nnue && python3 combined_net311x1024x64x32x1.py"
    # Workarounds if the run hangs at DDP ALLGATHER — add these via --env to
    # singularity exec just above the SIF line:
    #   --env "NCCL_P2P_DISABLE=1"   ← bypass GPU-to-GPU peer access
    #   --env "NCCL_IB_DISABLE=1"    ← bypass InfiniBand transport
    #   --env "NCCL_SHM_DISABLE=1"   ← bypass shared-memory transport (force TCP)

echo "[$(date '+%F %T')] === Done ==="
