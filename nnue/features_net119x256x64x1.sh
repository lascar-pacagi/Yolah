#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# features_net119x256x64x1.sh — submit with `sbatch features_net119x256x64x1.sh`
#
# Pipeline:
#   1. Build (or skip if already present) the encoded features cache via
#      preprocess_features.py — which itself invokes the baked-in
#      `encode_features` C++ binary to produce .features.txt files from the
#      games_* truth in /nnue/data, then consolidates them into a flat
#      (features.u8, values.i8) memmap with meta.json sidecar.
#   2. Train features_net119x256x64x1.py via DDP across all visible GPUs.
#      That script uses the chunked-shuffle loader (large sequential reads,
#      HDD-tolerant, single producer thread per rank).
#
# Adjust the #SBATCH lines below for your cluster's partition / account.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=features119
#SBATCH --output=features119_%j.out
#SBATCH --error=features119_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Tunables (override via env / `sbatch --export=...`) ─────────────────────
SIF="${SIF:-${SLURM_SUBMIT_DIR:-$PWD}/features_net119x256x64x1.sif}"
# Cache dir holds preprocess_features.py output (features.u8 / values.i8 /
# meta.json) and the intermediate feature_txt/ produced by the C++ encoder.
CACHE_DIR="${CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/cache_features119}"
MODEL_DIR="${MODEL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/models}"
# Number of training epochs; the trainer reads YOLAH_NB_EPOCHS at runtime
# so you can override with `sbatch --export=...,YOLAH_NB_EPOCHS=50` or by
# editing this default — no SIF rebuild needed.
YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-100}"

mkdir -p "${CACHE_DIR}" "${MODEL_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  SIF        : ${SIF}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Model dir  : ${MODEL_DIR}"
echo "  Epochs     : ${YOLAH_NB_EPOCHS}"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 1: preprocess (skip if cache already exists) ──────────────────────
if [[ ! -f "${CACHE_DIR}/meta.json" ]]; then
    echo "[$(date '+%F %T')] === Building features cache ==="
    singularity exec \
        --bind "${CACHE_DIR}:/cache" \
        "${SIF}" \
        bash -c "cd /nnue && python3 preprocess_features.py /cache"
else
    echo "[$(date '+%F %T')] === Cache exists, skipping preprocessing ==="
    python3 -c "
import json
m = json.load(open('${CACHE_DIR}/meta.json'))
print(f'  {m[\"n_positions\"]:,} positions')
print(f'  input_size      : {m[\"input_size\"]}')
"
fi

# ── Phase 2: train ──────────────────────────────────────────────────────────
# The chunked loader reads the cache in large sequential reads (HDD-tolerant)
# and uses a background producer thread — no DataLoader workers, no random
# reads. TORCH_NCCL_BLOCKING_WAIT=1 surfaces NCCL stalls as a clean timeout
# instead of letting them hang forever; NCCL_DEBUG=INFO prints what NCCL is
# actually doing on startup so a future hang at DDP wrap is diagnosable.
#
# The two NCCL_*_DISABLE envs below are commented out by default. Enable
# them if you hit a NCCL ALLGATHER timeout on the first DDP collective
# (run 705648 symptom): they force NCCL onto plain shared-mem / TCP and
# bypass P2P and InfiniBand — slower but very portable across cluster
# setups where cgroups or driver versions break the fast paths.
echo "[$(date '+%F %T')] === Training: features_net119x256x64x1.py ==="
singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --env "YOLAH_FEATURES_DIR=/cache" \
    --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS}" \
    --env "TORCH_NCCL_BLOCKING_WAIT=1" \
    --env "NCCL_DEBUG=INFO" \
    --env "NCCL_DEBUG_SUBSYS=INIT,COLL" \
    "${SIF}" \
    bash -c "cd /nnue && python3 features_net119x256x64x1.py"
    # Workarounds if the next run still hangs at DDP ALLGATHER — add these
    # via --env to singularity exec just above the SIF line:
    #   --env "NCCL_P2P_DISABLE=1"   ← bypass GPU-to-GPU peer access
    #   --env "NCCL_IB_DISABLE=1"    ← bypass InfiniBand transport
    #   --env "NCCL_SHM_DISABLE=1"   ← bypass shared-memory transport (force TCP)

echo "[$(date '+%F %T')] === Done ==="
