#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# space_features.sh — precompute the space features (nnue/space_features.h)
# of every position of the distillation cache, on a CPU node:
#
#     sbatch -p insa-cpu --exclude=crn07,crn14,crn15 nnue/space_features.sh
#
# Writes CACHE_DIR/space.u8 (N × 16 bytes, aligned with inputs.u8) and
# CACHE_DIR/space_meta.json, read by nnue_persp_distill.py when
# YOLAH_SPACE_MASK is set (step 2 of the NNUE: the second layer reads the
# corrected flood fill, the queen distance, the zones reached late and the
# mobility of both sides). ~0.35 µs per position and thread.
#
# COMMAND=compute-all: every candidate feature of the ablation (92 columns,
# space::compute_all) → CACHE_DIR/space_all.u8 + space_all_meta.json, ~1.5 µs
# per position and thread:
#     sbatch -p insa-cpu --exclude=crn07,crn14,crn15 --export=ALL,COMMAND=compute-all nnue/space_features.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=space_features
#SBATCH --output=space_features_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/SearchStudy}"   # its builds (shared with search_study.sh)
CACHE_DIR="${CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache}"
COMMAND="${COMMAND:-compute}"                  # or compute-all
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, build

[[ -f "${CACHE_DIR}/inputs.u8" ]] || { echo "missing: ${CACHE_DIR}/inputs.u8" >&2; exit 1; }
echo "[$(date '+%F %T')] $(hostname), Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?'), cache ${CACHE_DIR}"
build_alphazero_learn space_features
singularity exec "${WSL_BIND[@]}" \
    --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" --bind "${CACHE_DIR}:/cache" \
    "${SIF}" "${BUILD_DIR}/space_features" "${COMMAND}" /cache --threads "${SLURM_CPUS_PER_TASK:-16}"
echo "[$(date '+%F %T')] done"
