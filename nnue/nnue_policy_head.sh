#!/bin/bash
# nnue_policy_head.sh — experiment 1: move heads on the frozen dev NNUE (nnue_policy_head.py),
# trained on the NNUE cache + policies.i16 (STAGE=policies of transformer_value_policy.sh first).
#     sbatch -p insa-gpu --gres=gpu:1 --exclude=crn22,crn23 nnue/nnue_policy_head.sh
# Result: ${WORK_DIR}/report.txt (top-1 / top-3 / rank of the move played among the legal moves, per phase).
#SBATCH --job-name=nnue_ph
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=nnue_ph_%j.out
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/NNUEPolicyHead}"
CACHE_DIR="${CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache}"
NET="${NET:-/Yolah/models/nnue_persp_k8_them_g046_qat_ib4f_ref2u_az20_distill.pt}"
SAMPLES="${SAMPLES:-500e6}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"    # SIF, WSL_BIND
mkdir -p "${WORK_DIR}"
[[ -f "${CACHE_DIR}/policies.i16" ]] || { echo "no ${CACHE_DIR}/policies.i16: STAGE=policies of transformer_value_policy.sh first" >&2; exit 1; }
singularity exec --nv "${WSL_BIND[@]}" --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" --bind "${CACHE_DIR}:/cache" "${SIF}" \
    bash -c "cd /Yolah/nnue && python3 nnue_policy_head.py train --cache /cache --net ${NET} --out /work --samples ${SAMPLES} ${EXTRA:-}"
