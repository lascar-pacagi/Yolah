#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# transformer_value_policy.sh — train the transformer of the same size as the
# supervised ResNet 256x30 on the same cache (outcome z + move played), in
# the AlphaZero image, then export it for the MCTS (.ts + .test.bin).
#
#     sbatch -p insa-gpu --gres=gpu:1 --exclude=crn22,crn23 nnue/transformer_value_policy.sh
#     GPUS: --gres=gpu:2 works as is (DDP on the node's GPUs)
#
# Resumable: submit it again with the same WORK_DIR, it restarts at the last
# checkpoint (every 4 chunks ≈ 16 M positions per GPU).
# Compare with the ResNet on the same validation set (first 8 chunks):
#     STAGE=eval sbatch -p insa-gpu --gres=gpu:1 --exclude=crn22,crn23 nnue/transformer_value_policy.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tf_vp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=tf_vp_%j.out
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/TransformerVP}"
CACHE_DIR="${CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache_az256}"
RESNET_PT="${RESNET_PT:-/Yolah/nnue/cnn_resnet_256x30_value_policy.pt}"
STAGE="${STAGE:-train}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"    # SIF, WSL_BIND
mkdir -p "${WORK_DIR}"
[[ -f "${CACHE_DIR}/meta.json" ]] || { echo "no ${CACHE_DIR}/meta.json (CACHE_DIR=...)" >&2; exit 1; }

run_py() {
    singularity exec --nv "${WSL_BIND[@]}" --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" \
        --bind "${CACHE_DIR}:/cache" \
        --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS:-2},YOLAH_TF_LR=${YOLAH_TF_LR:-5e-4},YOLAH_TF_BATCH=${YOLAH_TF_BATCH:-1024}" \
        "${SIF}" bash -c "cd /Yolah/nnue && python3 transformer_value_policy.py $*"
}

case "${STAGE}" in
    train)
        echo "[$(date '+%F %T')] training on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
        run_py train --cache /cache --out /work
        run_py export --model /work/transformer.pt
        echo "[$(date '+%F %T')] done: ${WORK_DIR}/transformer.ts"
        ;;
    eval)
        for f in "${WORK_DIR}"/transformer.e*.pt "${WORK_DIR}"/transformer.pt; do
            [[ -f "${f}" ]] && run_py eval --cache /cache --model "/work/$(basename "${f}")" --chunks "${CHUNKS:-8}"
        done
        run_py eval --cache /cache --model "${RESNET_PT}" --resnet --chunks "${CHUNKS:-8}"
        ;;
    *) echo "STAGE=train|eval" >&2; exit 1 ;;
esac
