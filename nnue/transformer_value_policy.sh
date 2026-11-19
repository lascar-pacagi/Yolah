#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# transformer_value_policy.sh — train, with ONE recipe (AdamW, warmup + cosine,
# same cache, same split), three networks of the same size, then export each
# for the MCTS (.ts + .test.bin):
#     ARCH=transformer  d 512, 8 heads, 12 layers               (38.7 M params)
#     ARCH=resnet_ft    ResNet 256x30 + from-to policy head     (35.5 M)
#     ARCH=resnet       ResNet 256x30 unchanged (baseline)      (36.0 M)
# resnet_ft vs resnet = the effect of the policy head; transformer vs
# resnet_ft = the effect of attention.
#
#     for a in transformer resnet_ft resnet; do
#         ARCH=$a sbatch --job-name=vp_$a -p insa-gpu --gres=gpu:1 --exclude=crn22,crn23 nnue/transformer_value_policy.sh
#     done
#     GPUS: --gres=gpu:2 works as is (DDP on the node's GPUs)
#
# Resumable: submit it again with the same WORK_DIR, it restarts at the last
# checkpoint (every 4 chunks ≈ 16 M positions per GPU).
# Compare all of them, and the app's ResNet, on the same validation set (first 8 chunks):
#     STAGE=eval sbatch -p insa-gpu --gres=gpu:1 --exclude=crn22,crn23 nnue/transformer_value_policy.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tf_vp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=tf_vp_%j.out
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
ARCH="${ARCH:-transformer}"
BASE_DIR="${BASE_DIR:-${HOME}/TransformerVP}"
WORK_DIR="${WORK_DIR:-${BASE_DIR}/${ARCH}}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR=the ResNet cache of preprocess.py (positions.u8, values.i8, policies.i16)}"
RESNET_PT="${RESNET_PT:-/Yolah/nnue/cnn_resnet_256x30_value_policy.pt}"
STAGE="${STAGE:-train}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"    # SIF, WSL_BIND
mkdir -p "${WORK_DIR}"
[[ -f "${CACHE_DIR}/meta.json" ]] || { echo "no ${CACHE_DIR}/meta.json (CACHE_DIR=...)" >&2; exit 1; }
grep -q '"policies"' "${CACHE_DIR}/meta.json" && grep -q '"positions"' "${CACHE_DIR}/meta.json" || {
    echo "${CACHE_DIR} is not a preprocess.py cache (positions.u8 + values.i8 + policies.i16: the ResNet's);" >&2
    echo "find it with: find ~ -maxdepth 5 -name policies.i16" >&2; exit 1; }

run_py() {
    singularity exec --nv "${WSL_BIND[@]}" --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" --bind "${BASE_DIR}:/base" \
        --bind "${CACHE_DIR}:/cache" \
        --env "YOLAH_NB_EPOCHS=${YOLAH_NB_EPOCHS:-2},YOLAH_TF_LR=${YOLAH_TF_LR:-5e-4},YOLAH_TF_BATCH=${YOLAH_TF_BATCH:-1024},YOLAH_TF_ARCH=${ARCH}" \
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
        for a in transformer resnet_ft resnet; do
            for f in "${BASE_DIR}/${a}"/transformer.e*.pt "${BASE_DIR}/${a}"/transformer.pt; do
                [[ -f "${f}" ]] && { echo "── ${a}"; run_py eval --cache /cache --model "/base/${a}/$(basename "${f}")" --chunks "${CHUNKS:-8}"; }
            done
        done
        echo "── the app's ResNet (supervised, original recipe)"
        run_py eval --cache /cache --model "${RESNET_PT}" --resnet --chunks "${CHUNKS:-8}"
        ;;
    *) echo "STAGE=train|eval" >&2; exit 1 ;;
esac
