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
# Step 2, space features (YOLAH_PERSP_K=8 YOLAH_PERSP_THEM=1, starting from the
# trained k8_them network, models/nnue_persp_k8_them_distill.pt of the repo;
# needs CACHE_DIR/space.u8 from nnue/space_features.sh):
#   YOLAH_SPACE_MASK=65535             all 16 space features          → k8_them_space
#   YOLAH_SPACE_MASK=65278             without the corrected flood    → k8_them_spacelite
#                                      fill (its costliest, ~185 ns of ~336)
#   CONTINUE=1                         no space feature, k8_them trained the same
#                                      epochs more: the fair reference → k8_them_cont
# (defaults for these three: 10 epochs, lr 1e-4)
# Ablation of the candidate features (doc/nnue_roadmap.qmd; plan and report:
# nnue/ablation.py, which writes these commands; needs CACHE_DIR/space_all.u8
# from `COMMAND=compute-all nnue/space_features.sh`):
#   YOLAH_SPACE_GROUPS=0x1ff           the groups fc2 reads (bit g = group g of
#                                      space_features.h); 0 = none (= CONTINUE)
#   YOLAH_SEED=1                       another order of the data (noise of a run)
#   → variant k8_them_g<groups>[_s<seed>], same step-2 start and lr; the
#   ablation sets YOLAH_NB_EPOCHS=5 and MODEL_ROOT=…/models_ablation.
# Step 3, campaign P6 (doc/nnue_roadmap.qmd): fine-tunes spacelite (the repo's
# models/nnue_persp_k8_them_spacelite_distill.pt; needs CACHE_DIR/space.u8),
# 5 epochs, lr 1e-4, with
#   FROM_SPACELITE=1                   nothing new: the fair reference → k8_them_spacelite_cont5
#   FROM_SPACELITE=1 YOLAH_DIRECT=1    + the direct path of the space features → …_spacelite_d
#   FROM_SPACELITE=1 YOLAH_QAT=1       quantization-aware training         → …_spacelite_qat
#   FROM_SPACELITE=1 YOLAH_DIRECT=1 YOLAH_QAT=1                           → …_spacelite_d_qat
#   (QAT runs in float32, not bf16; not slower: 5.3 ms per 1024-position step on an
#   RTX 3080 against 9.0 ms in bf16 — the network is too small for bf16 to pay)
# Campaign P7 (doc/nnue_roadmap.qmd): the same budget for everyone and the new
# teacher. P7=1: 20 epochs from the distilled 193 network (lr 3e-4, as step 1)
# instead of chained fine-tunings, on the cache labelled by the best AlphaZero
# 256 network (CACHE_DIR default …/cache_az256, from nnue/relabel_teacher.sh),
# variant names …_az20:
#   P7=1 YOLAH_PERSP_THEM=1                                      → k8_them_az20
#   P7=1 YOLAH_PERSP_THEM=1 YOLAH_SPACE_MASK=65278               → k8_them_spacelite_az20
#   P7=1 YOLAH_PERSP_THEM=1 YOLAH_SPACE_MASK=65278 YOLAH_QAT=1   → k8_them_spacelite_qat_az20
#   P7=1 YOLAH_PERSP_THEM=1 YOLAH_SPACE_GROUPS=0x40 YOLAH_QAT=1  → k8_them_g040_qat_az20 (same for 0x7e, 0x46)
#   (~3-4 days each: submit with --time=5-00:00:00)
# On crn25 (L40S), torch does not see the GPU in this image (2026-11-06: "Cannot access accelerator
# device"); the AlphaZero image, used by the distill loop, trains there: SIF=$HOME/AlphaZeroLearn/alphazero_learn.sif
# Width of the first layer and random start (with P7=1, the P7 protocol otherwise):
#   YOLAH_L1_SIZE=512                  512 values per view instead of 1024      → …_l512…
#   SCRATCH=1                          no starting weights: PyTorch's random
#                                      initialisation instead of the 193 network → …_scratch…
#   P7=1 YOLAH_PERSP_THEM=1 YOLAH_SPACE_GROUPS=0x46 YOLAH_QAT=1 YOLAH_L1_SIZE=512 SCRATCH=1
#                                                      → k8_them_g046_qat_l512_scratch_az20
#   (the same without YOLAH_L1_SIZE: k8_them_g046_qat_scratch_az20, the 1024 network
#   under the same conditions; a 512 network cannot start from the 193 network)
# Architecture (SCRATCH=1 needed; validation only, no C++ for the last two yet):
#   YOLAH_L2_SIZE=32                   32 outputs of fc2 per head instead of 64  → …_l232…
#   YOLAH_L2_SQUARE=1                  w3 reads [h2, h2²]                        → …_sq…
#   YOLAH_INPUT_BUCKETS=4              4 sets of fc1 weights by phase            → …_ib4…
#   YOLAH_FC1_FACTOR=1 (with the above) shared set + a correction per phase       → …_ib4f…
# Evaluation only (no training): the float AND quantized validation mse of an
# already trained network, on the validation positions of every run, with the
# space columns of its training (YOLAH_SPACE_MASK / YOLAH_SPACE_GROUPS):
#   EVAL_MODEL=${HOME}/Yolah/models/nnue_persp_k8_them_spacelite_distill.pt YOLAH_SPACE_MASK=65278
#   EVAL_MODEL=${HOME}/Yolah/models/nnue_persp_k8_them_g040_distill.pt YOLAH_SPACE_GROUPS=0x40
#   → MODEL_ROOT/eval/<model name>.eval.jsonl (one line, epoch 0)
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
#SBATCH --time=72:00:00
# crn22 refuses containers (user namespace, 2026-11-07); an --exclude on the command line replaces this one.
#SBATCH --exclude=crn22

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
CONTINUE="${CONTINUE:-0}"
YOLAH_SPACE_MASK="${YOLAH_SPACE_MASK:-0}"
YOLAH_SPACE_GROUPS="${YOLAH_SPACE_GROUPS:-}"      # empty: not the ablation
YOLAH_SEED="${YOLAH_SEED:-0}"
FROM_SPACELITE="${FROM_SPACELITE:-0}"
YOLAH_DIRECT="${YOLAH_DIRECT:-0}"
YOLAH_QAT="${YOLAH_QAT:-0}"
P7="${P7:-0}"
YOLAH_L1_SIZE="${YOLAH_L1_SIZE:-1024}"
SCRATCH="${SCRATCH:-0}"
YOLAH_L2_SIZE="${YOLAH_L2_SIZE:-64}"
YOLAH_L2_SQUARE="${YOLAH_L2_SQUARE:-0}"
YOLAH_INPUT_BUCKETS="${YOLAH_INPUT_BUCKETS:-1}"
YOLAH_FC1_FACTOR="${YOLAH_FC1_FACTOR:-0}"
if [[ "${P7}" == "1" ]]; then CACHE_DIR="${P7_CACHE_DIR:-/home/pgarcia/NNUE/DistillationNNUE/cache_az256}"; fi
EVAL_MODEL="${EVAL_MODEL:-}"
if [[ "${FROM_SPACELITE}" == "1" ]]; then
    YOLAH_PERSP_K=8; YOLAH_PERSP_THEM=1; YOLAH_SPACE_MASK=65278
    INIT_MODEL_STEP2="${INIT_MODEL_STEP2:-${SRC_DIR}/../models/nnue_persp_k8_them_spacelite_distill.pt}"
    YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-5}"
fi
STEP2=0
if [[ "${YOLAH_SPACE_MASK}" != "0" || "${CONTINUE}" == "1" || -n "${YOLAH_SPACE_GROUPS}" ]]; then STEP2=1; fi
if [[ "${P7}" == "1" ]]; then STEP2=0; fi    # P7: from the 193 network, the space columns start at 0 (init_from_nnue193)
if [[ "${STEP2}" == "1" ]]; then
    # step 2 fine-tunes the trained k8_them network
    INIT_MODEL="${INIT_MODEL_STEP2:-${SRC_DIR}/../models/nnue_persp_k${YOLAH_PERSP_K}$([[ "${YOLAH_PERSP_THEM}" == "1" ]] && echo _them || true)_distill.pt}"
    YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-10}"
    YOLAH_LR="${YOLAH_LR:-1e-4}"
fi
YOLAH_NB_EPOCHS="${YOLAH_NB_EPOCHS:-20}"
YOLAH_DISTILL_ALPHA="${YOLAH_DISTILL_ALPHA:-0.8}"
YOLAH_LR="${YOLAH_LR:-3e-4}"
if [[ -n "${YOLAH_SPACE_GROUPS}" ]]; then
    YOLAH_NAME_SUFFIX="$(printf '_g%03x' "$(( YOLAH_SPACE_GROUPS ))")"
    if [[ "${YOLAH_SEED}" != "0" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_s${YOLAH_SEED}"; fi
else case "${YOLAH_SPACE_MASK}" in
    0)             YOLAH_NAME_SUFFIX="$([[ "${CONTINUE}" == "1" ]] && echo _cont || true)" ;;
    65535|0xffff|0xFFFF) YOLAH_NAME_SUFFIX="_space" ;;
    65278|0xfefe|0xFEFE) YOLAH_NAME_SUFFIX="_spacelite" ;;
    *)             YOLAH_NAME_SUFFIX="_space${YOLAH_SPACE_MASK}" ;;
esac; fi
if [[ "${FROM_SPACELITE}" == "1" ]]; then
    if [[ "${YOLAH_DIRECT}" == "0" && "${YOLAH_QAT}" == "0" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_cont5"; fi
    if [[ "${YOLAH_DIRECT}" == "1" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_d"; fi
    if [[ "${YOLAH_QAT}" == "1" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_qat"; fi
fi
WIDTH_SUFFIX=""
if [[ "${YOLAH_L1_SIZE}" != "1024" ]]; then
    if [[ "${SCRATCH}" != "1" ]]; then echo "YOLAH_L1_SIZE=${YOLAH_L1_SIZE} needs SCRATCH=1 (the starting networks have 1024)" >&2; exit 1; fi
    WIDTH_SUFFIX="_l${YOLAH_L1_SIZE}"
fi
if [[ "${YOLAH_L2_SIZE}" != "64" || "${YOLAH_L2_SQUARE}" == "1" || "${YOLAH_INPUT_BUCKETS}" != "1" ]] && [[ "${SCRATCH}" != "1" ]]; then
    echo "YOLAH_L2_SIZE, YOLAH_L2_SQUARE, YOLAH_INPUT_BUCKETS need SCRATCH=1" >&2; exit 1
fi
if [[ "${YOLAH_L2_SIZE}" != "64" ]]; then WIDTH_SUFFIX="${WIDTH_SUFFIX}_l2${YOLAH_L2_SIZE}"; fi
if [[ "${YOLAH_L2_SQUARE}" == "1" ]]; then WIDTH_SUFFIX="${WIDTH_SUFFIX}_sq"; fi
if [[ "${YOLAH_INPUT_BUCKETS}" != "1" ]]; then WIDTH_SUFFIX="${WIDTH_SUFFIX}_ib${YOLAH_INPUT_BUCKETS}"; fi
if [[ "${YOLAH_FC1_FACTOR}" == "1" ]]; then
    [[ "${YOLAH_INPUT_BUCKETS}" != "1" ]] || { echo "YOLAH_FC1_FACTOR needs YOLAH_INPUT_BUCKETS > 1" >&2; exit 1; }
    WIDTH_SUFFIX="${WIDTH_SUFFIX}f"
fi
if [[ "${SCRATCH}" == "1" ]]; then WIDTH_SUFFIX="${WIDTH_SUFFIX}_scratch"; fi
if [[ "${P7}" == "1" ]]; then
    if [[ "${YOLAH_DIRECT}" == "1" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_d"; fi
    if [[ "${YOLAH_QAT}" == "1" ]]; then YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}_qat"; fi
    YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}${WIDTH_SUFFIX}_az20"
else
    YOLAH_NAME_SUFFIX="${YOLAH_NAME_SUFFIX}${WIDTH_SUFFIX}"
fi

if [[ "${CONTROL}" == "1" ]]; then
    VARIANT="control193"
    SCRIPT="nnue193x1024x64x32x1_distill.py"
else
    # (not "$([[ … ]] && echo _them)": false when THEM=0, and set -e then stops
    # the script silently, with empty .out and .err)
    VARIANT="k${YOLAH_PERSP_K}"
    if [[ "${YOLAH_PERSP_THEM}" == "1" ]]; then VARIANT="${VARIANT}_them"; fi
    VARIANT="${VARIANT}${YOLAH_NAME_SUFFIX}"
    SCRIPT="nnue_persp_distill.py"
fi
if [[ -n "${EVAL_MODEL}" ]]; then VARIANT="eval"; INIT_MODEL="${EVAL_MODEL}"; fi
MODEL_DIR="${MODEL_ROOT}/${VARIANT}"
# The rendezvous port of torch.distributed, one per job: several jobs can share a
# node (with one fixed port, the second one fails with EADDRINUSE).
YOLAH_DDP_PORT="${YOLAH_DDP_PORT:-$(( 20000 + ${SLURM_JOB_ID:-$$} % 40000 ))}"
mkdir -p "${MODEL_DIR}"

if [[ "${SCRATCH}" == "1" ]]; then INIT_MODEL="none"; fi
for f in "${SIF}" "${CACHE_DIR}/meta.json" "${CACHE_DIR}/teacher_meta.json" \
         "${SRC_DIR}/${SCRIPT}" "${SRC_DIR}/nnue_persp.py" "${SRC_DIR}/distill_common.py"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ "${INIT_MODEL}" == "none" || -e "${INIT_MODEL}" ]] || { echo "missing: ${INIT_MODEL}" >&2; exit 1; }
if [[ "${YOLAH_SPACE_MASK}" != "0" ]]; then
    for f in "${CACHE_DIR}/space.u8" "${CACHE_DIR}/space_meta.json"; do
        [[ -e "$f" ]] || { echo "missing: $f (run nnue/space_features.sh first)" >&2; exit 1; }
    done
fi
if [[ -n "${YOLAH_SPACE_GROUPS}" && "$(( YOLAH_SPACE_GROUPS ))" != "0" ]]; then
    for f in "${CACHE_DIR}/space_all.u8" "${CACHE_DIR}/space_all_meta.json"; do
        [[ -e "$f" ]] || { echo "missing: $f (run COMMAND=compute-all nnue/space_features.sh first)" >&2; exit 1; }
    done
fi
# The starting weights, copied into the variant's directory (/mnt in the container).
INIT_NAME="$([[ "${STEP2}" == "1" ]] && echo init_persp.pt || echo init_193_distill.pt)"
if [[ -n "${EVAL_MODEL}" ]]; then
    INIT_NAME="$(basename "${EVAL_MODEL}")"; cp "${EVAL_MODEL}" "${MODEL_DIR}/${INIT_NAME}"
fi
if [[ "${SCRATCH}" == "1" ]]; then INIT_NAME="none"
else [[ -e "${MODEL_DIR}/${INIT_NAME}" ]] || cp "${INIT_MODEL}" "${MODEL_DIR}/${INIT_NAME}"; fi

echo "════════════════════════════════════════════════════════════════"
echo "  Job      : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  Variant  : ${VARIANT} (${SCRIPT})"
echo "  Cache    : ${CACHE_DIR}"
echo "  Models   : ${MODEL_DIR}"
echo "  Init     : ${INIT_MODEL}"
echo "  Epochs   : ${YOLAH_NB_EPOCHS}  alpha=${YOLAH_DISTILL_ALPHA}  lr=${YOLAH_LR}  ddp port=${YOLAH_DDP_PORT}"
echo "  Space    : mask ${YOLAH_SPACE_MASK}  groups=${YOLAH_SPACE_GROUPS:-none}  seed=${YOLAH_SEED}  continue=${CONTINUE}"
echo "  Step 3   : P7=${P7}  from spacelite=${FROM_SPACELITE}  direct=${YOLAH_DIRECT}  qat=${YOLAH_QAT}"
echo "  Layer 1  : ${YOLAH_L1_SIZE}  scratch=${SCRATCH}  input buckets=${YOLAH_INPUT_BUCKETS}  factorised=${YOLAH_FC1_FACTOR}"
echo "  Layer 2  : ${YOLAH_L2_SIZE}  square=${YOLAH_L2_SQUARE}"
echo "  Eval only: ${EVAL_MODEL:-no}"
echo "════════════════════════════════════════════════════════════════"

singularity exec --nv \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${MODEL_DIR}:/mnt" \
    --bind "${SRC_DIR}:/work" \
    --env "YOLAH_CACHE_DIR=/cache" \
    --env "YOLAH_MODEL_DIR=/mnt/" \
    --env "YOLAH_INIT_MODEL=$([[ "${INIT_NAME}" == "none" ]] && echo none || echo "/mnt/${INIT_NAME}")" \
    --env "YOLAH_L1_SIZE=${YOLAH_L1_SIZE}" \
    --env "YOLAH_L2_SIZE=${YOLAH_L2_SIZE}" \
    --env "YOLAH_L2_SQUARE=${YOLAH_L2_SQUARE}" \
    --env "YOLAH_INPUT_BUCKETS=${YOLAH_INPUT_BUCKETS}" \
    --env "YOLAH_FC1_FACTOR=${YOLAH_FC1_FACTOR}" \
    --env "YOLAH_SPACE_MASK=${YOLAH_SPACE_MASK}" \
    --env "YOLAH_SPACE_GROUPS=${YOLAH_SPACE_GROUPS:-0}" \
    --env "YOLAH_SEED=${YOLAH_SEED}" \
    --env "YOLAH_DIRECT=${YOLAH_DIRECT}" \
    --env "YOLAH_QAT=${YOLAH_QAT}" \
    --env "YOLAH_EVAL_ONLY=$([[ -n "${EVAL_MODEL}" ]] && echo 1 || echo 0)" \
    --env "YOLAH_NAME_SUFFIX=${YOLAH_NAME_SUFFIX}" \
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
