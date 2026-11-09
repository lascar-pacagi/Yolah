#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# distill_nnue_experiment.sh — trains NNUE_{r+1} of the distillation loop again
# with other settings, without touching the loop's own files, then plays it
# against NNUE_r. Run on the login node, from anywhere:
#
#     bash ~/Yolah/nnue/distill_nnue_experiment.sh TAG EPOCHS [LR] [ROUND]
#     bash ~/Yolah/nnue/distill_nnue_experiment.sh e20 20             # 20 epochs, lr 1e-4, round 0 → 1
#     bash ~/Yolah/nnue/distill_nnue_experiment.sh e8lr2e5 8 2e-5
#
# Submits, each job waiting for the one before (distill_loop.sh stages):
#   prepare    round ROUND's cache again (the stage next deleted it), skipped
#              if the cache is still there
#   relabel    the cache labelled by the existing ResNet_{ROUND+1} (not
#              trained again) and blended, skipped with the cache
#   nnue       NNUE_TAG=TAG: round_<ROUND+1>/exp_TAG/nnue.pt, from NNUE_ROUND
#   eval_nnue  the dev player with it against the one with NNUE_ROUND
#              (1000 rounds, 1 s per move) → eval/nnue_r<ROUND+1>_TAG__vs__nnue_r<ROUND>.csv
# Result: python3 ~/Yolah/nnue/loop_progress.py ~/DistillLoop  (line nnue_TAG)
# The loop itself is not restarted (no stage next).
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
TAG="${1:?usage: distill_nnue_experiment.sh TAG EPOCHS [LR] [ROUND]}"
EPOCHS="${2:?usage: distill_nnue_experiment.sh TAG EPOCHS [LR] [ROUND]}"
LR="${3:-1e-4}"
ROUND="${4:-0}"
[[ "${TAG}" =~ ^[A-Za-z0-9-]+$ ]] || { echo "TAG: letters, digits and '-' only" >&2; exit 1; }

LOOP_DIR="${LOOP_DIR:-${HOME}/DistillLoop}"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/distill_loop.sh"
EXCLUDE="${EXCLUDE:-crn04,crn05,crn07,crn13,crn14,crn15}"
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22}"
R=$(printf '%03d' "${ROUND}")
L="${LOOP_DIR}/logs"
CACHE="${LOOP_DIR}/round_${R}/cache"
mkdir -p "${L}"
cpu=(--parsable -p insa-cpu --exclude="${EXCLUDE}")
gpu=(--parsable -p insa-gpu --gres=gpu:1 --exclude="${GPU_EXCLUDE}")
env_base="ALL,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR}"

dep=()
if [[ -s "${CACHE}/blend_meta.json" ]]; then
    echo "round ${ROUND}: cache labelled and blended already (${CACHE}): no prepare, no relabel"
else
    j_prep=$(sbatch "${cpu[@]}" --export="${env_base},STAGE=prepare" --cpus-per-task=16 --mem=96G --time=12:00:00 \
        --job-name="dl${R}_prep" --output="${L}/${R}_prepare_%j.out" "${SCRIPT}")
    j_lab=$(sbatch "${gpu[@]}" --dependency=afterok:"${j_prep}" --export="${env_base},STAGE=relabel" --cpus-per-task=8 --mem=64G \
        --time=12:00:00 --job-name="dl${R}_relabel" --output="${L}/${R}_relabel_%j.out" "${SCRIPT}")
    echo "prepare ${j_prep}, relabel ${j_lab}"
    dep=(--dependency=afterok:"${j_lab}")
fi
j_nnue=$(sbatch "${gpu[@]}" ${dep[@]+"${dep[@]}"} --export="${env_base},STAGE=nnue,NNUE_TAG=${TAG},NNUE_EPOCHS=${EPOCHS},NNUE_LR=${LR}" \
    --cpus-per-task=8 --mem=64G --time=3-00:00:00 --job-name="dl${R}_nnue_${TAG}" --output="${L}/${R}_nnue_${TAG}_%j.out" "${SCRIPT}")
j_eval=$(sbatch "${cpu[@]}" --dependency=afterok:"${j_nnue}" --export="${env_base},STAGE=eval_nnue,NNUE_TAG=${TAG}" --array=0 \
    --cpus-per-task=32 --mem=32G --time=24:00:00 --job-name="dl${R}_ev_${TAG}" --output="${L}/${R}_eval_${TAG}_%A_%a.out" "${SCRIPT}")
echo "nnue ${j_nnue} (${EPOCHS} epochs, lr ${LR}), eval ${j_eval}"
echo "experiment ${TAG}: prepare/relabel → nnue ${j_nnue} → eval ${j_eval}" >> "${LOOP_DIR}/jobs.txt"
echo "result: python3 $(dirname "${SCRIPT}")/loop_progress.py ${LOOP_DIR}   (line nnue_${TAG})"
