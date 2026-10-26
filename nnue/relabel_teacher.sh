#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# relabel_teacher.sh — submit from ~/Yolah with `sbatch nnue/relabel_teacher.sh`
#
# Campaign P7 (doc/nnue_roadmap.qmd): a new teacher for the NNUE distillation,
# the best AlphaZero 256 network instead of the supervised ResNet.
#   1. NEW_CACHE gets HARD LINKS to every file of the current cache except its
#      teacher_* files (positions, outcomes, space features: no copy, no
#      preprocessing; the old cache and its labels stay as they are).
#   2. distill_teacher_labels.py labels it with TEACHER (resumable: a job
#      stopped by the time limit is simply resubmitted; one process per GPU).
#   3. compare_teachers.py compares the old and new labels on the validation
#      positions (the last 5 %, as the trainers): mse against the game
#      outcome, per phase, agreement between the two teachers.
# Then the P7 trainings read NEW_CACHE (nnue_persp_distill.sh P7=1).
#
#     sbatch nnue/relabel_teacher.sh
#     (teacher: the latest az_*.pt of the AlphaZero 256 run, named in the log and in
#      NEW_CACHE/teacher_meta.json; another one: --export=ALL,TEACHER=/path/az_<step>.pt)
#     sbatch --gres=gpu:2 nnue/relabel_teacher.sh          (two GPUs: half the time)
#
# Cost: one forward of the 256x30 ResNet per position, ~10k positions/s on an
# RTX 3080 in fp16 → about a day for the 988 M positions on one GPU.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=nnue_relabel
#SBATCH --output=nnue_relabel_%j.out
#SBATCH --partition=insa-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

set -euo pipefail
SIF="${SIF:-/home/pgarcia/NNUE/DistillationNNUE/nnue193x1024x64x32x1_distill.sif}"
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
OLD_CACHE="${OLD_CACHE:-/home/pgarcia/NNUE/DistillationNNUE/cache}"
NEW_CACHE="${NEW_CACHE:-/home/pgarcia/NNUE/DistillationNNUE/cache_az256}"
# Default teacher: the LATEST network of the AlphaZero 256 run (the run prunes its
# older az_*.pt, so a fixed step may be gone); TEACHER=… for another one.
# The teacher is copied into NEW_CACHE (teacher_net_<name>): a resubmitted job
# (resume) labels with that same copy, even after the run has pruned it.
AZ_MODELS="${AZ_MODELS:-/home/pgarcia/AlphaZeroLearn/work/models}"
KEPT="$(ls "${NEW_CACHE}"/teacher_net_*.pt 2>/dev/null | head -1 || true)"
if [[ -z "${TEACHER:-}" && -n "${KEPT}" ]]; then TEACHER="${KEPT}"; fi
TEACHER="${TEACHER:-$(ls "${AZ_MODELS}"/az_*.pt 2>/dev/null | sort | tail -1)}"
[[ -n "${TEACHER}" ]] || { echo "no az_*.pt in ${AZ_MODELS}" >&2; exit 1; }

for f in "${SIF}" "${OLD_CACHE}/meta.json" "${TEACHER}"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
mkdir -p "${NEW_CACHE}"
if [[ "$(basename "${TEACHER}")" != teacher_net_* ]]; then
    if [[ -n "${KEPT}" ]]; then
        echo "${NEW_CACHE} is being labelled with ${KEPT}: remove its teacher_* files to change teacher" >&2; exit 1
    fi
    cp "${TEACHER}" "${NEW_CACHE}/teacher_net_$(basename "${TEACHER}")"
    TEACHER="${NEW_CACHE}/teacher_net_$(basename "${TEACHER}")"
fi
for f in "${OLD_CACHE}"/*; do
    b="$(basename "$f")"
    case "$b" in teacher_*) continue ;; esac
    [[ -f "$f" && ! -e "${NEW_CACHE}/${b}" ]] || continue
    ln "$f" "${NEW_CACHE}/${b}" || { echo "cannot hard-link $f into ${NEW_CACHE} (another file system?)" >&2; exit 1; }
done

echo "════════════════════════════════════════════════════════════════"
echo "  Job      : ${SLURM_JOB_ID:-(local)} on $(hostname)"
echo "  Teacher  : ${TEACHER}"
echo "  Cache    : ${NEW_CACHE} (links to ${OLD_CACHE})"
echo "  Yolah    : $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "════════════════════════════════════════════════════════════════"

# The repo is bind-mounted (the teacher's code imports ../server/yolah.py).
singularity exec --nv \
    --bind "${YOLAH_DIR}:/Yolah" \
    --bind "${NEW_CACHE}:/cache" \
    --bind "${OLD_CACHE}:/old_cache" \
    "${SIF}" \
    bash -c "cd /Yolah/nnue && python3 distill_teacher_labels.py /cache --model /cache/$(basename "${TEACHER}") \
             && python3 compare_teachers.py /old_cache /cache"

echo "[$(date '+%F %T')] === Done: ${NEW_CACHE} ==="
