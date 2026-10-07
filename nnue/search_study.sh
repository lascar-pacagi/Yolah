#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# search_study.sh — one match between two minimax configurations per SLURM
# array task, on a CPU node (proxy cut / MPC / ProbCut study, see
# player/minmax_nnue_dev_player.h, N, O, P).
#
#     sbatch -p insa-cpu --exclude=crn14,crn15 --array=1-N nnue/search_study.sh campaign_Y1.txt
#
# Campaign file (config/search_study/): CANDIDATE REFERENCE TIME_US ROUNDS per
# line, the configurations being config/search_study/<name>.cfg. Each task
# plays ROUNDS rounds (one random opening, both colours) with one game per
# physical core, 1 search thread per player, and appends every game to
# WORK_DIR/search_study/<cand>__vs__<ref>__t<T>.csv (resumable: resubmit the
# task). Summary: python3 test/search_study_summary.py WORK_DIR/search_study
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=yolah_search
#SBATCH --output=yolah_search_%A_%a.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/SearchStudy}"   # its own builds: never touches the AlphaZero run's
CAMPAIGN="${1:?campaign file in config/search_study/}"
mkdir -p "${WORK_DIR}/search_study"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, run(), build

# One game per PHYSICAL core among the CPUs given to this task (24 "CPUs" can
# be 12 cores × 2 threads): the time per move must mean a core.
PHYS=$(lscpu -p=CPU,Core,Socket | grep -v '^#' | awk -F, -v allowed="$(grep Cpus_allowed_list /proc/self/status | cut -f2)" '
  BEGIN { n = split(allowed, r, ","); for (i = 1; i <= n; i++) { m = split(r[i], ab, "-"); for (c = ab[1]; c <= (m > 1 ? ab[2] : ab[1]); c++) ok[c] = 1 } }
  ($1 in ok) { core[$3 "," $2] = 1 }
  END { k = 0; for (x in core) k++; print k }')
PARALLEL=$(( ${SLURM_CPUS_PER_TASK:-24} < PHYS ? ${SLURM_CPUS_PER_TASK:-24} : PHYS ))

LINE=$(grep -v '^#' "${YOLAH_DIR}/config/search_study/${CAMPAIGN}" | grep . | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r CAND REF T ROUNDS <<< "${LINE}"
NAME="${CAND}__vs__${REF}__t${T}"
echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID} on $(hostname), CPU:$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), ${PHYS} physical cores, ${PARALLEL} games at a time"
echo "  ${CAND} vs ${REF}, ${T} µs/move, ${ROUNDS} rounds, Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"

build_alphazero_learn yolah_tournament

# The players list (paths inside the container; the weights "../models/…" are
# relative to the working directory /Yolah/config).
LIST="${WORK_DIR}/search_study/${NAME}.players"
printf '/Yolah/config/search_study/%s.cfg\n/Yolah/config/search_study/%s.cfg\n' "${CAND}" "${REF}" > "${LIST}"
start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/yolah_tournament \
    --players /work/search_study/${NAME}.players --results /work/search_study/${NAME}.csv \
    --time ${T} --threads 1 --parallel ${PARALLEL} --rounds ${ROUNDS} --opening-plies 4 \
    --seed ${SLURM_ARRAY_TASK_ID}"
wait_forwarding_signals
echo "[$(date '+%F %T')] done (exit code ${RC})"
