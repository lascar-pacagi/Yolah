#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# spsa.sh — SPSA tuning of the dev player's parameters (nnue/spsa.py) on ONE
# CPU node: θ+ against θ−, one game per physical core, 2 waves per batch.
#
#     PARAMS=dev_u8_params.json sbatch -p insa-cpu -w crn25 --cpus-per-task=128 nnue/spsa.sh
#     bash nnue/spsa.sh report            # where the parameters went (login node)
#
# Resumable: submit again (same NAME), it goes on from state.json. The
# recommended configuration is always WORK_DIR/spsa/NAME/current.cfg.
# 0.2 s per move: ~12 s per game; 128 cores ≈ 10 games/s → 100 000 games
# (PAIRS=50000) in about 3 h on crn25, ~15 h on a 24-core Xeon.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=yolah_spsa
#SBATCH --output=yolah_spsa_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@300
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${WORK_DIR:-${HOME}/SearchStudy}"
PARAMS="${PARAMS:-dev_u8_params.json}"
NAME="${NAME:-${PARAMS%_params.json}}"
PAIRS="${PAIRS:-50000}"
TIME_US="${TIME_US:-200000}"
BOOK="${BOOK:-/Yolah/config/openings/balanced.txt}"
if [[ "${1:-}" == "report" ]]; then
    exec python3 "${YOLAH_DIR}/nnue/spsa.py" --report --work "${WORK_DIR}/spsa/${NAME}" --params "${YOLAH_DIR}/config/spsa/${PARAMS}"
fi
mkdir -p "${WORK_DIR}/spsa/${NAME}"
source "${YOLAH_DIR}/nnue/alphazero_common.sh"
PHYS=$(lscpu -p=CPU,Core,Socket | grep -v '^#' | awk -F, -v allowed="$(grep Cpus_allowed_list /proc/self/status | cut -f2)" '
  BEGIN { n = split(allowed, r, ","); for (i = 1; i <= n; i++) { m = split(r[i], ab, "-"); for (c = ab[1]; c <= (m > 1 ? ab[2] : ab[1]); c++) ok[c] = 1 } }
  ($1 in ok) { core[$3 "," $2] = 1 }
  END { k = 0; for (x in core) k++; print k }')
PARALLEL=$(( ${SLURM_CPUS_PER_TASK:-24} < PHYS ? ${SLURM_CPUS_PER_TASK:-24} : PHYS ))
ROUNDS="${ROUNDS:-${PARALLEL}}"          # 2 · ROUNDS games per batch = 2 waves
echo "[$(date '+%F %T')] SPSA ${NAME} on $(hostname), ${PARALLEL} games at a time, ${ROUNDS} rounds per batch, ${PAIRS} pairs, ${TIME_US} µs/move, Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
build_alphazero_learn yolah_tournament
HOURS=$(max_hours_for_job 600)
start_in_container bash -c "cd /Yolah/config && exec python3 /Yolah/nnue/spsa.py --params /Yolah/config/spsa/${PARAMS} \
    --work /work/spsa/${NAME} --tournament ${BUILD_DIR}/yolah_tournament --pairs ${PAIRS} --rounds ${ROUNDS} \
    --parallel ${PARALLEL} --time ${TIME_US} --book ${BOOK} --hours ${HOURS}"
wait_forwarding_signals
echo "[$(date '+%F %T')] done (exit code ${RC})"
