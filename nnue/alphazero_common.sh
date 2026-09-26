# ────────────────────────────────────────────────────────────────────────────
# alphazero_common.sh — sourced by alphazero_learn.sh and alphazero_selfplay.sh
# (not a job by itself). Paths, the container launcher and the build step.
#
# SLURM runs a *copy* of the batch script from its spool directory, so the
# scripts source this file through YOLAH_DIR, never through a relative path.
# ────────────────────────────────────────────────────────────────────────────

# ── Paths (override via env / `sbatch --export=ALL,VAR=...`) ────────────────
# /data is the redundant storage visible from every node: the run lives there.
# Its I/O is light (~1.2 GB/day of self-play rows, a 141 MB network per export).
SIF="${SIF:-/data/${USER}/AlphaZeroLearn/alphazero_learn.sif}"
WORK_DIR="${WORK_DIR:-/data/${USER}/AlphaZeroLearn/work}"

# One build per CPU model: -march=native binaries are not portable across them.
CPU_TAG=$(grep -m1 'model name' /proc/cpuinfo | md5sum | cut -c1-8)
BUILD_DIR="/work/build/${CPU_TAG}"          # path inside the container

# WSL2 (local tests): the CUDA driver lives in /usr/lib/wsl/lib, which --nv
# does not bind. Harmless no-op on the cluster.
WSL_BIND=()
if [[ -d /usr/lib/wsl/lib ]]; then
    WSL_BIND=(--bind /usr/lib/wsl:/usr/lib/wsl --env LD_LIBRARY_PATH=/usr/lib/wsl/lib)
fi

# Run a command in the container: the repository at /Yolah, the run at /work.
run() {
    singularity exec --nv "${WSL_BIND[@]}" \
        --bind "${YOLAH_DIR}:/Yolah" \
        --bind "${WORK_DIR}:/work" \
        "${SIF}" "$@"
}

# Compile alphazero_learn for this node's CPU (skipped by make when up to
# date). flock: two jobs starting together on identical nodes share the build
# directory and must not run CMake in it at the same time.
build_alphazero_learn() {
    mkdir -p "${WORK_DIR}/build"
    echo "[$(date '+%F %T')] === Building alphazero_learn in ${BUILD_DIR} ==="
    run bash -c "set -e
        exec 9> /work/build/.lock_${CPU_TAG}
        flock 9
        mkdir -p ${BUILD_DIR}
        TORCH_PREFIX=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
        cmake -S /Yolah -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DENABLE_TORCH=ON \
              -DCMAKE_PREFIX_PATH=\${TORCH_PREFIX} -Wno-dev > /work/build/cmake_${CPU_TAG}.log 2>&1 \
            || { cat /work/build/cmake_${CPU_TAG}.log; exit 1; }
        cmake --build ${BUILD_DIR} --target alphazero_learn -j ${SLURM_CPUS_PER_TASK:-16} 2>&1 | tail -3"
}

# Start a command in the container in the background, as a direct child:
# `exec` makes $! the singularity process itself, which forwards SIGTERM into
# the container (a bash subshell in between would swallow the signal and
# orphan the processes inside). Sets PID.
start_in_container() {
    ( exec singularity exec --nv "${WSL_BIND[@]}" \
        --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" "${SIF}" "$@" ) &
    PID=$!
}

# Wait for PID, forwarding SIGUSR1 (SLURM's --signal) and SIGTERM (scancel)
# as SIGTERM: the program inside stops cleanly. Sets RC to its exit code.
wait_forwarding_signals() {
    trap 'echo "[$(date "+%F %T")] signal received, stopping cleanly"; kill -TERM ${PID} 2>/dev/null || true' USR1 TERM
    RC=0
    wait ${PID} || RC=$?                  # returns early when the trap fires...
    if kill -0 ${PID} 2>/dev/null; then   # ...then wait for the clean shutdown
        RC=0; wait ${PID} || RC=$?
    fi
}

# Hours left in this SLURM job minus a margin (0 outside SLURM), so that the
# program stops by itself even if the signal is lost.
max_hours_for_job() {
    local margin_s="$1" left secs
    [[ -z "${SLURM_JOB_ID:-}" ]] && { echo 0; return; }
    left=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null || echo "")
    # formats: [D-]HH:MM:SS, MM:SS
    if [[ "${left}" =~ ^(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        secs=$(( ${BASH_REMATCH[2]:-0}*86400 + 10#${BASH_REMATCH[3]}*3600 + 10#${BASH_REMATCH[4]}*60 + 10#${BASH_REMATCH[5]} ))
        awk -v s="${secs}" -v m="${margin_s}" 'BEGIN { h = (s - m) / 3600; if (h < 0.1) h = 0.1; printf "%.3f", h }'
    else
        echo 0
    fi
}
