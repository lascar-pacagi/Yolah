#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# tournament.sh — the grading tournament between all the players, on a GPU node.
#
#     sbatch tournament.sh                  # first job AND every following one
#
# Round robin between the players of config/tournament_players.txt: in each
# round every pair plays a random opening twice, once with each colour. Every
# player gets TIME_US per move and THREADS search threads. Games involving the
# convnet (AlphaZero MCTS on the GPU) are played one at a time, so that it never
# shares the GPU with itself; the others run PARALLEL at a time on the CPUs.
#
# Every finished game is appended to TOURNAMENT_DIR/games.csv: a job that ends
# (time limit, scancel) loses only the games in progress, and submitting the
# script again resumes the tournament exactly where it stopped.
#
# Ratings (any time, even while it runs):
#     python3 ~/Yolah/test/tournament_elo.py ~/Tournament/games.csv --matrix ~/Tournament/pairs.csv
# (the job also writes TOURNAMENT_DIR/ratings.txt when it stops).
#
# Needs: the image of alphazero_learn.def (SIF), the repository in YOLAH_DIR,
# and the convnet's weights nnue/cnn_resnet_256x30_value_policy.pt (its
# TorchScript export is made here if missing). The weights of the minimax
# players are the models/*.txt files of the repository.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=yolah_tournament
#SBATCH --output=yolah_tournament_%j.out
#SBATCH --error=yolah_tournament_%j.out
#SBATCH --partition=insa-gpu
#SBATCH -x crn23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
WORK_DIR="${TOURNAMENT_DIR:-${HOME}/Tournament}"      # games.csv, builds, ratings
source "${YOLAH_DIR}/nnue/alphazero_common.sh"         # SIF, run(), build, signals

PLAYERS="${PLAYERS:-/Yolah/config/tournament_players.txt}"   # path inside the container
TIME_US="${TIME_US:-2000000}"                  # 2 s per move
THREADS="${THREADS:-4}"                        # search threads per player
OPENING_PLIES="${OPENING_PLIES:-4}"
# CPU games at a time: one game uses THREADS cores (players move in turn); the
# GPU game uses the same, plus its GPU. Leave the GPU game its share.
CPUS="${SLURM_CPUS_PER_TASK:-32}"
PARALLEL="${PARALLEL:-$(( CPUS / THREADS - 1 ))}"
GPU_PARALLEL="${GPU_PARALLEL:-1}"

mkdir -p "${WORK_DIR}"
MAX_HOURS=$(max_hours_for_job 300)             # stop 5 min before the limit

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Yolah      : ${YOLAH_DIR} ($(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  Results    : ${WORK_DIR}/games.csv"
echo "  Players    : ${PLAYERS}"
echo "  Time/move  : ${TIME_US} µs, ${THREADS} threads, ${PARALLEL} CPU + ${GPU_PARALLEL} GPU games at a time"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# The convnet's TorchScript file (not in the repository): exported from the .pt.
if [[ ! -f "${YOLAH_DIR}/nnue/cnn_resnet_256x30_value_policy.ts" ]]; then
    echo "[$(date '+%F %T')] === Exporting the convnet to TorchScript ==="
    run bash -c "cd /Yolah/nnue && python3 resnet_export.py cnn_resnet_256x30_value_policy.pt"
fi

build_alphazero_learn yolah_tournament

echo "[$(date '+%F %T')] === Tournament ==="
# Run from config/: the configurations name their weights as ../models/… and ../nnue/….
start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/yolah_tournament --players ${PLAYERS} \
    --results /work/games.csv --time ${TIME_US} --threads ${THREADS} \
    --parallel ${PARALLEL} --gpu-parallel ${GPU_PARALLEL} --opening-plies ${OPENING_PLIES} \
    --hours ${MAX_HOURS}"
wait_forwarding_signals

echo "[$(date '+%F %T')] === Stopped (exit code ${RC}) — ratings ==="
run python3 /Yolah/test/tournament_elo.py /work/games.csv --matrix /work/pairs.csv | tee "${WORK_DIR}/ratings.txt" || true
