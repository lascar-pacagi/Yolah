#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# alphazero_learn.sh — AlphaZero/KataGo-style self-play learning on the cluster.
#
#     sbatch alphazero_learn.sh            # first job AND every following one
#
# The run lives in WORK_DIR (on /data). A job that ends (time limit,
# preemption, scancel) checkpoints; submitting the same script again continues
# it: 2 weeks + 2 weeks = one month of learning. Nothing is lost but the games
# in progress.
#
# Per job:
#   1. compile alphazero_learn (C++ self-play + matches) for this node's CPU,
#      inside the container, into WORK_DIR/build/<cpu> (skipped when cached);
#   2. run nnue/alphazero_learn.py, which starts one self-play process per GPU
#      of this job, trains on the first GPU, exports a new network every
#      EXPORT_EVERY samples and, every EVAL_HOURS, plays EVAL_GAMES games
#      (EVAL_SECONDS per move, colours alternated) on the last GPU against the
#      previously evaluated network.
#
# GPUs: 3 per job (the per-user maximum). The GPU nodes have 2 or 3 cards
# (A40 or RTX 8000, 48 GB); if a 3-GPU node is long to get, run this job with
# fewer GPUs and add the missing ones as alphazero_selfplay.sh jobs, which
# can land on any GPU node: everything goes through WORK_DIR on /data.
#     sbatch --gres=gpu:1 alphazero_learn.sh
#     sbatch alphazero_selfplay.sh ; sbatch alphazero_selfplay.sh
# To pin a GPU model, use the cluster's gres type, e.g. --gres=gpu:a40:3
# (`sinfo -o "%N %G"` lists them).
#
# Sizing (see "Sizing for the cluster" in doc/alphazero_mcts.pdf): self-play
# is the bottleneck; on an RTX 3080 self-play uses ~2.6 cores, and CPU work
# grows with the evaluation rate, so 3 GPUs twice as fast need ~16 cores (+
# trainer and matches): 32 CPUs. RAM holds a 16 M-row replay window (5.4 GB)
# and 768 network caches of 32 MB (25 GB): 128 GB.
#
# Follow the progress:  WORK_DIR/evals.csv (Elo), WORK_DIR/train_log.csv,
#                       alphazero_learn_<jobid>.out, WORK_DIR/logs/.
#
# SLURM sends SIGUSR1 30 min before the time limit (--signal); it is forwarded
# to the trainer, which saves everything and stops the self-play cleanly.
# MAX_HOURS is a second safety net computed from the job's time limit.
# ────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=alphazero_learn
#SBATCH --output=alphazero_learn_%j.out
#SBATCH --error=alphazero_learn_%j.out
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@1800

set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"                              # git checkout
source "${YOLAH_DIR}/nnue/alphazero_common.sh"                       # SIF, WORK_DIR, run(), ...

# ── Tunables (override via env / `sbatch --export=ALL,VAR=...`) ─────────────
# Initial network (used only when WORK_DIR is new), path on the host.
INIT_MODEL="${INIT_MODEL:-${YOLAH_DIR}/nnue/cnn_resnet_256x30_value_policy.pt}"

SELFPLAY_GAMES="${SELFPLAY_GAMES:-256}"       # concurrent games per GPU (128 saturate an RTX 3080)
SELFPLAY_NN_CACHE="${SELFPLAY_NN_CACHE:-32}"  # MB of network cache per game
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-1e-4}"
REUSE="${REUSE:-4}"                           # training samples per self-play row
MIN_ROWS="${MIN_ROWS:-250000}"                # N0: rows before training starts (KataGo: 250k)
MAX_WINDOW="${MAX_WINDOW:-16000000}"          # replay buffer capacity (336 B/row)
EXPORT_EVERY="${EXPORT_EVERY:-524288}"        # training samples between networks
EVAL_HOURS="${EVAL_HOURS:-4}"                 # time between evaluations
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_SECONDS="${EVAL_SECONDS:-2}"
EVAL_OPPONENTS="${EVAL_OPPONENTS:-previous}"  # previous, initial, lag:K (comma separated)
EXTRA_ARGS="${EXTRA_ARGS:-}"                  # anything else for alphazero_learn.py
RESUBMIT="${RESUBMIT:-0}"                     # 1 = sbatch this script again when the time is up

mkdir -p "${WORK_DIR}"
cp -n "${INIT_MODEL}" "${WORK_DIR}/init_model.pt" 2>/dev/null || true
MAX_HOURS=$(max_hours_for_job 1200)           # stop 20 min before the limit

echo "════════════════════════════════════════════════════════════════"
echo "  Job        : ${SLURM_JOB_ID:-(local)} on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Yolah      : ${YOLAH_DIR} ($(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  SIF        : ${SIF}"
echo "  Work dir   : ${WORK_DIR}"
echo "  Max hours  : ${MAX_HOURS}"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# ── 1. compile the C++ part for this node ───────────────────────────────────
build_alphazero_learn

# ── 2. learn ────────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] === Learning ==="
start_in_container bash -c "cd /Yolah/nnue && exec python3 -u alphazero_learn.py \
    --work /work --init-model /work/init_model.pt --bin ${BUILD_DIR}/alphazero_learn \
    --selfplay-config /Yolah/config/alphazero_mcts_selfplay_player.cfg \
    --eval-config /Yolah/config/alphazero_mcts_eval_player.cfg \
    --selfplay-games ${SELFPLAY_GAMES} --batch-size ${BATCH_SIZE} --lr ${LR} --reuse ${REUSE} \
    --selfplay-set 'nn cache=${SELFPLAY_NN_CACHE}' --max-window ${MAX_WINDOW} \
    --min-rows ${MIN_ROWS} --export-every ${EXPORT_EVERY} --eval-hours ${EVAL_HOURS} \
    --eval-games ${EVAL_GAMES} --eval-seconds ${EVAL_SECONDS} --eval-opponents ${EVAL_OPPONENTS} \
    --max-hours ${MAX_HOURS} ${EXTRA_ARGS}"
wait_forwarding_signals

echo "[$(date '+%F %T')] === Stopped (exit code ${RC}) ==="
tail -n 5 "${WORK_DIR}/evals.csv" 2>/dev/null || true
# Resubmit only after a clean stop (not after a crash: no resubmission loop).
if [[ "${RESUBMIT}" == "1" && "${RC}" == "0" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "resubmitting ${SLURM_SUBMIT_DIR}/alphazero_learn.sh"
    cd "${SLURM_SUBMIT_DIR}" && sbatch --export=ALL alphazero_learn.sh
fi
