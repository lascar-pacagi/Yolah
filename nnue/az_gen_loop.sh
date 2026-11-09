#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# az_gen_loop.sh — the generations loop for the ResNet: self-play by the
# current network (MCTS), exact results near the end, then training on a
# sliding window of the newest rows, generation after generation.
#
#     bash ~/Yolah/nnue/az_gen_loop.sh 1              # on the login node: submits generation 1
#     bash ~/Yolah/nnue/az_gen_loop.sh 3 train        # generation 3 again from its training (then eval, next)
#     bash ~/Yolah/nnue/az_gen_loop.sh 3 eval         # only its matches
#     bash ~/Yolah/nnue/az_gen_loop.sh status         # the progress so far
#
# Generation G (ROOT/gen_<G>, the network ROOT/models/az_<G>):
#
#   selfplay   GPU array of SELFPLAY_JOBS tasks: az_<G-1> plays (MCTS, the
#              self-play settings of config/alphazero_mcts_selfplay_player.cfg:
#              Dirichlet noise, moves sampled ∝ visits for the first 20 plies,
#              playout cap randomization) until GEN_ROWS rows in all:
#              every move is recorded, the fast searches too (flags FAST);
#              z of the positions with at most EXACT_FREE free squares is the
#              solver's exact result (flags EXACT) → gen_<G>/selfplay/*.bin.
#              A task stopped before its share (time limit) fails: run
#              `bash az_gen_loop.sh G` again, the tasks go on where they were.
#   train      GPU: az_<G-1> trained EPOCHS epochs on the WINDOW newest rows
#              (gen_<G>, gen_<G-1>, ..., then positions of OLD_CACHE to fill
#              the window: 3/4 at generation 1, 2/4, 1/4, none from generation
#              4 on with GEN_ROWS = WINDOW / 4) → az_<G> (nnue/az_gen_train.py;
#              resumes from its checkpoint). The policy is learnt on the rows
#              of the full searches only, the value on every row (z).
#   eval       GPU array: az_<G> against az_<G-1> (task 0) and az_0 (task 1),
#              EVAL_GAMES games at EVAL_SECONDS s per move, the balanced book
#              (alphazero_match.sh) → ROOT/matches.csv
#   next       (after train, with eval running) submits generation G + 1, until MAX_GEN
#
# Generation 0 is START_PT / START_TS (ResNet_1 of the distillation loop),
# copied to ROOT/models/az_00000000.{pt,ts} by the first submission.
# Sizes: a row is 336 bytes: 12.5 M rows = 4.2 GB per generation; the window
# of 50 M rows takes ~17 GB of memory in the training job.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
SCRIPT="${YOLAH_DIR}/nnue/az_gen_loop.sh"
ROOT="${ROOT:-${HOME}/AzGen}"
export WORK_DIR="${ROOT}"                       # alphazero_common.sh: /work in the container

# ── settings (sbatch --export / environment) ────────────────────────────────
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22}"             # crn22 refuses containers
START_PT="${START_PT:-${HOME}/DistillLoop/round_001/resnet.pt}"
START_TS="${START_TS:-${HOME}/DistillLoop/round_001/resnet.ts}"
OLD_CACHE="${OLD_CACHE:-/home/pgarcia/NNUE/DistillationNNUE/cache_az256}"  # fills the window ("none": no)
OLD_SELFPLAY="${OLD_SELFPLAY:-none}"            # older self-play rows, before OLD_CACHE ("none": no)
GEN_ROWS="${GEN_ROWS:-12500000}"                # new rows per generation
WINDOW="${WINDOW:-50000000}"                    # rows of a training (new + old)
EPOCHS="${EPOCHS:-10}"
LR="${LR:-5e-5}"
Q_WEIGHT="${Q_WEIGHT:-0}"                       # value target (1 − q)·z + q·root_q (not on exact rows)
FAST_POLICY_WEIGHT="${FAST_POLICY_WEIGHT:-0}"   # policy loss weight of the fast searches' rows
SELFPLAY_JOBS="${SELFPLAY_JOBS:-4}"             # GPUs of the self-play
SELFPLAY_GAMES="${SELFPLAY_GAMES:-256}"         # concurrent games per GPU
SIMS="${SIMS:-1600}"                            # simulations of a full search
SIMS_FAST="${SIMS_FAST:-200}"                   # of a fast search
EXACT_FREE="${EXACT_FREE:-22}"
EXACT_MS="${EXACT_MS:-300}"
EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_SECONDS="${EVAL_SECONDS:-1}"
MAX_GEN="${MAX_GEN:-20}"
AUTO="${AUTO:-1}"                               # 0: no next stage

STAGE="${STAGE:-submit}"
if [[ "${1:-}" == "status" ]]; then STAGE=status; set -- 0; fi
GEN="${GEN:-${1:-0}}"
FROM="${FROM:-${2:-selfplay}}"
G=$(printf '%03d' "${GEN}")
GD="gen_${G}"
mkdir -p "${ROOT}/logs" "${ROOT}/models"
if (( GEN >= 1 )); then mkdir -p "${ROOT}/${GD}/selfplay"; fi

# ── the progress so far (login node, no container) ─────────────────────────
if [[ "${STAGE}" == "status" ]]; then
    python3 - "${ROOT}" <<'EOF'
import json, os, sys, glob
root = sys.argv[1]
print("rows of the self-play of each generation")
for d in sorted(glob.glob(os.path.join(root, "gen_*"))):
    n = 0
    for f in glob.glob(os.path.join(d, "selfplay", "*.bin")):
        with open(f, "rb") as h:
            b = h.read(24)
        if len(b) == 24 and b[:8] == b"YOLAHSP1":
            n += int.from_bytes(b[16:20], "little")
    print(f"  {os.path.basename(d)}: {n:,} rows")
p = os.path.join(root, "train_log.jsonl")
if os.path.exists(p):
    print("\ntraining (validation on the generation's own rows; the last line per epoch)")
    for line in open(p):
        m = json.loads(line)
        print(f"  gen {m['gen']:3d} epoch {m['epoch']:5.2f}: mse z {m['val_mse_z']:.4f}  policy ce {m['val_policy_ce']:.4f}"
              f"  top1 {m['val_top1']:.3f}  mse exact {m['val_mse_exact']:.4f}")
p = os.path.join(root, "matches.csv")
if os.path.exists(p):
    print("\nmatches")
    for line in open(p):
        print("  " + line.rstrip())
EOF
    squeue -u "${USER}" -o "%.12i %.14j %.8T %.10M %R" 2>/dev/null | grep -E "JOBID|ag[0-9]" || true
    exit 0
fi

source "${YOLAH_DIR}/nnue/alphazero_common.sh"   # SIF, run(), build_alphazero_learn, start_in_container

# The old rows at /old in the container (if any).
OLD_ARGS=()
OLD_BIND=()
if [[ "${OLD_SELFPLAY}" != "none" && -d "${OLD_SELFPLAY}" ]]; then
    OLD_ARGS=(--old /old)
    OLD_BIND=(--bind "${OLD_SELFPLAY}:/old")
fi
if [[ "${OLD_CACHE}" != "none" && -d "${OLD_CACHE}" ]]; then
    OLD_ARGS+=(--old-cache /oldcache)
    OLD_BIND+=(--bind "${OLD_CACHE}:/oldcache")
fi

header() {
    echo "[$(date '+%F %T')] generations loop, generation ${GEN}, stage ${STAGE}, job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}} on $(hostname), Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
}

case "${STAGE}" in

# ── on the login node: the jobs of generation GEN ───────────────────────────
submit)
    (( GEN >= 1 )) || { echo "usage: az_gen_loop.sh GEN [selfplay|train|eval]   (GEN ≥ 1)" >&2; exit 1; }
    if [[ ! -s "${ROOT}/models/az_00000000.pt" ]]; then
        [[ -s "${START_PT}" && -s "${START_TS}" ]] || { echo "missing: ${START_PT} / ${START_TS}" >&2; exit 1; }
        cp "${START_PT}" "${ROOT}/models/az_00000000.pt"
        cp "${START_TS}" "${ROOT}/models/az_00000000.ts"
        echo "generation 0: ${START_PT}"
    fi
    PREV=$(printf 'az_%08d' $(( GEN - 1 )))
    if [[ "${FROM}" == "selfplay" ]]; then
        [[ -s "${ROOT}/models/${PREV}.ts" ]] || { echo "missing: ${ROOT}/models/${PREV}.ts (train generation $(( GEN - 1 )) first)" >&2; exit 1; }
        echo "{\"step\": $(( GEN - 1 )), \"ts\": \"../models/${PREV}.ts\"}" > "${ROOT}/${GD}/latest.json"
    fi
    L="${ROOT}/logs"
    common=(--parsable -p insa-gpu --gres=gpu:1 --exclude="${GPU_EXCLUDE}")
    env_base="ALL,GEN=${GEN},ROOT=${ROOT}"
    dep=()
    if [[ "${FROM}" == "selfplay" ]]; then
        j_sp=$(sbatch "${common[@]}" --export="${env_base},STAGE=selfplay" --array=0-$(( SELFPLAY_JOBS - 1 )) \
            --cpus-per-task=12 --mem=64G --time=7-00:00:00 --signal=B:USR1@600 \
            --job-name="ag${G}_sp" --output="${L}/${G}_selfplay_%A_%a.out" "${SCRIPT}")
        dep=(--dependency=afterok:"${j_sp}")
        echo "selfplay ${j_sp} (${SELFPLAY_JOBS} GPUs, ${GEN_ROWS} rows in all, by ${PREV})"
    fi
    if [[ "${FROM}" != "eval" ]]; then
        j_tr=$(sbatch "${common[@]}" ${dep[@]+"${dep[@]}"} --export="${env_base},STAGE=train" \
            --cpus-per-task=8 --mem=96G --time=4-00:00:00 --signal=B:USR1@900 \
            --job-name="ag${G}_train" --output="${L}/${G}_train_%j.out" "${SCRIPT}")
        dep=(--dependency=afterok:"${j_tr}")
        echo "train ${j_tr} (${EPOCHS} epochs on ${WINDOW} rows, lr ${LR})"
    fi
    LAST=$(( GEN >= 2 ? 1 : 0 ))
    j_ev=$(sbatch "${common[@]}" ${dep[@]+"${dep[@]}"} --export="${env_base},STAGE=eval" --array=0-${LAST} \
        --cpus-per-task=8 --mem=16G --time=16:00:00 \
        --job-name="ag${G}_eval" --output="${L}/${G}_eval_%A_%a.out" "${SCRIPT}")
    echo "eval ${j_ev}"
    if [[ "${AUTO}" == "1" && "${FROM}" != "eval" ]] && (( GEN < MAX_GEN )); then
        j_nx=$(sbatch --parsable -p insa-cpu --exclude=crn04,crn05,crn07,crn13,crn14,crn15 ${dep[@]+"${dep[@]}"} \
            --export="${env_base},STAGE=next" --cpus-per-task=1 --mem=1G --time=00:10:00 \
            --job-name="ag${G}_next" --output="${L}/${G}_next_%j.out" "${SCRIPT}")
        echo "next ${j_nx} (submits generation $(( GEN + 1 )))"
    fi
    echo "$(date '+%F %T') generation ${GEN} from ${FROM}: ${j_sp:-} ${j_tr:-} ${j_ev} ${j_nx:-}" >> "${ROOT}/jobs.txt"
    ;;

# ── self-play by az_<GEN-1>: this task's share of GEN_ROWS ──────────────────
selfplay)
    header
    TASK="${SLURM_ARRAY_TASK_ID:?array task}"
    NT="${SLURM_ARRAY_TASK_COUNT:-${SELFPLAY_JOBS}}"
    TAG="g${G}t$(printf '%02d' "${TASK}")"
    SHARE=$(( (GEN_ROWS + NT - 1) / NT ))
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    rows_done() { run python3 /Yolah/nnue/az_gen_data.py count "/work/${GD}/selfplay" --tag "_${TAG}_" | tail -1; }
    DONE=$(rows_done)
    LEFT=$(( SHARE - DONE ))
    echo "  task ${TASK}/${NT}: share ${SHARE} rows, ${DONE} already, ${LEFT} to play"
    (( LEFT > 0 )) || { echo "[$(date '+%F %T')] done already"; exit 0; }
    build_alphazero_learn
    start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/alphazero_learn selfplay \
        --config /Yolah/config/alphazero_mcts_selfplay_player.cfg --work /work/${GD} \
        --games ${SELFPLAY_GAMES} --set 'nn cache=32' --set 'nb simulations=${SIMS}' --set 'nb simulations fast=${SIMS_FAST}' \
        --max-samples ${LEFT} --record-fast --exact-free ${EXACT_FREE} --exact-ms ${EXACT_MS} \
        --report 900 --tag ${TAG}"
    wait_forwarding_signals
    DONE=$(rows_done)
    echo "[$(date '+%F %T')] stopped (exit code ${RC}): ${DONE} / ${SHARE} rows"
    if (( DONE < SHARE )); then
        echo "share not reached: run 'bash ${SCRIPT} ${GEN}' again on the login node (the tasks go on where they were)" >&2
        exit 1
    fi
    ;;

# ── az_<GEN-1> → az_<GEN> ───────────────────────────────────────────────────
train)
    header
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    trap 'echo "[$(date "+%F %T")] signal received, checkpointing"; kill -TERM ${PID} 2>/dev/null || true' USR1 TERM
    ( exec singularity exec --nv "${WSL_BIND[@]}" --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" \
        ${OLD_BIND[@]+"${OLD_BIND[@]}"} "${SIF}" python3 /Yolah/nnue/az_gen_train.py /work --gen "${GEN}" \
        --window "${WINDOW}" --epochs "${EPOCHS}" --lr "${LR}" --q-weight "${Q_WEIGHT}" \
        --fast-policy-weight "${FAST_POLICY_WEIGHT}" --seed "${GEN}" ${OLD_ARGS[@]+"${OLD_ARGS[@]}"} ) &
    PID=$!
    RC=0; wait ${PID} || RC=$?
    if kill -0 ${PID} 2>/dev/null; then RC=0; wait ${PID} || RC=$?; fi
    if [[ "${RC}" != "0" ]]; then
        echo "training stopped (exit code ${RC}): 'bash ${SCRIPT} ${GEN} train' on the login node resumes it from its checkpoint" >&2
        exit "${RC}"
    fi
    echo "[$(date '+%F %T')] done: models/$(printf 'az_%08d' "${GEN}")"
    ;;

# ── az_<GEN> against az_<GEN-1> (task 0) and az_0 (task 1) ──────────────────
eval)
    header
    B=$([[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]] && echo $(( GEN - 1 )) || echo 0)
    A="${GEN}" B="${B}" GAMES="${EVAL_GAMES}" MOVE_SECONDS="${EVAL_SECONDS}" bash "${YOLAH_DIR}/nnue/alphazero_match.sh"
    ;;

next)
    header
    cd "${YOLAH_DIR}"
    STAGE=submit GEN=$(( GEN + 1 )) FROM=selfplay bash "${SCRIPT}" $(( GEN + 1 ))
    ;;

*)
    echo "unknown stage ${STAGE}" >&2
    exit 1
    ;;
esac
