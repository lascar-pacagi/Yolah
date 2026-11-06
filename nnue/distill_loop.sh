#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# distill_loop.sh — the distillation loop: games of the dev player label
# positions, which improve the ResNet, which teaches the NNUE, which plays the
# next round's games.
#
#     bash nnue/distill_loop.sh 0          # on the login node: submits round 0
#     bash nnue/distill_loop.sh 3          # submits round 3 (needs round_003's networks)
#     python3 nnue/loop_progress.py ~/DistillLoop    # the progress so far
#
# Round r (LOOP_DIR/round_<r>, the networks NNUE_r and ResNet_r), one SLURM job
# per stage, each waiting for the one before (--dependency=afterok):
#
#   gen        CPU array, at most THROTTLE tasks at a time: GAMES games of the
#              dev player with NNUE_r (test/selfplay_label_main.cpp: MAX_NODES
#              nodes per move, the search value of every position, sampled
#              nodes of the search trees, the exact result near the end) →
#              round_<r>/games/task_<i>.bin. A task stopped (time limit,
#              preemption) starts again where it was: resubmit it.
#   prepare    CPU: the training cache round_<r>/cache (loop_prepare.py: the
#              games of round r, PREV_FRACTION of the games of each of the
#              WINDOW rounds before, and positions of OLD_CACHE: OLD_POSITIONS
#              at round 0, fewer each round, none from round OLD_ROUNDS on),
#              its space features (space_features compute-all)
#   resnet     GPU: ResNet_r fine-tuned on the cache → ResNet_{r+1}
#              (loop_resnet.py), which labels the cache
#              (distill_teacher_labels.py); the NNUE's target is then the
#              blend of that label and the search target (loop_blend.py)
#   nnue       GPU: NNUE_r distilled on the cache → NNUE_{r+1}
#              (nnue_persp_distill.py, export_value_nets.py)
#   eval_nnue  CPU: the dev player with NNUE_{r+1} against NNUE_r and NNUE_0
#   eval_resnet GPU: the MCTS player with ResNet_{r+1} against ResNet_r and ResNet_0
#              (alphazero_match.sh)
#   summary    LOOP_DIR/progress.csv (loop_progress.py)
#   next       (AUTO=1, after nnue) deletes round r's cache and the games of
#              the round that leaves the window (r − WINDOW + 1 is the oldest
#              round r + 1 reads; KEEP_DATA=1 keeps everything) and submits
#              round r + 1, until MAX_ROUND
#
# Round 0 is created from INIT_NNUE (the g046 network) and INIT_RESNET (the
# latest network of the AlphaZero 256 run). Everything lives in LOOP_DIR; the
# networks of every round are kept (round_<r>/nnue.*, round_<r>/resnet.*,
# resnet/models/az_<r>.*), the games of the last WINDOW + 1 rounds, the cache
# of the current round only.
#
# Sizes: a game ≈ 250 positions × 40 bytes ≈ 10 KB (200 k games: 2 GB); the
# cache ≈ 300 bytes per position: about 100 M positions (round r ≈ 42 M after
# the tree nodes on the games' paths are dropped, half of each of the 2 rounds
# before, the OLD_CACHE part), 30 GB.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
SCRIPT="${YOLAH_DIR}/nnue/distill_loop.sh"
LOOP_DIR="${LOOP_DIR:-${HOME}/DistillLoop}"
export WORK_DIR="${LOOP_DIR}"                     # alphazero_common.sh: /work in the container

# ── settings (sbatch --export / environment) ────────────────────────────────
EXCLUDE="${EXCLUDE:-crn04,crn05,crn07,crn13,crn14,crn15}"
# The gen stage only (a search has a fixed number of nodes, its time is the
# CPU's speed: 48 s per game and core on crn16's E5-2698 v3, 14.5 s on an
# i7-12700K): the Xeons out, the EPYCs (crn12, crn13, crn25) in.
GEN_EXCLUDE="${GEN_EXCLUDE:-crn01,crn02,crn03,crn04,crn05,crn07,crn13,crn14,crn15,crn16}"
GAMES="${GAMES:-200000}"                          # games per round
GAMES_PER_TASK="${GAMES_PER_TASK:-4000}"
THROTTLE="${THROTTLE:-8}"                         # gen tasks at a time (scontrol update JobId=… ArrayTaskThrottle=N)
GEN_CPUS="${GEN_CPUS:-16}"                        # logical CPUs per gen task (one game per physical core)
GEN_MEM="${GEN_MEM:-8G}"                          # per gen task (8 games: 64 MB tables each, the solver, the network)
MAX_NODES="${MAX_NODES:-650000}"                  # ≈ 0.5 s on an i7-12700K
RECORD_DEPTH="${RECORD_DEPTH:-10}"
TREE_PER_SEARCH="${TREE_PER_SEARCH:-8}"
EXACT_FREE="${EXACT_FREE:-22}"
EXACT_MS="${EXACT_MS:-300}"
RANDOM_PLIES="${RANDOM_PLIES:-2}"
BOOK="${BOOK:-/Yolah/config/openings/balanced.txt}"
OLD_CACHE="${OLD_CACHE:-/home/pgarcia/NNUE/DistillationNNUE/cache_az256}"
OLD_POSITIONS="${OLD_POSITIONS:-50000000}"         # at round 0; × (1 − r / OLD_ROUNDS) at round r
OLD_ROUNDS="${OLD_ROUNDS:-3}"                     # no OLD_CACHE position from this round on
WINDOW="${WINDOW:-2}"                             # previous rounds whose games are in the cache
PREV_FRACTION="${PREV_FRACTION:-0.5}"             # of their games (drawn at random, whole games)
RESNET_EPOCHS="${RESNET_EPOCHS:-1}"
RESNET_LR="${RESNET_LR:-5e-5}"
Q_WEIGHT="${Q_WEIGHT:-0.7}"                       # ResNet value target: (1 − q)·z + q·search
SEARCH_WEIGHT="${SEARCH_WEIGHT:-0.5}"             # NNUE teacher: (1 − w)·ResNet + w·search
NNUE_EPOCHS="${NNUE_EPOCHS:-2}"
NNUE_LR="${NNUE_LR:-1e-4}"
NNUE_ALPHA="${NNUE_ALPHA:-0.8}"
EVAL_NNUE_TIME="${EVAL_NNUE_TIME:-1000000}"       # µs per move
EVAL_NNUE_ROUNDS="${EVAL_NNUE_ROUNDS:-1000}"      # pairs of games
EVAL_CPUS="${EVAL_CPUS:-32}"
EVAL_RESNET_GAMES="${EVAL_RESNET_GAMES:-400}"
EVAL_RESNET_SECONDS="${EVAL_RESNET_SECONDS:-1}"
AUTO="${AUTO:-1}"
MAX_ROUND="${MAX_ROUND:-10}"
KEEP_DATA="${KEEP_DATA:-0}"
INIT_NNUE="${INIT_NNUE:-${YOLAH_DIR}/models/nnue_persp_k8_them_g046_qat_az20_distill}"   # .pt and .quantized.txt
AZ_WORK="${AZ_WORK:-${HOME}/AlphaZeroLearn/work}"
INIT_RESNET="${INIT_RESNET:-}"                    # empty: the latest network of AZ_WORK

STAGE="${STAGE:-submit}"
ROUND="${ROUND:-${1:?round number}}"
R=$(printf '%03d' "${ROUND}")
N=$(printf '%03d' $(( ROUND + 1 )))
RD="round_${R}"
ND="round_${N}"
STEP_NEXT=$(printf '%08d' $(( ROUND + 1 )))
mkdir -p "${LOOP_DIR}/logs" "${LOOP_DIR}/players" "${LOOP_DIR}/eval" "${LOOP_DIR}/resnet/models"

source "${YOLAH_DIR}/nnue/alphazero_common.sh"   # SIF, run(), build_alphazero_learn, start_in_container

# Run in the container with the old cache visible at /old.
run_old() {
    singularity exec --nv "${WSL_BIND[@]}" --bind "${YOLAH_DIR}:/Yolah" --bind "${WORK_DIR}:/work" \
        --bind "${OLD_CACHE}:/old" "${SIF}" "$@"
}

# The dev player's configuration with the network of round $1 (paths inside the container).
write_player() {
    local r="$1"
    sed -e "s|\"weights\": *\"[^\"]*\"|\"weights\": \"/work/round_${r}/nnue.quantized.txt\"|" \
        -e "s|\"description\": *\"[^\"]*\"|\"description\": \"dev player, network of round ${r} of the distillation loop\"|" \
        "${YOLAH_DIR}/config/search_study/dev_final.cfg" > "${LOOP_DIR}/players/nnue_r${r}.cfg"
}

physical_cores() {
    lscpu -p=CPU,Core,Socket | grep -v '^#' | awk -F, -v allowed="$(grep Cpus_allowed_list /proc/self/status | cut -f2)" '
      BEGIN { n = split(allowed, r, ","); for (i = 1; i <= n; i++) { m = split(r[i], ab, "-"); for (c = ab[1]; c <= (m > 1 ? ab[2] : ab[1]); c++) ok[c] = 1 } }
      ($1 in ok) { core[$3 "," $2] = 1 }
      END { k = 0; for (x in core) k++; print k }'
}

header() {
    echo "[$(date '+%F %T')] distill loop, round ${ROUND}, stage ${STAGE}, job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}} on $(hostname), Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
}

case "${STAGE}" in

# ── on the login node: round 0's networks, then the jobs of the round ───────
submit)
    if [[ "${ROUND}" == "0" && ! -f "${LOOP_DIR}/${RD}/nnue.pt" ]]; then
        mkdir -p "${LOOP_DIR}/${RD}"
        for ext in pt quantized.txt; do
            [[ -s "${INIT_NNUE}.${ext}" ]] || { echo "missing: ${INIT_NNUE}.${ext}" >&2; exit 1; }
            cp "${INIT_NNUE}.${ext}" "${LOOP_DIR}/${RD}/nnue.${ext}"
        done
        if [[ -z "${INIT_RESNET}" ]]; then
            step=$(grep -o '"step": *[0-9]*' "${AZ_WORK}/latest.json" | grep -o '[0-9]*$')
            INIT_RESNET="${AZ_WORK}/models/$(printf 'az_%08d' "${step}")"
        fi
        for ext in pt ts; do
            [[ -s "${INIT_RESNET}.${ext}" ]] || { echo "missing: ${INIT_RESNET}.${ext}" >&2; exit 1; }
            cp "${INIT_RESNET}.${ext}" "${LOOP_DIR}/${RD}/resnet.${ext}"
            cp "${INIT_RESNET}.${ext}" "${LOOP_DIR}/resnet/models/az_00000000.${ext}"
        done
        echo "{\"step\": 0, \"ts\": \"models/az_00000000.ts\"}" > "${LOOP_DIR}/resnet/latest.json"
        write_player "${R}"
        echo "round 0: NNUE ${INIT_NNUE}, ResNet ${INIT_RESNET}" | tee "${LOOP_DIR}/${RD}/origin.txt"
    fi
    for f in nnue.pt nnue.quantized.txt resnet.pt; do
        [[ -s "${LOOP_DIR}/${RD}/${f}" ]] || { echo "missing: ${LOOP_DIR}/${RD}/${f} (round ${ROUND} not ready)" >&2; exit 1; }
    done
    [[ -f "${LOOP_DIR}/players/nnue_r${R}.cfg" ]] || write_player "${R}"
    TASKS=$(( (GAMES + GAMES_PER_TASK - 1) / GAMES_PER_TASK ))
    L="${LOOP_DIR}/logs"
    common=(--parsable)
    cpu=(-p insa-cpu --exclude="${EXCLUDE}")
    gpu=(-p insa-gpu --gres=gpu:1)
    j_gen=$(sbatch "${common[@]}" -p insa-cpu --exclude="${GEN_EXCLUDE}" --export=ALL,STAGE=gen,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --array=0-$(( TASKS - 1 ))%${THROTTLE} --cpus-per-task="${GEN_CPUS}" --mem="${GEN_MEM}" --time=24:00:00 \
        --signal=B:USR1@300 --job-name="dl${R}_gen" --output="${L}/${R}_gen_%A_%a.out" "${SCRIPT}")
    j_prep=$(sbatch "${common[@]}" "${cpu[@]}" --export=ALL,STAGE=prepare,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterok:${j_gen} --cpus-per-task=16 --mem=96G --time=12:00:00 \
        --job-name="dl${R}_prep" --output="${L}/${R}_prepare_%j.out" "${SCRIPT}")
    j_res=$(sbatch "${common[@]}" "${gpu[@]}" --export=ALL,STAGE=resnet,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterok:${j_prep} --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
        --job-name="dl${R}_resnet" --output="${L}/${R}_resnet_%j.out" "${SCRIPT}")
    j_nnue=$(sbatch "${common[@]}" "${gpu[@]}" --export=ALL,STAGE=nnue,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterok:${j_res} --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
        --job-name="dl${R}_nnue" --output="${L}/${R}_nnue_%j.out" "${SCRIPT}")
    evals=$([[ "${ROUND}" == "0" ]] && echo 0 || echo 0-1)    # against the previous round and round 0
    j_evn=$(sbatch "${common[@]}" "${cpu[@]}" --export=ALL,STAGE=eval_nnue,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterok:${j_nnue} --array="${evals}" --cpus-per-task="${EVAL_CPUS}" --mem=32G --time=24:00:00 \
        --job-name="dl${R}_evnnue" --output="${L}/${R}_eval_nnue_%A_%a.out" "${SCRIPT}")
    j_evr=$(sbatch "${common[@]}" "${gpu[@]}" --export=ALL,STAGE=eval_resnet,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterok:${j_res} --array="${evals}" --cpus-per-task=8 --mem=16G --time=24:00:00 \
        --job-name="dl${R}_evresnet" --output="${L}/${R}_eval_resnet_%A_%a.out" "${SCRIPT}")
    j_sum=$(sbatch "${common[@]}" "${cpu[@]}" --export=ALL,STAGE=summary,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
        --dependency=afterany:${j_evn}:${j_evr} --cpus-per-task=1 --mem=2G --time=00:30:00 \
        --job-name="dl${R}_summary" --output="${L}/${R}_summary_%j.out" "${SCRIPT}")
    echo "round ${ROUND}: gen ${j_gen} (${TASKS} tasks of ${GAMES_PER_TASK} games, ${THROTTLE} at a time), prepare ${j_prep}, resnet ${j_res}, nnue ${j_nnue}, eval nnue ${j_evn}, eval resnet ${j_evr}, summary ${j_sum}" \
        | tee -a "${LOOP_DIR}/jobs.txt"
    if [[ "${AUTO}" == "1" && $(( ROUND + 1 )) -lt "${MAX_ROUND}" ]]; then
        j_next=$(sbatch "${common[@]}" "${cpu[@]}" --export=ALL,STAGE=next,ROUND=${ROUND},LOOP_DIR=${LOOP_DIR} \
            --dependency=afterok:${j_nnue} --cpus-per-task=1 --mem=2G --time=00:30:00 \
            --job-name="dl${R}_next" --output="${L}/${R}_next_%j.out" "${SCRIPT}")
        echo "round ${ROUND}: next ${j_next} (submits round $(( ROUND + 1 )))" | tee -a "${LOOP_DIR}/jobs.txt"
    fi
    ;;

# ── games of the dev player ─────────────────────────────────────────────────
gen)
    header
    PHYS=$(physical_cores)
    THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} < PHYS ? ${SLURM_CPUS_PER_TASK:-16} : PHYS ))
    TASK="${SLURM_ARRAY_TASK_ID:?array task}"
    FIRST=$(( TASK * GAMES_PER_TASK ))
    COUNT=$(( GAMES - FIRST < GAMES_PER_TASK ? GAMES - FIRST : GAMES_PER_TASK ))
    echo "  CPU:$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), ${PHYS} physical cores: ${THREADS} games at a time; games ${FIRST} .. $(( FIRST + COUNT - 1 ))"
    mkdir -p "${LOOP_DIR}/${RD}/games"
    build_alphazero_learn selfplay_label
    start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/selfplay_label \
        --player /work/players/nnue_r${R}.cfg --out /work/${RD}/games/task_$(printf '%04d' "${TASK}").bin \
        --first-game ${FIRST} --games ${COUNT} --threads ${THREADS} --seed $(( ROUND + 1 )) \
        --book ${BOOK} --random-plies ${RANDOM_PLIES} --max-nodes ${MAX_NODES} --record-depth ${RECORD_DEPTH} \
        --tree-per-search ${TREE_PER_SEARCH} --exact-free ${EXACT_FREE} --exact-ms ${EXACT_MS}"
    wait_forwarding_signals
    echo "[$(date '+%F %T')] done (exit code ${RC})"
    exit "${RC}"
    ;;

# ── the training cache and its space features ───────────────────────────────
prepare)
    header
    rm -rf "${LOOP_DIR}/${RD}/cache"
    N_OLD=$(( ROUND >= OLD_ROUNDS ? 0 : OLD_POSITIONS * (OLD_ROUNDS - ROUND) / OLD_ROUNDS ))
    PREV=""
    for (( p = ROUND - 1; p >= 0 && p >= ROUND - WINDOW; p-- )); do
        PREV+=" --prev /work/round_$(printf '%03d' "${p}")"
    done
    echo "  round ${ROUND}: ${N_OLD} positions of ${OLD_CACHE}, previous rounds:${PREV:- none}"
    run_old bash -c "set -e; cd /Yolah/nnue && python3 loop_prepare.py /work/${RD}${PREV} \
        --prev-fraction ${PREV_FRACTION} --old-cache /old --old-positions ${N_OLD} --seed ${ROUND}"
    build_alphazero_learn space_features
    run bash -c "${BUILD_DIR}/space_features compute-all /work/${RD}/cache --threads ${SLURM_CPUS_PER_TASK:-16}"
    du -sh "${LOOP_DIR}/${RD}/games" "${LOOP_DIR}/${RD}/cache"
    echo "[$(date '+%F %T')] done"
    ;;

# ── ResNet_{r+1}, its labels, the NNUE's target ─────────────────────────────
resnet)
    header
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    run bash -c "set -e; cd /Yolah/nnue
        python3 loop_resnet.py /work/${RD}/cache --init /work/${RD}/resnet.pt --out-dir /work/resnet \
            --step $(( ROUND + 1 )) --epochs ${RESNET_EPOCHS} --lr ${RESNET_LR} --q-weight ${Q_WEIGHT} --seed ${ROUND}
        rm -f /work/${RD}/cache/teacher_values.f16 /work/${RD}/cache/teacher_done.u8 /work/${RD}/cache/teacher_resnet.f16
        python3 distill_teacher_labels.py /work/${RD}/cache --model /work/resnet/models/az_${STEP_NEXT}.pt
        python3 loop_blend.py /work/${RD}/cache --search-weight ${SEARCH_WEIGHT}"
    mkdir -p "${LOOP_DIR}/${ND}"
    for ext in pt ts; do cp "${LOOP_DIR}/resnet/models/az_${STEP_NEXT}.${ext}" "${LOOP_DIR}/${ND}/resnet.${ext}"; done
    echo "{\"step\": $(( ROUND + 1 )), \"ts\": \"models/az_${STEP_NEXT}.ts\"}" > "${LOOP_DIR}/resnet/latest.json"
    echo "[$(date '+%F %T')] done: ${ND}/resnet.pt"
    ;;

# ── NNUE_{r+1} ──────────────────────────────────────────────────────────────
nnue)
    header
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    mkdir -p "${LOOP_DIR}/${ND}/nnue_train"
    rm -f "${LOOP_DIR}/${ND}/nnue_train/"*.pt
    NAME=nnue_persp_k8_them_g046_qat_distill
    run env YOLAH_CACHE_DIR=/work/${RD}/cache YOLAH_MODEL_DIR=/work/${ND}/nnue_train/ \
        YOLAH_INIT_MODEL=/work/${RD}/nnue.pt YOLAH_PERSP_K=8 YOLAH_PERSP_THEM=1 YOLAH_SPACE_GROUPS=0x46 \
        YOLAH_QAT=1 YOLAH_NB_EPOCHS=${NNUE_EPOCHS} YOLAH_LR=${NNUE_LR} YOLAH_DISTILL_ALPHA=${NNUE_ALPHA} \
        YOLAH_SEED=${ROUND} YOLAH_DDP_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 )) \
        bash -c "set -e; cd /Yolah/nnue && python3 nnue_persp_distill.py"
    cp "${LOOP_DIR}/${ND}/nnue_train/${NAME}.$(( NNUE_EPOCHS - 1 )).pt" "${LOOP_DIR}/${ND}/nnue.pt"
    run bash -c "set -e; cd /Yolah/nnue && python3 export_value_nets.py --persp /work/${ND}/nnue.pt /work/${ND}"
    [[ -s "${LOOP_DIR}/${ND}/nnue.quantized.txt" ]] || { echo "export failed: no ${ND}/nnue.quantized.txt" >&2; exit 1; }
    write_player "${N}"
    echo "[$(date '+%F %T')] done: ${ND}/nnue.pt, players/nnue_r${N}.cfg"
    ;;

# ── the dev player with NNUE_{r+1} against NNUE_r (task 0) and NNUE_0 (task 1) ──
eval_nnue)
    header
    REF=$([[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]] && echo "${R}" || echo 000)
    NAME="nnue_r${N}__vs__nnue_r${REF}"
    PHYS=$(physical_cores)
    PARALLEL=$(( ${SLURM_CPUS_PER_TASK:-32} < PHYS ? ${SLURM_CPUS_PER_TASK:-32} : PHYS ))
    printf '/work/players/nnue_r%s.cfg\n/work/players/nnue_r%s.cfg\n' "${N}" "${REF}" > "${LOOP_DIR}/eval/${NAME}.players"
    build_alphazero_learn yolah_tournament
    start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/yolah_tournament \
        --players /work/eval/${NAME}.players --results /work/eval/${NAME}.csv \
        --time ${EVAL_NNUE_TIME} --threads 1 --parallel ${PARALLEL} --rounds ${EVAL_NNUE_ROUNDS} \
        --opening-plies 4 --book ${BOOK} --seed $(( 1000 + ROUND ))"
    wait_forwarding_signals
    echo "[$(date '+%F %T')] done (exit code ${RC})"
    exit "${RC}"
    ;;

# ── the MCTS player with ResNet_{r+1} against ResNet_r (task 0) and ResNet_0 (task 1) ──
eval_resnet)
    header
    B=$([[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]] && echo "${ROUND}" || echo 0)
    WORK_DIR="${LOOP_DIR}/resnet" A=$(( ROUND + 1 )) B="${B}" GAMES="${EVAL_RESNET_GAMES}" \
        MOVE_SECONDS="${EVAL_RESNET_SECONDS}" bash "${YOLAH_DIR}/nnue/alphazero_match.sh"
    ;;

summary)
    header
    run bash -c "cd /Yolah/nnue && python3 loop_progress.py /work"
    ;;

# ── round r is trained: its positions go, round r + 1 starts ────────────────
next)
    header
    if [[ "${KEEP_DATA}" != "1" ]]; then
        du -sh "${LOOP_DIR}/${RD}/cache" 2>/dev/null || true
        rm -rf "${LOOP_DIR}/${RD}/cache"
        echo "deleted the cache of round ${ROUND}"
        OUT=$(( ROUND - WINDOW ))                  # round r + 1 reads the games of r + 1 − WINDOW .. r
        if (( OUT >= 0 )); then
            OD="round_$(printf '%03d' "${OUT}")"
            du -sh "${LOOP_DIR}/${OD}/games" 2>/dev/null || true
            rm -rf "${LOOP_DIR}/${OD}/games"
            echo "deleted the games of round ${OUT} (out of the window)"
        fi
    fi
    cd "${YOLAH_DIR}"
    STAGE=submit ROUND=$(( ROUND + 1 )) bash "${SCRIPT}" $(( ROUND + 1 ))
    ;;

*)
    echo "unknown stage ${STAGE}" >&2
    exit 1
    ;;
esac
