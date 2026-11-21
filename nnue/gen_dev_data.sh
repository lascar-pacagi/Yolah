#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# gen_dev_data.sh — a new data set for the value / policy networks (ResNet,
# transformer): games of the dev player (dev_u9_spsa), as in 2025 (the data of
# the ResNet: games at 10 s per move on one thread, 0 to 9 random plies at the
# start, 4 symmetries), with today's much stronger player.
#
#     bash nnue/gen_dev_data.sh            # on the login node: submits the array
#     bash nnue/gen_dev_data.sh status     # games and positions so far, per CPU model, ETA
#     bash nnue/gen_dev_data.sh resubmit   # the tasks not finished and not queued (time limit, node failure)
#
# Each task plays GAMES_PER_TASK games (test/selfplay_label_main.cpp), one per
# physical core, and writes DATA_DIR/games/task_<i>.bin (40-byte records,
# resumable: a task started again goes on where it stopped):
#   • game g: RANDOM_PLIES = 0-9 → g mod 10 random plies from the initial
#     position (no book: as in 2025), then the dev player to the end;
#   • every move a search of MAX_NODES nodes (± NODE_JITTER drawn per search,
#     so that games from the same start do not repeat each other: with a fixed
#     node count the player is deterministic). 4 M nodes ≈ 2 s on an
#     i7-12700K or an EPYC 9534 core (measured 2026-11-21: 2.6 M nodes ≈ 1.3 s),
#     ~6.5 s on the old Xeons (fixed nodes: same quality, only slower);
#   • only the positions where the player searched (the root of each search;
#     TREE_PER_SEARCH = 0: none of the tree), with the search's value and depth,
#     the move played, the game's result z, and with ≤ EXACT_FREE free squares
#     the exact result by the endgame solver (EXACT_MS per position).
# The random plies' positions are not recorded (no search value, no move of the player).
#
# Sizes: ~53 positions per game (4 test games: 211) → GAMES = 4.8 M games ≈ 250 M
# positions × 40 bytes = 10 GB. Time: ~110 s per game and EPYC core → ~150 k
# core-hours: about 2 weeks with ~500 cores, longer when the cluster is busy.
# The 4 symmetries that keep the initial position (identity, 180° rotation,
# both diagonal reflections) are applied later, when the training cache is made.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=devdata
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
SCRIPT="${YOLAH_DIR}/nnue/gen_dev_data.sh"
DATA_DIR="${DATA_DIR:-${HOME}/DevData}"
export WORK_DIR="${DATA_DIR}"                     # alphazero_common.sh: /work in the container

# ── settings (environment) ──────────────────────────────────────────────────
PLAYER="${PLAYER:-/Yolah/config/search_study/dev_u9_spsa.cfg}"   # path inside the container
GAMES="${GAMES:-4800000}"
GAMES_PER_TASK="${GAMES_PER_TASK:-6000}"          # 800 tasks (SLURM arrays: < 1000)
THROTTLE="${THROTTLE:-32}"                        # tasks at a time (scontrol update JobId=… ArrayTaskThrottle=N)
GEN_CPUS="${GEN_CPUS:-16}"                        # logical CPUs per task (one game per physical core)
GEN_MEM="${GEN_MEM:-8G}"
GEN_TIME="${GEN_TIME:-3-00:00:00}"                # ~12 h on an EPYC, ~40 h on an old Xeon (16 games at a time)
EXCLUDE="${EXCLUDE:-crn04,crn05,crn06,crn07,crn13,crn14,crn15}"   # broken for containers / no AVX2
MAX_NODES="${MAX_NODES:-4000000}"
NODE_JITTER="${NODE_JITTER:-0.1}"
RANDOM_PLIES="${RANDOM_PLIES:-0-9}"
TREE_PER_SEARCH="${TREE_PER_SEARCH:-0}"
EXACT_FREE="${EXACT_FREE:-22}"
EXACT_MS="${EXACT_MS:-300}"
SEED="${SEED:-2026}"
STAGE="${STAGE:-${1:-submit}}"

TASKS=$(( (GAMES + GAMES_PER_TASK - 1) / GAMES_PER_TASK ))
mkdir -p "${DATA_DIR}/games" "${DATA_DIR}/logs"

# Games of task i, and whether its file holds all of them.
task_count() { local first=$(( $1 * GAMES_PER_TASK )); echo $(( GAMES - first < GAMES_PER_TASK ? GAMES - first : GAMES_PER_TASK )); }

submit_array() {   # $1: the array spec (e.g. 0-799%32 or 3,17,250%32)
    sbatch --parsable -p insa-cpu --exclude="${EXCLUDE}" --array="$1" \
        --cpus-per-task="${GEN_CPUS}" --mem="${GEN_MEM}" --time="${GEN_TIME}" --signal=B:USR1@300 \
        --export=ALL,STAGE=gen,DATA_DIR="${DATA_DIR}",GAMES="${GAMES}",GAMES_PER_TASK="${GAMES_PER_TASK}",PLAYER="${PLAYER}",MAX_NODES="${MAX_NODES}",NODE_JITTER="${NODE_JITTER}",RANDOM_PLIES="${RANDOM_PLIES}",TREE_PER_SEARCH="${TREE_PER_SEARCH}",EXACT_FREE="${EXACT_FREE}",EXACT_MS="${EXACT_MS}",SEED="${SEED}" \
        --job-name=devdata --output="${DATA_DIR}/logs/gen_%A_%a.out" "${SCRIPT}"
}

# Per task file: the number of distinct games in it (the record's last field).
games_in_files() {
    python3 - "${DATA_DIR}/games" <<'EOF'
import os, sys
import numpy as np
d = sys.argv[1]
rec = np.dtype([("b", "<u8"), ("w", "<u8"), ("e", "<u8"), ("ply", "<u2"), ("bs", "u1"), ("ws", "u1"),
                ("value", "<i2"), ("move", "<u2"), ("depth", "u1"), ("kind", "u1"), ("z", "i1"), ("exact", "i1"), ("game", "<u4")])
for f in sorted(os.listdir(d)):
    if not f.endswith(".bin"): continue
    a = np.fromfile(os.path.join(d, f), dtype=rec)
    print(f[5:9], len(np.unique(a["game"])) if len(a) else 0, len(a))
EOF
}

case "${STAGE}" in

submit)
    j=$(submit_array "0-$(( TASKS - 1 ))%${THROTTLE}")
    echo "devdata ${j}: ${TASKS} tasks of ${GAMES_PER_TASK} games (${GAMES} games, ${MAX_NODES} nodes per move ± ${NODE_JITTER}, ${RANDOM_PLIES} random plies), ${THROTTLE} at a time → ${DATA_DIR}/games"
    ;;

status)
    games_in_files | awk -v tasks="${TASKS}" -v gpt="${GAMES_PER_TASK}" -v total="${GAMES}" '
        { g += $2; p += $3; if ($2 >= gpt || ($1 + 0 == tasks - 1 && $2 >= total - (tasks - 1) * gpt)) done++; started++ }
        END { printf "%d / %d games (%.1f %%), %d positions (%.1f per game); tasks: %d done, %d started, %d in all\n",
                     g, total, 100 * g / total, p, (g ? p / g : 0), done, started, tasks
              if (g) printf "at this rate: %.0f M positions for the %d games\n", p / g * total / 1e6, total }'
    # Speed: games per hour of the last hour, from the logs of the running tasks.
    squeue -h -u "${USER}" -n devdata -t R -o "%i %N %M" | head -40
    echo "(running tasks: $(squeue -h -u "${USER}" -n devdata -t R | wc -l), pending: $(squeue -h -u "${USER}" -n devdata -t PD -r | wc -l))"
    ;;

resubmit)
    # Tasks whose file is missing or incomplete, and not in the queue.
    queued=$(squeue -h -u "${USER}" -n devdata -r -o "%K" | sort -u)
    declare -A have
    while read -r t n _; do have[$((10#$t))]=$n; done < <(games_in_files)
    todo=()
    for (( t = 0; t < TASKS; t++ )); do
        n="${have[$t]:-0}"
        (( n >= $(task_count "$t") )) && continue
        grep -qx "$t" <<< "${queued}" && continue
        todo+=("$t")
    done
    if (( ${#todo[@]} == 0 )); then echo "nothing to resubmit"; exit 0; fi
    spec=$(IFS=,; echo "${todo[*]}")
    j=$(submit_array "${spec}%${THROTTLE}")
    echo "devdata ${j}: ${#todo[@]} tasks resubmitted (${spec:0:200}$([[ ${#spec} -gt 200 ]] && echo …))"
    ;;

gen)
    source "${YOLAH_DIR}/nnue/alphazero_common.sh"   # SIF, build_alphazero_learn, start_in_container
    PHYS=$(lscpu -p=CPU,Core,Socket | grep -v '^#' | awk -F, -v allowed="$(grep Cpus_allowed_list /proc/self/status | cut -f2)" '
      BEGIN { n = split(allowed, r, ","); for (i = 1; i <= n; i++) { m = split(r[i], ab, "-"); for (c = ab[1]; c <= (m > 1 ? ab[2] : ab[1]); c++) ok[c] = 1 } }
      ($1 in ok) { core[$3 "," $2] = 1 }
      END { k = 0; for (x in core) k++; print k }')
    THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} < PHYS ? ${SLURM_CPUS_PER_TASK:-16} : PHYS ))
    TASK="${SLURM_ARRAY_TASK_ID:?array task}"
    FIRST=$(( TASK * GAMES_PER_TASK ))
    COUNT=$(task_count "${TASK}")
    echo "[$(date '+%F %T')] devdata task ${TASK} on $(hostname) (job ${SLURM_JOB_ID:-local}), Yolah $(git -C "${YOLAH_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "  CPU:$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), ${PHYS} physical cores: ${THREADS} games at a time; games ${FIRST} .. $(( FIRST + COUNT - 1 ))"
    build_alphazero_learn selfplay_label
    start_in_container bash -c "cd /Yolah/config && exec ${BUILD_DIR}/selfplay_label \
        --player ${PLAYER} --out /work/games/task_$(printf '%04d' "${TASK}").bin \
        --first-game ${FIRST} --games ${COUNT} --threads ${THREADS} --seed ${SEED} \
        --random-plies ${RANDOM_PLIES} --max-nodes ${MAX_NODES} --node-jitter ${NODE_JITTER} \
        --tree-per-search ${TREE_PER_SEARCH} --exact-free ${EXACT_FREE} --exact-ms ${EXACT_MS}"
    wait_forwarding_signals
    echo "[$(date '+%F %T')] done (exit code ${RC})"
    exit "${RC}"
    ;;

*) echo "unknown stage ${STAGE} (submit, status, resubmit)" >&2; exit 1 ;;
esac
