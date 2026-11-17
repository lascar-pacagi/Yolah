#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# top3_tournament.sh — rank the strongest AIs of the application: Minimax NNUE
# dev (dev_u8), MCTS ResNet and AlphaZero, with the application's settings
# (config/app_top3/), plus the dev of before campaigns U1-U10 (dev_final), 1 s
# per move, balanced openings, both colours. The games without an MCTS (dev
# against dev_final) run on the CPUs alongside the GPU game.
#
#     bash ~/Yolah/nnue/top3_tournament.sh            # submit JOBS jobs (login node)
#     bash ~/Yolah/nnue/top3_tournament.sh status     # ratings of all the games so far
#
# Every game involves an MCTS on the GPU, and a GPU plays one game at a time
# (the MCTS must not share it: its strength depends on its simulations per
# second). So the games are spread over JOBS jobs on JOBS GPU nodes, each with
# its own TOURNAMENT_DIR and its own SEED (another order of the openings); the
# ratings pool their games.csv. One game takes about a minute: ROUNDS 150 per
# job = 1800 games (6 pairs × 2 colours × 150; 5 of the 6 pairs on the GPU)
# in about 25 h; 4 jobs: 7200 games. Changing the players: use a new BASE
# (BASE=~/TournamentTop4); "status" pools every BASE given in POOL (space-separated).
# A job that stops (time limit, scancel) is resumed by submitting it again with
# the same JOB number (TOURNAMENT_DIR keeps its games.csv).
#
#     JOBS=6 ROUNDS=100 bash ~/Yolah/nnue/top3_tournament.sh
#     ONLY=3 bash ~/Yolah/nnue/top3_tournament.sh     # (re)submit job 3 only
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
BASE="${BASE:-${HOME}/TournamentTop3}"
JOBS="${JOBS:-4}"
ROUNDS="${ROUNDS:-150}"
AZ_MODEL="${AZ_MODEL:-${HOME}/AlphaZeroLearn/work/models/az_00271360.ts}"   # the application's AlphaZero
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22,crn23}"                                    # crn22: no containers; crn23: excluded by tournament.sh

if [[ "${1:-submit}" == "status" ]]; then
    files=()
    for b in ${POOL:-${BASE}}; do for f in "${b}"/job*/games.csv; do [[ -f "${f}" ]] && files+=("${f}"); done; done
    (( ${#files[@]} )) || { echo "no games yet in ${POOL:-${BASE}}/job*/games.csv"; exit 1; }
    for f in "${files[@]}"; do echo "  ${f}: $(( $(wc -l < "${f}") - 1 )) games"; done
    exec python3 "${YOLAH_DIR}/test/tournament_elo.py" "${files[@]}" --matrix "${BASE}/pairs.csv"
fi

[[ -s "${AZ_MODEL}" ]] || { echo "missing ${AZ_MODEL} (AZ_MODEL=...)" >&2; exit 1; }
for k in $(seq 1 "${JOBS}"); do
    [[ -n "${ONLY:-}" && "${ONLY}" != "${k}" ]] && continue
    dir="${BASE}/job${k}"
    mkdir -p "${dir}"
    [[ -f "${dir}/az256.ts" ]] || { cp "${AZ_MODEL}" "${dir}/az256.ts"; basename "${AZ_MODEL}" > "${dir}/az256.name"; }
    jid=$(cd "${dir}" && sbatch --parsable --job-name="top3_${k}" --time=2-00:00:00 --exclude="${GPU_EXCLUDE}" \
        --output="${dir}/top3_%j.out" --error="${dir}/top3_%j.out" \
        --export=ALL,TOURNAMENT_DIR="${dir}",PLAYERS=/Yolah/config/app_top3/players.txt,THREADS=4,TIME_US=1000000,BOOK=/Yolah/config/openings/balanced.txt,SEED="${k}",ROUNDS="${ROUNDS}" \
        "${YOLAH_DIR}/nnue/tournament.sh")
    echo "job ${k}: ${jid} → ${dir}/games.csv"
done
echo "ratings: bash ${YOLAH_DIR}/nnue/top3_tournament.sh status"
