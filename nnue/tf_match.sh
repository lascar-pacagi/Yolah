#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# tf_match.sh — MCTS with the transformer against MCTS with the ResNet trained
# alongside it (transformer_value_policy.sh, experiment 2).
#
# Both networks come from ~/TransformerVP (same data, same recipe):
#   A = transformer_reg/transformer.ts   the transformer (augmentation + dropout)
#   B = resnet_ft/transformer.ts         ResNet 256×30 with the same from-to policy head
# Same MCTS for both (config/alphazero_mcts_eval_player.cfg: no noise, no
# sampling, fp16), same time per move: the transformer pays its own speed
# (fp16 batch 64 on a 3080: 7.7 ms vs 8.1 ms, about the same). Balanced
# openings, each played with both colours, split over an array of GPUs
# (alphazero_match.sh with A_TS / B_TS, 16 h each).
#
#     bash ~/Yolah/nnue/tf_match.sh                  # submit (login node)
#     MOVE_SECONDS=0.2 bash ~/Yolah/nnue/tf_match.sh # another time per move
#     bash ~/Yolah/nnue/tf_match.sh status           # sum of the tasks so far
#     A_RUN=transformer B_RUN=resnet_ft bash ~/Yolah/nnue/tf_match.sh   # other runs of BASE_DIR
#     A_RUN=resnet_ft B_RUN=app bash ~/Yolah/nnue/tf_match.sh           # against the app's ResNet
#
# A run name is a directory of BASE_DIR (its transformer.ts), or "app": the
# app's MCTS ResNet (nnue/cnn_resnet_256x30_value_policy.ts, original 2-channel
# policy head; exported from its .pt, which git keeps, at the first use).
#
# status (also while the tasks run: it reads their logs): each task's games so
# far, A's W/D/L, score and Elo with its 95 % interval. JOB=<array job id>
# keeps one submission only; the time per move is part of the log names.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

YOLAH_DIR="${YOLAH_DIR:-${HOME}/Yolah}"
BASE_DIR="${BASE_DIR:-${HOME}/TransformerVP}"
A_RUN="${A_RUN:-transformer_reg}"
B_RUN="${B_RUN:-resnet_ft}"
GAMES="${GAMES:-2000}"
TASKS="${TASKS:-5}"
MOVE_SECONDS="${MOVE_SECONDS:-1}"
GPU_EXCLUDE="${GPU_EXCLUDE:-crn22,crn23}"
# The build and matches.csv of the AlphaZero run (alphazero_learn is already built there).
WORK_DIR="${WORK_DIR:-${HOME}/AlphaZeroLearn/work}"
TAG="${A_RUN}_vs_${B_RUN}_${MOVE_SECONDS}s"
LOGS="${BASE_DIR}/logs"
mkdir -p "${LOGS}"

if [[ "${1:-submit}" == "status" ]]; then
    python3 - "${LOGS}" "${TAG}" "${JOB:-}" <<'EOF'
import glob, math, re, sys
d, tag, job = sys.argv[1:4]
logs = sorted(glob.glob(f"{d}/tf_match_{tag}_{job or '*'}_*.out"))
if not logs:
    sys.exit(f"no log yet ({d}/tf_match_{tag}_*.out)")
pat = re.compile(r"^game (\d+)/(\d+): A (black|white) score (\d+)-(\d+) \(\d+ plies\) -> A (win|draw|loss)")
w = dr = l = 0
print(tag.replace("_vs_", " (A) vs "))
for f in logs:
    done = total = 0
    for line in open(f, errors="replace"):
        m = pat.match(line)
        if not m: continue
        done, total = int(m.group(1)), int(m.group(2))
        res = m.group(6)
        if res == "win": w += 1
        elif res == "loss": l += 1
        else: dr += 1
    print(f"  {f.rsplit('/', 1)[1]}: {done}/{total or '?'} games")
n = w + dr + l
if n == 0:
    sys.exit("no finished game yet")
s = (w + dr / 2) / n
se = math.sqrt(max((w + dr / 4) / n - s * s, 0) / n)
clip = lambda x: min(max(x, 1e-3), 1 - 1e-3)
elo = lambda x: 400 * math.log10(clip(x) / (1 - clip(x)))
print(f"{n} games: A {w}W {dr}D {l}L, score {s:.3f} ± {1.96 * se:.3f}  →  Elo {elo(s):+.0f}  [{elo(s - 1.96 * se):+.0f}, {elo(s + 1.96 * se):+.0f}]")
EOF
    exit 0
fi

ts_of() { if [[ "$1" == app ]]; then echo "${YOLAH_DIR}/nnue/cnn_resnet_256x30_value_policy.ts"; else echo "${BASE_DIR}/$1/transformer.ts"; fi; }
A_TS=$(ts_of "${A_RUN}")
B_TS=$(ts_of "${B_RUN}")
APP_TS=$(ts_of app)
if [[ ( "${A_RUN}" == app || "${B_RUN}" == app ) && ! -s "${APP_TS}" ]]; then
    # git keeps the .pt only: trace the .ts once (CPU, about a minute), in the AlphaZero image.
    SIF="${SIF:-${HOME}/AlphaZeroLearn/alphazero_learn.sif}"
    echo "exporting ${APP_TS} from its .pt"
    singularity exec --bind "${YOLAH_DIR}:/Yolah" "${SIF}" \
        bash -c "cd /Yolah/nnue && python3 resnet_export.py cnn_resnet_256x30_value_policy.pt" \
        || { echo "export failed; run it on a compute node: srun -p insa-cpu singularity exec --bind ${YOLAH_DIR}:/Yolah ${SIF} bash -c 'cd /Yolah/nnue && python3 resnet_export.py cnn_resnet_256x30_value_policy.pt'" >&2; exit 1; }
fi
for f in "${A_TS}" "${B_TS}"; do
    [[ -s "${f}" ]] || { echo "missing: ${f} (training not finished, or not exported)" >&2; exit 1; }
done
(( GAMES % (2 * TASKS) == 0 )) || { echo "GAMES must be a multiple of 2 × TASKS" >&2; exit 1; }
cd "${LOGS}"
JID=$(A_TS="${A_TS}" B_TS="${B_TS}" A_NAME="${A_RUN}" B_NAME="${B_RUN}" WORK_DIR="${WORK_DIR}" \
    sbatch --parsable --export=ALL,GAMES=$(( GAMES / TASKS )),MOVE_SECONDS="${MOVE_SECONDS}" \
    --array=0-$(( TASKS - 1 )) --time=16:00:00 --exclude="${GPU_EXCLUDE}" \
    --job-name="tf_match" --output="${LOGS}/tf_match_${TAG}_%A_%a.out" --error="${LOGS}/tf_match_${TAG}_%A_%a.out" \
    "${YOLAH_DIR}/nnue/alphazero_match.sh")
echo "tf_match ${JID}: ${A_RUN} vs ${B_RUN}, ${GAMES} games on ${TASKS} GPUs, ${MOVE_SECONDS} s/move"
echo "logs: ${LOGS}/tf_match_${TAG}_${JID}_*.out   —   result: MOVE_SECONDS=${MOVE_SECONDS} bash ${YOLAH_DIR}/nnue/tf_match.sh status"
echo "$(date '+%F %T') tf_match ${JID} ${A_RUN} vs ${B_RUN} GAMES=${GAMES} MOVE_SECONDS=${MOVE_SECONDS}" >> "${BASE_DIR}/tf_match_jobs.txt"
