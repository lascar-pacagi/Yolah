# AlphaZero self-play learning

Improves the value/policy ResNet (`nnue/cnn_resnet_256x30_value_policy.pt`) by
self-play, AlphaZero-style with KataGo's refinements. The full explanation —
formulas, figures, annotated sources — is chapter "Learning by self-play" of
`doc/alphazero_mcts.pdf` / `.html`.

## Files

| file | role |
|---|---|
| `player/alphazero_selfplay.{h,cpp}` | self-play driver: 128 games on one batched GPU network, hot swap of weights, training rows |
| `test/alphazero_learn_main.cpp` | `alphazero_learn selfplay` / `alphazero_learn match` |
| `nnue/alphazero_learn.py` | trainer + orchestrator (starts self-play and evaluations) |
| `nnue/alphazero_learn.def` / `.sh` | cluster image (toolchain only) and main SLURM job |
| `nnue/alphazero_selfplay.sh` | optional extra self-play GPU on another node |
| `nnue/alphazero_common.sh` | paths and shell functions shared by the two jobs |
| `config/alphazero_mcts_selfplay_player.cfg` | self-play search settings |
| `config/alphazero_mcts_eval_player.cfg` | evaluation-match settings (2 s/move comes from the command line) |

## Cluster

5 GPU nodes (7 A40 + 6 RTX 8000, 48 GB), partition `insa-gpu`. The run lives
in `~/AlphaZeroLearn/work` (the home is visible from every node). Plan for
~60 GB.

```bash
# once, where you are root (or --fakeroot):
cd nnue && sudo singularity build alphazero_learn.sif alphazero_learn.def
# on the cluster: ~/Yolah = git checkout, the .sif in ~/AlphaZeroLearn/
# (paths: nnue/alphazero_common.sh). The network is in Git LFS, which the
# cluster lacks: the clone holds a pointer, so copy the real file over it:
#     scp nnue/cnn_resnet_256x30_value_policy.pt cluster:Yolah/nnue/
sbatch alphazero_learn.sh       # 3 GPUs on one node, 14 days
sbatch alphazero_learn.sh       # again, same WORK_DIR: continues where it stopped
```

If a node with 3 free GPUs is long to get, split the run over nodes:

```bash
sbatch --gres=gpu:1 alphazero_learn.sh    # trainer + self-play + evaluations
sbatch alphazero_selfplay.sh              # + 1 self-play GPU on any node
sbatch alphazero_selfplay.sh              # + 1 more
```

The extra self-play jobs stop by themselves 6 h after the last new network.

The main job asks for 3 GPUs, 32 CPUs, 128 GB, 14 days: one self-play process
of 256 games per GPU, the trainer on the first GPU, the evaluation matches on
the last. The C++ is compiled on the compute node at the start of each job (the
flags use `-march=native`; cached per CPU model). A40 (bf16) and RTX 8000
(fp16) jobs can follow each other on the same run. `RESUBMIT=1` makes a job
resubmit itself after a clean stop. Sizing rationale: "Sizing for the cluster"
in the doc.

Progress: `WORK_DIR/evals.csv` (every 4 h: 20 games, 2 s/move, newest vs
previously evaluated network, 10 random openings × both colours, chained Elo),
`WORK_DIR/train_log.csv`, `WORK_DIR/logs/selfplay_gpu<k>.log` (check `evals/s`
there: it sets the data rate, ~72 × evals/s rows per day).

## Locally

```bash
cd build-torch && make alphazero_learn
cd ../nnue && python3 alphazero_learn.py --work /tmp/az --bin ../build-torch/alphazero_learn
```

Quick smoke test (tiny budgets): add
`--selfplay-games 32 --selfplay-set 'nb simulations=100' --selfplay-set 'nb simulations fast=25'
--batch-size 128 --min-rows 1000 --export-every 2048 --eval-every 2 --eval-games 2 --eval-seconds 0.2 --max-hours 0.1`.

Measured on an RTX 3080 (128 games, 800/200 simulations, steady state):
~0.51 games/s, ~6.9 training rows/s (~600k rows/day, ~44k games/day),
~8 300 network evaluations/s, 55 plies per game; 256 games or two processes do
not beat 128 games in one process there (the GPU is saturated). Trainer:
~3 400 samples/s at batch 1024 (4.9 GB).
