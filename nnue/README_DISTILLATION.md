# Distilling the ResNet into the small networks

Teacher: `cnn_resnet_256x30_value_policy.pt` (two-headed ResNet).
Students: `nnue_193x1024x64x32x1.pt` (193 → 1024 → 64 → 32 → 1) and
`features_119x256x64x1.pt` (119 → 256 → 64 → 1), value heads only.
Method and metrics: see the docstring of `distill_common.py`.

```
distill_common.py                    shared: teacher loading, board→planes, symmetries (TTA),
                                     DistillLoader (X, z, v_teacher), Metrics
distill_teacher_labels.py            cache → teacher_values.f16 / teacher_done.u8 / teacher_meta.json
distill_boards_for_features.py       features cache → boards.u8 (aligned, verified)
nnue193x1024x64x32x1_distill.py      student trainer (NNUE)          + .sh/.def for SLURM
features_net119x256x64x1_distill.py  student trainer (features net)  + .sh/.def for SLURM
```

## Cluster

```bash
singularity build nnue193x1024x64x32x1_distill.sif nnue193x1024x64x32x1_distill.def
cp models/cnn_resnet_256x30_value_policy.pt models/nnue_193x1024x64x32x1.pt <MODEL_DIR>/
sbatch nnue193x1024x64x32x1_distill.sh          # reuses cache_nnue193 of the plain run
sbatch features_net119x256x64x1_distill.sh      # reuses cache_features119
```
Tunables (env or `sbatch --export`): `YOLAH_NB_EPOCHS`, `YOLAH_DISTILL_ALPHA`
(0.8), `YOLAH_LR` (3e-4), `DISTILL_TTA=1` (teacher averaged over the 8
symmetries, 8× labelling cost), `LABEL_END=N` (label only a prefix; training
uses labelled chunks only). Labelling is resumable and runs ≈ 10k positions/s
per RTX-3080-class GPU (≈ 1 day per 10⁹ positions); a job that hits the time
limit is resubmitted and continues.

## Local test (small cache)

```bash
cd nnue
YOLAH_GAME_DIR=<dir with a few games_* files> ~/env/bin/python preprocess_nnue.py /tmp/cache_nnue
~/env/bin/python distill_teacher_labels.py /tmp/cache_nnue --model cnn_resnet_256x30_value_policy.pt
YOLAH_CACHE_DIR=/tmp/cache_nnue YOLAH_MODEL_DIR=/tmp/models/ YOLAH_INIT_MODEL=$PWD/nnue_193x1024x64x32x1.pt \
YOLAH_NB_EPOCHS=2 YOLAH_CHUNK_SIZE=262144 ~/env/bin/python nnue193x1024x64x32x1_distill.py
```
(`YOLAH_CHUNK_SIZE` must be ≤ the labelled size; the default 4M is for the
full cache.)

## Output per epoch

```
epoch 1 train [vs outcome] loss … mse … sign-acc … bucket-acc … signed-acc …   ← the training set
epoch 1 train [vs teacher] mse … sign-agree … bucket-agree …                    ← the augmented training set
epoch 1 train [teacher   ] mse … sign-acc … bucket-acc … signed-acc …          ← ceiling
epoch 1 val   … (same three lines)
```
