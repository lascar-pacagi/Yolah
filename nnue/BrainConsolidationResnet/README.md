# BrainConsolidationResnet

A `rustlings`-style learning path for `cnn_resnet_value_policy_chunked.py`.
The goal: by the end, you can type the whole file from memory and explain
every line.

## How it works

Each exercise is a snapshot of the trainer file with **everything filled in
except one section** — the section you're learning. You complete the TODO,
remove the `# I AM NOT DONE` marker at the top, and the runner's tests
check your work.

```
exercises/03_bitboard_to_plane/
├── exercise.py   ← you edit this file
├── notes.md      ← concepts, API, ASCII diagrams, progressive hints
└── test.py       ← what the runner runs to check your code
```

Progress goes from leaves to root: encoders → loader → network →
training loop → `main()`.

## Usage

```bash
# from inside this directory
python3 runner.py                  # show progress + run current exercise's tests
python3 runner.py --hint           # print notes.md for the current exercise
python3 runner.py --only 03_       # run just one exercise (substring match)
python3 runner.py --watch          # re-run whenever a file is saved
python3 runner.py --reset 05_      # restore exercise 05's exercise.py from the solution-blank
```

The runner builds a small synthetic cache on first run (under `fixtures/cache/`)
so tests don't need the real ~253 GB dataset. CPU-only — no GPU is required
for any of the unit tests. The DDP-wrap step is exercised with `world_size=1`,
which NCCL handles fine on a single GPU.

## Status legend

```
[PASS]   you've removed `# I AM NOT DONE`, tests green.
[FAIL]   you've removed `# I AM NOT DONE`, tests red — open the diff.
[TODO]   `# I AM NOT DONE` is still there → runner skips this one.
```

## Curriculum (20 steps)

| #  | Section                                             |
|----|-----------------------------------------------------|
| 01 | Imports & module setup                              |
| 02 | Constants (NUM_ACTIONS, POLICY_IGNORE_INDEX, …)     |
| 03 | `_bitboard_to_plane`                                |
| 04 | `encode_cnn`                                        |
| 05 | `ChunkedShuffleLoader.__init__` — chunk sharding    |
| 06 | `ChunkedShuffleLoader._open` — memmaps              |
| 07 | `set_epoch` / `__len__`                             |
| 08 | `_producer` part A: read + permute                  |
| 09 | `_producer` part B: slice + pin + enqueue           |
| 10 | `__iter__` — thread launch + queue consume          |
| 11 | `ResBlock`                                          |
| 12 | `Net.__init__`                                      |
| 13 | `Net.forward`                                       |
| 14 | `ddp_setup`                                         |
| 15 | `TrainerDDP.__init__` part A: optimizer / loss      |
| 16 | `TrainerDDP.__init__` part B: AMP + DDP + compile   |
| 17 | `_h2d_async` + `_compute_loss` + `_policy_correct`  |
| 18 | `_run_epoch` — prefetch loop                        |
| 19 | `_validate` + `train()`                             |
| 20 | `main()` + `__main__`                               |

## Conventions

- Each `exercise.py` starts with `# I AM NOT DONE` on line 1. Remove that
  line when you think you're done. The runner will then run the tests.
- Each `notes.md` has four sections: **Concepts**, **API**, **Diagram**,
  **Hints** (progressive — only peek if stuck).
- Reference solutions live in `solutions/NN_<name>.py`. Resist looking at
  them until you've genuinely tried.

## When you finish an exercise

Tell Claude "done with exercise NN" — Claude will:
1. Diff your `exercise.py` against the canonical solution.
2. Flag anything sub-optimal (style, idiom, perf, robustness).
3. Quiz you on one related concept to make sure you understood the *why*.
