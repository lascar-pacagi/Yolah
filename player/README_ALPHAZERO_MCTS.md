# AlphaZero-style MCTS player

`AlphaZeroMCTSPlayer` (config name `"AlphaZeroMCTSPlayer"`) plays with a PUCT
tree search guided by the two-headed ResNet trained by
`nnue/cnn_resnet_value_policy_chunked.py`. No learning happens in the player;
everything a future AlphaZero self-play loop needs is exposed (see below).

## Files

| file | role |
|------|------|
| `player/alphazero_mcts.h/.cpp` | the search: PUCT, batched leaf gathering with virtual loss and multi-visit collisions, lock-free multi-threading on a pool-allocated tree, transposition cache, tree reuse, Dirichlet noise / temperature for self-play |
| `player/nn_cache.h` | Zobrist-keyed cache of network outputs (value + priors): transpositions are never re-evaluated |
| `player/alphazero_mcts_player.h/.cpp` | `Player` wrapper, JSON config, factory registration (`player.cpp`) |
| `nnue/nn_evaluator.h/.cpp` | backend interface (`nn::Evaluator`) + factory, board encoding shared with training |
| `nnue/resnet.h/.cpp` | pure C++ backend (Eigen + OpenMP, BN folded), no dependency |
| `nnue/resnet_torch.h/.cpp` | libtorch backend (GPU, fp16), built with `-DENABLE_TORCH=ON` |
| `nnue/batched_evaluator.h/.cpp` | merges the batches of several search threads into one backend call |
| `nnue/resnet_export.py` | `.pt` → `.bin` (CPU), `.ts` (libtorch), `.test.bin` (reference outputs) |
| `test/alphazero_check_main.cpp`, `test/resnet_check.*`, `test/alphazero_mcts_check.*` | self-checks and benchmarks (`alphazero_check` executable) |
| `config/alphazero_mcts_player.cfg`, `config/alphazero_mcts_torch_player.cfg` | example configs |

## Exporting the network

```bash
cd nnue && ~/env/bin/python resnet_export.py cnn_resnet_256x30_value_policy.pt
# → cnn_resnet_256x30_value_policy.bin / .ts / .test.bin
```

## Building

```bash
# CPU backend only (no new dependency)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# + libtorch GPU backend (use the libtorch inside a pip torch package)
cmake -S . -B build-torch -DCMAKE_BUILD_TYPE=Release -DENABLE_TORCH=ON \
      -DCMAKE_PREFIX_PATH=$HOME/env/lib/python3.12/site-packages/torch
cmake --build build-torch -j --target alphazero_check Yolah
```

## Checking / benchmarking

```bash
cd build && ./alphazero_check                 # network vs PyTorch reference, throughput, search invariants, tree reuse
./alphazero_check --backend torch --threads 4 --batch 64 --sims 800   # GPU (from build-torch/)
./alphazero_check --games 2 --time 1000000    # 2 games vs MCTSMemPlayer at 1 s/move
```

Measured on this machine (i7-12700K, RTX 3080): CPU backend ≈ 130–160
positions/s → ≈ 160 simulations/s; GPU fp16 ≈ 7,600 positions/s at batch
256 → ≈ 2,200 simulations/s with 4 search threads.

## Configuration

See the comment block in `player/alphazero_mcts_player.h`. Recommended
settings: CPU backend → `"nb threads": 1, "batch size": 32`; GPU backend →
`"nb threads": 4-8, "batch size": 64`. `"nn cache"` (MB) is the transposition
cache.

## Future learning (self-play loop)

1. Set `"nb simulations": 800, "dirichlet epsilon": 0.25, "temperature": 1.0,
   "temperature cutoff": 20` and play games; after every `play()`,
   `last_result().policy` is the visit distribution π over
   `last_result().children` (the policy target, action index `from*64+to` as
   in `preprocess.py`) and the game result gives z.
2. Train with `cnn_resnet_value_policy_chunked.py` (same encoding), export
   with `resnet_export.py`, hot-swap with `reload_weights()` — the tree and
   the transposition cache are reset automatically.

## Design notes

* Values are stored per node from the point of view of the player who moved
  into the node, so `Q(s,a) = W/N` of the child is what PUCT needs; the
  network value (player to move at the leaf) is negated once at the leaf and
  at every ply on the way up.
* Virtual loss = pending visits counted as losses. A descent that reaches a
  leaf already gathered by the same thread adds a visit to it (multiplicity)
  instead of being discarded, so batches stay full; collisions with other
  threads' leaves are undone and the thread joins the evaluator's barrier
  when it could gather nothing.
* Tree reuse: the new root is looked up among the root, its children and its
  grandchildren by position equality, so it works for normal play, self-play
  with one search object for both sides, and repeated searches.
* Transpositions: network outputs are cached by Zobrist hash; the tree
  statistics are not merged (a DAG backup is not exact), so the tree stays a
  tree while the expensive part is shared.
