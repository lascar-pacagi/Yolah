# AlphaZero-style MCTS player

`AlphaZeroMCTSPlayer` (config name `"AlphaZeroMCTSPlayer"`) plays with a PUCT
graph search guided by the two-headed ResNet trained by
`nnue/cnn_resnet_value_policy_chunked.py`. The selection rule, the first-play
urgency and the root logic follow KataGo rather than plain AlphaZero. No
learning happens in the player itself; the self-play learning loop built on
top of it is described in [`README_ALPHAZERO_LEARN.md`](README_ALPHAZERO_LEARN.md).

**Full write-up — algorithm, formulas, diagrams and a code walkthrough — in
[`doc/alphazero_mcts.qmd`](../doc/alphazero_mcts.qmd)** (rendered to
`doc/alphazero_mcts.html` and `doc/alphazero_mcts.pdf`).

## Files

| file | role |
|------|------|
| `player/alphazero_mcts.h/.cpp` | the search: KataGo-style PUCT, graph search (one node per position), batched leaf gathering with virtual loss, lock-free multi-threading, network cache, tree reuse, LCB move selection, Dirichlet noise / forced playouts / policy target pruning / playout cap randomization for self-play |
| `player/nn_cache.h` | Zobrist-keyed cache of network outputs (value + priors): transpositions are never re-evaluated |
| `player/alphazero_mcts_player.h/.cpp` | `Player` wrapper, JSON config, factory registration (`player.cpp`) |
| `nnue/nn_evaluator.h/.cpp` | backend interface (`nn::Evaluator`) + factory, board encoding shared with training |
| `nnue/resnet.h/.cpp` | pure C++ backend (Eigen + OpenMP, BN folded), no dependency |
| `nnue/resnet_torch.h/.cpp` | libtorch backend (GPU, fp16), built with `-DENABLE_TORCH=ON` |
| `nnue/batched_evaluator.h/.cpp` | merges the batches of several search threads into one backend call |
| `nnue/resnet_export.py` | `.pt` → `.bin` (CPU), `.ts` (libtorch), `.test.bin` (reference outputs) |
| `test/alphazero_check_main.cpp`, `test/resnet_check.*`, `test/alphazero_mcts_check.*` | self-checks and benchmarks (`alphazero_check` executable) |
| `config/alphazero_mcts_player.cfg`, `config/alphazero_mcts_torch_player.cfg` | example configs (competitive play, CPU / GPU) |
| `config/alphazero_mcts_selfplay_player.cfg` | example config for self-play (noise, forced playouts, playout cap) |
| `doc/alphazero_mcts.qmd` | the full write-up, ending with the complete annotated source of everything in this table |
| `doc/make_appendix.py`, `doc/figs/build.sh` | regenerate that appendix and the diagrams — re-run both after changing the search |

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
./alphazero_check --no-bench --set 'graph search=false'   # A/B any config key
```

Measured on this machine (i7-12700K, RTX 3080): CPU backend ≈ 130–160
positions/s → ≈ 160 simulations/s; GPU fp16 ≈ 7,600 positions/s at batch
256 → ≈ 2,200 simulations/s with 4 search threads.

### What graph search costs and buys

CPU backend, one thread, 3,200 simulations per position:

| | graph | tree (`"graph search": false`) |
|---|---|---|
| network calls | 3,141–3,195 | 2,552–2,947 |
| network-cache hits | 0 | 231–646 |
| **distinct positions** | **3,141–3,195** | 2,552–2,947 |
| simulations/s | 155–159 | 162–181 |

Graph search is **not** a speedup — it is 5–15 % slower per simulation. In tree
mode, 7–20 % of the leaves are transpositions: the search allocates a duplicate
node, the network cache serves its value for free, and the simulation backs that
known value up along its own path. Those are cheap, genuine MCTS updates, just
redundant ones. In graph mode the descent links to the existing node and
continues deeper instead, ending at a position nobody has evaluated — which
costs a real network call.

Same simulation count, same node count, 20–25 % more distinct positions
examined, and every shared node pools the statistics of all its parents.
Correcting for the lower simulation rate, the advantage is **4–8 % more distinct
positions per wall-clock second**, consistently across the four test positions.

That is a proxy, not a strength result. Resolving it properly means a head-to-head
(`./alphazero_check --games N --time … --opponent <tree-mode.cfg>`), and the
effect is small enough that a few hundred games would not separate the two. The
default is `true` on the strength of the proxy and of the pooling; the flag is
there to re-test the decision cheaply when the GPU backend makes a long match
affordable.

## Configuration

Every key is listed in the comment block of
`player/alphazero_mcts_player.h`, with a table of defaults and meanings in the
parameter reference of [`doc/alphazero_mcts.qmd`](../doc/alphazero_mcts.qmd).
Recommended settings: CPU backend → `"nb threads": 1, "batch size": 32`; GPU
backend → `"nb threads": 4-8, "batch size": 64`. `"nn cache"` (MB) is the
network-output cache.

The search defaults are KataGo's. To get plain AlphaZero behaviour back for a
comparison: `"cpuct exploration log": 0`, `"cpuct utility stdev scale": 0`,
`"graph search": false`, `"use lcb": false`, `"policy target pruning": false`.

## Self-play learning

Implemented: `player/alphazero_selfplay.{h,cpp}` (the self-play driver),
`test/alphazero_learn_main.cpp` (`alphazero_learn selfplay|match`) and
`nnue/alphazero_learn.py` (trainer + orchestrator), run on the cluster with
`nnue/alphazero_learn.sh`. See [`README_ALPHAZERO_LEARN.md`](README_ALPHAZERO_LEARN.md)
and the "Learning by self-play" chapter of the doc.

What the driver relies on from this player's search:

* **π is the *play values*, not the raw visit counts** — the forced playouts
  have been taken back out and the LCB winner has been given its bonus.
  `ChildStat::visits` still reports the raw counts for display.
* **`SearchResult::full_search`** — a playout-cap fast search ran a small budget
  with no noise and no forced playouts; its policy target is not recorded.
* **`AlphaZeroMCTSPlayer::search_params(json)`** builds the search parameters
  from a config without building a player (the driver runs `az::Search`
  directly, many games on one shared evaluator).

## Design notes

* **Selection** is KataGo's PUCT:
  `Q + [c + c_log·log((W+c_base)/c_base)]·sigma_hat·P·sqrt(W+0.01)/(1+W(s,a))`,
  where `W` is the total weight of the edges leaving the node and `sigma_hat`
  scales exploration by how much the values backed up through the node
  disagree. Setting `"cpuct exploration log": 0` and
  `"cpuct utility stdev scale": 0` recovers plain AlphaZero.
* **Graph search**: one node per *position*, in a sharded hash table, so every
  path that transposes into it shares its statistics. The PUCT denominator uses
  the per-edge visit count (this parent's share), `Q` uses the node's average.
  Yolah's state graph is acyclic — every move burns a square — so no cycle
  handling is needed. `"graph search": false` reverts to a plain tree.
  Child nodes are created lazily on first descent, which is why a 400-simulation
  search holds ~400 nodes rather than ~20,000.
* **Node key** = Zobrist hash (board + side to move) mixed with the black score.
  The hash alone is right for the network cache — it is exactly the network's
  input — but not for a node, whose terminal value depends on how the points
  split between the players, which passes can skew.
* Values are stored per node from the point of view of the player who moved
  into the node, so `Q(s,a) = W/N` of the child is what PUCT needs; the
  network value (player to move at the leaf) is negated once at the leaf and
  at every ply on the way up. Each node also accumulates the sum of squares,
  used by `sigma_hat` and by the root LCB.
* Virtual loss = pending visits counted as losses, on both nodes and edges. Two
  descents of one batch that reach the same leaf each keep their own path (under
  graph search they can arrive through different parents), and the single
  evaluation is backed up along each of them; collisions with another thread's
  leaves are undone, and a thread that could gather nothing joins the
  evaluator's barrier instead of spinning.
* **Tree reuse** is a lookup in the node table: whatever depth the new position
  sat at, its statistics are kept. Everything the new root cannot reach is then
  swept (mark and sweep, once per move).
* **At the root**: forced playouts (`"forced playouts k"`, self-play only) push
  a move the noise made interesting up to `sqrt(k·P·W)` visits; policy target
  pruning takes those visits back out of the training target by inverting the
  PUCT formula; LCB selection plays the best `Q - 5·sigma(Q)` among the children
  holding at least 20 % of the leader's play value. The pruned, LCB-adjusted
  *play values* — not the raw visit counts — are what becomes pi and what the
  move is chosen from.
