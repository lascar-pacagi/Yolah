#!/usr/bin/env python3
"""
make_appendix.py — generate doc/appendix.qmd from the real sources.

The appendix of doc/alphazero_mcts.qmd lists every file of the AlphaZero MCTS
player in full. Rather than pasting the code into the document (where it would
rot on the first edit), the listings are generated from the working tree: each
file gets an orientation paragraph, a table of the symbols it defines, and its
complete source. Re-run this and re-render after touching any of them.

    ./make_appendix.py && quarto render alphazero_mcts.qmd

The prose lives here, next to the file list; the line-by-line commentary lives
in the sources themselves, which is the only place it can stay correct.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix.qmd")

# (path, language, orientation, [(symbol, role), ...])
PARTS = [
("The search", """
These five files are the search proper. They depend on `game.h`, `misc.h`,
`zobrist.h` and the `nn::Evaluator` interface of the next part, and on nothing
else.
""", [

("player/alphazero_mcts.h", "cpp", """
The public surface: the parameter block, what a search reports, and the class.
The header comment is the algorithm overview — the same one @sec-puct through
@sec-playoutcap expand on.

`SearchParams` is a plain aggregate with the KataGo defaults baked in, so
`az::SearchParams{}` is a usable configuration once a budget is set. Everything
private is forward-declared: `Node`, `Edge`, `PathStep` and `Worker` are defined
in the `.cpp`, so nothing that includes this header sees the graph
representation.
""", [
 ("`SearchParams`", "every knob, grouped: budget, parallelism, PUCT, FPU, graph, root, self-play"),
 ("`ChildStat`", "one root move as reported: move, prior, raw edge visits, `q`, `q2`, play value, LCB"),
 ("`SearchResult`", "the move, the root value, the counters, `full_search`, the children and $\\pi$"),
 ("`SearchResult::to_string`", "the one-line-plus-children summary printed by `\"verbose\": true`"),
 ("`Search`", "the search object: owns the node table, the network cache and the parameters"),
 ("`Search::search`", "run one search from a position and pick a move"),
 ("`Search::reset`", "drop the graph (new game); the network cache survives"),
 ("`Search::clear_cache`", "drop the cached network outputs (after new weights)"),
 ("`Search::params`", "read/write access to the live parameters"),
 ("`Search::Shard`", "one stripe of the node table: a mutex and a `hash → Node*` map"),
]),

("player/alphazero_mcts.cpp", "cpp", """
The implementation, in the order the search uses it: the graph representation,
the node table and its garbage collection, re-rooting, the selection rule, the
per-thread worker loop, and finally the root post-processing that turns visit
counts into a move and a training target.

Two invariants hold everywhere and explain most of the code:

* a node's `w` is the sum of values **for the player who moved into it**, so
  `w/n` read from the parent is already the parent's $Q(s,a)$ and the sign flips
  once per ply on the way up;
* `state` is the only atomic with ordering that matters. `EXPANDED` is stored
  with release and read with acquire, which is what makes the edge vector safe
  to read without a lock; every statistic is relaxed, because a stale read only
  changes which leaf a descent picks.
""", [
 ("`Search::Edge`", "one per (position, move): prior, successor pointer, this parent's visit and pending counts"),
 ("`Search::Node`", "one per position: `n`, `pending`, `w`, `w2`, `state`, the GC mark, the network value, the edges"),
 ("`Search::PathStep`", "one ply of a descent: the node reached and the edge taken to reach it"),
 ("`Search::Worker`", "per-thread buffers: PRNG, the batch's leaves, its descents, the path arena, the request/result spans"),
 ("`same_position`", "full position equality, used by the tree-mode re-rooting scan"),
 ("`mix`", "splitmix64 finaliser, used to fold the score into the node key"),
 ("`node_key`", "Zobrist hash $\\oplus$ mixed black score — see @sec-nodekey for why the score is needed"),
 ("`Search::terminal_value`", "the exact result of a finished game, for the player to move"),
 ("`Search::new_node`", "allocate one node from the pool resource and count it"),
 ("`Search::lookup`", "find a position in the node table (used by re-rooting)"),
 ("`Search::destroy_all`", "destroy every node; the table is the ownership list"),
 ("`Search::mark` / `collect`", "mark and sweep from the new root; safe as a plain recursion because the graph is acyclic"),
 ("`Search::get_or_create`", "table lookup or insert, under the shard lock"),
 ("`Search::new_root`", "start a fresh graph at a position"),
 ("`Search::reuse`", "re-root by table lookup (graph mode) or by the two-ply scan (tree mode), then collect"),
 ("`Search::expand_root`", "the one unbatched network call, for the root itself"),
 ("`Search::apply_root_noise`", "mix Dirichlet noise into a *copy* of the root priors, leaving the graph's own priors alone"),
 ("`Search::utility_stdev_factor`", "$\\hat\\sigma(s)$ from the node's `w` and `w2` — @sec-puct"),
 ("`Search::fpu_value`", "what an unvisited child is assumed to be worth — @sec-fpu"),
 ("`Search::select_child`", "one PUCT argmax, plus the forced-playout override at the root"),
 ("`Search::priors_from_logits`", "softmax over the legal moves only, with `policy_temperature`"),
 ("`Search::expand`", "build the edge vector, store the network value, publish `EXPANDED`"),
 ("`Search::backup`", "negamax accumulation into every node and edge of one path"),
 ("`Search::undo_pending`", "roll the virtual loss back on an abandoned descent"),
 ("`Search::run_worker`", "gather → evaluate → expand → back up, until the budget is spent"),
 ("`Search::root_play_values`", "policy target pruning and the LCB adjustment — @sec-root"),
 ("`Search::make_result`", "assemble the statistics, $\\pi$ and the move choice"),
]),

("player/alphazero_mcts_player.h", "cpp", """
The `Player` wrapper. The header comment is the authoritative JSON schema — the
parameter reference in @sec-parameters is the same list with the defaults spelled
out.
""", [
 ("`AlphaZeroMCTSPlayer`", "`Player` implementation: owns the pool resource, the evaluator and the `az::Search`"),
 ("`AlphaZeroMCTSPlayer::play`", "one search, one move; the result is kept for `last_result()`"),
 ("`AlphaZeroMCTSPlayer::game_over`", "reset the graph at the end of a game"),
 ("`AlphaZeroMCTSPlayer::last_result`", "the full root statistics of the last search — the self-play hook"),
 ("`AlphaZeroMCTSPlayer::reload_weights`", "hot-swap the network: resets the graph and the network cache"),
 ("`AlphaZeroMCTSPlayer::config`", "round-trip the live configuration back to JSON"),
]),

("player/alphazero_mcts_player.cpp", "cpp", """
JSON in, `SearchParams` out, plus the decision of whether to wrap the backend in
a `BatchingEvaluator`. The only subtlety is the ordering in the constructor: the
backend has to exist before `batch size` can default to
`preferred_batch_size()`, and the search has to be destroyed before the pool
resource it allocated the graph from — hence the explicit destructor.
""", [
 ("`read_nb_threads`", "accepts a number or the string `\"hardware concurrency\"`"),
 ("`AlphaZeroMCTSPlayer::AlphaZeroMCTSPlayer`", "parse the config, build the backend, wrap it if multi-threaded, build the search"),
 ("`AlphaZeroMCTSPlayer::~AlphaZeroMCTSPlayer`", "destroy the search first: the graph lives in `memory`"),
]),

("player/nn_cache.h", "cpp", """
A direct-mapped cache of network outputs, keyed by the Zobrist hash — which is
exactly the network's input, so a hit is always valid. It is the second line of
defence behind the node table: it also catches positions swept out of the graph,
and it survives across moves and games.

Two details worth noting: the stored move count is verified on lookup, so a
64-bit collision cannot hand back another position's priors; and priors are
16-bit fixed point, which costs 1/65535 of resolution and halves the entry size.
""", [
 ("`NNCache::Entry`", "key, value, move count, and the priors as `uint16` fixed point"),
 ("`NNCache::NNCache`", "size in bytes, rounded down to a power-of-two entry count; 0 disables it"),
 ("`NNCache::lookup`", "hit only if both the key and the move count match"),
 ("`NNCache::store`", "always overwrite — the table is direct-mapped"),
 ("`NNCache::clear`", "take every stripe lock, then wipe (used by `reload_weights`)"),
]),
]),

("The network interface and its backends", """
Everything below `nn::Evaluator` is interchangeable: the search never learns
which backend it is talking to, and `make_evaluator` picks one from the same
JSON the player was given.
""", [

("nnue/nn_evaluator.h", "cpp", """
The interface, plus the two encodings that *must* agree with the training code:
`encode_planes` (the four input planes, in the reversed cell order of
@sec-encoding) and `action_index` (`from*64 + to`). If either drifts from the
Python side, the network silently reads a permuted board.
""", [
 ("`PLANES`, `PLANE_CELLS`, `INPUT_FLOATS`, `NUM_ACTIONS`", "the shapes, shared with the training script"),
 ("`encode_planes`", "position → 256 floats, NCHW, cell $i$ = bit $63-i$"),
 ("`action_index`", "move → policy index; a pass maps to 0 (A1→A1)"),
 ("`Request` / `Result`", "a position with its legal moves / a value and the logits of exactly those moves"),
 ("`Evaluator`", "the abstract backend: `evaluate`, `preferred_batch_size`, `info`, `reload`"),
 ("`make_evaluator`", "factory over the `\"backend\"` key"),
]),

("nnue/nn_evaluator.cpp", "cpp", """
The factory. Nothing more: it reads `\"backend\"`, `\"weights\"`, `\"device\"`,
`\"fp16\"` and `\"nb eval threads\"`, and fails loudly when the libtorch backend
is asked for in a build that does not have it.
""", [
 ("`make_evaluator`", "`\"cpu\"` → `ResNetCPU`, `\"torch\"` → `ResNetTorch` (only with `ENABLE_TORCH`)"),
]),

("nnue/batched_evaluator.h", "cpp", """
The wrapper that merges several search threads into one backend call. The
header comment explains the three conditions that fire a batch; the empty-request
barrier is the part worth remembering, because it is what a search thread whose
descents all collided uses instead of spinning.
""", [
 ("`BatchingEvaluator`", "`Evaluator` that owns a backend and a worker thread"),
 ("`BatchingEvaluator::Job`", "one caller's request/result spans plus its completion flag"),
 ("`BatchingEvaluator::evaluate`", "enqueue and block; an empty span is a barrier join"),
 ("`BatchingEvaluator::set_nb_clients`", "how many threads must be waiting before a batch fires early"),
 ("`BatchingEvaluator::worker_loop`", "wait, concatenate, call the backend once, scatter, wake the callers"),
]),

("nnue/batched_evaluator.cpp", "cpp", """
The queue. FIFO, so no client can starve; the worker fires on whichever of the
three conditions comes first, and `reload` takes the same lock as the worker so
weights cannot change mid-batch.
""", []),

("nnue/resnet.h", "cpp", """
The pure-C++ backend's declaration. The header comment is the performance
rationale: NHWC activations, im2col + one GEMM per convolution, BatchNorm folded
at export time, OpenMP over chunks of the batch.
""", [
 ("`ResNetCPU`", "`Evaluator` over the `.bin` written by `resnet_export.py`"),
 ("`ResNetCPU::Block`", "one residual block's folded weights: two (scale, shift) pairs and two kernels"),
 ("`ResNetCPU::Scratch`", "per-thread activation and im2col buffers, grown lazily"),
 ("`ResNetCPU::forward`", "the reference path used by the self-check: planes in, values and all 4096 logits out"),
 ("`ResNetCPU::forward_chunk`", "trunk plus heads on at most `MAX_CHUNK` positions"),
 ("`ResNetCPU::policy_logit`", "one logit from the 128-d policy feature vector — only the legal moves are computed"),
]),

("nnue/resnet.cpp", "cpp", """
The inference itself. The shape to keep in mind is
$\\text{Out}_{64n \\times C} = \\text{Col}_{64n \\times 9C}\\cdot W^{\\mathsf T}_{9C \\times C}$:
every $3\\times3$ convolution is one GEMM, and the pre-activation BatchNorm and
ReLU are applied while the im2col patches are gathered, so the activations are
read once per layer.

`flush_denormals` is not an optimisation, it is a correctness-of-performance
fix: the trained network produces activations around $10^{-41}$, and denormal
arithmetic is ~100× slower on x86. FTZ/DAZ are per-thread MXCSR flags, so every
OpenMP worker sets them itself rather than relying on link flags the LTO link
line does not carry.
""", [
 ("`TapTable`", "precomputed source pixel for each (output pixel, kernel tap), $-1$ outside the board"),
 ("`Reader`", "little-endian reader for the `.bin` format documented in `resnet_export.py`"),
 ("`bn_relu`", "$y = \\max(0, s_c x + t_c)$ over a row-major block — the folded pre-activation BN"),
 ("`im2col`", "gather the nine taps of every output pixel into one GEMM operand row"),
 ("`flush_denormals`", "set FTZ/DAZ on the calling thread"),
 ("`ResNetCPU::load`", "read the header, then every folded tensor in export order"),
 ("`ResNetCPU::forward_chunk`", "stem, $B$ residual blocks, output BN, then both heads"),
 ("`ResNetCPU::evaluate`", "split the batch into chunks, run them under OpenMP, fill in only the legal logits"),
]),

("nnue/resnet_torch.h", "cpp", """
The libtorch backend's declaration, compiled only under `-DENABLE_TORCH=ON`.
""", [
 ("`ResNetTorch`", "`Evaluator` over the TorchScript module, on CUDA or CPU, optionally in fp16"),
]),

("nnue/resnet_torch.cpp", "cpp", """
Load the traced module, stage the batch into a pinned CPU tensor, one forward,
read the two outputs back. The mutex serialises forwards — callers are batched
upstream anyway.
""", [
 ("`ResNetTorch::load`", "`torch::jit::load`, move to the device, `eval()`, optionally `to(half)`"),
 ("`ResNetTorch::evaluate`", "encode into the staging tensor, forward, scatter the legal logits back"),
]),
]),

("Training and export", """
The Python half. `cnn_resnet_value_policy_chunked.py` is the script the shipped
weights were trained with and the one `resnet_export.py` imports; its sibling
`cnn_resnet_value_policy.py` defines the *same* network and differs only in the
data pipeline (a stock `DataLoader` with worker processes, which is the right
choice when the cache fits in page cache).
""", [

("nnue/cnn_resnet_value_policy_chunked.py", "python", """
The network definition, the encoding, and the training loop.

The loader is the part that is specific to this project rather than to AlphaZero.
The position cache is a ~253 GB file; sampling it uniformly at random turns
training into millions of scattered 256-byte reads, which on a spinning disk is
seek-bound. `ChunkedShuffleLoader` reads ~1 GB **contiguous** chunks
sequentially, shuffles within a chunk and reshuffles the chunk order each epoch
— an approximate shuffle, but a chunk holds ~4 M positions, and sequential reads
are fast without pre-warming 253 GB of RAM. One background thread per rank does
the reading and batch construction while the main thread drives the GPU; the
overlap is real because numpy's memcpy and CUDA's pin-memory both release the
GIL.

The DDP detail that bites: every rank must yield *exactly* the same number of
batches, or the first rank to finish hangs the others on the gradient
all-reduce. Hence the floor division in the chunk sharding and the dropped
trailing batches.
""", [
 ("`_bitboard_to_plane`, `encode_cnn`", "the Python side of @sec-encoding — must match `nn::encode_planes`"),
 ("`ChunkedShuffleLoader`", "sequential chunked reader over the memmap cache, double-buffered, DDP-sharded"),
 ("`ChunkedShuffleLoader._producer`", "the background thread: read a chunk, shuffle it, emit pinned batches"),
 ("`ResBlock`", "pre-activation residual block (@fig-resblock)"),
 ("`Net`", "stem, $B$ blocks, output BN, value head and policy head (@fig-net)"),
 ("`Net.forward`", "returns `(value, policy_logits)`"),
 ("`TrainerDDP`", "one process per GPU: channels-last, AMP, `torch.compile`, cosine LR"),
 ("`TrainerDDP._compute_loss`", "$\\lambda_v\\,\\mathrm{MSE}(v,z) + \\lambda_p\\,\\mathrm{CE}(\\ell, a^\\star)$"),
 ("`TrainerDDP._run_epoch` / `_validate`", "the loop, with value sign accuracy and policy top-1 as the readable metrics"),
 ("`main`", "contiguous train/val split, build the loaders inside the spawned process, train"),
]),

("nnue/resnet_export.py", "python", """
One checkpoint in, three artefacts out. The module docstring is the
authoritative `.bin` layout; @sec-export is the same thing with the algebra
spelled out.

The asymmetry to remember: a BatchNorm that *follows* a convolution folds into
its weights and bias; a BatchNorm that *precedes* one cannot, so its $(s,t)$
pair is written out for the C++ side to apply.
""", [
 ("`bn_affine`", "eval-mode BatchNorm → $(s, t)$ with $s = \\gamma/\\sqrt{\\sigma^2+\\epsilon}$, $t = \\beta - \\mu s$"),
 ("`conv_to_out_tap_in`", "$(out, in, 3, 3)$ → $(out, 9, in)$, the layout the CPU GEMM wants"),
 ("`Writer`", "little-endian float32 sink, counting what it wrote"),
 ("`export_bin`", "header, stem, blocks, output BN, both heads — the order `ResNetCPU::load` reads"),
 ("`export_torchscript`", "trace in eval mode on a dummy batch and save"),
 ("`export_test_vectors`", "random positions with PyTorch's exact fp32 outputs, for the C++ self-check"),
 ("`main`", "tolerates a DDP-wrapped checkpoint, infers $C$, $B$, $F$, $A$ from the tensors"),
]),
]),

("The self-checks", """
`alphazero_check` is the regression suite: it verifies the C++ backends against
PyTorch's own outputs, benchmarks them, and asserts the search's invariants.
""", [

("test/resnet_check.cpp", "cpp", """
Replays `*.test.bin` through a backend and compares. The tolerance is 2e-3 for
the fp32 paths and 2e-2 for GPU fp16; the policy argmax is compared separately,
because a logit difference that never changes the argmax is harmless to the
search.
""", [
 ("`resnet_check`", "max |value error|, max |logit error| and argmax disagreements vs the reference"),
 ("`resnet_bench`", "positions per second at a given batch size"),
]),

("test/alphazero_mcts_check.cpp", "cpp", """
The search's invariants. `check_result` is the interesting part: it asserts that
the child visits sum to the root visits, that $\\pi$ sums to 1, that Q and the priors
are in range, that the budget was honoured exactly — unless this was a
playout-cap fast search — and that the move played is the first child in the
reported (play-value) order.

`alphazero_reuse_check` deserves its shape. Asserting that a *random* opponent
reply lands on a subtree the search explored is a coin flip — with ~100 legal
replies and a few hundred simulations it usually does not — so that walk only
reports, and the two cases that must always reuse are checked explicitly:
searching the same position twice must re-root on the root itself, and playing
our own move must re-root on a child.
""", [
 ("`random_position`", "a random legal position, for the search check"),
 ("`check_result`", "the invariants above"),
 ("`alphazero_search_check`", "four positions, fixed simulation budget, no tree reuse"),
 ("`alphazero_reuse_check`", "the random walk (reported) plus the root and child reuse cases (asserted)"),
 ("`alphazero_vs`", "a head-to-head match against another player config, colours alternating"),
]),
]),
("The learning loop", """
The self-play learning loop of @sec-selfplay: the C++ self-play driver and its
command-line front end, the Python trainer that orchestrates everything, the
two search configurations, and the cluster image and job scripts.
""", [

("player/alphazero_selfplay.h", "cpp", """
The row format and the self-play entry point. `TrainingSample` is written to
disk as is (336 bytes, no padding — the `static_assert`s guard that) and read by
numpy through `SAMPLE_DTYPE` in `alphazero_learn.py`: the two must change
together.
""", [
 ("`TrainingSample`", "one training row: the bitboards, the side to move, z, the sparse $\\pi$, the search value, the network step"),
 ("`SampleFileHeader`", "24-byte file header: magic, version, row size, row and game counts"),
 ("`ModelRef` / `read_latest_model`", "parse `latest.json`, resolving a relative path against the work directory"),
 ("`SelfPlayOptions`", "the driver's knobs: concurrency, flush policy, poll period"),
 ("`run_selfplay`", "play until stopped — @sec-sp-driver"),
]),

("player/alphazero_selfplay.cpp", "cpp", """
One thread per game, all sharing one `BatchingEvaluator`; the calling thread is
the supervisor (hot swap, periodic flush, statistics). Games stopped mid-way are
dropped; finished ones always reach the disk before `run_selfplay` returns.
""", [
 ("`SampleWriter`", "collects finished games; writes ~2000-row files under a temporary name and renames them"),
 ("`make_sample`", "a `SearchResult` → a `TrainingSample` ($\\pi$ quantised to 16 bits)"),
 ("`run_selfplay` / `game_loop`", "one game after another; z filled in at the end of each game"),
]),

("test/alphazero_learn_main.cpp", "cpp", """
The `alphazero_learn` binary. `selfplay` wraps `run_selfplay` with a signal
handler; `match` plays two networks against each other with paired random
openings (@sec-eval) and writes the full record as JSON.
""", [
 ("`selfplay_main`", "parse options, install SIGINT/SIGTERM, run"),
 ("`match_main`", "A vs B, colours alternated, a random opening per pair of games"),
 ("`apply_set`", "`--set 'key=value'`: override a key of the JSON config"),
]),

("nnue/alphazero_learn.py", "python", """
The trainer, which is also the orchestrator: it starts the self-play processes
and the evaluation thread, trains, exports and checkpoints. Read `main` last:
it is the glue between the classes above it.
""", [
 ("`SAMPLE_DTYPE`", "numpy mirror of `az::TrainingSample`"),
 ("`symmetry_tables`", "the 8 plane permutations and the 8 move-index maps — @sec-symmetries"),
 ("`ReplayWindow`", "ring buffer + KataGo's window formula; reads new files"),
 ("`BatchMaker`", "rows → input planes (on the GPU), sparse $\\pi$, $z$, with a random symmetry per row"),
 ("`export_model` / `publish_latest`", "write `.pt` + `.ts`, then point `latest.json` at them"),
 ("`State`", "`state.json`: counters, exports, pending and done evaluations, Elo"),
 ("`elo_from_score`", "score → Elo difference with a half-game prior"),
 ("`SelfPlayPool`", "one self-play process per GPU, restarted if it dies"),
 ("`Evaluator`", "thread running the queued matches, appending to `evals.csv`"),
 ("`Evaluator.ensure_ts`", "re-trace a pruned `.ts` from its `.pt` when an old network is needed again"),
 ("`cleanup_models` (in `main`)", "disk budget: drop unevaluated exports, keep the `.pt` of evaluated ones"),
 ("`main`", "setup, resume, the training loop, the clean shutdown"),
]),

("config/alphazero_mcts_selfplay_player.cfg", "json", """
Self-play search settings (the `weights` key is replaced by `latest.json`).
""", []),

("config/alphazero_mcts_eval_player.cfg", "json", """
Evaluation-match settings: competitive play, no noise, no sampling. The time per
move comes from `alphazero_learn match --time`.
""", []),

("nnue/alphazero_learn.def", "bash", """
The cluster image: a toolchain only, the sources are bind-mounted (@sec-cluster).
""", []),

("nnue/alphazero_common.sh", "bash", """
Sourced by the two job scripts: the paths (`~/AlphaZeroLearn`), the container launcher,
the build step (with its lock), and the signal forwarding that makes a clean
stop possible.
""", [
 ("`run`", "run a command in the container, repository at `/Yolah`, run at `/work`"),
 ("`build_alphazero_learn`", "compile for this node's CPU model, serialised by `flock`"),
 ("`start_in_container`", "start in the background with `exec`, so that `$!` is Singularity itself"),
 ("`wait_forwarding_signals`", "wait, forwarding SIGUSR1/SIGTERM as SIGTERM for a clean stop"),
 ("`max_hours_for_job`", "the job's remaining time minus a margin, from `squeue`"),
]),

("nnue/alphazero_learn.sh", "bash", """
The main SLURM job: 3 GPUs, 32 CPUs, 128 GB, 14 days (sizing:
@sec-learn-sizing). It compiles for the node, runs the trainer, and stops
cleanly on the time-limit signal.
""", []),

("nnue/alphazero_selfplay.sh", "bash", """
An extra self-play GPU on any node, for a run split over several jobs
(@fig-split).
""", []),
]),
]


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read().rstrip("\n")


def main():
    out = []
    w = out.append
    w("# Annotated source {#sec-source}\n")
    w("""
Every file the document describes, in full, generated from the working tree by
`doc/make_appendix.py`. Each one carries an orientation note, a table of the
symbols it defines, and its complete source with line numbers — so the sections
above can point at `alphazero_mcts.cpp:212` and mean it.

The running commentary lives in the sources themselves rather than being
duplicated here; what the tables add is the one-line answer to "what is this
for" for every symbol, which a comment sitting next to a definition cannot give
you at a glance.
""".strip() + "\n")

    total = 0
    for part_title, part_intro, files in PARTS:
        w(f"\n## {part_title}\n")
        w(part_intro.strip() + "\n")
        for path, lang, orientation, symbols in files:
            src = read(path)
            n = src.count("\n") + 1
            total += n
            anchor = "sec-src-" + path.replace("/", "-").replace(".", "-").replace("_", "-")
            w(f"\n### `{path}` {{#{anchor}}}\n")
            w(f"*{n} lines.*\n")
            w(orientation.strip() + "\n")
            if symbols:
                w("\n| symbol | role |")
                w("|---|---|")
                for sym, role in symbols:
                    w(f"| {sym} | {role} |")
                w("")
            w(f"\n``` {{.{lang} .numberLines}}")
            w(src)
            w("```\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    nb_files = sum(len(files) for _, _, files in PARTS)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {nb_files} files, {total:,} lines of source")

    # A listing that no longer matches the tree is worse than no listing, so
    # shout if the working tree has uncommitted changes to a listed file.
    try:
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--"] +
                               [p for _, _, fs in PARTS for p, _, _, _ in fs],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        if dirty:
            print("note: listed sources with uncommitted changes:")
            print(dirty)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
