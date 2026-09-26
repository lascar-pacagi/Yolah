#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H
// alphazero_mcts.h — AlphaZero-style Monte-Carlo tree search for Yolah, with
// the refinements KataGo added on top of it.
//
// Inspired by MCTS_mem_player.h (same pool-allocated storage, same lock-free
// multi-threading style) but the leaf is evaluated by the two-headed ResNet
// instead of a random playout, and children are selected with PUCT, in the
// KataGo formulation (cpp/search/searchexplorehelpers.cpp):
//
//     a* = argmax_a  Q(s,a) + c(s) · P(s,a) · √(W(s) + 0.01) / (1 + W(s,a))
//     c(s) = [cpuct + cpuct_log · log((W(s) + cpuct_base) / cpuct_base)] · σ̂(s)
//     σ̂(s) = 1 + stdev_scale · (stdev(s) / stdev_prior − 1)
//
// W(s) = Σ_a W(s,a) is the total weight of the *edges* leaving s (one per
// visit, plus one per in-flight virtual loss) and stdev(s) is the running
// standard deviation of the values backed up through s, blended with a prior.
// The σ̂ factor explores more where the children disagree and less where the
// node's value has settled. AlphaZero is the special case
// cpuct_log = stdev_scale = 0.
//
// GRAPH SEARCH ─────────────────────────────────────────────────────────────
// The search space is a DAG, not a tree. Every Yolah move turns the square it
// leaves into a hole (`empty ^= from`), so the number of holes increases by
// exactly one per move and a position can never repeat: no cycles, ever. At
// the same time the four stones of a side move independently, so move orders
// commute and the same position is reached along very many paths.
//
// The search therefore stores ONE node per distinct position, in a shared
// hash table keyed by the Zobrist hash (plus the score split, which the hash
// does not cover — see node_key in the .cpp). A node reached through k
// different parents accumulates the statistics of all k of them: a line
// refuted in one branch is refuted in every branch that transposes into it.
//
//   • Node  — one per position: visit count, ΣV, ΣV², the network's value,
//             and the edges to its successors.
//   • Edge  — one per (position, move): prior, pointer to the successor node
//             (created on first descent), and this parent's own visit count.
//
// PUCT uses the EDGE visits in the denominator (this parent's share of the
// exploration) and the CHILD NODE's average for Q (everything the search
// knows about that position, whoever explored it). That is exactly KataGo's
// childWeight = rawChildWeight · edgeVisits / childVisits, which for a search
// weighting every visit by 1 collapses to "edge visits".
//
// Search loop (per thread, repeated until the time / simulation budget is spent):
//   1. GATHER up to `batch_size` leaves. Each descent applies a *virtual loss*
//      (a pending visit that counts as a loss) to every node and edge on its
//      path so that the following descents — of this thread and of the others
//      — are steered towards different leaves. A descent that reaches a leaf
//      some other thread is already expanding is a "collision"; it is undone
//      and, after `max_collisions` of them, the batch is closed early.
//      Terminal leaves are scored exactly and backed up at once.
//   2. EVALUATE the batch with one call to nn::Evaluator (a GPU wants a few
//      hundred positions per call; the CPU backend a few dozen).
//   3. EXPAND every leaf with the network's priors (softmax over the legal
//      moves only, optional temperature) and BACK UP its value along every
//      path that reached it, flipping the sign at each ply (negamax
//      convention), converting the pending visits into real ones.
//
// Values are stored per node as W = Σ v from the point of view of the player
// who moved INTO the node, i.e. Q(s,a) = W(child)/N(child) is exactly the
// quantity PUCT needs at the parent. The network's value is for the player to
// move at the evaluated position, so the leaf itself receives -v.
//
// Network cache: before asking the network for a leaf, the search looks the
// position up in an NNCache keyed by its Zobrist hash (player/nn_cache.h). It
// is a second line of defence behind the node table — it also catches
// positions that have dropped out of the graph, and it survives across moves
// and games.
//
// Tree reuse: search(state) looks `state` up in the node table and, if it is
// there, re-roots on it — keeping every statistic gathered so far, however
// deep in the old graph the position was. The nodes that are no longer
// reachable from the new root are then collected (mark and sweep).
//
// ROOT REFINEMENTS ─────────────────────────────────────────────────────────
// Three things happen at the root that do not happen anywhere else:
//   • FORCED PLAYOUTS (KataGo paper §5.1, self-play only): a child that has
//     been visited at least once is forced up to √(k·P(s,a)·W(s)) visits, so
//     that a move the noise made interesting is actually tried a few times
//     instead of being dropped after one bad evaluation.
//   • POLICY TARGET PRUNING (same section): those forced visits are taken
//     back out afterwards — each non-best child is reduced to the visit count
//     that PUCT would retrospectively have given it — so the training target
//     is not polluted by playouts the search was forced to make.
//   • LCB MOVE SELECTION: the move played is the one with the best lower
//     confidence bound Q − λ·σ(Q), not simply the most visited, among the
//     children with enough visits for the bound to mean anything.
//
// Self-play: everything the driver needs is on SearchResult — the play-value
// distribution π over the root's moves (policy target), the root value, and
// move sampling with a temperature. Dirichlet noise on the root priors and
// PLAYOUT CAP RANDOMIZATION (a large budget with noise and a recorded policy
// target on a fraction of the moves, a small one without on the rest) are
// parameters. A new network is exported with nnue/resnet_export.py and
// swapped in through nn::Evaluator::reload(). See
// player/alphazero_mcts_player.h for the JSON keys.
#include "game.h"
#include "misc.h"
#include "nn_evaluator.h"
#include "nn_cache.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory_resource>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace az {

struct SearchParams {
    // ── budget: the search stops when either bound is hit (0 = no bound) ──
    uint64_t microseconds     = 1'000'000;   // wall-clock budget
    uint64_t nb_simulations   = 0;           // e.g. 800 for AlphaZero-style self-play
    // ── parallelism ──
    size_t   nb_threads       = 1;           // search threads sharing the graph
    size_t   batch_size       = 32;          // leaves gathered per thread per network call
    size_t   max_collisions   = 0;           // cross-thread collisions closing a batch (0 = batch_size)
    size_t   nn_cache_mb      = 64;          // transposition cache of network outputs (0 = off)

    // ── PUCT (KataGo defaults, cpp/search/searchparams.cpp) ──
    float    cpuct_exploration      = 1.0f;    // c
    float    cpuct_exploration_log  = 0.45f;   // c_log (0 = AlphaZero: no log term)
    float    cpuct_exploration_base = 500.0f;  // c_base
    // Exploration is scaled by σ̂ = 1 + scale · (stdev / prior − 1), where stdev
    // is the standard deviation of the values backed up through the node,
    // regularised towards `prior` with weight `prior_weight` visits. Scale 0
    // disables it (σ̂ = 1) and gives the plain AlphaZero/LC0 behaviour.
    float    cpuct_utility_stdev_scale        = 0.85f;
    float    cpuct_utility_stdev_prior        = 0.40f;
    float    cpuct_utility_stdev_prior_weight = 2.0f;

    // ── First play urgency ──
    // An unvisited child is worth
    //     baseline − fpu_reduction · √(prior mass of the visited siblings)
    // instead of 0, so that the search does not spread over every child of a
    // clearly bad position. `baseline` is the value of the parent for the
    // player to move there: its running average Q blended with its raw network
    // value, the average weighing (visited prior mass)^pow — an unexplored
    // node trusts the network, a well explored one its own statistics.
    // fpu_loss_prop finally drags the estimate a proportion of the way towards
    // a loss (−1). The root uses its own, gentler, constants.
    float    fpu_reduction      = 0.2f;
    float    root_fpu_reduction = 0.1f;
    float    fpu_loss_prop      = 0.0f;
    float    root_fpu_loss_prop = 0.0f;
    bool     fpu_parent_weight_by_visited_policy     = true;
    float    fpu_parent_weight_by_visited_policy_pow = 1.0f;
    // Used only when the above is false: a constant weight for the network
    // value in the blend (0 = use the running average alone).
    float    fpu_parent_weight  = 0.0f;

    // Softmax temperature applied to the policy logits at expansion (>1
    // flattens the priors; LC0 uses ~1.36 for its nets).
    float    policy_temperature = 1.0f;

    // ── graph search ──
    // Share one node between every path that reaches the same position. Off
    // reverts to a plain tree (each path gets its own node), which is only
    // useful for measuring what the sharing is worth.
    bool     graph_search     = true;

    // ── root: move choice ──
    // Play the move with the best lower confidence bound Q − lcb_stdevs·σ(Q)
    // among the children holding at least min_visit_prop_for_lcb of the most
    // visited child's visits, instead of the most visited move.
    bool     use_lcb                = true;
    float    lcb_stdevs             = 5.0f;
    float    min_visit_prop_for_lcb = 0.20f;
    // Take the forced playouts back out of the root's play values, so that the
    // policy target reflects what PUCT would have wanted (KataGo always does
    // this at the root; it is what makes forced playouts safe to train on).
    bool     policy_target_pruning  = true;

    // ── self-play exploration ──
    float    dirichlet_alpha    = 0.3f;      // Dir(α) noise mixed into the root priors
    float    dirichlet_epsilon  = 0.0f;      // 0 = no noise (competitive play)
    float    temperature        = 0.0f;      // move sampling ∝ π^(1/τ); 0 = argmax
    int      temperature_cutoff = 0;         // sample only while ply < cutoff (0 = never sample)
    // Force every root child that has been visited at least once up to
    // √(k·P·W) visits. 0 = off; the paper uses k = 2 during self-play only.
    float    forced_playouts_k  = 0.0f;
    // Playout cap randomization: with probability `fast_prob` the search runs
    // `nb_simulations_fast` simulations with no root noise, no forced playouts
    // and SearchResult::full_search = false — the driver must not record a
    // policy target for those moves. Needs nb_simulations > 0. 0 = off.
    float    playout_cap_fast_prob = 0.0f;
    uint64_t nb_simulations_fast   = 0;

    // ── misc ──
    bool     reuse_tree       = true;
    uint64_t seed             = 0;           // 0 = random_device
};

struct ChildStat {
    Move  move;
    float prior;        // network prior P(s,a) (after noise if any)
    uint32_t visits;    // edge visits N(s,a), before any pruning
    float q;            // Q(s,a) for the side to move at the root, ∈ [-1, 1]
    float q2;           // mean of Q² over the child's visits (for the LCB variance)
    float play_value;   // visits after policy target pruning / LCB adjustment
    float lcb;          // Q(s,a) − lcb_stdevs · σ(Q), or -2 if not computed
};

struct SearchResult {
    Move     best_move  = Move::none();
    float    root_value = 0.0f;         // Q of the root for the side to move (from visits)
    float    net_value  = 0.0f;         // raw network value of the root position
    uint64_t nb_simulations = 0;        // simulations run in this call (reused ones excluded)
    uint32_t root_visits = 0;           // total visits of the root (including reused)
    uint64_t nb_collisions = 0;         // cross-thread collisions (undone descents)
    uint64_t nb_cache_hits = 0;         // leaves expanded from the network cache
    uint64_t nb_evaluations = 0;        // positions sent to the network
    uint64_t nb_transpositions = 0;     // edges that linked to an existing node
    bool     full_search = true;        // false = a fast playout-cap search: no policy target
    double   seconds = 0.0;
    size_t   tree_nodes = 0;            // distinct positions currently in the graph
    std::vector<ChildStat> children;    // sorted by play value, descending
    // Training target π(a): the root's play values, normalised. With
    // policy_target_pruning these are the visit counts with the forced
    // playouts taken back out.
    std::vector<float> policy;
    std::string to_string(size_t max_children = 8) const;
};

class Search {
public:
    Search(nn::Evaluator& evaluator, const SearchParams& params, std::pmr::memory_resource* memory);
    ~Search();
    Search(const Search&) = delete;
    Search& operator=(const Search&) = delete;

    // Run a search from `state` (must not be game over) and pick a move.
    SearchResult search(const Yolah& state);
    // Forget the graph (call at the start of a new game). The network cache is
    // kept: it only depends on the weights.
    void reset();
    // Drop the cached network outputs (after loading new weights).
    void clear_cache();

    SearchParams&       params()       { return prm; }
    const SearchParams& params() const { return prm; }
    size_t nb_nodes() const { return node_count.load(std::memory_order_relaxed); }

private:
    struct Node;
    struct Edge;
    struct Worker;   // per-thread state (buffers, PRNG)
    struct PathStep; // one ply of a descent: the node reached and the edge taken

    enum : uint8_t { UNEXPANDED = 0, EXPANDING = 1, EXPANDED = 2, TERMINAL = 3 };

    // ── node table (one entry per distinct position) ──
    static constexpr size_t NB_SHARDS = 64;
    struct Shard {
        std::mutex mutex;
        std::unordered_map<uint64_t, Node*> nodes;
    };
    Node* get_or_create(uint64_t key, Worker* w);
    Node* lookup(uint64_t key);
    Node* new_node();
    void  destroy_all();
    void  collect(Node* keep);        // mark and sweep from the new root
    void  mark(Node* node);

    // graph management
    bool  reuse(const Yolah& state, uint64_t key);
    void  new_root(const Yolah& state, uint64_t key);
    void  expand_root(const Yolah& state);
    void  apply_root_noise(Worker& w);
    // search
    void  run_worker(Worker& w, const Yolah& root_state);
    uint32_t select_child(const Node& node, bool is_root) const;
    float utility_stdev_factor(const Node& node, uint32_t visits) const;
    float fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const;
    void  priors_from_logits(const nn::Result& result, size_t nb_moves, float* priors) const;
    void  expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value);
    void  backup(const PathStep* path, size_t length, float leaf_value);
    void  undo_pending(const PathStep* path, size_t length);
    static float terminal_value(const Yolah& state);
    // result
    SearchResult make_result(const Yolah& state, double seconds, uint64_t sims,
                             uint64_t collisions, Worker& w) const;
    void root_play_values(std::vector<ChildStat>& children) const;

    nn::Evaluator& evaluator;
    SearchParams prm;
    std::pmr::polymorphic_allocator<> alloc;
    Shard shards[NB_SHARDS];
    Node* root = nullptr;
    Yolah root_state;                      // position at the root (valid when root != nullptr)
    uint64_t root_key = 0;                 // its node-table key
    std::vector<float> root_priors;        // root priors after Dirichlet noise (empty = none)
    uint32_t mark_counter = 0;
    std::atomic<uint64_t> node_count{0};
    std::atomic<uint64_t> simulations{0};  // simulations completed in the current search
    std::atomic<uint64_t> reserved{0};     // simulations claimed by the workers (budget accounting)
    std::atomic<uint64_t> collisions{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> evaluations{0};
    std::atomic<uint64_t> transpositions{0};
    std::unique_ptr<NNCache> cache;
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point deadline;
    uint64_t sim_budget = 0;               // simulations for the current search (playout cap)
    bool     full_search = true;           // false = fast search: no noise, no forced playouts
    PRNG     rng;                          // search-level PRNG (playout cap randomization)
    uint64_t base_seed;
    uint64_t search_counter = 0;
};

} // namespace az

#endif
