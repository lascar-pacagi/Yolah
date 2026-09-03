#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H
// alphazero_mcts.h — AlphaZero-style Monte-Carlo tree search for Yolah.
//
// Inspired by MCTS_mem_player.h (same pool-allocated tree, same lock-free
// multi-threading style) but the leaf is evaluated by the two-headed ResNet
// instead of a random playout, and children are selected with PUCT:
//
//     a* = argmax_a  Q(s,a) + c_puct(s) · P(s,a) · √N(s) / (1 + N(s,a))
//     c_puct(s) = c_init + log((N(s) + c_base + 1) / c_base)          [AlphaZero]
//
// Search loop (per thread, repeated until the time / simulation budget is spent):
//   1. GATHER up to `batch_size` leaves. Each descent applies a *virtual loss*
//      (a pending visit that counts as a loss) to every node on its path so
//      that the following descents — of this thread and of the others — are
//      steered towards different leaves. A descent that reaches a leaf some
//      other descent is already expanding is a "collision"; it is undone and,
//      after `max_collisions` of them, the batch is closed early.
//      Terminal leaves are scored exactly and backed up at once.
//   2. EVALUATE the batch with one call to nn::Evaluator (a GPU wants a few
//      hundred positions per call; the CPU backend a few dozen).
//   3. EXPAND every leaf with the network's priors (softmax over the legal
//      moves only, optional temperature) and BACK UP its value along the path,
//      flipping the sign at each ply (negamax convention), converting the
//      pending visit into a real one.
//
// Values are stored per node as W = Σ v from the point of view of the player
// who moved INTO the node, i.e. Q(s,a) = W(child)/N(child) is exactly the
// quantity PUCT needs at the parent. The network's value is for the player to
// move at the evaluated position, so the leaf itself receives -v.
//
// Transpositions: the same position is reached through many move orders (the
// four stones of a side move independently). Before asking the network for a
// leaf, the search looks the position up in an NNCache keyed by its Zobrist
// hash (player/nn_cache.h); a hit expands the node with the cached value and
// priors at zero cost. Statistics are not merged between transposed nodes —
// the tree stays a tree, which keeps the backups exact — but the expensive
// part, the network call, is shared.
//
// Collisions and batching: when two descents of the same batch pick the same
// unexpanded leaf, the second one is recorded as an extra visit of that leaf
// (multiplicity) rather than discarded, so the batch still fills up and the
// evaluation is backed up once per visit — this is what a sequential search
// would have done, up to the child the second visit would have expanded.
// Collisions with leaves owned by *another* thread are undone; after
// `max_collisions` of those the batch is closed early.
//
// Tree reuse: search(state) first looks for `state` among the root, its
// children and its grandchildren (same position ⇒ same subtree) and promotes
// the matching node to root, keeping every statistic gathered so far. This
// covers the normal case (we played, the opponent replied → grandchild), a
// self-play driver that calls the same Search for both sides (→ child) and a
// repeated call on the same position (→ root). Anything else resets the tree.
//
// Future learning (AlphaZero loop). Everything the self-play driver needs is
// exposed on SearchResult: the visit distribution π over the root's moves
// (policy target), the root value estimate, and move sampling with a
// temperature; Dirichlet noise on the root priors is a parameter. A new
// network is exported with nnue/resnet_export.py and swapped in through
// nn::Evaluator::reload(). See player/alphazero_mcts_player.h for the JSON
// keys.
#include "game.h"
#include "nn_evaluator.h"
#include "nn_cache.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

namespace az {

struct SearchParams {
    // ── budget: the search stops when either bound is hit (0 = no bound) ──
    uint64_t microseconds     = 1'000'000;   // wall-clock budget
    uint64_t nb_simulations   = 0;           // e.g. 800 for AlphaZero-style self-play
    // ── parallelism ──
    size_t   nb_threads       = 1;           // search threads sharing the tree
    size_t   batch_size       = 32;          // leaves gathered per thread per network call
    size_t   max_collisions   = 0;           // cross-thread collisions closing a batch (0 = batch_size)
    size_t   nn_cache_mb      = 64;          // transposition cache of network outputs (0 = off)
    // ── PUCT ──
    float    c_puct_init      = 1.25f;
    float    c_puct_base      = 19652.0f;
    // First play urgency: an unvisited child is assumed to be worth
    // Q(parent) - fpu_reduction · √(prior mass of the visited siblings)
    // instead of 0, so that the search does not spread over every child of a
    // clearly bad position (Leela Chess Zero style).
    float    fpu_reduction    = 0.3f;
    // Softmax temperature applied to the policy logits at expansion (>1
    // flattens the priors; LC0 uses ~1.36 for its nets).
    float    policy_temperature = 1.0f;
    // ── self-play exploration ──
    float    dirichlet_alpha    = 0.3f;      // Dir(α) noise mixed into the root priors
    float    dirichlet_epsilon  = 0.0f;      // 0 = no noise (competitive play)
    float    temperature        = 0.0f;      // move sampling ∝ N^(1/τ); 0 = argmax N
    int      temperature_cutoff = 0;         // sample only while ply < cutoff (0 = never sample)
    // ── misc ──
    bool     reuse_tree       = true;
    uint64_t seed             = 0;           // 0 = random_device
};

struct ChildStat {
    Move  move;
    float prior;        // network prior P(s,a) (after noise if any)
    uint32_t visits;    // N(s,a)
    float q;            // Q(s,a) for the side to move at the root, ∈ [-1, 1]
};

struct SearchResult {
    Move     best_move  = Move::none();
    float    root_value = 0.0f;         // Q of the root for the side to move (from visits)
    float    net_value  = 0.0f;         // raw network value of the root position
    uint64_t nb_simulations = 0;        // simulations run in this call (reused ones excluded)
    uint32_t root_visits = 0;           // total visits of the root (including reused)
    uint64_t nb_collisions = 0;         // cross-thread collisions (undone descents)
    uint64_t nb_cache_hits = 0;         // leaves expanded from the transposition cache
    uint64_t nb_evaluations = 0;        // positions sent to the network
    double   seconds = 0.0;
    size_t   tree_nodes = 0;            // nodes currently allocated in the tree
    std::vector<ChildStat> children;    // sorted by visits, descending
    // Training target π(a) = N(s,a)^(1/τ) / Σ — visit distribution over the
    // root's legal moves (same order as `children`).
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
    // Forget the tree (call at the start of a new game). The transposition
    // cache is kept: it only depends on the network.
    void reset();
    // Drop the cached network outputs (after loading new weights).
    void clear_cache();

    SearchParams&       params()       { return prm; }
    const SearchParams& params() const { return prm; }
    size_t nb_nodes() const { return node_count.load(std::memory_order_relaxed); }

private:
    struct Node;
    struct Worker;   // per-thread state (buffers, PRNG)

    enum : uint8_t { UNEXPANDED = 0, EXPANDING = 1, EXPANDED = 2, TERMINAL = 3 };

    // tree management
    bool  reuse(const Yolah& state);
    void  new_root(const Yolah& state);
    void  promote(Node& target);
    void  expand_root(const Yolah& state);
    void  apply_root_noise(Worker& w);
    // search
    void  run_worker(Worker& w, const Yolah& root_state);
    uint32_t select_child(const Node& node, bool is_root) const;
    void  priors_from_logits(const nn::Result& result, size_t nb_moves, float* priors) const;
    void  expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value);
    void  backup(Node* const* path, size_t length, float leaf_value, uint32_t multiplicity = 1);
    void  undo_pending(Node* const* path, size_t length);
    static float terminal_value(const Yolah& state);
    // result
    SearchResult make_result(const Yolah& state, double seconds, uint64_t sims,
                             uint64_t collisions, Worker& w) const;

    nn::Evaluator& evaluator;
    SearchParams prm;
    std::pmr::polymorphic_allocator<> alloc;
    std::unique_ptr<Node> root;
    Yolah root_state;                      // position at the root (valid when root != nullptr)
    std::vector<float> root_priors;        // root priors after Dirichlet noise (empty = none)
    float root_net_value = 0.0f;
    std::atomic<uint64_t> node_count{0};
    std::atomic<uint64_t> simulations{0};  // simulations completed in the current search
    std::atomic<uint64_t> reserved{0};     // simulations claimed by the workers (budget accounting)
    std::atomic<uint64_t> collisions{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> evaluations{0};
    std::unique_ptr<NNCache> cache;
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point deadline;
    uint64_t base_seed;
    uint64_t search_counter = 0;
};

} // namespace az

#endif
