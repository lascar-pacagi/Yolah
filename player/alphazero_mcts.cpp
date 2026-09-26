// alphazero_mcts.cpp — see alphazero_mcts.h for the algorithm overview.
#include "alphazero_mcts.h"
#include "misc.h"
#include "zobrist.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <thread>

namespace az {

// ── Graph node and edge ────────────────────────────────────────────────────
// One Node per distinct position, one Edge per (position, move). All
// statistics are atomics updated with relaxed ordering: they are heuristics, a
// slightly stale read only changes which leaf a descent picks, never the
// graph's integrity. The only synchronisation that matters is `state`
// (acquire/release) — a thread that observes EXPANDED is guaranteed to see the
// fully built edge vector — and Edge::child, published the same way.
struct Search::Edge {
    Move  action = Move::none();        // move leading to the successor
    float prior  = 0.0f;                // P(s,a) from the policy head
    std::atomic<Search::Node*> child{nullptr};   // successor, created on first descent
    std::atomic<uint32_t> n{0};         // edge visits: this parent's share of the child
    std::atomic<uint32_t> pending{0};   // in-flight visits through this edge

    Edge() = default;
    Edge(Move m, float p) : action(m), prior(p) {}
    // Atomics are not movable; the vector needs a move constructor anyway.
    Edge(const Edge& o)
        : action(o.action), prior(o.prior), child(o.child.load()),
          n(o.n.load()), pending(o.pending.load()) {}
    Edge& operator=(const Edge& o) {
        action = o.action; prior = o.prior;
        child = o.child.load(); n = o.n.load(); pending = o.pending.load();
        return *this;
    }
};

struct Search::Node {
    std::atomic<uint32_t> n{0};         // completed visits N(s), from every parent
    std::atomic<uint32_t> pending{0};   // in-flight visits (virtual loss), each counted as a loss
    std::atomic<float>    w{0.0f};      // Σ backed-up values for the player who moved INTO this node
    std::atomic<float>    w2{0.0f};     // Σ of their squares (sign-free: stdev of the subgraph's values)
    std::atomic<uint8_t>  state{UNEXPANDED};
    uint32_t mark = 0;                  // garbage collection: last mark_counter that reached it
    float    net_v = 0.0f;              // network value at this node (player to move here)
    std::pmr::vector<Edge> children;

    using allocator_type = std::pmr::polymorphic_allocator<>;
    explicit Node(allocator_type a = {}) : children(a) {}
};

// One ply of a descent: the node reached and the edge that was taken to reach
// it (null for the root, which no edge leads to).
struct Search::PathStep {
    Node* node;
    Edge* edge;
};

// ── Per-thread state ───────────────────────────────────────────────────────
struct Search::Worker {
    static constexpr size_t MAX_PATH = Yolah::MAX_NB_PLIES + 8;   // root → deepest leaf
    struct Leaf {
        Node*    node;
        uint64_t hash;           // Zobrist key, for the network cache
        Yolah    state;
        Yolah::MoveList moves;
    };
    // A descent that reached `leaf`: the value evaluated there is backed up
    // along this path. Two descents of the same batch can reach the same leaf
    // through different paths — under graph search even through different
    // parents — so every descent keeps its own path.
    struct Descent {
        uint32_t leaf;
        uint32_t path_len;
    };
    PRNG prng;
    std::vector<Leaf> leaves;            // distinct leaves to evaluate
    std::vector<Descent> descents;       // one per gathered visit
    std::vector<PathStep> paths;         // (batch_size + 1) × MAX_PATH steps
    std::vector<nn::Request> requests;
    std::vector<nn::Result> results;

    Worker(uint64_t seed, size_t batch_size)
        : prng(seed ? seed : 0x9E3779B97F4A7C15ull),
          paths((batch_size + 1) * MAX_PATH, PathStep{nullptr, nullptr}),
          results(batch_size) {
        leaves.reserve(batch_size);
        descents.reserve(batch_size);
        requests.reserve(batch_size);
    }
    PathStep* path(size_t k) { return paths.data() + k * MAX_PATH; }
};

// ── helpers ────────────────────────────────────────────────────────────────
namespace {
    bool same_position(const Yolah& a, const Yolah& b) {
        return a.nb_plies() == b.nb_plies() &&
               a.bitboard(Yolah::BLACK) == b.bitboard(Yolah::BLACK) &&
               a.bitboard(Yolah::WHITE) == b.bitboard(Yolah::WHITE) &&
               a.empty_bitboard() == b.empty_bitboard() &&
               a.score() == b.score();
    }

    uint64_t mix(uint64_t x) {
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27; x *= 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    // Key of a position in the node table. zobrist::hash covers the board and
    // the side to move — exactly what the network sees, which is why the
    // network cache can key on it alone. It does NOT cover how the points
    // scored so far split between the two players: two positions with the same
    // board can differ there (a pass scores nothing), and the terminal value
    // depends on the split. Nodes carry statistics, not just network outputs,
    // so their key has to separate them.
    uint64_t node_key(uint64_t zobrist_hash, const Yolah& state) {
        return zobrist_hash ^ mix(uint64_t(state.score().first) + 1);
    }
}

float Search::terminal_value(const Yolah& state) {
    const auto [black, white] = state.score();
    const int diff = (state.current_player() == Yolah::BLACK ? 1 : -1) * (int(black) - int(white));
    return diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
}

// ── construction / node table ──────────────────────────────────────────────
Search::Search(nn::Evaluator& evaluator, const SearchParams& params, std::pmr::memory_resource* memory)
    : evaluator(evaluator), prm(params), alloc(memory),
      rng(params.seed ? params.seed : 0x2545F4914F6CDD1Dull),
      base_seed(params.seed ? params.seed : std::random_device{}()) {
    if (prm.microseconds == 0 && prm.nb_simulations == 0)
        throw std::invalid_argument("az::Search: either microseconds or nb_simulations must be > 0");
    prm.nb_threads = std::max<size_t>(1, prm.nb_threads);
    prm.batch_size = std::max<size_t>(1, prm.batch_size);
    cache = std::make_unique<NNCache>(prm.nn_cache_mb << 20);
}

Search::~Search() {
    destroy_all();
}

void Search::clear_cache() {
    cache->clear();
}

Search::Node* Search::new_node() {
    node_count.fetch_add(1, std::memory_order_relaxed);
    return alloc.new_object<Node>();
}

Search::Node* Search::lookup(uint64_t key) {
    Shard& sh = shards[key % NB_SHARDS];
    std::lock_guard lock(sh.mutex);
    auto it = sh.nodes.find(key);
    return it == sh.nodes.end() ? nullptr : it->second;
}

void Search::destroy_all() {
    for (Shard& sh : shards) {
        for (auto& [key, node] : sh.nodes) alloc.delete_object(node);
        sh.nodes.clear();
    }
    root = nullptr;
    node_count = 0;
}

void Search::reset() {
    destroy_all();
    root_priors.clear();
}

// Depth-first marking of everything still reachable from the new root. The
// position graph is acyclic (every move fills a square, so the number of holes
// strictly increases), hence a plain recursion with a "already marked" test
// terminates and visits each node once.
void Search::mark(Node* node) {
    if (!node || node->mark == mark_counter) return;
    node->mark = mark_counter;
    if (node->state.load(std::memory_order_relaxed) != EXPANDED) return;
    for (Edge& e : node->children) mark(e.child.load(std::memory_order_relaxed));
}

void Search::collect(Node* keep) {
    ++mark_counter;
    mark(keep);
    uint64_t freed = 0;
    for (Shard& sh : shards) {
        for (auto it = sh.nodes.begin(); it != sh.nodes.end();) {
            if (it->second->mark != mark_counter) {
                alloc.delete_object(it->second);
                it = sh.nodes.erase(it);
                ++freed;
            } else {
                ++it;
            }
        }
    }
    node_count.fetch_sub(freed, std::memory_order_relaxed);
}

// Find (or create) the node for `key`. Under graph search the key identifies
// the position, so every path reaching it gets the same node; in tree mode the
// caller passes a fresh unique key and every edge gets its own node.
Search::Node* Search::get_or_create(uint64_t key, Worker* w) {
    Shard& sh = shards[key % NB_SHARDS];
    std::lock_guard lock(sh.mutex);
    auto [it, inserted] = sh.nodes.try_emplace(key, nullptr);
    if (!inserted) {
        transpositions.fetch_add(1, std::memory_order_relaxed);
        return it->second;
    }
    (void)w;
    it->second = new_node();
    return it->second;
}

void Search::new_root(const Yolah& state, uint64_t key) {
    destroy_all();
    root = get_or_create(key, nullptr);
    root_state = state;
    root_key = key;
}

bool Search::reuse(const Yolah& state, uint64_t key) {
    if (!root) return false;
    if (prm.graph_search) {
        // The node table is the whole tree-reuse mechanism: whatever depth the
        // position sat at in the old graph, its statistics are right there.
        Node* node = lookup(key);
        if (!node) return false;
        root = node;
        root_state = state;
        root_key = key;
        collect(root);                 // drop everything the new root cannot reach
        return true;
    }
    // Tree mode: the position can only be the root, one of its children or one
    // of its grandchildren.
    if (same_position(state, root_state)) return true;
    if (root->state.load(std::memory_order_acquire) != EXPANDED) return false;
    for (Edge& e : root->children) {
        Node* child = e.child.load(std::memory_order_acquire);
        if (!child) continue;
        Yolah s1 = root_state;
        s1.play(e.action);
        if (same_position(s1, state)) {
            root = child; root_state = state; root_key = key; collect(root);
            return true;
        }
        if (child->state.load(std::memory_order_acquire) != EXPANDED) continue;
        for (Edge& g : child->children) {
            Node* grandchild = g.child.load(std::memory_order_acquire);
            if (!grandchild) continue;
            Yolah s2 = s1;
            s2.play(g.action);
            if (same_position(s2, state)) {
                root = grandchild; root_state = state; root_key = key; collect(root);
                return true;
            }
        }
    }
    return false;
}

void Search::expand_root(const Yolah& state) {
    if (state.game_over()) {
        root->state.store(TERMINAL, std::memory_order_release);
        return;
    }
    Yolah::MoveList moves;
    state.moves(moves);
    nn::Request request{&state, moves.begin(), static_cast<uint16_t>(moves.size())};
    nn::Result result;
    evaluator.evaluate(std::span<const nn::Request>(&request, 1), std::span<nn::Result>(&result, 1));
    root->state.store(EXPANDING, std::memory_order_relaxed);
    float priors[Yolah::MAX_NB_MOVES];
    priors_from_logits(result, moves.size(), priors);
    expand(*root, moves, priors, result.value);
}

void Search::apply_root_noise(Worker& w) {
    root_priors.clear();
    if (!full_search || prm.dirichlet_epsilon <= 0.0f || root->children.empty()) return;
    std::mt19937_64 gen(w.prng.rand<uint64_t>());
    std::gamma_distribution<float> gamma(prm.dirichlet_alpha, 1.0f);
    const size_t nb = root->children.size();
    root_priors.resize(nb);
    float sum = 0.0f;
    for (size_t i = 0; i < nb; i++) { root_priors[i] = gamma(gen); sum += root_priors[i]; }
    for (size_t i = 0; i < nb; i++) {
        const float noise = sum > 0.0f ? root_priors[i] / sum : 1.0f / nb;
        root_priors[i] = (1.0f - prm.dirichlet_epsilon) * root->children[i].prior + prm.dirichlet_epsilon * noise;
    }
}

// ── search primitives ──────────────────────────────────────────────────────
// KataGo's σ̂ factor: the exploration constant is scaled by how much the values
// backed up through this node disagree, measured against a prior stdev.
float Search::utility_stdev_factor(const Node& node, uint32_t visits) const {
    if (prm.cpuct_utility_stdev_scale <= 0.0f) return 1.0f;
    const float prior = prm.cpuct_utility_stdev_prior;
    float stdev = prior;
    if (visits > 1) {
        const float n = static_cast<float>(visits);
        // The sign of the mean is irrelevant here, only its square is used.
        const float mean    = node.w.load(std::memory_order_relaxed) / n;
        const float mean_sq = mean * mean;
        // Relaxed reads of w and w2 can be slightly out of step; clamp so that
        // the variance never comes out negative.
        const float sq_avg  = std::max(node.w2.load(std::memory_order_relaxed) / n, mean_sq);
        const float pw = prm.cpuct_utility_stdev_prior_weight;
        stdev = std::sqrt(std::max(0.0f,
            ((mean_sq + prior * prior) * pw + sq_avg * n) / (pw + n - 1.0f) - mean_sq));
    }
    return 1.0f + prm.cpuct_utility_stdev_scale * (stdev / prior - 1.0f);
}

// Value assumed for a child that has not been visited yet (first play urgency).
float Search::fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const {
    // Everything here is from the point of view of the player to move at
    // `node`: node.w is for the player who moved into it, hence the sign flip,
    // and node.net_v is the network's value for the player to move.
    const float parent_q = visits > 0 ? -node.w.load(std::memory_order_relaxed) / visits : node.net_v;
    float baseline = parent_q;
    if (prm.fpu_parent_weight_by_visited_policy) {
        const float p = prm.fpu_parent_weight_by_visited_policy_pow;
        const float mass = p == 1.0f ? visited_policy_mass : std::pow(visited_policy_mass, p);
        const float avg_weight = std::min(1.0f, mass);
        baseline = avg_weight * parent_q + (1.0f - avg_weight) * node.net_v;
    } else if (prm.fpu_parent_weight > 0.0f) {
        baseline = prm.fpu_parent_weight * node.net_v + (1.0f - prm.fpu_parent_weight) * parent_q;
    }
    const float reduction = (is_root ? prm.root_fpu_reduction : prm.fpu_reduction) * std::sqrt(visited_policy_mass);
    const float loss_prop = is_root ? prm.root_fpu_loss_prop : prm.fpu_loss_prop;
    const float fpu = baseline - reduction;
    return fpu + (-1.0f - fpu) * loss_prop;    // values live in [-1, 1]: a loss is -1
}

uint32_t Search::select_child(const Node& node, bool is_root) const {
    const auto& children = node.children;
    const uint32_t nb = static_cast<uint32_t>(children.size());
    const uint32_t n_parent = node.n.load(std::memory_order_relaxed);
    const float* priors = (is_root && !root_priors.empty()) ? root_priors.data() : nullptr;

    // Total weight of the edges leaving this node (KataGo's totalChildWeight —
    // the node's own visit is not part of it) and prior mass of the moves that
    // have been tried at least once.
    float total_child_weight = 0.0f, visited_mass = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const Edge& e = children[i];
        const uint32_t w = e.n.load(std::memory_order_relaxed) + e.pending.load(std::memory_order_relaxed);
        total_child_weight += static_cast<float>(w);
        if (w > 0) visited_mass += priors ? priors[i] : e.prior;
    }

    const float cpuct = prm.cpuct_exploration + prm.cpuct_exploration_log *
        std::log((total_child_weight + prm.cpuct_exploration_base) / prm.cpuct_exploration_base);
    // The √W of the PUCT formula, with KataGo's offset keeping it positive at W = 0.
    const float explore_scaling = cpuct * utility_stdev_factor(node, n_parent)
                                * std::sqrt(total_child_weight + 0.01f);
    const float fpu = fpu_value(node, n_parent, visited_mass, is_root);
    // Forced playouts (self-play only): a move that has been tried once is
    // pushed up to √(k·P·W) visits before PUCT is allowed to drop it.
    const bool forced = is_root && full_search && prm.forced_playouts_k > 0.0f;
    const float forced_scale = forced ? prm.forced_playouts_k * total_child_weight : 0.0f;

    uint32_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < nb; i++) {
        const Edge& e = children[i];
        const uint32_t n = e.n.load(std::memory_order_relaxed);
        const uint32_t pending = e.pending.load(std::memory_order_relaxed);
        const uint32_t n_eff = n + pending;
        const float p = priors ? priors[i] : e.prior;
        // Q comes from the child NODE — everything the search knows about that
        // position, whichever parent paid for it — while the PUCT denominator
        // is this EDGE's own visit count.
        float q = fpu;
        const Node* child = e.child.load(std::memory_order_acquire);
        if (child) {
            const uint32_t cn = child->n.load(std::memory_order_relaxed);
            const uint32_t cp = child->pending.load(std::memory_order_relaxed);
            // Every pending visit is counted as a loss (-1): the virtual loss.
            if (cn + cp > 0)
                q = (child->w.load(std::memory_order_relaxed) - static_cast<float>(cp)) / (cn + cp);
        }
        float score = q + explore_scaling * p / (1.0f + n_eff);
        if (forced && n_eff > 0 && static_cast<float>(n_eff) < std::sqrt(forced_scale * p))
            score += 1e6f;
        if (score > best_score) { best_score = score; best = i; }
    }
    return best;
}

void Search::priors_from_logits(const nn::Result& result, size_t nb_moves, float* priors) const {
    // Softmax over the legal moves only, with the policy temperature.
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t m = 0; m < nb_moves; m++) max_logit = std::max(max_logit, result.logits[m]);
    float sum = 0.0f;
    const float inv_t = 1.0f / prm.policy_temperature;
    for (size_t m = 0; m < nb_moves; m++) {
        priors[m] = std::exp((result.logits[m] - max_logit) * inv_t);
        sum += priors[m];
    }
    for (size_t m = 0; m < nb_moves; m++) priors[m] /= sum;
}

void Search::expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value) {
    const size_t nb = moves.size();
    node.children.reserve(nb);
    for (size_t m = 0; m < nb; m++) node.children.emplace_back(moves[m], priors[m]);
    node.net_v = value;
    node.state.store(EXPANDED, std::memory_order_release);   // publishes the edges
}

void Search::backup(const PathStep* path, size_t length, float leaf_value) {
    // The leaf's value is for the player to move at the leaf; a node's W is
    // for the player who moved into it, hence the initial sign flip, and one
    // more flip per ply on the way up (negamax).
    float v = -leaf_value;
    // Σ v² is the same at every ply: the negamax flip disappears in the square.
    const float v2 = leaf_value * leaf_value;
    for (size_t i = length; i-- > 0;) {
        Node* node = path[i].node;
        node->w.fetch_add(v, std::memory_order_relaxed);
        node->w2.fetch_add(v2, std::memory_order_relaxed);
        node->n.fetch_add(1, std::memory_order_relaxed);
        node->pending.fetch_sub(1, std::memory_order_relaxed);
        if (Edge* e = path[i].edge) {
            e->n.fetch_add(1, std::memory_order_relaxed);
            e->pending.fetch_sub(1, std::memory_order_relaxed);
        }
        v = -v;
    }
}

void Search::undo_pending(const PathStep* path, size_t length) {
    for (size_t i = 0; i < length; i++) {
        path[i].node->pending.fetch_sub(1, std::memory_order_relaxed);
        if (Edge* e = path[i].edge) e->pending.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ── the per-thread search loop ─────────────────────────────────────────────
void Search::run_worker(Worker& w, const Yolah& root_position) {
    Node* const root_node = root;
    const uint64_t root_hash = zobrist::hash(root_position);
    const size_t max_collisions = prm.max_collisions ? prm.max_collisions : prm.batch_size;
    uint64_t local_collisions = 0, local_hits = 0, local_evals = 0;
    float priors[Yolah::MAX_NB_MOVES];
    // Tree mode: every edge gets its own node, under a key nothing else uses.
    static std::atomic<uint64_t> unique_key{0x8000000000000000ull};

    while (!stop.load(std::memory_order_relaxed)) {
        // ── 1. gather ──
        w.leaves.clear();
        w.descents.clear();
        w.requests.clear();
        size_t budget = prm.batch_size;
        if (sim_budget) {
            // Claim visits from the shared budget so that the threads together
            // never exceed it (unused claims are returned below).
            uint64_t claimed = reserved.load(std::memory_order_relaxed);
            for (;;) {
                if (claimed >= sim_budget) { budget = 0; break; }
                const uint64_t want = std::min<uint64_t>(budget, sim_budget - claimed);
                if (reserved.compare_exchange_weak(claimed, claimed + want, std::memory_order_relaxed)) {
                    budget = want;
                    break;
                }
            }
            if (budget == 0) {
                if (simulations.load(std::memory_order_relaxed) >= sim_budget) break;
                // Other threads still hold claims: wait for their batch instead of spinning.
                evaluator.evaluate({}, {});
                std::this_thread::yield();
                continue;
            }
        }
        size_t batch_collisions = 0, gathered_visits = 0;
        while (gathered_visits < budget && batch_collisions <= max_collisions) {
            PathStep* path = w.path(w.descents.size());
            size_t len = 0;
            Node* node = root_node;
            Yolah state = root_position;
            uint64_t hash = root_hash;
            node->pending.fetch_add(1, std::memory_order_relaxed);
            path[len++] = {node, nullptr};
            for (;;) {
                uint8_t st = node->state.load(std::memory_order_acquire);
                if (st == EXPANDED) {
                    Edge& e = node->children[select_child(*node, node == root_node)];
                    const uint8_t mover = state.current_player();
                    state.play(e.action);
                    hash = zobrist::update(hash, mover, e.action);
                    e.pending.fetch_add(1, std::memory_order_relaxed);
                    Node* child = e.child.load(std::memory_order_acquire);
                    if (!child) {
                        // Link the edge to its successor, sharing the node with
                        // every other path that reaches the same position. The
                        // shard lock serialises the linking of this edge: in
                        // graph mode it is picked by the position key, in tree
                        // mode by the edge itself.
                        const uint64_t key = prm.graph_search ? node_key(hash, state)
                                                              : unique_key.fetch_add(1, std::memory_order_relaxed);
                        Shard& sh = shards[(prm.graph_search ? key : std::hash<const void*>{}(&e)) % NB_SHARDS];
                        std::unique_lock lock(sh.mutex);
                        child = e.child.load(std::memory_order_acquire);
                        if (!child) {
                            auto [it, inserted] = sh.nodes.try_emplace(key, nullptr);
                            if (inserted) it->second = new_node();
                            else          transpositions.fetch_add(1, std::memory_order_relaxed);
                            child = it->second;
                            e.child.store(child, std::memory_order_release);
                        }
                    }
                    child->pending.fetch_add(1, std::memory_order_relaxed);
                    path[len++] = {child, &e};
                    node = child;
                    continue;
                }
                if (st == TERMINAL) {
                    backup(path, len, terminal_value(state));
                    simulations.fetch_add(1, std::memory_order_relaxed);
                    ++gathered_visits;
                    break;
                }
                if (st == UNEXPANDED) {
                    uint8_t expected = UNEXPANDED;
                    if (node->state.compare_exchange_strong(expected, EXPANDING, std::memory_order_acq_rel,
                                                            std::memory_order_acquire)) {
                        ++gathered_visits;
                        if (state.game_over()) {
                            node->state.store(TERMINAL, std::memory_order_release);
                            backup(path, len, terminal_value(state));
                            simulations.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                        Worker::Leaf& leaf = w.leaves.emplace_back();
                        leaf.node = node;
                        leaf.state = state;
                        state.moves(leaf.moves);
                        leaf.hash = hash;
                        // Transposition the node table missed (a position that
                        // dropped out of the graph, or tree mode): expand from
                        // the network cache, no network call.
                        float value;
                        if (cache->lookup(leaf.hash, leaf.moves.size(), value, priors)) {
                            expand(*node, leaf.moves, priors, value);
                            backup(path, len, value);
                            simulations.fetch_add(1, std::memory_order_relaxed);
                            ++local_hits;
                            w.leaves.pop_back();
                        } else {
                            w.descents.push_back({static_cast<uint32_t>(w.leaves.size() - 1),
                                                  static_cast<uint32_t>(len)});
                        }
                        break;
                    }
                    if (expected != EXPANDING) continue;   // became EXPANDED/TERMINAL meanwhile: go on
                }
                // st == EXPANDING: the leaf is being evaluated. If it is ours
                // (same batch), this descent rides on the same evaluation —
                // its path is kept separately, it may well differ. Otherwise
                // the descent is undone.
                uint32_t mine = UINT32_MAX;
                for (uint32_t i = 0; i < w.leaves.size(); i++)
                    if (w.leaves[i].node == node) { mine = i; break; }
                if (mine != UINT32_MAX) {
                    w.descents.push_back({mine, static_cast<uint32_t>(len)});
                    ++gathered_visits;
                } else {
                    undo_pending(path, len);
                    ++batch_collisions;
                }
                break;
            }
        }
        local_collisions += batch_collisions;
        if (sim_budget && gathered_visits < budget)
            reserved.fetch_sub(budget - gathered_visits, std::memory_order_relaxed);

        // ── 2. evaluate ──
        const size_t nb = w.leaves.size();
        if (nb == 0 && gathered_visits == 0) {
            // Every descent collided with leaves another thread is evaluating:
            // join the evaluator's barrier (an empty request) so we resume
            // right after that batch has been expanded, instead of spinning.
            evaluator.evaluate({}, {});
            std::this_thread::yield();
        }
        if (nb > 0) {
            for (Worker::Leaf& leaf : w.leaves)
                w.requests.push_back({&leaf.state, leaf.moves.begin(), static_cast<uint16_t>(leaf.moves.size())});
            evaluator.evaluate(w.requests, std::span<nn::Result>(w.results.data(), nb));
            local_evals += nb;
            // ── 3. expand ... ──
            for (size_t i = 0; i < nb; i++) {
                Worker::Leaf& leaf = w.leaves[i];
                const nn::Result& r = w.results[i];
                priors_from_logits(r, leaf.moves.size(), priors);
                cache->store(leaf.hash, leaf.moves.size(), r.value, priors);
                expand(*leaf.node, leaf.moves, priors, r.value);
            }
            // ── ... and back up, once per descent, along its own path ──
            for (size_t d = 0; d < w.descents.size(); d++) {
                const Worker::Descent& desc = w.descents[d];
                backup(w.path(d), desc.path_len, w.results[desc.leaf].value);
            }
            simulations.fetch_add(w.descents.size(), std::memory_order_relaxed);
        }

        if (prm.microseconds && std::chrono::steady_clock::now() >= deadline) stop.store(true, std::memory_order_relaxed);
        if (sim_budget && simulations.load(std::memory_order_relaxed) >= sim_budget)
            stop.store(true, std::memory_order_relaxed);
    }
    collisions.fetch_add(local_collisions, std::memory_order_relaxed);
    cache_hits.fetch_add(local_hits, std::memory_order_relaxed);
    evaluations.fetch_add(local_evals, std::memory_order_relaxed);
}

// ── top level ──────────────────────────────────────────────────────────────
SearchResult Search::search(const Yolah& state) {
    using namespace std::chrono;
    const auto start = steady_clock::now();

    const uint64_t key = node_key(zobrist::hash(state), state);
    if (!(prm.reuse_tree && reuse(state, key))) new_root(state, key);
    if (root->state.load(std::memory_order_acquire) == UNEXPANDED) expand_root(state);

    // Playout cap randomization: most self-play moves get a small budget, no
    // root noise and no policy target; a minority get the full treatment.
    full_search = true;
    sim_budget = prm.nb_simulations;
    if (prm.nb_simulations && prm.nb_simulations_fast && prm.playout_cap_fast_prob > 0.0f) {
        const float u = (rng.rand<uint64_t>() >> 11) * 0x1.0p-53f;
        if (u < prm.playout_cap_fast_prob) {
            full_search = false;
            sim_budget = prm.nb_simulations_fast;
        }
    }

    std::vector<std::unique_ptr<Worker>> workers;
    for (size_t t = 0; t < prm.nb_threads; t++) {
        // Distinct, non-zero seeds per thread and per search.
        const uint64_t seed = (base_seed + 0x9E3779B97F4A7C15ull * (search_counter * prm.nb_threads + t + 1)) | 1;
        workers.push_back(std::make_unique<Worker>(seed, prm.batch_size));
    }
    ++search_counter;

    simulations = 0;
    reserved = 0;
    collisions = 0;
    cache_hits = 0;
    evaluations = 0;
    transpositions = 0;
    stop = false;
    if (root->state.load(std::memory_order_acquire) == EXPANDED) {
        apply_root_noise(*workers[0]);
        deadline = prm.microseconds ? start + microseconds(prm.microseconds) : steady_clock::time_point::max();
        if (prm.nb_threads == 1) {
            run_worker(*workers[0], state);
        } else {
            std::vector<std::jthread> threads;
            for (size_t t = 0; t < prm.nb_threads; t++)
                threads.emplace_back([this, &workers, t, &state] { run_worker(*workers[t], state); });
        }
    }
    const double seconds = duration<double>(steady_clock::now() - start).count();
    SearchResult res = make_result(state, seconds, simulations.load(), collisions.load(), *workers[0]);
    res.root_visits = root->n.load();
    return res;
}

// Play values of the root's moves: the edge visits, with the forced playouts
// taken back out (KataGo's getReducedPlaySelectionWeight) and the LCB bonus
// applied. They serve both as the training target π and as the move choice.
void Search::root_play_values(std::vector<ChildStat>& children) const {
    const size_t nb = children.size();
    if (nb == 0) return;

    float total = 0.0f;
    for (const ChildStat& c : children) total += static_cast<float>(c.visits);
    for (ChildStat& c : children) c.play_value = static_cast<float>(c.visits);

    // ── 1. the most stably explored child (KataGo's "non LCB best") ──
    size_t best = 0;
    {
        float best_goodness = -1e30f;
        for (size_t i = 0; i < nb; i++) {
            const float v = static_cast<float>(children[i].visits);
            // Discount one visit's worth: the most recent one may be an outlier.
            const float g = v * std::max(0.0f, v - 1.0f) / std::max(1.0f, v) + 2.0f * children[i].prior;
            if (g > best_goodness) { best_goodness = g; best = i; }
        }
    }

    // ── 2. policy target pruning ──
    // How many visits would PUCT have given each child, in retrospect, if it
    // had never been forced? Invert the selection formula at the best child's
    // selection value and keep the smaller of the two counts.
    if (prm.policy_target_pruning && total > 0.0f) {
        const float cpuct = prm.cpuct_exploration + prm.cpuct_exploration_log *
            std::log((total + prm.cpuct_exploration_base) / prm.cpuct_exploration_base);
        const float explore_scaling = cpuct * utility_stdev_factor(*root, root->n.load(std::memory_order_relaxed))
                                    * std::sqrt(total + 0.01f);
        const ChildStat& b = children[best];
        const float best_value = b.q + explore_scaling * b.prior / (1.0f + static_cast<float>(b.visits));
        for (size_t i = 0; i < nb; i++) {
            if (i == best) continue;
            ChildStat& c = children[i];
            if (c.visits == 0) { c.play_value = 0.0f; continue; }
            const float excess = best_value - c.q;     // the exploration term it would need
            if (excess <= 0.0f) continue;              // still the best move on value alone: keep it
            const float wanted = explore_scaling * c.prior / excess - 1.0f;
            c.play_value = std::min(c.play_value, std::ceil(std::max(0.0f, wanted)));
        }
    }

    // ── 3. LCB: give the best lower bound enough play value to be chosen ──
    if (!prm.use_lcb || total <= 0.0f) return;
    std::vector<float> radius(nb, 0.0f);
    const float best_play = children[best].play_value;
    float best_lcb = -1e10f;
    size_t best_lcb_index = SIZE_MAX;
    for (size_t i = 0; i < nb; i++) {
        ChildStat& c = children[i];
        if (c.visits == 0) { c.lcb = -2.0f; radius[i] = 2.0f * prm.lcb_stdevs; continue; }
        // Weighted-sample machinery from KataGo, specialised to unit weights:
        // every visit weighs 1, so weightSum = weightSqSum = edge visits and
        // the effective sample size is the edge visit count itself.
        const float n = static_cast<float>(c.visits);
        const float mean_sq = c.q * c.q;
        float sq_avg = std::max(c.q2, mean_sq + 1e-8f);
        // A prior saying the variance could be as large as the whole utility
        // range (radius 1), with a weight that vanishes as the visits grow.
        const float prior_w = 1.0f / (n * n);
        sq_avg = (sq_avg * n + (sq_avg + 1.0f) * prior_w) / (n + prior_w);
        const float weight_sum = n + prior_w;
        const float ess = weight_sum * weight_sum / (n + prior_w * prior_w);
        const float var = std::max(0.0f, sq_avg - mean_sq);
        radius[i] = std::sqrt(var / ess) * prm.lcb_stdevs;
        c.lcb = c.q - radius[i];
        if (c.play_value > 0.0f && c.play_value >= prm.min_visit_prop_for_lcb * best_play) {
            if (c.lcb > best_lcb) { best_lcb = c.lcb; best_lcb_index = i; }
        }
    }
    if (best_lcb_index == SIZE_MAX) return;
    // The winner is given just enough play value to beat every other child:
    // a child whose bound is `excess` worse would need its radius to grow by
    // `factor` to catch up, and the radius shrinks like 1/√weight.
    float adjusted = children[best_lcb_index].play_value;
    for (size_t i = 0; i < nb; i++) {
        if (i == best_lcb_index) continue;
        const float excess = best_lcb - children[i].lcb;
        if (excess < 0.0f) continue;   // that child only lost on the visit threshold
        const float factor = (radius[i] + excess) / (radius[i] + 0.20f * excess);
        adjusted = std::max(adjusted, factor * factor * children[i].play_value);
    }
    children[best_lcb_index].play_value = adjusted;
}

SearchResult Search::make_result(const Yolah& state, double seconds, uint64_t sims,
                                 uint64_t nb_collisions, Worker& w) const {
    SearchResult res;
    res.seconds = seconds;
    res.nb_simulations = sims;
    res.nb_collisions = nb_collisions;
    res.nb_cache_hits = cache_hits.load();
    res.nb_evaluations = evaluations.load();
    res.nb_transpositions = transpositions.load();
    res.full_search = full_search;
    res.tree_nodes = node_count.load();
    res.net_value = root->net_v;
    const uint32_t root_n = root->n.load();
    res.root_value = root_n > 0 ? -root->w.load() / root_n : root->net_v;

    const auto& children = root->children;
    res.children.reserve(children.size());
    for (size_t i = 0; i < children.size(); i++) {
        const Edge& e = children[i];
        const Node* c = e.child.load(std::memory_order_acquire);
        const uint32_t cn = c ? c->n.load() : 0;
        ChildStat st{};
        st.move   = e.action;
        st.prior  = root_priors.empty() ? e.prior : root_priors[i];
        st.visits = e.n.load();
        st.q      = cn > 0 ? c->w.load() / cn : 0.0f;
        st.q2     = cn > 0 ? c->w2.load() / cn : 0.0f;
        st.lcb    = -2.0f;
        res.children.push_back(st);
    }
    root_play_values(res.children);
    std::stable_sort(res.children.begin(), res.children.end(), [](const ChildStat& a, const ChildStat& b) {
        return a.play_value != b.play_value ? a.play_value > b.play_value : a.q > b.q;
    });
    if (res.children.empty()) return res;   // terminal root

    // Training target π: the play values, normalised. If the search could not
    // complete a single simulation, fall back to the priors.
    float total = 0.0f;
    for (const ChildStat& c : res.children) total += c.play_value;
    res.policy.resize(res.children.size());
    for (size_t i = 0; i < res.children.size(); i++)
        res.policy[i] = total > 0 ? res.children[i].play_value / total : res.children[i].prior;

    // Move choice: the best play value or, in the self-play opening phase, a
    // sample ∝ π^(1/τ).
    size_t pick = 0;
    if (total > 0.0f && prm.temperature > 0.0f && state.nb_plies() < prm.temperature_cutoff) {
        std::vector<double> weights(res.children.size());
        double sum = 0.0;
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = std::pow(static_cast<double>(res.children[i].play_value), 1.0 / prm.temperature);
            sum += weights[i];
        }
        double r = (w.prng.rand<uint64_t>() >> 11) * 0x1.0p-53 * sum;
        for (pick = 0; pick + 1 < weights.size(); pick++) {
            r -= weights[pick];
            if (r <= 0.0) break;
        }
    } else if (total == 0.0f) {
        for (size_t i = 1; i < res.children.size(); i++)
            if (res.children[i].prior > res.children[pick].prior) pick = i;
    }
    res.best_move = res.children[pick].move;
    return res;
}

std::string SearchResult::to_string(size_t max_children) const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "[ move ]: " << best_move << "  Q: " << std::showpos << root_value << std::noshowpos
       << "  net: " << std::showpos << net_value << std::noshowpos
       << "  sims: " << nb_simulations << (full_search ? "" : " (fast)")
       << " (" << root_visits << " root visits, " << nb_evaluations
       << " net evals, " << nb_cache_hits << " cache hits, " << nb_transpositions
       << " transpositions, " << nb_collisions << " collisions)  "
       << std::setprecision(1) << seconds << "s, "
       << std::setprecision(0) << (seconds > 0 ? nb_simulations / seconds : 0) << " sims/s, nodes: " << tree_nodes << "\n";
    os << std::setprecision(3);
    for (size_t i = 0; i < std::min(max_children, children.size()); i++) {
        const ChildStat& c = children[i];
        os << "  " << c.move << "  N: " << std::setw(6) << c.visits << "  Q: " << std::showpos << c.q
           << "  lcb: " << c.lcb << std::noshowpos << "  P: " << c.prior
           << "  pi: " << (policy.empty() ? 0.0f : policy[i]) << "\n";
    }
    return os.str();
}

} // namespace az
