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

// ── Tree node ──────────────────────────────────────────────────────────────
// 48 bytes + the children vector's heap block. All statistics are atomics
// updated with relaxed ordering: they are heuristics, a slightly stale read
// only changes which leaf a descent picks, never the tree's integrity. The
// only synchronisation that matters is `state` (acquire/release): a thread
// that observes EXPANDED is guaranteed to see the fully built children vector.
struct Search::Node {
    std::atomic<uint32_t> n{0};         // completed visits N(s,a)
    std::atomic<uint32_t> pending{0};   // in-flight visits (virtual loss), each counted as a loss
    std::atomic<float>    w{0.0f};      // Σ backed-up values for the player who moved INTO this node
    std::atomic<uint8_t>  state{UNEXPANDED};
    Move  action  = Move::none();       // move that leads here from the parent
    float prior   = 0.0f;               // P(s,a) from the policy head
    float net_v   = 0.0f;               // network value at this node (player to move here)
    std::pmr::vector<Node> children;

    using allocator_type = std::pmr::polymorphic_allocator<>;
    explicit Node(allocator_type a = {}) : children(a) {}
    Node(Move m, float p, allocator_type a) : action(m), prior(p), children(a) {}
    // Uses-allocator move: steals the children (O(1) with the same resource).
    Node(Node&& o, allocator_type a) noexcept
        : n(o.n.load()), pending(o.pending.load()), w(o.w.load()), state(o.state.load()),
          action(o.action), prior(o.prior), net_v(o.net_v), children(std::move(o.children), a) {}
    Node(Node&& o) noexcept : Node(std::move(o), o.children.get_allocator()) {}
    Node& operator=(Node&& o) noexcept {
        n = o.n.load(); pending = o.pending.load(); w = o.w.load(); state = o.state.load();
        action = o.action; prior = o.prior; net_v = o.net_v;
        children = std::move(o.children);
        return *this;
    }
    size_t subtree_size() const {
        size_t s = 1;
        for (const Node& c : children) s += c.subtree_size();
        return s;
    }
};

// ── Per-thread state ───────────────────────────────────────────────────────
struct Search::Worker {
    static constexpr size_t MAX_PATH = Yolah::MAX_NB_PLIES + 8;   // root → deepest leaf
    struct Leaf {
        Node*    node;
        size_t   path_len;
        uint32_t multiplicity;   // visits of this leaf in the batch (collisions)
        uint64_t hash;           // Zobrist key for the transposition cache
        Yolah    state;
        Yolah::MoveList moves;
    };
    PRNG prng;
    std::vector<Leaf> leaves;            // leaves gathered for the current batch
    std::vector<Node*> paths;            // (batch_size + 1) × MAX_PATH node pointers
    std::vector<nn::Request> requests;
    std::vector<nn::Result> results;

    Worker(uint64_t seed, size_t batch_size)
        : prng(seed ? seed : 0x9E3779B97F4A7C15ull), paths((batch_size + 1) * MAX_PATH, nullptr),
          results(batch_size) {
        leaves.reserve(batch_size);
        requests.reserve(batch_size);
    }
    Node** path(size_t k) { return paths.data() + k * MAX_PATH; }
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
}

float Search::terminal_value(const Yolah& state) {
    const auto [black, white] = state.score();
    const int diff = (state.current_player() == Yolah::BLACK ? 1 : -1) * (int(black) - int(white));
    return diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
}

// ── construction / tree management ─────────────────────────────────────────
Search::Search(nn::Evaluator& evaluator, const SearchParams& params, std::pmr::memory_resource* memory)
    : evaluator(evaluator), prm(params), alloc(memory),
      base_seed(params.seed ? params.seed : std::random_device{}()) {
    if (prm.microseconds == 0 && prm.nb_simulations == 0)
        throw std::invalid_argument("az::Search: either microseconds or nb_simulations must be > 0");
    prm.nb_threads = std::max<size_t>(1, prm.nb_threads);
    prm.batch_size = std::max<size_t>(1, prm.batch_size);
    cache = std::make_unique<NNCache>(prm.nn_cache_mb << 20);
}

void Search::clear_cache() {
    cache->clear();
}

Search::~Search() = default;

void Search::reset() {
    root.reset();
    root_priors.clear();
    node_count = 0;
}

void Search::new_root(const Yolah& state) {
    root = std::make_unique<Node>(alloc);
    root_state = state;
    node_count = 1;
}

void Search::promote(Node& target) {
    // Steal the subtree, then let the old tree die with the old root.
    auto promoted = std::make_unique<Node>(std::move(target), alloc);
    root = std::move(promoted);
    node_count = root->subtree_size();
}

bool Search::reuse(const Yolah& state) {
    if (same_position(state, root_state)) return true;            // same position again
    for (Node& child : root->children) {
        Yolah s1 = root_state;
        s1.play(child.action);
        if (same_position(s1, state)) {                            // we are asked to play the reply
            promote(child);
            root_state = state;
            return true;
        }
        if (child.state.load(std::memory_order_acquire) != EXPANDED) continue;
        for (Node& grandchild : child.children) {
            Yolah s2 = s1;
            s2.play(grandchild.action);
            if (same_position(s2, state)) {                        // our move, then the opponent's
                promote(grandchild);
                root_state = state;
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
    if (prm.dirichlet_epsilon <= 0.0f || root->children.empty()) return;
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
uint32_t Search::select_child(const Node& node, bool is_root) const {
    const auto& children = node.children;
    const uint32_t nb = static_cast<uint32_t>(children.size());
    const uint32_t n_parent = node.n.load(std::memory_order_relaxed);
    const float N = static_cast<float>(n_parent + node.pending.load(std::memory_order_relaxed));
    const float sqrt_N = std::sqrt(std::max(N, 1.0f));
    const float c_puct = prm.c_puct_init + std::log((N + prm.c_puct_base + 1.0f) / prm.c_puct_base);
    // Q of this node for the player to move here (the sign flips: node.w is
    // from the opponent's point of view), used as the FPU baseline.
    const float parent_q = n_parent > 0 ? -node.w.load(std::memory_order_relaxed) / n_parent : node.net_v;
    const float* priors = (is_root && !root_priors.empty()) ? root_priors.data() : nullptr;

    float visited_mass = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const Node& c = children[i];
        if (c.n.load(std::memory_order_relaxed) + c.pending.load(std::memory_order_relaxed) > 0)
            visited_mass += priors ? priors[i] : c.prior;
    }
    const float fpu = parent_q - prm.fpu_reduction * std::sqrt(visited_mass);

    uint32_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < nb; i++) {
        const Node& c = children[i];
        const uint32_t n = c.n.load(std::memory_order_relaxed);
        const uint32_t pending = c.pending.load(std::memory_order_relaxed);
        const uint32_t n_eff = n + pending;
        // Every pending visit is counted as a loss (-1): the virtual loss.
        const float q = n_eff > 0 ? (c.w.load(std::memory_order_relaxed) - static_cast<float>(pending)) / n_eff : fpu;
        const float p = priors ? priors[i] : c.prior;
        const float score = q + c_puct * p * sqrt_N / (1.0f + n_eff);
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
    node_count.fetch_add(nb, std::memory_order_relaxed);
    node.state.store(EXPANDED, std::memory_order_release);   // publishes the children
}

void Search::backup(Node* const* path, size_t length, float leaf_value, uint32_t multiplicity) {
    // The leaf's value is for the player to move at the leaf; the node's W is
    // for the player who moved into it, hence the initial sign flip, and one
    // more flip per ply on the way up (negamax). `multiplicity` visits were
    // descended (and left pending) along this same path.
    float v = -leaf_value * static_cast<float>(multiplicity);
    for (size_t i = length; i-- > 0;) {
        Node* node = path[i];
        node->w.fetch_add(v, std::memory_order_relaxed);
        node->n.fetch_add(multiplicity, std::memory_order_relaxed);
        node->pending.fetch_sub(multiplicity, std::memory_order_relaxed);
        v = -v;
    }
}

void Search::undo_pending(Node* const* path, size_t length) {
    for (size_t i = 0; i < length; i++) path[i]->pending.fetch_sub(1, std::memory_order_relaxed);
}

// ── the per-thread search loop ─────────────────────────────────────────────
void Search::run_worker(Worker& w, const Yolah& root_position) {
    Node* const root_node = root.get();
    const size_t max_collisions = prm.max_collisions ? prm.max_collisions : prm.batch_size;
    uint64_t local_collisions = 0, local_hits = 0, local_evals = 0;
    float priors[Yolah::MAX_NB_MOVES];

    while (!stop.load(std::memory_order_relaxed)) {
        // ── 1. gather ──
        w.leaves.clear();
        w.requests.clear();
        size_t budget = prm.batch_size;
        if (prm.nb_simulations) {
            // Claim visits from the shared budget so that the threads together
            // never exceed it (unused claims are returned below).
            uint64_t claimed = reserved.load(std::memory_order_relaxed);
            for (;;) {
                if (claimed >= prm.nb_simulations) { budget = 0; break; }
                const uint64_t want = std::min<uint64_t>(budget, prm.nb_simulations - claimed);
                if (reserved.compare_exchange_weak(claimed, claimed + want, std::memory_order_relaxed)) {
                    budget = want;
                    break;
                }
            }
            if (budget == 0) {
                if (simulations.load(std::memory_order_relaxed) >= prm.nb_simulations) break;
                // Other threads still hold claims: wait for their batch instead of spinning.
                evaluator.evaluate({}, {});
                std::this_thread::yield();
                continue;
            }
        }
        size_t batch_collisions = 0, gathered_visits = 0;
        while (gathered_visits < budget && batch_collisions <= max_collisions) {
            Node** path = w.path(w.leaves.size());
            size_t len = 0;
            Node* node = root_node;
            Yolah state = root_position;
            node->pending.fetch_add(1, std::memory_order_relaxed);
            path[len++] = node;
            for (;;) {
                uint8_t st = node->state.load(std::memory_order_acquire);
                if (st == EXPANDED) {
                    node = &node->children[select_child(*node, node == root_node)];
                    state.play(node->action);
                    node->pending.fetch_add(1, std::memory_order_relaxed);
                    path[len++] = node;
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
                        leaf.path_len = len;
                        leaf.multiplicity = 1;
                        leaf.state = state;
                        state.moves(leaf.moves);
                        leaf.hash = zobrist::hash(state);
                        // Transposition: expand from the cache, no network call.
                        float value;
                        if (cache->lookup(leaf.hash, leaf.moves.size(), value, priors)) {
                            expand(*node, leaf.moves, priors, value);
                            backup(path, len, value);
                            simulations.fetch_add(1, std::memory_order_relaxed);
                            ++local_hits;
                            w.leaves.pop_back();
                        }
                        break;
                    }
                    if (expected != EXPANDING) continue;   // became EXPANDED/TERMINAL meanwhile: go on
                }
                // st == EXPANDING: the leaf is being evaluated. If it is ours
                // (same batch), count one more visit of it; otherwise undo.
                Worker::Leaf* mine = nullptr;
                for (Worker::Leaf& l : w.leaves) if (l.node == node) { mine = &l; break; }
                if (mine) {
                    ++mine->multiplicity;
                    ++gathered_visits;
                } else {
                    undo_pending(path, len);
                    ++batch_collisions;
                }
                break;
            }
        }
        local_collisions += batch_collisions;
        if (prm.nb_simulations && gathered_visits < budget)
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
            // ── 3. expand + back up ──
            uint64_t visits = 0;
            for (size_t i = 0; i < nb; i++) {
                Worker::Leaf& leaf = w.leaves[i];
                const nn::Result& r = w.results[i];
                priors_from_logits(r, leaf.moves.size(), priors);
                cache->store(leaf.hash, leaf.moves.size(), r.value, priors);
                expand(*leaf.node, leaf.moves, priors, r.value);
                backup(w.path(i), leaf.path_len, r.value, leaf.multiplicity);
                visits += leaf.multiplicity;
            }
            simulations.fetch_add(visits, std::memory_order_relaxed);
        }

        if (prm.microseconds && std::chrono::steady_clock::now() >= deadline) stop.store(true, std::memory_order_relaxed);
        if (prm.nb_simulations && simulations.load(std::memory_order_relaxed) >= prm.nb_simulations)
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

    if (!(prm.reuse_tree && root && reuse(state))) new_root(state);
    if (root->state.load(std::memory_order_acquire) == UNEXPANDED) expand_root(state);

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

SearchResult Search::make_result(const Yolah& state, double seconds, uint64_t sims,
                                 uint64_t nb_collisions, Worker& w) const {
    SearchResult res;
    res.seconds = seconds;
    res.nb_simulations = sims;
    res.nb_collisions = nb_collisions;
    res.nb_cache_hits = cache_hits.load();
    res.nb_evaluations = evaluations.load();
    res.tree_nodes = node_count.load();
    res.net_value = root->net_v;
    const uint32_t root_n = root->n.load();
    res.root_value = root_n > 0 ? -root->w.load() / root_n : root->net_v;

    const auto& children = root->children;
    res.children.reserve(children.size());
    uint64_t total = 0;
    for (size_t i = 0; i < children.size(); i++) {
        const Node& c = children[i];
        const uint32_t n = c.n.load();
        total += n;
        res.children.push_back({c.action, root_priors.empty() ? c.prior : root_priors[i], n,
                                n > 0 ? c.w.load() / n : 0.0f});
    }
    std::stable_sort(res.children.begin(), res.children.end(), [](const ChildStat& a, const ChildStat& b) {
        return a.visits != b.visits ? a.visits > b.visits : a.q > b.q;
    });
    if (res.children.empty()) return res;   // terminal root

    // Training target π: raw visit distribution (τ = 1), or the priors if the
    // search could not complete a single simulation.
    res.policy.resize(res.children.size());
    for (size_t i = 0; i < res.children.size(); i++)
        res.policy[i] = total > 0 ? static_cast<float>(res.children[i].visits) / total : res.children[i].prior;

    // Move choice: argmax visits (ties → higher Q, already sorted) or, in the
    // self-play opening phase, a sample ∝ N^(1/τ).
    size_t pick = 0;
    if (total > 0 && prm.temperature > 0.0f && state.nb_plies() < prm.temperature_cutoff) {
        std::vector<double> weights(res.children.size());
        double sum = 0.0;
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = std::pow(static_cast<double>(res.children[i].visits), 1.0 / prm.temperature);
            sum += weights[i];
        }
        double r = (w.prng.rand<uint64_t>() >> 11) * 0x1.0p-53 * sum;
        for (pick = 0; pick + 1 < weights.size(); pick++) {
            r -= weights[pick];
            if (r <= 0.0) break;
        }
    } else if (total == 0) {
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
       << "  sims: " << nb_simulations << " (" << root_visits << " root visits, " << nb_evaluations
       << " net evals, " << nb_cache_hits << " cache hits, " << nb_collisions << " collisions)  "
       << std::setprecision(1) << seconds << "s, "
       << std::setprecision(0) << (seconds > 0 ? nb_simulations / seconds : 0) << " sims/s, nodes: " << tree_nodes << "\n";
    os << std::setprecision(3);
    for (size_t i = 0; i < std::min(max_children, children.size()); i++) {
        const ChildStat& c = children[i];
        os << "  " << c.move << "  N: " << std::setw(6) << c.visits << "  Q: " << std::showpos << c.q
           << std::noshowpos << "  P: " << c.prior << "  pi: " << (policy.empty() ? 0.0f : policy[i]) << "\n";
    }
    return os.str();
}

} // namespace az
