#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H
#include "player.h"
#include <atomic>
#include <memory_resource>
#include "BS_thread_pool.h"
#include <numbers>
#include <chrono>
#include "misc.h"
#include "types.h"
#include <array>

namespace {
    thread_local PRNG prng(std::random_device{}());

    float game_value(const Yolah& yolah, uint8_t player) {
        const auto [black_score, white_score] = yolah.score();
        if (int v = (player == Yolah::BLACK ? 1 : -1) * (black_score - white_score); v > 0) {
            return 1;
        }
        else if (v < 0) {
            return -1;
        }
        return 0;
    }

    struct Playout {
        float operator()(Yolah& yolah) const {
            uint8_t player = yolah.current_player();
            Yolah::MoveList moves;
            while (!yolah.game_over()) {
                yolah.moves(moves);
                Move m = moves[reduce(prng.rand<uint32_t>(), moves.size())];
                yolah.play(m);
            }
            return game_value(yolah, player);
        }
    };
}

template<typename EvalLeaf = Playout>
class MCTSPlayer : public Player {
    struct Node {
        static size_t NB_NODES;
        static constexpr float C = std::numbers::sqrt2_v<float>;
        std::atomic<uint32_t> nb_visits = 1;
        std::atomic<uint32_t> virtual_loss = 0;
        std::atomic<float> value = 0;
        std::atomic<uint8_t> expanded = 0;
        Move action;
        std::pmr::vector<Node> nodes;

        using allocator_type = std::pmr::polymorphic_allocator<>;
        explicit Node(allocator_type alloc = {}) : Node(Move::none(), alloc) {
        }
        explicit Node(Move action, allocator_type alloc = {}) : action(action), nodes(alloc) {
            ++Node::NB_NODES;
        }
        Node(Node&& n, allocator_type) noexcept
            : nb_visits(n.nb_visits.load()), virtual_loss(n.virtual_loss.load()), value(n.value.load()),
              expanded(n.expanded.load()), action(std::move(n.action)), nodes(std::move(n.nodes)) {
        }

        bool is_leaf() const {
            return expanded.load(std::memory_order_acquire) != 2;
        }

        void expand(const Yolah& yolah) {
            if (uint8_t desired = 0; expanded == 0 && nb_visits >= NB_VISITS_BEFORE_EXPANSION &&
                expanded.compare_exchange_strong(desired, 1, std::memory_order::acq_rel, std::memory_order_relaxed)) {
                Yolah::MoveList moves;
                yolah.moves(moves);
                nodes.resize(moves.size());
                NB_NODES += moves.size();
                for (size_t i = 0; i < moves.size(); i++) {
                    nodes[i].action = moves[i];
                }
                expanded.store(2, std::memory_order_release);
            }
        }

        void update(float v) {
            nb_visits += 1;
            virtual_loss -= VIRTUAL_LOSS;
            value += v;
        }

        uint32_t select() const {
            auto N = static_cast<double>(nb_visits);
            double log_N = std::log(N);
            auto nb_children = static_cast<uint32_t>(nodes.size());
            //uint32_t k = prng.rand<uint32_t>() % nodes.size();
            uint32_t k = reduce(prng.rand<uint32_t>(), nodes.size());
            uint32_t res = k;
            double best_value = std::numeric_limits<double>::lowest();
            for (uint32_t i = 0; i < nb_children; i++) {
                auto n = static_cast<double>(nodes[k].nb_visits + nodes[k].virtual_loss);
                if (double v = -nodes[k].value / n + C * std::sqrt(log_N / n); v > best_value) {
                    best_value = v;
                    res = k;
                }
                ++k;
                if (k == nb_children) {
                    k = 0;
                }
            }
            return res;
        }
    };
    friend std::ostream& operator<<(std::ostream& os, const Node& n) {
        os << "[ # visits ]: " << n.nb_visits << '\n';
        os << "[   value  ]: " << static_cast<double>(n.value) / n.nb_visits << '\n';
        os << "[ children ]:\n";
        for (const auto& node : n.nodes) {
            os << "  "  << node.action << " (" << node.nb_visits << " / " << node.value << ")\n";
        }
        return os;
    }
    static constexpr uint32_t VIRTUAL_LOSS = 4;
    static constexpr uint32_t NB_VISITS_BEFORE_EXPANSION = 8;
    const uint64_t thinking_time;
    EvalLeaf eval_leaf;
    Node root;
    BS::thread_pool pool;

    void think(Yolah yolah) {
        using namespace std::chrono;
        const steady_clock::time_point start = steady_clock::now();
        duration<uint64_t, std::micro> mu;
        Yolah backup = yolah;
        std::array<Node*, SQUARE_NB> visited;
        size_t nb_iter = 0;
        for(;;) {
            size_t size = 1;
            Node* current = &root;
            visited[0] = current;
            current->virtual_loss += VIRTUAL_LOSS;
            while (!yolah.game_over() && !current->is_leaf()) {
                current = &current->nodes[current->select()];
                yolah.play(current->action);
                current->virtual_loss += VIRTUAL_LOSS;
                visited[size++] = current;
            }
            float game_val = 0;
            if (yolah.game_over()) {
                game_val = ::game_value(yolah, yolah.current_player());
            } else {
                current->expand(yolah);
                game_val = eval_leaf(yolah);
            }
            for (size_t i = size - 1; i < size; --i) {
                visited[i]->update(game_val);
                game_val = -game_val;
            }
            yolah = backup;
            ++nb_iter;
            if ((nb_iter & 0x1F) == 0) {
                mu = duration_cast<std::chrono::microseconds>(steady_clock::now() - start);
                if (mu.count() > thinking_time) break;
            }
        }
    }

    void reset() {
        Node::NB_NODES = 1;
        root.nb_visits = 1;
        root.value = 0;
        root.expanded = false;
        root.nodes.resize(0);
    }

public:
    explicit MCTSPlayer(uint64_t microseconds, EvalLeaf eval_leaf = EvalLeaf{}, std::pmr::polymorphic_allocator<> alloc = {})
        : MCTSPlayer(microseconds, std::thread::hardware_concurrency(), std::move(eval_leaf), alloc) {
    }
    explicit MCTSPlayer(uint64_t microseconds, std::size_t nb_threads, EvalLeaf eval_leaf = EvalLeaf{}, std::pmr::polymorphic_allocator<> alloc = {})
        : thinking_time(microseconds), eval_leaf(std::move(eval_leaf)), root(Move::none(), alloc), pool(static_cast<BS::concurrency_t>(nb_threads)) {
    }

    uint64_t microseconds() const {
        return thinking_time;
    }

    size_t nb_threads() const {
        return pool.get_thread_count();
    }

    Move play(Yolah yolah) override {
        BS::multi_future<void> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t) {
            think(yolah);
        });
        futures.get();
        auto nb_children = static_cast<uint32_t>(root.nodes.size());
        if (nb_children == 0) [[unlikely]] {
            std::cout << "random " << root.nb_visits << "\n";
            root.expand(yolah);
        }
        //uint32_t k = prng.rand<uint32_t>() % root.nodes.size();
        uint32_t k = reduce(prng.rand<uint32_t>(), root.nodes.size());
        Move res = Move::none();
        uint32_t best_nb_visits = 0;
        for (uint32_t i = 0; i < nb_children; i++) {
            uint32_t n = root.nodes[k].nb_visits;
            if (n > best_nb_visits) {
                best_nb_visits = n;
                res = root.nodes[k].action;
            }
            ++k;
            if (k == nb_children) {
                k = 0;
            }
        }
        //std::cout << root;
        //std::cout << Node::NB_NODES << '\n';
        reset();
        return res;
    }

    std::string info() override {
        return "mcts player";
    }

    json config() override {
        json j;
        j["name"] = "MCTSPlayer";
        j["microseconds"] = thinking_time;
        if (pool.get_thread_count() == std::thread::hardware_concurrency()) {
            j["nb threads"] = "hardware concurrency";
        } else {
            j["nb threads"] = pool.get_thread_count();
        }
        return j;
    }
};

template<typename EvalLeaf>
size_t MCTSPlayer<EvalLeaf>::Node::NB_NODES = 0;

#endif
