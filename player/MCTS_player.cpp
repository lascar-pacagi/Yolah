#include "MCTS_player.h"
#include <chrono>
#include "misc.h"
#include "types.h"
#include <array>

using std::size_t;

namespace {
    thread_local PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
}

size_t MCTSPlayer::Node::NB_NODES = 0;

void MCTSPlayer::Node::expand(const Yolah& yolah) { 
    if (bool desired = false; nodes.empty() && nb_visits >= NB_VISITS_BEFORE_EXPANSION && 
        expanded.compare_exchange_strong(desired, true, std::memory_order::acq_rel, std::memory_order_relaxed)) {
        Yolah::MoveList moves;
        yolah.moves(moves);
        nodes.resize(moves.size());            
        for (size_t i = 0; i < moves.size(); i++) {
            nodes[i].action = moves[i];
        }
    }
}

bool MCTSPlayer::Node::is_leaf() const {
    return nodes.empty();
}
        
void MCTSPlayer::Node::update(float v) {
    nb_visits += 1;
    virtual_loss -= VIRTUAL_LOSS;
    value += v;
}

uint32_t MCTSPlayer::Node::select() const {
    auto N = static_cast<float>(nb_visits);
    float log_N = std::log(N);
    // TO DO: SIMD
    auto nb_children = static_cast<uint32_t>(nodes.size());
    uint32_t k = prng.rand<uint32_t>() % nodes.size();
    uint32_t res = k;
    float best_value = std::numeric_limits<float>::lowest(); 
    for (uint32_t i = 0; i < nb_children; i++) {        
        auto n = static_cast<float>(nodes[k].nb_visits + nodes[k].virtual_loss);
        if (float v = -nodes[k].value / n + C * std::sqrt(log_N / n); v > best_value) {
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

std::ostream& operator<<(std::ostream& os, const MCTSPlayer::Node& n) {
    os << "[ # visits ]: " << n.nb_visits << '\n';
    os << "[   value  ]: " << n.value << '\n';
    os << "[ children ]:\n";
    for (const auto& node : n.nodes) {
        os << "  "  << node.action << " (" << node.nb_visits << " / " << node.value << ")\n"; 
    }
    return os;
}

namespace {
    float game_value(const Yolah& yolah, uint8_t player) {
        const auto [black_score, white_score] = yolah.score();        
        if (int v = (player == Yolah::BLACK ? 1 : -1) * (black_score - white_score); v > 0) return 1;
        else if (v < 0) return -1;
        return 0;
    }
}

float MCTSPlayer::playout(Yolah yolah) const {
    uint8_t player = yolah.current_player();
    Yolah::MoveList moves;
    while (!yolah.game_over()) {
        yolah.moves(moves);
        if (moves.size() == 0) continue;
        Move m = moves[prng.rand<size_t>() % moves.size()];
        yolah.play(m);
    }
    return game_value(yolah, player);
}

void MCTSPlayer::think(Yolah yolah) {
    using namespace std::chrono;
    const steady_clock::time_point start = steady_clock::now();
    duration<uint64_t, std::micro> mu;        
    Yolah backup = yolah;    
    std::array<Node*, SQUARE_NB> visited;
    size_t nb_iter = 0;
    do {
        size_t size = 1;
        Node* current = &root;
        visited[0] = current;
        ++current->nb_visits;
        current->virtual_loss += VIRTUAL_LOSS;
        while (!yolah.game_over() && !current->is_leaf()) {
            current = &current->nodes[current->select()];
            yolah.play(current->action);
            ++current->nb_visits;
            current->virtual_loss += VIRTUAL_LOSS;
            visited[size++] = current;
        }
        float game_value = 0;
        if (yolah.game_over()) {            
            game_value = ::game_value(yolah, yolah.current_player());
        } else {
            current->expand(yolah);
            game_value = playout(yolah);
        }
        for (size_t i = size - 1; i < size; --i) {
            visited[i]->update(game_value);
            game_value = -game_value;
        }
        yolah = backup;
        ++nb_iter;
        mu = duration_cast<microseconds>(steady_clock::now() - start);
    } while (mu.count() < thinking_time);
}

void MCTSPlayer::reset() {
    Node::NB_NODES = 1;
    root.nb_visits = 1;
    root.value = 0;
    root.expanded = false;
    root.nodes.resize(0);
}

Move MCTSPlayer::play(Yolah yolah) {
    BS::multi_future<void> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t) { 
        think(yolah);
    });
    futures.get();
    auto nb_children = static_cast<uint32_t>(root.nodes.size());
    if (nb_children == 0) [[unlikely]] {
        std::cout << "random " << root.nb_visits << "\n";
        root.expand(yolah);
    }
    uint32_t k = prng.rand<uint32_t>() % root.nodes.size();
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
    std::cout << root;
    std::cout << Node::NB_NODES << '\n';
    reset();    
    return res;
}
