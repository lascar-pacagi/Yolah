#include "MCTS_player.h"
#include <chrono>
#include "misc.h"

using std::size_t;

namespace {
    thread_local PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
}

void MCTSPlayer::Node::expand(Yolah yolah) {
    uint32_t n = NB_VISITS_BEFORE_EXPANSION - 1;
    if (nb_visits == n) {            
        if (nb_visits.compare_exchange_strong(&n, NB_VISITS_BEFORE_EXPANSION, std::memory_order::acq_rel, std::memory_order_relaxed)) {
            Yolah::MoveList moves = yolah.moves();
            nodes.resize(moves.size());
            auto alloc = nodes.get_allocator();
            for (size_t i = 0; i < moves.size(); i++) {
                nodes[i] = alloc.new_object<Node>(moves[i], alloc);
            }
        }
    }
}

bool MCTSPlayer::Node::is_leaf() const {
    return nodes.size() == 0;
}
        
void MCTSPlayer::Node::update(float v) {
    nb_visit -= static_cast<int32_t>(VIRTUAL_LOSS) - 1;
    value += v;
}

uint32_t MCTSPlayer::Node::select() const {
    float N = nb_visits;
    float log_N = std::log(N);
    // TO DO: SIMD
    uint32_t nb_children = nodes.size();
    uint32_t k = prng.rand() % nodes.size();
    uint32_t res = k;
    float best_value = std::numeric_limits<float>::lowest; 
    for (uint32_t i = 0; i < nb_children; i++) {
        float v = -nodes[k].value / nodes[k].nb_visits + C * std::sqrt(logN / nodes[k].nb_visits);
        if (v > best_value) {
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

namespace {
    float game_value(Yolah yolah, uint8_t player) {
        const auto [black_score, white_score] = yolah.score();
        int v = (player == Yolah::BLACK ? 1 : -1) * (static_cast<int>(black_score) - static_cast<int>(white_score));
        if (v > 0) return 1;
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
    steady_clock::time_point now;
    duration<uint64_t, std::micro> mu;        
    Yolah backup = yolah;    
    Node* visited[NB_SQUARE];
    size_t nb_iter = 0;
    do {
        size_t size = 1;
        Node* current = &root;
        visited[0] = current;
        current->nb_visits += VIRTUAL_LOSS;
        while (!yolah.game_over() && !current->is_leaf()) {
            current = current->nodes[current->select()];
            yolah.play(current->action);
            current->nb_visits += VIRTUAL_LOSS;
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
        mu = duration_cast<microseconds>(steady_clock::now() - t1);
    } while (mu.count() < thinking_time);
}

Move MCTSPlayer::play(Yolah yolah) {
    BS::multi_future<void> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t) { 
        think(yolah);
    });
    futures.get();
    uint32_t nb_children = root.nodes.size();
    uint32_t k = prng.rand() % root.nodes.size();
    Move res = Move::none();
    uint32_t best_nb_visits = 0; 
    for (uint32_t i = 0; i < nb_children; i++) {
        if (root.nodes[k].nb_visits > best_nb_visits) {
            best_nb_visits = root.nodes[k].nb_visits;
            res = root.nodes[k].action;
        }
        ++k;
        if (k == nb_children) {
            k = 0;
        }
    }
    return res;
}
