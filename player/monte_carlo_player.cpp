#include "monte_carlo_player.h"
#include <chrono>
#include <future>
#include <algorithm>

using std::size_t;

namespace {
    thread_local PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
}

MonteCarloPlayer::MonteCarloPlayer(size_t nb_iter) : MonteCarloPlayer(nb_iter, std::thread::hardware_concurrency()) {
}

MonteCarloPlayer::MonteCarloPlayer(size_t nb_iter, size_t nb_threads) 
    : nb_iter(nb_iter), pool(nb_threads) {
}

uint64_t MonteCarloPlayer::random_game(Yolah& yolah, uint8_t player) {
    Yolah::MoveList moves;
    while (!yolah.game_over()) {                 
        yolah.moves(moves);
        if (moves.size() == 0) continue;
        Move m = moves[prng.rand<size_t>() % moves.size()];
        yolah.play(m);
    }
    const auto [black_score, white_score] = yolah.score();
    return player == Yolah::BLACK ? black_score : white_score;
}

Move MonteCarloPlayer::play(Yolah yolah) {
    uint8_t player = yolah.current_player();
    Yolah::MoveList moves;
    yolah.moves(moves);
    BS::multi_future<uint64_t> futures = pool.submit_sequence<size_t>(0, moves.size(), [&](size_t i) { 
        uint64_t res = 0;
        for (size_t iter = 0; iter < nb_iter; iter++) {
            Yolah y = yolah;
            y.play(moves[i]);
            res += random_game(y, player);
        }
        return res;
    });
    std::vector<uint64_t> action_values = futures.get();
    auto pos = std::distance(begin(action_values), std::max_element(begin(action_values), end(action_values)));
    return moves[pos];
}
