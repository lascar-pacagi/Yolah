#include "monte_carlo_player.h"
#include <chrono>
#include <future>
#include <algorithm>
#include <ratio>

using std::size_t;

namespace {
    thread_local PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
}

MonteCarloPlayer::MonteCarloPlayer(uint64_t microseconds) : MonteCarloPlayer(microseconds, std::thread::hardware_concurrency()) {
}

MonteCarloPlayer::MonteCarloPlayer(uint64_t microseconds, size_t nb_threads) 
    : thinking_time(microseconds), pool(nb_threads) {
}

int16_t MonteCarloPlayer::random_game(Yolah& yolah, uint8_t player) {
    Yolah::MoveList moves;
    while (!yolah.game_over()) {                 
        yolah.moves(moves);
        if (moves.size() == 0) continue;
        Move m = moves[prng.rand<size_t>() % moves.size()];
        yolah.play(m);
    }
    //const auto [black_score, white_score] = yolah.score();
    //return player == Yolah::BLACK ? black_score : white_score;
    return yolah.score(player);
}

// Move MonteCarloPlayer::play(Yolah yolah) {
//     using namespace std::chrono;
//     uint8_t player = yolah.current_player();
//     Yolah::MoveList moves;
//     yolah.moves(moves);
//     BS::multi_future<double> futures = pool.submit_sequence<size_t>(0, moves.size(), [&](size_t i) { 
//         double res = 0;
//         steady_clock::time_point t1 = steady_clock::now();
//         duration<uint64_t, std::micro> mu;
//         uint32_t n = 0;
//         do {
//             Yolah y = yolah;
//             y.play(moves[i]);
//             res += random_game(y, player);
//             n++;
//             mu = duration_cast<microseconds>(steady_clock::now() - t1);
//         } while (mu.count() < thinking_time);
//         return res / n;
//     });
//     std::vector<double> action_values = futures.get();
//     auto pos = std::distance(begin(action_values), std::max_element(begin(action_values), end(action_values)));
//     return moves[pos];
// }

Move MonteCarloPlayer::play(Yolah yolah) {
    using namespace std::chrono;
    using std::vector, std::pair;
    uint8_t player = yolah.current_player();
    Yolah::MoveList moves;
    yolah.moves(moves);
    BS::multi_future<vector<double>> futures = pool.submit_sequence<size_t>(0, pool.get_thread_count(), [&](size_t) { 
        vector<double> res(moves.size());
        steady_clock::time_point t1 = steady_clock::now();
        duration<uint64_t, std::micro> mu;
        do {
            for (size_t i = 0; i < moves.size(); i++) {
                Yolah y = yolah;
                y.play(moves[i]);
                res[i] += random_game(y, player);
            }       
            mu = duration_cast<microseconds>(steady_clock::now() - t1);
        } while (mu.count() < thinking_time);
        return res;
    });
    vector<double> action_values(moves.size());
    for (const auto& v: futures.get()) {
        for (size_t i = 0; i < moves.size(); i++) {
            action_values[i] += v[i];
        }        
    }
    auto pos = std::distance(begin(action_values), std::max_element(begin(action_values), end(action_values)));
    return moves[pos];
}

std::string MonteCarloPlayer::info() {
    return "monte carlo player";
}

json MonteCarloPlayer::config() {
    json j;
    j["name"] = "MonteCarloPlayer";
    j["microseconds"] = thinking_time;
    if (pool.get_thread_count() == std::thread::hardware_concurrency()) {
        j["nb threads"] = "hardware concurrency";
    } else {
        j["nb threads"] = pool.get_thread_count();
    } 
    return j;
}
