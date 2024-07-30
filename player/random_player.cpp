#include "random_player.h"
#include <chrono>

RandomPlayer::RandomPlayer() : prng(std::chrono::system_clock::now().time_since_epoch().count()) {
}

RandomPlayer::RandomPlayer(uint64_t seed) : prng(seed), init_with_clock(false) {
}

Move RandomPlayer::play(Yolah yolah) {
    if (yolah.game_over()) return Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    if (moves.size() == 0) return Move::none();
    return moves[prng.rand<std::size_t>() % moves.size()];
}

std::string RandomPlayer::info() {
    return "random player";
}

json RandomPlayer::config() {
    json j;
    j["name"] = "RandomPlayer";
    if (init_with_clock) {
        j["seed"] = "clock";
    } else {
        j["seed"] = prng.seed();
    }
    return j;
}
