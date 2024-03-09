#include "random_player.h"
#include <chrono>

RandomPlayer::RandomPlayer() : prng(std::chrono::system_clock::now().time_since_epoch().count()) {
}

Move RandomPlayer::play(Yolah yolah) {
    if (yolah.game_over()) return Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    if (moves.size() == 0) return Move::none();
    return moves[prng.rand<std::size_t>() % moves.size()];
}
