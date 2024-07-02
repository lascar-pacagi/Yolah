#include "MCTS_mem_player.h"

Move MCTSMemPlayer::play(Yolah yolah) {
    return player.play(yolah);
}

std::string MCTSMemPlayer::info() {
    return "mcts synchronized memory pool player";
}
