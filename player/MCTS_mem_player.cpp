#include "MCTS_mem_player.h"

Move MCTSMemPlayer::play(Yolah yolah) {
    return player.play(yolah);
}

std::string MCTSMemPlayer::info() {
    return "mcts synchronized memory pool player";
}

json MCTSMemPlayer::config() {
    json j;
    j["name"] = "MCTSMemPlayer";
    j["microseconds"] = player.microseconds();
    if (player.nb_threads() == std::thread::hardware_concurrency()) {
        j["nb threads"] = "hardware concurrency";
    } else {
        j["nb threads"] = player.nb_threads();
    }
    return j;
}
