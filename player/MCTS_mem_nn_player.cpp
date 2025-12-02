#include "MCTS_mem_nn_player.h"

std::string MCTSMemNNPlayer::info() {
    return "mcts synchronized memory pool with neural network player";
}

json MCTSMemNNPlayer::config() {
    json j;
    j["name"] = "MCTSMemNNPlayer";
    j["microseconds"] = player.microseconds();
    if (player.nb_threads() == std::thread::hardware_concurrency()) {
        j["nb threads"] = "hardware concurrency";
    } else {
        j["nb threads"] = player.nb_threads();
    }
    j["weights"] = nnue_parameters_filename;
    return j;
}