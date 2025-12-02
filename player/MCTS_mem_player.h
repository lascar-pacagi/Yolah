#ifndef MCTS_MEM_PLAYER_H
#define MCTS_MEM_PLAYER_H
#include "MCTS_player.h"
#include <memory_resource>

template<typename EvalLeaf = Playout>
class MCTSMemPlayer : public Player {
    std::pmr::synchronized_pool_resource resource;
    MCTSPlayer<EvalLeaf> player;    
public:
    explicit MCTSMemPlayer(uint64_t microseconds, EvalLeaf eval_leaf = EvalLeaf{})
        : MCTSMemPlayer(microseconds, std::thread::hardware_concurrency(), std::move(eval_leaf)) {
    }
    explicit MCTSMemPlayer(uint64_t microseconds, std::size_t nb_threads, EvalLeaf eval_leaf = EvalLeaf{})
        : player(microseconds, nb_threads, std::move(eval_leaf), &resource) {
    }
    uint64_t microseconds() const {
        return player.microseconds();
    }
    size_t nb_threads() const {
        return player.nb_threads();
    } 
    Move play(Yolah yolah) override {
        return player.play(yolah);
    }
    std::string info() override {
        return "mcts synchronized memory pool player";
    }
    json config() override {
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
};

#endif