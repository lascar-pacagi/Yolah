#ifndef MCTS_MEM_PLAYER_H
#define MCTS_MEM_PLAYER_H
#include "MCTS_player.h"
#include <memory_resource>

class MCTSMemPlayer : public Player {
    std::pmr::synchronized_pool_resource resource;
    MCTSPlayer player;    
public:
    explicit MCTSMemPlayer(uint64_t microseconds)
        : MCTSMemPlayer(microseconds, std::thread::hardware_concurrency()) {
    }
    explicit MCTSMemPlayer(uint64_t microseconds, std::size_t nb_threads) : player(microseconds, nb_threads, &resource) {        
    } 
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

#endif