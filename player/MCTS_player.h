#ifndef _MCTS_PLAYER_H
#define _MCTS_PLAYER_H
#include "player.h"
#include <atomic>
#include <memory_resource>

class MCTSPlayer : public Player {
    struct Node {
        std::atomic<uint32_t> nb_visits;
        std::atomic<float> value;
        Move action;
        std::pmr::vector<Node*> nodes;
    };
    Node root;  
public:
    Move play(Yolah) override;
};

#endif