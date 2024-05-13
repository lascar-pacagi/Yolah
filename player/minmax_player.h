#ifndef MINMAX_PLAYER_H
#define MINMAX_PLAYER_H
#include "player.h"
#include "heuristic.h"

class MinMaxPlayer : public Player {

public:
    Move play(Yolah) override;
};

#endif