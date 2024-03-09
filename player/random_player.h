#ifndef RANDOM_PLAYER_H
#define RANDOM_PLAYER_H
#include "player.h"
#include "misc.h"

class RandomPlayer : public Player {
    PRNG prng;
public:
    RandomPlayer();
    Move play(Yolah yolah) override;
};

#endif
