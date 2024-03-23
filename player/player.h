#ifndef PLAYER_H 
#define PLAYER_H
#include "game.h"

struct Player {
    virtual ~Player() = default;
    virtual Move play(Yolah yolah) = 0;
};

#endif
