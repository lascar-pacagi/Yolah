#ifndef PLAYER_H 
#define PLAYER_H
#include "game.h"

struct Player {
    virtual ~Player() = default;
    virtual Move play(Yolah) = 0;
    virtual void game_over(Yolah) {
    }
};

#endif
