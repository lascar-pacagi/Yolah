#ifndef PLAYER_H 
#define PLAYER_H
#include "game.h"
#include <string>

struct Player {
    virtual ~Player() = default;
    virtual Move play(Yolah) = 0;
    std::string info() {
        return "";
    }
    virtual void game_over(Yolah) {
    }
};

#endif
