#ifndef PLAYER_H 
#define PLAYER_H
#include "game.h"
#include <string>

struct Player {
    virtual ~Player() = default;
    virtual Move play(Yolah) = 0;
    virtual std::string info() {
        return "";
    }
    virtual void game_over(Yolah) {
    }
    static std::unique_ptr<Player> create(const json&);
};

#endif
