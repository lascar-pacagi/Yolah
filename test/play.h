#ifndef PLAY_H
#define PLAY_H
#include "player.h"
#include <memory>

namespace test {
    void play(std::unique_ptr<Player> player1, std::unique_ptr<Player> player2);    
    void play(std::unique_ptr<Player> player1, std::unique_ptr<Player> player2, size_t nb_games);
}

#endif
