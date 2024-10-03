#ifndef GENERATE_GAMES_H
#define GENERATE_GAMES_H
#include <iostream>
#include <memory>
#include "player.h"

namespace data {
    void generate_games(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, size_t nb_random_moves, size_t nb_games_per_thread, size_t nb_threads = 1);
}

#endif