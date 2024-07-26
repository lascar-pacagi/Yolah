#ifndef HEURISTIC_H
#define HEURISTIC_H
#include "game.h"
#include <array>
#include <set>

namespace heuristic {
    /*
        1. No move
        2. Number of moves
        3. Connectivity set: sum of squares connected to each piece without counting the same square twice.
        4. Alone: number of squares owns by the player.
        5. First: number of squares we can reach first in one move.
        6. Blocked: number of pieces that cannot move.
        7. Influence: number of squares closer to us moving one square in each direction.
    */
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_WEIGHT,
        CONNECTIVITY_SET_WEIGHT,
        ALONE_WEIGHT,
        FIRST_WEIGHT,
        BLOCKED_WEIGHT,
        INFLUENCE_WEIGHT,
        NB_WEIGHTS
    };
    constexpr int16_t MAX_VALUE = 30000;
    constexpr int16_t MIN_VALUE = -MAX_VALUE;
    // Weights from Noisy Cross Entropy Method: {-56.2339, 144.1800889829667, 39.21721908854634, 63.51208868079438, 47.71966011602488, -175.0342140878195, 176.1105939010502}
    // Weights from Nelder Mead:
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-56.2339, 144.1800889829667, 39.21721908854634, 63.51208868079438, 47.71966011602488, -175.0342140878195, 176.1105939010502};
    uint64_t floodfill(uint64_t player_bb, uint64_t free);
    int16_t connectivity_set(uint64_t player_bb, uint64_t free);    
    int16_t alone(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> first(const Yolah::MoveList&, const Yolah::MoveList&);
    int16_t blocked(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> influence(const Yolah&);
    int16_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int16_t evaluation(uint8_t player, const Yolah&);

    std::set<int16_t> sampling_heuristic_values(size_t nb_random_games);    
}

#endif
