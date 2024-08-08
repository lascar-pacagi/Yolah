#ifndef HEURISTIC_H
#define HEURISTIC_H
#include "game.h"
#include <array>
#include <set>

namespace heuristic {
    /*
        1. No move
        2. Number of moves
        3. Connectivity: sum of squares connected to each piece.
        4. Connectivity set: sum of squares connected to each piece without counting the same square twice.
        5. Alone: number of squares owns by the player.
        6. First: number of squares we can reach first in one move.
        7. Blocked: number of pieces that cannot move.
        8. Influence: number of squares closer to us moving one square in each direction.
    */
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_WEIGHT,
        CONNECTIVITY_WEIGHT,
        CONNECTIVITY_SET_WEIGHT,
        ALONE_WEIGHT,
        FIRST_WEIGHT,
        BLOCKED_WEIGHT,
        INFLUENCE_WEIGHT,
        NB_WEIGHTS
    };
    constexpr int16_t MAX_VALUE = 30000;
    constexpr int16_t MIN_VALUE = -MAX_VALUE;
    // Weights from Noisy Cross Entropy Method: {-28.0347, 72.51912935405451, 46.21936424416053, -51.40212464523143, 120.7286565586756, 129.4251135508776, -303.5946772763174, 319.5712180572979}
    // Weights from Nelder Mead:
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-10.4281, 135.8600193093234, 25.74332821032138, -21.43164318580857, 42.39057311554222, 121.7537626624634, -244.0065210343193, 383.5705511402678};
    uint64_t floodfill(uint64_t player_bb, uint64_t free);
    int16_t connectivity(uint8_t player, const Yolah& yolah);
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
