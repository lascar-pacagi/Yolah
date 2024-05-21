#ifndef HEURISTIC_H
#define HEURISTIC_H
#include "game.h"
#include <array>

namespace heuristic {
    /*
        1. No move
        2. Mobility: sum of squares reachable by each piece.
        3. Mobility set: sum of squares reachable by each piece without counting the same square twice.
        4. Connectivity: sum of squares connected to each piece.
        5. Connectivity set: sum of squares connected to each piece without counting the same square twice.
        6. Alone: number of squares owns by the player.
    */
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_WEIGHT,
        MOBILITY_WEIGHT,
        CONNECTIVITY_WEIGHT,
        CONNECTIVITY_SET_WEIGHT,
        ALONE_WEIGHT,
        NB_WEIGHTS
    };
    constexpr int32_t MAX_VALUE = 1000000;
    constexpr int32_t MIN_VALUE = -MAX_VALUE;
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{36.6023, 65.04737770450171, 554.1119972762815, 86.54276988986744, 42.05588289904526, 69.83660444100575};
    int32_t mobility(const Yolah::MoveList&);
    int32_t connectivity(uint8_t player, const Yolah&);
    int32_t connectivity_set(uint8_t player, const Yolah&);
    int32_t alone(uint8_t player, const Yolah&);
    int32_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int32_t evaluation(uint8_t player, const Yolah&);
}

#endif
