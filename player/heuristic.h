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
        7. Closer: number of squares closer to us moving one square in each direction.
        8. First: number of squares we can reach first in one move.
        9. Blocked: number of pieces that cannot move.
    */
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_WEIGHT,
        MOBILITY_WEIGHT,
        CONNECTIVITY_WEIGHT,
        CONNECTIVITY_SET_WEIGHT,
        ALONE_WEIGHT,
        CLOSER_WEIGHT,
        FIRST_WEIGHT,
        BLOCKED_WEIGHT,
        NB_WEIGHTS
    };
    constexpr int32_t MAX_VALUE = 1000000;
    constexpr int32_t MIN_VALUE = -MAX_VALUE;
    // Weights from Noisy Cross Entropy Method: {-997.806, 996.6839706629841, 721.546333616339, 300.9970181811095, -546.8805808633228, 762.7951319268694, 3038.873560054843, -30.41860359538886, -1057.454889954856}
    // Weights from Nelder Mead:
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-997.806, 996.6839706629841, 721.546333616339, 300.9970181811095, -546.8805808633228, 762.7951319268694, 3038.873560054843, -30.41860359538886, -1057.454889954856};
    int32_t mobility(const Yolah::MoveList&);
    int32_t connectivity(uint8_t player, const Yolah&);
    int32_t connectivity_set(uint8_t player, const Yolah&);
    int32_t alone(uint8_t player, const Yolah&);
    int32_t closer(uint8_t player, const Yolah&);
    int32_t first(const Yolah::MoveList&, const Yolah::MoveList&);
    int32_t blocked(uint8_t player, const Yolah&);
    int32_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int32_t evaluation(uint8_t player, const Yolah&);
}

#endif
