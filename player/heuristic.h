#ifndef HEURISTIC_H
#define HEURISTIC_H
#include "game.h"
#include <array>
#include <set>

namespace heuristic {
    // /*
    //     1. No move
    //     2. Mobility: sum of squares reachable by each piece.
    //     3. Mobility set: sum of squares reachable by each piece without counting the same square twice.
    //     4. Connectivity: sum of squares connected to each piece.
    //     5. Connectivity set: sum of squares connected to each piece without counting the same square twice.
    //     6. Alone: number of squares owns by the player.
    //     7. Closer: number of squares closer to us moving one square in each direction.
    //     8. First: number of squares we can reach first in one move.
    //     9. Blocked: number of pieces that cannot move.
    // */
    // enum {
    //     NO_MOVE_WEIGHT,
    //     NB_MOVES_WEIGHT,
    //     MOBILITY_WEIGHT,
    //     CONNECTIVITY_WEIGHT,
    //     CONNECTIVITY_SET_WEIGHT,
    //     ALONE_WEIGHT,
    //     CLOSER_WEIGHT,
    //     FIRST_WEIGHT,
    //     BLOCKED_WEIGHT,
    //     NB_WEIGHTS
    // };
    // constexpr int16_t MAX_VALUE = 30000;
    // constexpr int16_t MIN_VALUE = -MAX_VALUE;
    // // Weights from Noisy Cross Entropy Method: {-43.6389, 186.0898207582441, 212.3614455787967, 99.4115486162837, -26.71184784352851, 129.6799782468804, 376.060077184592, 73.21603455275256, -40.82948984807577}
    // // Weights from Nelder Mead:
    // constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-43.6389, 186.0898207582441, 212.3614455787967, 99.4115486162837, -26.71184784352851, 129.6799782468804, 376.060077184592, 73.21603455275256, -40.82948984807577};
    // int16_t mobility(const Yolah::MoveList&);
    // int16_t connectivity(uint8_t player, const Yolah&);
    // int16_t connectivity_set(uint8_t player, const Yolah&);
    // int16_t alone(uint8_t player, const Yolah&);
    // int16_t closer(uint8_t player, const Yolah&);
    // int16_t first(const Yolah::MoveList&, const Yolah::MoveList&);
    // int16_t blocked(uint8_t player, const Yolah&);
    // int16_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    // int16_t evaluation(uint8_t player, const Yolah&);

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
    // enum {
    //     NO_MOVE_WEIGHT,
    //     NB_MOVES_WEIGHT,
    //     MOBILITY_WEIGHT,
    //     CONNECTIVITY_WEIGHT,
    //     CONNECTIVITY_SET_WEIGHT,
    //     ALONE_WEIGHT,
    //     CLOSER_WEIGHT,
    //     FIRST_WEIGHT,
    //     BLOCKED_WEIGHT,
    //     NB_WEIGHTS
    // };
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_WEIGHT,
        CONNECTIVITY_SET_WEIGHT,
        FIRST_WEIGHT,
        BLOCKED_WEIGHT,
        NB_WEIGHTS
    };
    constexpr int16_t MAX_VALUE = 30000;
    constexpr int16_t MIN_VALUE = -MAX_VALUE;
    // Weights from Noisy Cross Entropy Method: {-43.6389, 186.0898207582441, 212.3614455787967, 99.4115486162837, -26.71184784352851, 129.6799782468804, 376.060077184592, 73.21603455275256, -40.82948984807577}
    // Weights from Nelder Mead:
    //constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-43.6389, 186.0898207582441, 212.3614455787967, 99.4115486162837, -26.71184784352851, 129.6799782468804, 376.060077184592, 73.21603455275256, -40.82948984807577};
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{-1, 1, 1, 1, -1};
    // int16_t mobility(const Yolah::MoveList&);
    // int16_t connectivity(uint8_t player, const Yolah&);
    // int16_t connectivity_set(uint8_t player, const Yolah&);
    // int16_t alone(uint8_t player, const Yolah&);
    // int16_t closer(uint8_t player, const Yolah&);
    int16_t first(const Yolah::MoveList&, const Yolah::MoveList&);
    int16_t blocked(uint8_t player, const Yolah&);
    int16_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int16_t evaluation(uint8_t player, const Yolah&);

    std::set<int16_t> sampling_heuristic_values(size_t nb_random_games);    
}

#endif
