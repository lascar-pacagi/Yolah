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
        7. Influence: number of squares closer to us moving one square in each direction.
        8. Freedom: number of free squares around a stone. 
    */
    enum {
        NO_MOVE_WEIGHT,
        NB_MOVES_OPENING_WEIGHT,
        NB_MOVES_MIDDLE_WEIGHT,
        NB_MOVES_END_WEIGHT,
        CONNECTIVITY_OPENING_WEIGHT,
        CONNECTIVITY_MIDDLE_WEIGHT,
        CONNECTIVITY_END_WEIGHT,
        CONNECTIVITY_SET_OPENING_WEIGHT,
        CONNECTIVITY_SET_MIDDLE_WEIGHT,
        CONNECTIVITY_SET_END_WEIGHT,
        ALONE_OPENING_WEIGHT,
        ALONE_MIDDLE_WEIGHT,
        ALONE_END_WEIGHT,
        FIRST_OPENING_WEIGHT,
        FIRST_MIDDLE_WEIGHT,
        FIRST_END_WEIGHT,
        INFLUENCE_OPENING_WEIGHT,
        INFLUENCE_MIDDLE_WEIGHT,
        INFLUENCE_END_WEIGHT,
        FREEDOM_0_OPENING_WEIGTH,
        FREEDOM_1_OPENING_WEIGTH,
        FREEDOM_2_OPENING_WEIGTH,
        FREEDOM_3_OPENING_WEIGTH,
        FREEDOM_4_OPENING_WEIGTH,
        FREEDOM_5_OPENING_WEIGTH,
        FREEDOM_6_OPENING_WEIGTH,
        FREEDOM_7_OPENING_WEIGTH,
        FREEDOM_8_OPENING_WEIGTH,
        FREEDOM_0_MIDDLE_WEIGTH,
        FREEDOM_1_MIDDLE_WEIGTH,
        FREEDOM_2_MIDDLE_WEIGTH,
        FREEDOM_3_MIDDLE_WEIGTH,
        FREEDOM_4_MIDDLE_WEIGTH,
        FREEDOM_5_MIDDLE_WEIGTH,
        FREEDOM_6_MIDDLE_WEIGTH,
        FREEDOM_7_MIDDLE_WEIGTH,
        FREEDOM_8_MIDDLE_WEIGTH,
        FREEDOM_0_END_WEIGTH,
        FREEDOM_1_END_WEIGTH,
        FREEDOM_2_END_WEIGTH,
        FREEDOM_3_END_WEIGTH,
        FREEDOM_4_END_WEIGTH,
        FREEDOM_5_END_WEIGTH,
        FREEDOM_6_END_WEIGTH,
        FREEDOM_7_END_WEIGTH,
        FREEDOM_8_END_WEIGTH,
        NB_WEIGHTS
    };
    constexpr int16_t MAX_VALUE = 30000;
    constexpr int16_t MIN_VALUE = -MAX_VALUE;
    constexpr uint64_t MIDDLE_GAME = 36;
    constexpr uint64_t END_GAME    = 18; 
    // Weights from Noisy Cross Entropy Method:
    // Weights from Nelder Mead:
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS {
        -255.745,           // NO_MOVE_WEIGHT
        81.62132957543031,  // NB_MOVES_OPENING_WEIGHT
        91.57609768985716,  // NB_MOVES_MIDDLE_WEIGHT
        -177.6773616734466, // NB_MOVES_END_WEIGHT
        68.16486241472865,  // CONNECTIVITY_OPENING_WEIGHT
        26.70495498502099,  // CONNECTIVITY_MIDDLE_WEIGHT
        154.383379630677,   // CONNECTIVITY_END_WEIGHT
        -530.8242077186487, // CONNECTIVITY_SET_OPENING_WEIGHT
        -313.2023757332665, // CONNECTIVITY_SET_MIDDLE_WEIGHT
        -43.87849657315125, // CONNECTIVITY_SET_END_WEIGHT
        172.0314980997644,  // ALONE_OPENING_WEIGHT
        326.4480489128537,  // ALONE_MIDDLE_WEIGHT
        310.909906521125,   // ALONE_END_WEIGHT
        230.0669166683616,  // FIRST_OPENING_WEIGHT
        220.9179293732371,  // FIRST_MIDDLE_WEIGHT
        289.5340889519555,  // FIRST_END_WEIGHT
        387.3041149887017,  // INFLUENCE_OPENING_WEIGHT
        825.3446537589967,  // INFLUENCE_MIDDLE_WEIGHT
        1028.305233286851,  // INFLUENCE_END_WEIGHT
        -180.1927551775643, // FREEDOM_0_OPENING_WEIGTH
        -1069.837309882286, // FREEDOM_1_OPENING_WEIGTH
        -14.56689415550916, // FREEDOM_2_OPENING_WEIGTH
        -345.1710839652004, // FREEDOM_3_OPENING_WEIGTH
        375.3764302746833,  // FREEDOM_4_OPENING_WEIGTH
        291.2553310488145,  // FREEDOM_5_OPENING_WEIGTH
        -137.9246980219349, // FREEDOM_6_OPENING_WEIGTH
        -288.4070807306961, // FREEDOM_7_OPENING_WEIGTH
        -289.8783023241542, // FREEDOM_8_OPENING_WEIGTH
        -666.1384943743948, // FREEDOM_0_MIDDLE_WEIGTH
        -112.2225426176193, // FREEDOM_1_MIDDLE_WEIGTH
        369.0052859270172,  // FREEDOM_2_MIDDLE_WEIGTH
        552.4179521303455,  // FREEDOM_3_MIDDLE_WEIGTH
        517.3756779669266,  // FREEDOM_4_MIDDLE_WEIGTH
        -2.68655764395052,  // FREEDOM_5_MIDDLE_WEIGTH
        -725.3576607550269, // FREEDOM_6_MIDDLE_WEIGTH
        -784.0368752945366, // FREEDOM_7_MIDDLE_WEIGTH
        -1084.969212245822, // FREEDOM_8_MIDDLE_WEIGTH
        -563.2524584917178, // FREEDOM_0_END_WEIGTH
        237.5401061901083,  // FREEDOM_1_END_WEIGTH
        -108.1002928009579, // FREEDOM_2_END_WEIGTH
        177.788676880064,   // FREEDOM_3_END_WEIGTH
        277.559496857502,   // FREEDOM_4_END_WEIGTH
        -48.35322675721582, // FREEDOM_5_END_WEIGTH
        -342.6502992564743, // FREEDOM_6_END_WEIGTH
        -288.0894324261568, // FREEDOM_7_END_WEIGTH
        -705.161239285647   // FREEDOM_8_END_WEIGTH
    };
    uint64_t floodfill(uint64_t player_bb, uint64_t free);
    int16_t connectivity(uint8_t player, const Yolah& yolah);
    int16_t connectivity_set(uint64_t player_bb, uint64_t free);        
    int16_t alone(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> first(const Yolah::MoveList&, const Yolah::MoveList&);
    int16_t blocked(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> influence(const Yolah&);
    double freedom(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights);
    int16_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int16_t evaluation(uint8_t player, const Yolah&);

    std::set<int16_t> sampling_heuristic_values(size_t nb_random_games);    
}

#endif
