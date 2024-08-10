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
        8. Freedom: number of squares around a stone that are free. 
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
    // Weights from Noisy Cross Entropy Method: {-38.6887, 115.9957255897555, 70.67848988704004, 56.36922097254798, 70.38977155176737, 189.6693161686363, -387.9124933123701, 514.5944087588214}
    // Weights from Nelder Mead:
    constexpr std::array<double, NB_WEIGHTS> WEIGHTS{};
    uint64_t floodfill(uint64_t player_bb, uint64_t free);
    int16_t connectivity(uint8_t player, const Yolah& yolah);
    int16_t connectivity_set(uint64_t player_bb, uint64_t free);        
    int16_t alone(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> first(const Yolah::MoveList&, const Yolah::MoveList&);
    int16_t blocked(uint8_t player, const Yolah&);
    std::pair<uint64_t, uint64_t> influence(const Yolah&);
    double freedom(uint8_t player, const Yolah&);
    int16_t eval(uint8_t player, const Yolah&, const std::array<double, NB_WEIGHTS>& weights = WEIGHTS);
    int16_t evaluation(uint8_t player, const Yolah&);

    std::set<int16_t> sampling_heuristic_values(size_t nb_random_games);    
}

#endif
