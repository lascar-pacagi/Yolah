#ifndef MINMAX_PLAYER_H
#define MINMAX_PLAYER_H
#include "player.h"
#include "heuristic.h"

class MinMaxPlayer : public Player {
    const uint64_t thinking_time;
    // int32_t negamax(Yolah& yolah, int32_t alpha, int32_t beta, uint16_t depth);
    // int32_t search(Yolah&, Move&);
    // void sort_moves(Yolah&, Yolah::MoveList&);
public:
    //MinMaxPlayer(uint64_t microseconds, std::size_t nb_threads = 1);
    Move play(Yolah) override;
};

#endif