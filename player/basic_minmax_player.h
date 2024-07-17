#ifndef BASIC_MINMAX_PLAYER_H
#define BASIC_MINMAX_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include <functional>

class BasicMinMaxPlayer : public Player {
    using heuristic_eval = std::function<int32_t(uint8_t, const Yolah&)>;
    uint8_t depth;
    heuristic_eval heuristic;
    size_t nb_nodes = 0;
    int32_t negamax(Yolah& yolah, int32_t alpha, int32_t beta, uint8_t depth);
    int32_t search(Yolah&, Move&);
    void sort_moves(Yolah&, Yolah::MoveList&);
public:
    BasicMinMaxPlayer(uint8_t depth, heuristic_eval = heuristic::evaluation);
    Move play(Yolah) override;
    std::string info() override;
};

#endif
