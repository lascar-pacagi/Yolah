#ifndef MINMAX_PLAYER_V8_H
#define MINMAX_PLAYER_V8_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>

class MinMaxPlayerV8 : public Player {
    using heuristic_eval = std::function<int16_t(uint8_t, const Yolah&)>;
    const uint64_t thinking_time;
    TranspositionTable table;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    heuristic_eval heuristic;
    std::atomic_bool stop = false;
    size_t nb_nodes = 0;
    size_t nb_hits  = 0;
    static constexpr size_t NB_PLIES = 128;
    Move killer1[NB_PLIES]{};
    Move killer2[NB_PLIES]{};
    static constexpr int16_t HISTORY_MAX = 10000;
    int16_t history[SQUARE_NB][SQUARE_NB]{};
    
    int16_t negamax(Yolah& yolah, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth);
    int16_t root_search(Yolah&, uint64_t hash, int8_t depth, Move&);
    void sort_moves(Yolah&, uint64_t hash, Yolah::MoveList&);
    Move iterative_deepening(Yolah&);
    void print_pv(Yolah, uint64_t hash, int8_t depth);
    
public:
    MinMaxPlayerV8(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction, 
                   heuristic_eval heuristic = heuristic::evaluation);
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

#endif
