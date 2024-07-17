#ifndef MINMAX_PLAYER_H
#define MINMAX_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
//#include "tt.h"
#include <atomic>

class MinMaxPlayerV1 : public Player {
    using heuristic_eval = std::function<int32_t(uint8_t, const Yolah&)>;
    const uint64_t thinking_time;
    TranspositionTable table;
    uint8_t R;
    heuristic_eval heuristic;
    uint64_t hash = 0;
    std::atomic_bool stop = false;
    size_t nb_nodes = 0;
    size_t nb_hits  = 0;

    int32_t negamax(Yolah& yolah, uint64_t hash, int32_t alpha, int32_t beta, uint8_t depth);
    int32_t root_search(Yolah&, uint64_t hash, uint8_t depth, Move&);
    void sort_moves(Yolah&, uint64_t hash, Yolah::MoveList&);
    Move iterative_deepening(Yolah&);
    void print_pv(Yolah, uint64_t hash, uint8_t depth);
public:
    MinMaxPlayerV1(uint64_t microseconds, size_t tt_size_mb/*nb_bits*/, uint8_t R = 3, heuristic_eval heuristic = heuristic::evaluation);
    Move play(Yolah) override;
    std::string info() override;
};

#endif
