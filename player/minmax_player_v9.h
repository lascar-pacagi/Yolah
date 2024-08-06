#ifndef MINMAX_PLAYER_V9_H
#define MINMAX_PLAYER_V9_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include "BS_thread_pool.h"

class MinMaxPlayerV9 : public Player {
    using heuristic_eval = std::function<int16_t(uint8_t, const Yolah&)>;
    const uint64_t thinking_time;
    TranspositionTable table;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    BS::thread_pool pool;
    heuristic_eval heuristic;
    std::atomic_bool stop = false;
    
    struct Search {
        uint8_t depth   = 0;
        int16_t value   = 0;
        Move move       = Move::none();
    };

    int16_t negamax(Yolah& yolah, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth);
    int16_t root_search(Yolah&, uint64_t hash, int8_t depth, Move&);
    void sort_moves(Yolah&, uint64_t hash, Yolah::MoveList&);
    void iterative_deepening(Yolah, Search&);
    void print_pv(Yolah, uint64_t hash, int8_t depth);
    
public:
    MinMaxPlayerV9(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction, 
                   size_t nb_threads = 1, heuristic_eval heuristic = heuristic::evaluation);
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

#endif
