#ifndef MINMAX_PLAYER_V11_H
#define MINMAX_PLAYER_V11_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include "BS_thread_pool.h"

class MinMaxPlayerV11 : public Player {
    using heuristic_eval = std::function<int16_t(uint8_t, const Yolah&)>;
    const uint64_t thinking_time;
    TranspositionTable table;
    BS::thread_pool pool;
    heuristic_eval heuristic;
    std::atomic_bool stop = false;
    
    static constexpr size_t NB_PLIES = 128;
    struct Search {
        uint8_t depth   = 0;
        int16_t value   = 0;
        Move move       = Move::none();    
        Move killer1[NB_PLIES]{};
        Move killer2[NB_PLIES]{};
        size_t nb_nodes = 0;
        size_t nb_hits  = 0;
    };

    int16_t negamax(Yolah& yolah, Search&, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth);
    int16_t root_search(Yolah&, Search&, uint64_t hash, int8_t depth, Move&);
    void sort_moves(Yolah&, const Search& s, uint64_t hash, Yolah::MoveList&);
    void iterative_deepening(Yolah, Search&);
    void print_pv(Yolah, uint64_t hash, int8_t depth);
    
public:
    MinMaxPlayerV11(uint64_t microseconds, size_t tt_size_mb, size_t nb_threads = 1, 
                    heuristic_eval heuristic = heuristic::evaluation);
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

#endif
