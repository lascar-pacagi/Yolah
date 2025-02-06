#ifndef MINMAX_NNUE_PLAYER_H
#define MINMAX_NNUE_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include "BS_thread_pool.h"
#include "nnue.h"

class MinMaxNNUEPlayer : public Player {
    const uint64_t thinking_time;
    TranspositionTable table;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    const std::string nnue_parameters_filename;
    std::vector<NNUE<4096, 64, 64>> nnues;
    BS::thread_pool pool;
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
        size_t nnue_index = 0;        
    };

    int16_t negamax(Yolah& yolah, Search&, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth);
    int16_t root_search(Yolah&, Search&, uint64_t hash, int8_t depth, Move&);
    void sort_moves(Yolah&, const Search& s, uint64_t hash, Yolah::MoveList&);
    void iterative_deepening(Yolah, Search&);
    void print_pv(Yolah, uint64_t hash, int8_t depth);
    
public:
    MinMaxNNUEPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction, 
                 const std::string& nnue_parameters_filename, size_t nb_threads = 1);
    Move play(Yolah) override;
    std::string info() override;
    json config() override;
};

#endif
