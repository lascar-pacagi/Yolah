#ifndef MINMAX_NNUE_BASELINE_PLAYER_H
#define MINMAX_NNUE_BASELINE_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include "nnue_quantized.h"

// The reference ("étalon") for the work on the search algorithm: the search
// of MinMaxNNUE_QuantizedPlayer, copied as is, on ONE thread (no lazy SMP).
// Do not change its behaviour: every improvement goes into
// MinMaxNNUE_DevPlayer and is measured against this one.
class MinMaxNNUE_BaselinePlayer : public Player {
public:
    struct Result {
        Move     move     = Move::none();
        int16_t  value    = 0;
        uint8_t  depth    = 0;     // last fully searched depth
        uint64_t nb_nodes = 0;
        uint64_t nb_hits  = 0;
        double   seconds  = 0;
    };

private:
    const uint64_t thinking_time;
    TranspositionTable table;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    const std::string nnue_q_parameters_filename;
    NNUE_Quantized nnue;
    bool verbose;
    std::atomic_bool stop = false;

    struct Search {
        uint8_t depth   = 0;
        int16_t value   = 0;
        Move move       = Move::none();
        Move killer1[Yolah::MAX_NB_PLIES]{};
        Move killer2[Yolah::MAX_NB_PLIES]{};
        size_t nb_nodes = 0;
        size_t nb_hits  = 0;
        NNUE_Quantized::Accumulator acc;
    };

    int16_t negamax(Yolah& yolah, Search&, uint64_t hash, int16_t alpha, int16_t beta, int8_t depth);
    int16_t root_search(Yolah&, Search&, uint64_t hash, int8_t depth, Move&);
    void sort_moves(Yolah&, const Search& s, uint64_t hash, Yolah::MoveList&);
    void iterative_deepening(Yolah, Search&, uint8_t max_depth);
    void print_pv(Yolah, uint64_t hash, int8_t depth);

public:
    MinMaxNNUE_BaselinePlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction,
                              const std::string& nnue_q_parameters_filename, bool verbose = false);
    Move play(Yolah) override;
    // Iterative deepening up to max_depth, stopped after `microseconds`
    // (0 = no time limit). For the benchmarks (test/search_bench_main.cpp).
    Result search(const Yolah&, uint8_t max_depth, uint64_t microseconds);
    void clear_table();            // forget everything (reproducible benchmarks)
    std::string info() override;
    json config() override;
};

#endif
