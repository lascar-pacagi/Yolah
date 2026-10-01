#ifndef MINMAX_NNUE_DEV_PLAYER_H
#define MINMAX_NNUE_DEV_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include "nnue_quantized.h"

// The search being improved, step by step. It started as an exact copy of
// MinMaxNNUE_BaselinePlayer (the reference, which never changes); every change
// is measured against it with test/search_bench_main.cpp and matches.
//
// Changes so far (see the .cpp for the details):
//   A. fixes: no evaluation stored in the transposition table (those entries
//      were invisible and evicted real ones); nothing stored once the clock has
//      stopped the search; the best move of an interrupted iteration is kept;
//      killer moves only come from beta cutoffs; fail-soft values.
//   B. principal variation search ("pvs") and aspiration windows at the root
//      ("aspiration window"), each one can be switched off in the config.

// The switches of the improvements (config keys in brackets), so that each
// one can be measured on its own. (Outside the class: a nested struct with
// default member initializers cannot be a default argument of the class.)
struct MinMaxNNUE_DevOptions {
    bool pvs = true;                 // ["pvs"] null windows for all moves but the first
    int  aspiration_window = 300;    // ["aspiration window"] half width at the root, 0 = full window
};

class MinMaxNNUE_DevPlayer : public Player {
public:
    struct Result {
        Move     move     = Move::none();
        int16_t  value    = 0;
        uint8_t  depth    = 0;     // last fully searched depth
        uint64_t nb_nodes = 0;
        uint64_t nb_hits  = 0;
        double   seconds  = 0;
    };

    using Options = MinMaxNNUE_DevOptions;

private:
    // Search values are computed in int (no overflow when a window is widened
    // or shifted by one) and stored in the table as int16.
    static constexpr int INFINITE = 32000;           // > any game value
    static constexpr int WIN = heuristic::MAX_VALUE; // a finished game is worth ±(WIN + score)

    const uint64_t thinking_time;
    TranspositionTable table;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    const std::string nnue_q_parameters_filename;
    NNUE_Quantized nnue;
    Options options;
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

    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    int  negamax(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);
    int  root_search(Yolah&, Search&, uint64_t hash, int alpha, int beta, int depth, Move&);
    int  search_move(Yolah&, Search&, uint64_t hash, Move m, size_t i, int alpha, int beta, int depth);
    void sort_moves(Yolah&, const Search& s, Move tt_move, Yolah::MoveList&);
    void iterative_deepening(Yolah, Search&, uint8_t max_depth);
    void print_pv(Yolah, uint64_t hash, int8_t depth);

public:
    MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction,
                         const std::string& nnue_q_parameters_filename, bool verbose = false, Options options = {});
    Move play(Yolah) override;
    // Iterative deepening up to max_depth, stopped after `microseconds`
    // (0 = no time limit). For the benchmarks (test/search_bench_main.cpp).
    Result search(const Yolah&, uint8_t max_depth, uint64_t microseconds);
    void clear_table();            // forget everything (reproducible benchmarks)
    std::string info() override;
    json config() override;
};

#endif
