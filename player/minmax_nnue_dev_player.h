#ifndef MINMAX_NNUE_DEV_PLAYER_H
#define MINMAX_NNUE_DEV_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include <atomic>
#include <vector>
#include "nnue_quantized.h"
#include "endgame_solver.h"

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
//      ("aspiration window").
//   C. move ordering: history heuristic ("history"), countermoves
//      ("countermove"), root moves ordered by the previous iteration
//      ("root ordering").
//   D. lazy NNUE accumulators ("lazy accumulator"): a stack of accumulators,
//      one per ply, computed only when a leaf is evaluated; no undo.
//      Evaluation cache ("eval cache"): leaf values by position hash.
//   E. late move reductions growing with log(depth)·log(move number), smaller
//      for PV nodes, killers / countermoves and moves with a good history
//      ("lmr", "lmr base", "lmr divisor"); the reference's fixed reduction
//      ("nb moves at full depth", "late move reduction") when "lmr" is false.
//   G. exact endgame solver (player/endgame_solver.h): at the root, a win /
//      draw / loss proof when few free squares remain ("endgame root",
//      "endgame root time"); in the tree, exact values instead of the
//      network's for the nodes with very few free squares ("endgame tree").
// Each improvement can be switched off in the config.

// The switches of the improvements (config keys in brackets), so that each
// one can be measured on its own. (Outside the class: a nested struct with
// default member initializers cannot be a default argument of the class.)
struct MinMaxNNUE_DevOptions {
    bool pvs = true;                 // ["pvs"] null windows for all moves but the first
    int  aspiration_window = 300;    // ["aspiration window"] half width at the root, 0 = full window
    bool history = true;             // ["history"] quiet moves ordered by their history of cutoffs
    bool countermove = true;         // ["countermove"] the move that refuted the opponent's last move
    bool root_ordering = true;       // ["root ordering"] root moves: best first, then by subtree size
    bool lazy_accumulator = true;    // ["lazy accumulator"] NNUE accumulators updated only when needed
    int  eval_cache_bits = 20;       // ["eval cache"] 2^bits cached leaf values (8 bytes each), 0 = none
    bool lmr = true;                 // ["lmr"] logarithmic late move reductions (false: the reference's)
    double lmr_base = 1.0;           // ["lmr base"]    reduction = base + log(depth)·log(move number) / divisor
    double lmr_divisor = 1.75;       // ["lmr divisor"]
    int  endgame_root = 0;           // ["endgame root"] try to prove the result at the root with at most
                                     //   this many free squares (0 = never)
    double endgame_root_time = 0.5;  // ["endgame root time"] share of the thinking time given to that proof
    int  endgame_tree = 0;           // ["endgame tree"] solve exactly the nodes with at most this many
                                     //   free squares (0 = never)
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

    // ── Move ordering (C) ──
    // History: for each colour and each (from, to), a score that goes up when
    // the move causes a beta cutoff and down when it was tried before the
    // move that did. It is kept between the moves of a game (halved at each
    // new search: old knowledge fades). Bounded in [−HISTORY_MAX, HISTORY_MAX]
    // by the "gravity" update of update_history.
    static constexpr int HISTORY_MAX = 16384;
    // Move scores of score_moves: the special moves above any history score.
    static constexpr int SCORE_TT = 1 << 30, SCORE_KILLER1 = 1 << 29,
                         SCORE_KILLER2 = SCORE_KILLER1 - 1, SCORE_COUNTER = SCORE_KILLER1 - 2;
    // Late move reductions (E): base reduction for (depth, move number), in
    // 1/1024 of a ply so that the adjustments can be fractional.
    int lmr_table[64][Yolah::MAX_NB_MOVES]{};
    int16_t history[2][SQUARE_NB][SQUARE_NB]{};
    // Countermove: for each opponent move (from, to), the last move that
    // refuted it (caused a cutoff right after it).
    Move countermoves[SQUARE_NB][SQUARE_NB]{};

    struct RootMove {
        Move     move;
        uint64_t nodes = 0;       // size of its subtree in the last iteration
    };

    struct Search {
        uint8_t depth   = 0;
        int16_t value   = 0;
        Move move       = Move::none();
        Move killer1[Yolah::MAX_NB_PLIES]{};
        Move killer2[Yolah::MAX_NB_PLIES]{};
        size_t nb_nodes = 0;
        size_t nb_hits  = 0;
        Move played[Yolah::MAX_NB_PLIES + 1]{};   // played[p]: the move played at ply p (to find the countermove)
        std::vector<RootMove> root_moves;
        // Without the lazy accumulators: ONE accumulator, updated by
        // nnue.play / nnue.undo around every move (the reference's way).
        NNUE_Quantized::Accumulator acc;
        // With them (D): accs[p] is the accumulator of the position at ply p
        // of the current line, valid only if acc_ok[p]. played[p] (above)
        // tells how to get accs[p + 1] from accs[p].
        NNUE_Quantized::Accumulator accs[Yolah::MAX_NB_PLIES + 1];
        bool acc_ok[Yolah::MAX_NB_PLIES + 1]{};
    };

    // Evaluation cache (D): one entry per slot, the newest wins. 32 bits of
    // the hash check the position (the other bits choose the slot).
    struct EvalEntry {
        uint32_t key = 0;
        int16_t  value = 0;
        bool     used = false;
    };
    std::vector<EvalEntry> eval_cache;
    // The exact endgame solver (G), with its own table, kept between moves.
    EndgameSolver endgame;
    uint64_t nb_evals = 0, nb_eval_hits = 0;

    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    int  evaluate(const Yolah& yolah, Search& s, uint64_t hash);
    int  network_value(const Yolah& yolah, Search& s);
    void update_accumulator(const int16_t* in, int16_t* out, uint8_t player, Move m) const;
    int  negamax(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);
    int  root_search(Yolah&, Search&, uint64_t hash, int alpha, int beta, int depth, Move&);
    int  late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const;
    int  search_move(Yolah&, Search&, uint64_t hash, Move m, size_t i, int reduction, int alpha, int beta, int depth);
    void score_moves(const Yolah&, const Search& s, Move tt_move, const Yolah::MoveList&, int* scores) const;
    static Move pick_move(Yolah::MoveList&, int* scores, size_t i);
    void update_history(uint8_t player, Move m, int bonus);
    void iterative_deepening(Yolah, Search&, uint8_t max_depth);
    void print_pv(Yolah, uint64_t hash, int8_t depth);

public:
    MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction,
                         const std::string& nnue_q_parameters_filename, bool verbose = false, Options options = {});
    Move play(Yolah) override;
    // Iterative deepening up to max_depth, stopped after `microseconds`
    // (0 = no time limit). For the benchmarks (test/search_bench_main.cpp).
    Result search(const Yolah&, uint8_t max_depth, uint64_t microseconds);
    // Leaf evaluations and how many came from the cache, since the start.
    std::pair<uint64_t, uint64_t> eval_stats() const { return {nb_evals, nb_eval_hits}; }
    void clear_table();            // forget everything: table, history, countermoves (reproducible benchmarks)
    std::string info() override;
    json config() override;
};

#endif
