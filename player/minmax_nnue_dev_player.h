#ifndef MINMAX_NNUE_DEV_PLAYER_H
#define MINMAX_NNUE_DEV_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include "search_table.h"
#include <memory>
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
//   F. pruning at non-PV nodes, from the static evaluation (the network's
//      value of the node itself): reverse futility pruning ("rfp depth",
//      "rfp margin"), null move pruning ("null move", "null move reduction"),
//      late move pruning ("lmp depth", "lmp moves"). Measured: reverse
//      futility +76 Elo, late move pruning +35, both +95; null move −81
//      (Yolah is full of zugzwangs: every move spoils one's own space), off.
//   H. a transposition table designed for Yolah (player/search_table.h):
//      both bounds per entry, replacement of the positions that can no longer
//      be reached, 64-byte clusters, prefetch ("yolah table"). Off: no gain
//      measured at 0.2 s/move (−15 ± 23 Elo), to try again at longer times.
//   I. articulation moves: a move whose destination cuts its region of free
//      squares in two (local test, see is_articulation_move) is treated like
//      a capture in chess: ordered early ("articulation ordering"), never
//      reduced nor pruned ("articulation lmr"). Off: −80 to −110 Elo — the
//      local test flags up to 40 % of the free squares, and even the exact
//      articulation points are a third of them in the middle game (mostly
//      cutting off small pockets): too many "tactical" moves.
//   J. territory ordering: at deep nodes, the quiet moves are also ordered by
//      the territory they leave (influence = king-step Voronoi, or queen
//      distance), see territory_after ("territory ordering", "territory
//      depth", "territory weight"). Measured (2000 games, 0.2 s/move):
//      influence weight 512 +36 Elo [+9, +59] (the default), influence 2048
//      +21, mobility 512 +12, queen distance 128 +1.
//   K. staged move generation ("staged"): the table's move is searched before
//      the other moves are even generated (see negamax, step 4). Same tree in
//      pure alpha-beta (209/209 values); with reductions −4 % nodes, −6 % time
//      (the other moves are then scored with fresher killers / history).
//   G. exact endgame solver (player/endgame_solver.h): at the root, a win /
//      draw / loss proof when few free squares remain ("endgame root",
//      "endgame root time"); in the tree, exact values instead of the
//      network's for the nodes with very few free squares ("endgame tree").
//   L. lazy SMP ("nb threads"): several threads search the same root with a
//      shared transposition table and evaluation cache; each has its own
//      history, killers, accumulators and root ordering (see search()).
//   M. evaluation grain ("eval grain"): the network's value rounded to a
//      multiple of G, to see whether a coarser evaluation cuts more. It does
//      (depth 13: −34 % nodes at G = 2048), but it loses (2000 games, 0.2 s):
//      G 512 +3, 1024 −5, 2048 −32, 4096 −108 Elo. Off (G = 1).
//   N. "proxy cut", a null move without the zugzwang problem: a REAL but
//      ordinary move searched at reduced depth; if even it reaches beta, cut
//      ("proxy cut" 1, "proxy rank"). Variant 2: ProbCut (the table's move
//      against beta + a margin, "probcut margin"). "proxy depth",
//      "proxy reduction". Measured (2000 games, 0.2 s/move): variant 1 with
//      the best ordinary move +47 Elo [+20, +71] (on by default; −29 % nodes
//      at depth 13), with the 4th ordinary move ±0, ProbCut −45 / −54.
//      At 1 s/move (1200 games): +59 [+30, +94] over no proxy cut; depth ≥ 3
//      (+51), R = 2 (+48) or R = 4 (+60): no difference, defaults kept.
//   O. Multi-ProbCut (Buro) without its regression ("mpc", "mpc ratio",
//      "mpc margin", "mpc margin per ply"): this position searched at a
//      shallow depth d' against beta + margin (and alpha − margin), the
//      margin set by matches as Stockfish sets its own. Off by default.
//   P. Stockfish's ProbCut generalized ("probcut" moves, "probcut margin",
//      "probcut reduction", "probcut filter"): the best-ordered moves at
//      depth − 4 against beta + margin. Off by default. O and P are the
//      references for the proxy cut study (config/search_study/).
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
    double lmr_base = 1.25;          // ["lmr base"]    reduction = base + log(depth)·log(move number) / divisor
    double lmr_divisor = 1.5;        // ["lmr divisor"]
    int  rfp_depth = 4;              // ["rfp depth"] reverse futility pruning up to this depth (0 = never)
    int  rfp_margin = 3000;          // ["rfp margin"] per ply of depth (values: ±30000 = tanh ±1)
    bool null_move = false;          // ["null move"] null move pruning (−81 Elo in a match: off)
    int  null_move_reduction = 3;    // ["null move reduction"] R at depth 4; R = this + depth / 4 − 1
    int  lmp_depth = 3;              // ["lmp depth"] late move pruning up to this depth (0 = never)
    int  lmp_moves = 4;              // ["lmp moves"] moves searched before pruning: this + depth²
    bool pass_rule = false;          // ["pass rule"] a player who must pass has lost (see negamax)
    bool articulation_ordering = false;   // ["articulation ordering"] articulation moves after the killers
    bool articulation_lmr = false;   // ["articulation lmr"] articulation moves never reduced nor pruned
    int  territory_ordering = 1;     // ["territory ordering"] 0 = off, 1 = influence (king steps),
                                     //   2 = queen distance, 3 = mobility
    int  territory_depth = 4;        // ["territory depth"] only at nodes of at least this depth
    int  territory_weight = 512;     // ["territory weight"] score = history + weight · territory
    bool fast_influence = true;      // ["fast influence"] influence_fast (same values, less work)
    int  proxy_cut = 1;              // ["proxy cut"] 0 = off, 1 = ordinary move vs beta, 2 = ProbCut
    int  proxy_rank = 0;             // ["proxy rank"] 1: which ordinary move (0 = the best-ordered one)
    int  probcut_margin = 3000;      // ["probcut margin"] 2: searched against beta + this
    int  proxy_depth = 4;            // ["proxy depth"] only at nodes of at least this depth
    int  proxy_reduction = 3;        // ["proxy reduction"] searched at depth − 1 − (this + depth / 4 − 1)
    int  proxy_delta = 0;            // ["proxy delta"] 1: the witness must reach beta + this (0: the proxy cut)
    int  proxy_witness = 0;          // ["proxy witness"] 1: 0 = best-ordered ordinary move, 1 = killer 1,
                                     //   2 = countermove, 3 = best ordinary move by history ONLY (no scoring
                                     //   of the moves: the territory term is the costly part)
    int  proxy_multi_c = 1;          // ["proxy multi c"] 1: cut if >= c of the first m witnesses reach the bound
    int  proxy_multi_m = 1;          // ["proxy multi m"]
    bool proxy_diverse = false;      // ["proxy diverse"] 1: successive witnesses move different pieces
    int  proxy_second = -1;          // ["proxy second"] 1: >= 0: after a failure with v >= bound − this,
                                     //   a second witness gets a chance (fail soft values)
    int  proxy_prefilter = 0;        // ["proxy prefilter"] 1: a probe at depth − this first; if it fails, no cut
    int  proxy_weval = 0;            // ["proxy weval"] 1: > 1: among the first K ordinary moves, the witness is
                                     //   the one with the best static value after it
    int  proxy_verify = 0;           // ["proxy verify"] 1: cuts at nodes of at least this depth verified by a
                                     //   search of the node itself at the probe depth (0 = never)
    int  proxy_lazy = 0;             // ["proxy lazy"] 1: a CHEAP witness, probed before any move generation:
                                     //   1 = killer 1 if legal (else the usual witness), 2 = killer 1 only
    bool proxy_first = false;        // ["proxy first"] the proxy cut before MPC (O) instead of after it
    int  proxy_tail = 0;             // ["proxy tail"] 1: after a FAILED proxy test, the ordinary moves ordered
                                     //   after the witness: 0 searched as usual, 1 pruned, 2 pruned only if the
                                     //   witness failed by more than "proxy tail margin", 3 reduced by
                                     //   "proxy tail reduction" more, 4 reduced progressively (a heuristic:
                                     //   unlike the cut, value(witness) >= value(later moves) is only likely,
                                     //   from the move ordering)
    int  proxy_tail_keep = 0;        // ["proxy tail keep"] ordinary moves after the witness still searched as usual
    int  proxy_tail_margin = 2000;   // ["proxy tail margin"] mode 2
    int  proxy_tail_reduction = 1;   // ["proxy tail reduction"] modes 3 and 4
    int  proxy_tail_step = 4;        // ["proxy tail step"] mode 4, progressive: the k-th move after the witness
    int  proxy_tail_cap = 3;         // ["proxy tail cap"]   is reduced by min(cap, reduction + k / step) more,
                                     //   one more after a clear failure (by more than "proxy tail margin")
    int  mpc = 0;                    // ["mpc"] O. Multi-ProbCut without regression: 0 off, 1 fail high only,
                                     //   2 both directions (see negamax)
    int  mpc_depth = 5;              // ["mpc depth"] only at nodes of at least this depth
    int  mpc_ratio = 40;             // ["mpc ratio"] shallow depth d' = max(1, depth · ratio / 100)
    int  mpc_margin = 3000;          // ["mpc margin"] margin = this + per ply · (depth − d')
    int  mpc_margin_per_ply = 0;     // ["mpc margin per ply"]
    int  probcut = 0;                // ["probcut"] P. Stockfish's ProbCut generalized: number of moves tried
                                     //   (0 = off), each against beta + "probcut margin"
    int  probcut_depth = 5;          // ["probcut depth"] only at nodes of at least this depth (SF: > 4)
    int  probcut_reduction = 4;      // ["probcut reduction"] the moves are searched at depth − this (SF: 4)
    bool probcut_filter = true;      // ["probcut filter"] only if the static value after the move already
                                     //   reaches the bound (stands for Stockfish's qsearch test)
    int  eval_grain = 1;             // ["eval grain"] network values rounded to a multiple of this (1 = exact)
    bool staged = true;              // ["staged"] table's move first, the others generated only if needed
    bool yolah_table = false;        // ["yolah table"] SearchTable instead of the reference's table
                                     //   (−15 ± 23 Elo at 0.2 s/move: the table is hardly loaded there)
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
    // The transposition table: the reference's, or the one of step H. Only one
    // is allocated; tt_probe / tt_store / tt_move hide which.
    std::unique_ptr<TranspositionTable> table;
    std::unique_ptr<SearchTable> yolah_table;
    const size_t tt_size_mb;
    struct TTView {
        bool found = false;
        Move move = Move::none();
        int  depth = 0;
        int  lower = -32767, upper = 32767;   // value ∈ [lower, upper]
    };
    TTView tt_probe(uint64_t hash) const;
    void   tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta, int value, Move move);
    Move   tt_move(uint64_t hash) const;
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
                         SCORE_KILLER2 = SCORE_KILLER1 - 1, SCORE_COUNTER = SCORE_KILLER1 - 2,
                         SCORE_ARTICULATION = 1 << 28;   // + history: between the special moves and the others
    // Late move reductions (E): base reduction for (depth, move number), in
    // 1/1024 of a ply so that the adjustments can be fractional.
    int lmr_table[64][Yolah::MAX_NB_MOVES]{};
    // What each thread learns about the moves and keeps between the moves of
    // the game (L: one per thread — sharing them would need locks, and
    // different orderings make the threads explore different trees, which is
    // what lazy SMP relies on).
    struct Worker {
        int16_t history[2][SQUARE_NB][SQUARE_NB]{};
        // Countermove: for each opponent move (from, to), the last move that
        // refuted it (caused a cutoff right after it).
        Move countermoves[SQUARE_NB][SQUARE_NB]{};
    };
    const size_t nb_threads;
    std::vector<std::unique_ptr<Worker>> workers;

    struct RootMove {
        Move     move;
        uint64_t nodes = 0;       // size of its subtree in the last iteration
    };

    // Everything one thread needs during one search (L: one per thread).
    struct Search {
        int     id      = 0;      // thread number, 0 = the main thread
        Worker* w       = nullptr;
        uint64_t nb_evals = 0, nb_eval_hits = 0;
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
        bool proxy_off = false;                    // during a proxy cut verification: no proxy cut inside
    };

    // Evaluation cache (D): one entry per slot, the newest wins. 32 bits of
    // the hash check the position (the other bits choose the slot). Each entry
    // is ONE 64-bit word, read and written atomically (L: the threads share
    // the cache; a torn entry — the key of one position with the value of
    // another — is impossible):
    //     bits 63..32: key (high 32 bits of the hash)
    //     bits 31..16: value (int16)
    //     bit  0     : used
    std::vector<std::atomic<uint64_t>> eval_cache;
    // The exact endgame solver (G), with its own table, kept between moves.
    EndgameSolver endgame;
    uint64_t nb_evals = 0, nb_eval_hits = 0;

    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    int  evaluate(const Yolah& yolah, Search& s, uint64_t hash);
    int  network_value(const Yolah& yolah, Search& s);
    int  raw_network_value(const Yolah& yolah, Search& s);
    void update_accumulator(const int16_t* in, int16_t* out, uint8_t player, Move m) const;
    int  negamax(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);
    int  root_search(Yolah&, Search&, uint64_t hash, int alpha, int beta, int depth, Move&);
    int  late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const;
    int  search_move(Yolah&, Search&, uint64_t hash, Move m, size_t i, int reduction, int alpha, int beta, int depth);
    void score_moves(const Yolah&, const Search& s, Move tt_move, const Yolah::MoveList&, int* scores,
                     int depth = 0) const;
    int  territory_after(const Yolah&, Move) const;
    static bool is_legal(const Yolah&, Move);
    static Move pick_move(Yolah::MoveList&, int* scores, size_t i, size_t n);
    bool is_articulation_move(const Yolah&, Move) const;
    void update_history(Search&, uint8_t player, Move m, int bonus);
    void iterative_deepening(Yolah, Search&, uint8_t max_depth);
    void print_pv(Yolah, uint64_t hash, int8_t depth);

public:
    MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction,
                         const std::string& nnue_q_parameters_filename, bool verbose = false, Options options = {},
                         size_t nb_threads = 1);
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
