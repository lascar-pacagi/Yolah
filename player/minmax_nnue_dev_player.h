#ifndef MINMAX_NNUE_DEV_PLAYER_H
#define MINMAX_NNUE_DEV_PLAYER_H
#include "player.h"
#include "heuristic.h"
#include "transposition_table.h"
#include "search_table.h"
#include <array>
#include <memory>
#include <atomic>
#include <vector>
#include <fstream>
#include <chrono>
#include <bit>
#include <string>
#include "nnue_quantized.h"
#include "nnue_persp.h"
#include "endgame_solver.h"

struct SearchTracer;
int lmr_bucket(int free, int depth, size_t i, int hist, int tried);   // U5 (6)   // player/search_trace.h, only used by the search_trace executable

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
//      +21, mobility 512 +12, queen distance 128 +1. Modes 4–6 (2026-10-15,
//      after test/territory_study_main.cpp): the queen distance by occluded
//      fill, ~80 ns for both sides, and a linear score of the 5 queen measures.
//      search_bench depth 9 (209 positions): influence 2.67 M nodes 1.36 s,
//      mode 2 2.76 M 1.59 s, mode 4 the same nodes 1.48 s, mode 5 8.05 M
//      (the opening weights swamp the history: dropped), mode 6 2.84 M 1.53 s.
//      Campaign Q1.
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
//      margin set by matches as Stockfish sets its own. On by default since
//      the study (Y1–Y6, 2026-10): both directions, margin 4000 + 500 per ply
//      of depth − d', +57 over none at 1 s/move; the proxy cut kept after it
//      (+8 [−2, +19]), in Yolah as in Edax and NTest a tuned MPC leaves the
//      proxy cut little to add.
//   P. Stockfish's ProbCut generalized ("probcut" moves, "probcut margin",
//      "probcut reduction", "probcut filter"): the best-ordered moves at
//      depth − 4 against beta + margin. Off by default. O and P are the
//      references for the proxy cut study (config/search_study/).
//   S. The sparse product of the network's 1024 → 64 layer ("sparse h1",
//      nnue/sparse_matvec.h): only the groups of 4 inputs of h1 with a
//      non-zero value are multiplied. Same values, same tree (search_bench
//      209/209), value() −34 %, −21 % time at depth 9. Campaign S1 (cluster,
//      E5-2650 v4): +61 [+53, +69] at 0.2 s/move, +35 [+28, +42] at 1 s
//      (4000 games each). On by default since 2026-10-12.
//   T. history prior ("history prior", "history prior mode", "history prior
//      weight"): what the history learns during a search, learnt beforehand
//      from the cutoffs of many searches (search_bench cutstats, then
//      nnue/history_prior.py), one table per phase (free squares) and colour.
//      Only 71 % of the cut nodes cut on their first move (search_bench,
//      depth 9): the ordering has room. search_bench depth 9 (empty history
//      per position): nodes −11 to −13 %. Campaign T1 (vs dev_y7_sparse, 4000
//      games each): mode 1 50 % +4.3 [−3.5, +12.2] (0.2 s), +3.6 [−3.7, +10.8]
//      (1 s); mode 2 100 % −7.6 / −3.3; mode 2 200 % +2.6 / +0.3. No gain: in
//      a game the history is already warm from the previous moves. Off.
//      Mode 4 (the 1 or 2 moves the players choose most in the phase, from
//      data/games, before the history; campaign T2, 4000 games each): top 1
//      −25.0 [−32.8, −17.2] (0.2 s), −16.2 [−23.5, −8.9] (1 s); top 2 −24.0,
//      −25.2. The history's best moves pushed back (and reduced more by LMR,
//      which grows with the move's rank). Step T closed: off.
//   U. "fly reply" (the user's idea, 2026-11-17): a late move searched first
//      at a reduced depth against a STRONGER reply (the opponent may also
//      jump over pieces and holes along a queen's line); pruned if even so it
//      stays <= alpha (mode 1), the node cut if it beats alpha (mode 2).
//      Campaign U1.
//      Campaigns U1-U10 (2026-11-17, config/search_study/campaign_U*.txt, about
//      200 variants, 2000 games each, all behind options, off by default):
//      the "extra power" family (U1-U3: fly reply, optimistic test, proxy
//      witness against a stronger reply, region bounds…) brought nothing (best
//      ≈ 0; storing the bound of a node with pruned moves cost −340). What
//      paid is ordering by the LOCAL reply to the opponent's last move. Kept in
//      config/search_study/dev_u8.cfg (+54.1 [+45.0, +63.3] against dev_final
//      at 1 s): continuation history by the last move ("cont hist" 1, U4
//      +32.6), sealed pockets as tempo reserves ("reserve", exact, U5 +10 at
//      1 s), killers as a bonus ("killer mode" 1, U7 +17.7), intercept of the
//      last move with one ply less of LMR ("intercept", U7 +17.0), history of
//      reductions that proved wrong ("lmrh", U7 +12.3), correction history
//      ("corr hist" 2, U6 +10.6). Neutral or worse: IIR, improving, singular
//      extension / multi-cut, quiescence, razoring, futility per move, root
//      audit, symmetry, learnt LMR, TT provenance, the reviews' fixes (U9, U10).
// Each improvement can be switched off in the config.

// The switches of the improvements (config keys in brackets), so that each
// one can be measured on its own. (Outside the class: a nested struct with
// default member initializers cannot be a default argument of the class.)
struct MinMaxNNUE_DevOptions {
    bool pvs = true;                 // ["pvs"] null windows for all moves but the first
    int  aspiration_window = 300;    // ["aspiration window"] half width at the root, 0 = full window
    double aspiration_k = 1.25;      // ["aspiration k"] > 0: adaptive half width, k × the mean change of value between depths (at least "aspiration window")
    int  tempo = 0;                  // ["tempo"] added to the network's value for the side to move (values: ±30000 = tanh ±1)
    double aspiration_grow = 1.5;    // ["aspiration grow"] adaptive windows: factor of the half width after each failure
    // Campaign A2 (2026-10-31, from an audit of the aspiration failures; all off by default):
    bool root_fix = true;            // ["root fix"] root order after a FAILED attempt: fail low keeps the order (the
                                     //   previous best stays first), fail high only brings the new best to the front;
                                     //   the nodes of a move are summed over the attempts of a depth
    bool root_order_gap = false;     // ["root order gap"] after a completed root search, the moves after the best by
                                     //   (rank by nodes + rank by fail-soft value) (root log: fewer far misses)
    int  root_close_full = 0;        // ["root close full"] > 0: at the root, the N moves (after the first) with the best
                                     //   fail-soft values of the last search are not reduced
    bool root_nodes_last = false;    // ["root nodes last"] with "root fix": the nodes of a move are those of the last
                                     //   attempt only (summed, the moves before a fail-high cut count twice)
    bool aspiration_keep_beta = false; // ["aspiration keep beta"] adaptive windows: a fail low does not lower beta
    int  aspiration_stats_depth = 4; // ["aspiration stats depth"] adaptive windows: |v_d − v_{d−1}| counted from this depth
    bool aspiration_center2 = false; // ["aspiration center 2"] window centred on (v_{d−1} + v_{d−2}) / 2 (odd/even effect)
    double aspiration_last = 0;      // ["aspiration last"] > 0, adaptive windows: half width at least this × |v_{d−1} − v_{d−2}|
                                     //   (after a surprise the next depth often fails: root log, nnue/root_surprise.py)
    bool pvs_research_high = false;  // ["pvs research high"] PV nodes: full-window re-search also when the null window
                                     //   gave ≥ beta (Stockfish today), not only alpha < v < beta
    bool tt_cut_pv = false;          // ["tt cut pv"] false: at PV nodes, the table cuts only with an exact value or a
                                     //   proven result (Stockfish: no table cut at PV nodes)
    bool tt_cut_pv_exact = true;     // ["tt cut pv exact"] with "tt cut pv" false: false = not even an exact value cuts
                                     //   at PV nodes (with the Yolah table, lower == upper can be two merged bounds
                                     //   of non-PV searches), only a proven result
    bool tt_pv = false;              // ["tt pv"] Yolah table: entries remember that their position was searched as a
                                     //   PV node (Stockfish's ttPv); at such a node, even with a null window: no
                                     //   reverse futility pruning, LMR as at PV nodes
    bool tt_depth_asym = false;      // ["tt depth asym"] a table cut on a lower bound (≥ beta) needs a strictly
                                     //   deeper entry, an upper bound or exact value the same depth (Stockfish)
    bool table_tiebreak = false;     // ["table tiebreak"] Yolah table: among entries of equal depth, replace the one
                                     //   with the most free squares (the oldest), not always slot 0
    bool proxy_after_tt = false;     // ["proxy after tt"] staged: the proxy test comes after the table's move
    bool lmr_territory = false;      // ["lmr territory"] false: LMR adjusted by the history only (the move score
                                     //   also holds the territory term at depth ≥ "territory depth")
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
    // U. "Fly reply" (the user's idea, 2026-11-17): at a non-PV node, a move m after the first is
    //    first searched at a reduced depth against a STRONGER reply: the opponent's next move may also
    //    jump over pieces and holes along a queen's line, landing on a free square. With a stronger
    //    opponent m can only look worse: value_fly(m) <= value(m). Mode 1: if even that is <= alpha,
    //    m is pruned (heuristic, like late move pruning); mode 2: if it is > alpha (>= beta at a
    //    non-PV node), the node cuts (safe direction, a proxy cut with a handicap); 3: both.
    //    Otherwise m is searched as usual. Never in a branch where a fly reply is already active.
    int  fly_reply = 0;              // ["fly reply"] 0 off, 1 prune m, 2 cut the node, 3 both
    int  fly_reply_jumps = 1;        // ["fly reply jumps"] 1: every free square of the line past the
                                     //   obstacles, 2: only the first free square past the first obstacles
    int  fly_reply_depth = 3;        // ["fly reply depth"] only at nodes of at least this depth
    int  fly_reply_reduction = 2;    // ["fly reply reduction"] m searched at depth − 1 − R, R = this + depth / div
    int  fly_reply_div = 4;          // ["fly reply div"]
    int  fly_reply_from = 1;         // ["fly reply from"] moves of rank >= this are tested (0 = the first one too)
    bool fly_reply_store = false;    // ["fly reply store"] a node with moves pruned by the fly reply stores its
                                     //   upper bound in the table (off: not stored, fail-hard)
    int  fly_reply_margin = 0;       // ["fly reply margin"] mode 1: pruned if the value is <= threshold − this
    bool fly_reply_relative = false; // ["fly reply relative"] mode 1: threshold = the FIRST move's value against the
                                     //   stronger reply (same handicap: moves compared with each other), else alpha
    int  fly_reply_order = 0;
    bool fly_reply_filter = false;   // ["fly reply filter"] 6: mode 2 only if one ply with extra power, statically (the
    int  fly_reply_filter_margin = 0;//   network), already gives >= beta + "fly reply filter margin"
    // U, pivots of the second round (campaign U2):
    //  1 optimistic test (the right direction to prune): m pruned if, with OUR next move given the extra
    //    power, a reduced search still says <= alpha
    int  fly_prune = 0;              // ["fly prune"] 0 off, 1 on
    int  fly_prune_jumps = 2;        // ["fly prune jumps"] see fly_moves (1 all, 2 first, 3 holes only, 4 one obstacle)
    int  fly_prune_depth = 8;        // ["fly prune depth"]
    int  fly_prune_reduction = 4;    // ["fly prune reduction"] R = this + depth / div
    int  fly_prune_div = 3;          // ["fly prune div"]
    int  fly_prune_from = 1;         // ["fly prune from"]
    //  3 the proxy cut's witness searched against the stronger reply (allows bigger "proxy reduction")
    bool proxy_fly = false;          // ["proxy fly"]
    int  proxy_fly_jumps = 2;        // ["proxy fly jumps"]
    //  4 openness: gap between the opponent's best static reply with and without the extra power (after a
    //    pass); closed (gap < closed): late moves reduced one ply more; open (gap > open): one ply less
    int  fly_lmr = 0;                // ["fly lmr"] 0 off, 1 on
    int  fly_lmr_depth = 6;          // ["fly lmr depth"]
    int  fly_lmr_closed = 1000;      // ["fly lmr closed"]
    int  fly_lmr_open = 8000;        // ["fly lmr open"]
    int  fly_lmr_jumps = 2;          // ["fly lmr jumps"]
    //  5 bounds of the final score from the regions of free squares (region_bounds): cut if low >= margin,
    //    fail low if high <= −margin, at non-PV nodes
    int  region_bounds = 0;          // ["region bounds"] 0 off, 1 on
    int  region_margin = 1;          // ["region margin"]
    int  region_max_free = 40;       // ["region max free"] only with at most this many free squares
    //  7 ordering: the ordinary moves ordered by our static value after the opponent's best reply with
    //    extra power (instead of the history), at nodes of depth >= "fly order depth"
    int  fly_order = 0;              // ["fly order"]
    int  fly_order_depth = 6;        // ["fly order depth"]
    int  fly_order_jumps = 2;        // ["fly order jumps"]
    // U, third round (campaign U3, ideas from Gemini and Codex):
    //  G region bounds 2 = SOUND: our future moves >= a legal greedy harvest of the regions only we touch,
    //    the opponent's <= the free squares of the regions it touches (8-connected, pieces are walls)
    //  A refutation by a restricted reply: after a late move, the opponent limited to its "refute replies"
    //    likeliest replies (child's table move, countermove, history); if one of them keeps us <= alpha
    //    (search at depth − 2 − "refute reduction"), the move is pruned (exact at reduction 0)
    int  refute = 0;                 // ["refute"]
    int  refute_replies = 1;         // ["refute replies"]
    int  refute_depth = 4;           // ["refute depth"]
    int  refute_from = 4;            // ["refute from"]
    int  refute_reduction = 0;       // ["refute reduction"]
    //  B the proxy cut's witness with OUR next turn restricted to our k best-ordered moves (pessimistic),
    //    searched "proxy restrict deeper" plies deeper
    int  proxy_restrict = 0;         // ["proxy restrict"] k, 0 off
    int  proxy_restrict_deeper = 0;  // ["proxy restrict deeper"]
    //  E free pass to fail low: if even passing (searched at depth − 1 − R) stays <= alpha − margin, the node
    //    fails low (in a zugzwang the pass is >= the best move; elsewhere the margin covers a move's worth)
    int  pass_razor = 0;             // ["pass razor"]
    int  pass_razor_depth = 4;       // ["pass razor depth"]
    int  pass_razor_margin = 3000;   // ["pass razor margin"]
    int  pass_razor_reduction = 3;   // ["pass razor reduction"]
    int  pass_razor_guard = 0;       // ["pass razor guard"] only if static value <= alpha − this
    //  C no-hole move to fail low: the table's move (else killer 1) played WITHOUT leaving a hole, searched at
    //    depth − 1 − R; if even so <= alpha, the node fails low (only if static value <= alpha − guard)
    int  nohole_razor = 0;           // ["nohole razor"]
    int  nohole_razor_depth = 4;     // ["nohole razor depth"]
    int  nohole_razor_guard = 2000;  // ["nohole razor guard"]
    int  nohole_razor_reduction = 3; // ["nohole razor reduction"]
    //  D the proxy cut's witness against opponent replies that leave no hole (pessimistic-ish)
    bool proxy_nohole = false;       // ["proxy nohole"]
    //  H validity statistics (search_bench): each prune / cut decision of a test re-searched normally
    bool verify_tests = false;       // ["verify tests"]
    // U4 (2026-11-17, ideas of Gemini and Codex, campaign U4): classical search techniques adapted to Yolah
    int  iir = 0;                        // ["iir"] 9 internal iterative reduction: no table move at depth >= "iir depth": one ply less (1 every node, 2 PV and cut nodes)
    int  iir_depth = 4;                  // ["iir depth"] 
    int  cutnode_lmr = 0;                // ["cutnode lmr"] 9 late moves of an expected cut node reduced this much more
    bool improving = false;              // ["improving"] 6 static value better than two plies ago: reverse futility margin (depth − improving); not improving: late moves reduced one ply more
    int  corr_hist = 0;                  // ["corr hist"] 6 correction history: the static value corrected by a table (pieces, side) learnt from (best − static); 1 for the pruning, 2 at the leaves too
    int  corr_scale = 8;                 // ["corr scale"]   bonus = (best − static) · depth / this
    int  se = 0;                         // ["se"] 8 singular extension: the table move extended one ply if every other move fails below (table lower bound − margin · depth) at depth (d−1)/2
    int  se_depth = 7;                   // ["se depth"] 
    int  se_margin = 30;                 // ["se margin"]   per ply of depth
    bool se_double = false;              // ["se double"]   two plies if they fail below by twice the margin
    bool multicut = false;               // ["multicut"] 8 multi-cut: if that search reaches beta, the node cuts
    int  ext_max = 0;                    // ["ext max"]   extensions per line at most (0: half the iteration depth)
    int  cont_hist = 0;                  // ["cont hist"] 7 continuation history in the ordering: 1 by the opponent's last move, 2 also by our previous move
    int  cont_weight = 100;               // ["cont weight"] U6: % of the continuation history (last move) in the ordering
    int  cont2_weight = 100;              // ["cont2 weight"] U6: % of the 2-ply one
    int  cont_bonus = 100;                // ["cont bonus"] U6: % of the history bonus given to the continuation history
    bool cont_lmr = false;                // ["cont lmr"] U6: the continuation history also in the LMR's history term
    // U7 (2026-11-17, third round of Gemini and Codex around the continuation history)
    bool cont_age = false;               // ["cont age"] U7 fix: the continuation tables halved at each new move, like the history
    int  cont_fix = 0;                   // ["cont fix"] U7 fix: maluses only for the moves really searched, never on the cutting move's destination (1 continuation tables, 2 the history too)
    int  malus_pct = 100;                // ["malus pct"] U7: malus = this % of the bonus
    int  malus_cap = 0;                  // ["malus cap"] U7: at most this many maluses per cutoff (the last ones tried; 0 no cap)
    int  bonus_margin = 0;               // ["bonus margin"] U7 (10): bonus × (1 + clamp((v − beta) / this, 0, 2)) (0 off)
    int  rel_weight = 0;                 // ["rel weight"] U7 (A1): % weight of the relative continuation (our destination − the opponent's last destination, 15×15)
    int  hole_weight = 0;                // ["hole weight"] U7 (A2, C1): % weight of the continuation by the square the opponent just destroyed (its last origin) → our destination
    int  from_weight = 0;                // ["from weight"] U7 (A3): % weight of the continuation opponent's last destination → our ORIGIN (which piece to move)
    bool cont2_same = false;             // ["cont2 same"] U7 (C3): the 2-ply continuation split by "the same piece moves again"
    int  geo_weight = 0;                 // ["geo weight"] U7 (C4): % weight of the history by the holes around the origin, direction and distance
    int  hpv_weight = 0;                 // ["hpv weight"] U7 (C6): % weight of the history of PV best moves in the first 4 plies
    int  intercept = 0;                  // ["intercept"] U7 (B8): a move landing on a line of, or next to, the opponent's last destination: ordering bonus (this), and with "intercept lmr" one ply less
    bool intercept_lmr = false;          // ["intercept lmr"] 
    int  killer2ply = 0;                 // ["killer2ply"] U7 (B9): the killer of two plies before, tried after the killers
    int  killer_mode = 0;                // ["killer mode"] U7 (C2): 0 killers first; 1 killers as a bonus of "killer bonus" on the history; 2 killers first only if their continuation is positive
    int  killer_bonus = 4000;            // ["killer bonus"] 
    int  hist_prune = 0;                 // ["hist prune"] U7 (A4): at depth <= 3, non-PV, from the 3rd move: an ordinary move whose history (+ continuation) < −this·depth is pruned (0 off)
    int  lmp_protect = 0;                // ["lmp protect"] U7 (A4): late move pruning keeps the moves whose continuation >= this (0 off)
    int  fp_margin = 0;                  // ["fp margin"] U7 (B6): futility pruning per move at depth <= 3: static + 1000 + this·depth <= alpha, ordinary move: pruned (0 off)
    int  razor_margin = 0;               // ["razor margin"] U7 (B7): razoring at depth <= 2: static + this·depth <= alpha: the static value returned (0 off)
    int  lmrh = 0;                       // ["lmrh"] U7 (C5): moves whose reduced search was wrong (failed high after reduction) learnt by context; history >= this: one ply less (0 off)
    int  corr_cont = 0;                  // ["corr cont"] U7 (C7): correction of the static value (pruning only) by (our previous destination, the opponent's last destination)
    // U9 (2026-11-17, Gemini's code review)
    bool root_context = false;            // ["root context"] the last two moves before the root (found by comparing with the previous root) fill played[], so that the continuation, intercept and countermove also work at the root and the first plies
    bool killer_special = false;          // ["killer special"] with "killer mode" 1, the killers stay special moves for the LMR, the proxy witnesses, the tail and the per-move pruning
    bool lmr_hist_full = false;           // ["lmr hist full"] the LMR's history term is the ordering score (history + every context term)
    int  reserve_max_free = 64;           // ["reserve max free"] the sealed pockets only looked for with at most this many free squares (speed)
    // U10 (2026-11-17, Astra's review; kept only if >= +10 Elo)
    bool malus_per_key = false;           // ["malus per key"] each context table penalises each of its keys once, never the rewarded key (cfrom got its bonus cancelled by the other moves of the same piece)
    int  root_cont_lmr = 0;               // ["root cont lmr"] the root's LMR history term + this % of the continuation (needs "root context")
    int  prec_weight = 0;                 // ["prec weight"] % weight of the precise continuation (opponent's last destination, our origin, our destination)
    bool prec_conf = false;               // ["prec conf"] that weight grows with the entry's observations (full from 8)
    int  fn_sample = 0;                   // ["fn sample"] one reduced fail-low in this many re-searched at full depth; the real misses learnt by context (0 off)
    int  fn_thr = 2000;                   // ["fn thr"] that table >= this: one ply less of LMR
    int  tt_bonus = 0;                    // ["tt bonus"] % of the bonus given to the table's move when the table cuts
    int  proxy_bonus = 0;                 // ["proxy bonus"] % of the bonus given to a proxy witness that cuts
    // U12 (2026-11-18, user's idea): blocking an opponent piece's ray toward a zone of free squares
    int  block_weight = 0;                // ["block weight"] ordering weight per square of the blocked cone zone (0 off)
    int  block_mode = 0;                  // ["block mode"] 0 bonus added to the move's score, 1 tactical tier (SCORE_ARTICULATION) if zone >= "block min"
    int  block_min = 4;                   // ["block min"] minimum blocked zone score for mode 1 or "block lmr"
    int  block_steps = 2;                 // ["block steps"] king expansion steps in the cone behind our destination (0: ray only)
    int  block_cone = 1;                  // ["block cone"] 1: 90-degree cone behind the destination, 2: half-plane
    int  block_multi = 50;                // ["block multi"] extra % per additional opponent piece blocked by the same move
    int  block_depth = 1;                 // ["block depth"] only at nodes of at least this depth
    bool block_lmr = false;               // ["block lmr"] one ply less of LMR when the blocked zone score >= "block min"
    int  piece_lmr = 0;                  // ["piece lmr"] 4 a late move of a piece already tried this many times in the node: one ply more (0 off)
    bool family_lmr = false;             // ["family lmr"] 4b the first move of each piece (family of the destroyed square): one ply less
    int  gate_lmr = 0;                   // ["gate lmr"] 5 a move that shrinks the contested region (reached by both) by >= "gate min" squares: not reduced
    int  gate_min = 4;                   // ["gate min"] 
    int  gate_depth = 3;                 // ["gate depth"] 
    int  gate_ext = 0;                   // ["gate ext"] 5b one of the first 4 moves at depth >= 6 whose destroyed square cuts the opponent off >= "gate ext min" free squares: extended one ply
    int  gate_ext_min = 6;               // ["gate ext min"] 
    int  opp_lmr = 0;                    // ["opp lmr"] 10 opponent mobility: a late move removing >= "opp lmr k" opponent moves reduced one ply less, a passive one (none removed, not next to its pieces) one more
    int  opp_lmr_k = 3;                  // ["opp lmr k"] 
    int  split_solve = 0;                // ["split solve"] 1 once no region of free squares is reached by both sides: the exact value (each side's longest harvest, solved exactly)
    int  split_max_free = 36;            // ["split max free"] 
    int  split_budget = 200000;          // ["split budget"]   nodes per harvest problem
    int  pocket = 0;                     // ["pocket"] 2 one move per private region (only we reach it) at non-PV nodes, the others pruned
    int  pocket_depth = 3;               // ["pocket depth"] 
    int  qs = 0;                         // ["qs"] 3 quiescence: 1 tension moves (an opponent piece left with <= 1 free neighbour, or a move of our piece with <= 2) with stand pat; 2 mobility, no stand pat: a side with <= "qs moves" moves plays them all
    int  qs_depth = 2;                   // ["qs depth"] 
    int  qs_moves = 2;                   // ["qs moves"] 
    bool sibling_ref = false;            // ["sibling ref"] 5c the replies that cut after a sibling move are tried early (a killer local to the parent)        // ["fly reply order"] the opponent's moves: 0 ordinary ones first, 1 jumps first, 2 history only
    // U5 (2026-11-17, second list of Codex, campaign U5)
    int  tt_fragile = 0;                 // ["tt fragile"] 7 provenance: bounds of nodes whose moves were pruned (LMP, proxy tail, pockets) marked fragile; 1 they need one more ply to cut within "tt fragile ply" plies of the root, 2 everywhere
    int  tt_fragile_ply = 4;             // ["tt fragile ply"] 
    int  root_audit = 0;                 // ["root audit"] 8 after each iteration of depth >= "root audit depth": the second root move searched with another policy (no LMP, LMR one ply less, no table cut near the root) against the best; switched if better under the same policy
    int  root_audit_depth = 6;           // ["root audit depth"] 
    int  root_audit_frac = 10;           // ["root audit frac"]   only while less than (100 − this) % of the time is used
    int  sym = 0;                        // ["sym"] 9 symmetry: near a cut (|static − beta| < "sym window"), the network on the position mirrored; if they disagree by > "sym thr", no static pruning here and LMR one ply less
    int  sym_depth = 4;                  // ["sym depth"] 
    int  sym_window = 3000;              // ["sym window"] 
    int  sym_thr = 2000;                 // ["sym thr"] 
    int  reserve = 0;                    // ["reserve"] 1 sealed pockets (pieces that can only move into a region only they reach): only one move kept, one that keeps the maximal harvest (exact: the pocket is a reserve of tempos)
    int  reserve_max = 20;               // ["reserve max"]   pockets of at most this many free squares
    int  lmr_log = 0;                    // ["lmr log"] 6 (search_bench) one reduced fail-low in this many re-searched at full depth and logged with its features
    std::string lmr_table;                // ["lmr table"] 6: the learnt LMR adjustments (nnue/lmr_learn.py), empty: none
    int  lmp_depth = 3;              // ["lmp depth"] late move pruning up to this depth (0 = never)
    int  lmp_moves = 4;              // ["lmp moves"] moves searched before pruning: this + depth²
    bool pass_rule = false;          // ["pass rule"] a player who must pass has lost (see negamax)
    bool articulation_ordering = false;   // ["articulation ordering"] articulation moves after the killers
    bool articulation_lmr = false;   // ["articulation lmr"] articulation moves never reduced nor pruned
    int  territory_ordering = 1;     // ["territory ordering"] 0 = off, 1 = influence (king steps),
                                     //   2 = queen distance, 3 = mobility, 4 = queen distance by
                                     //   occluded fill (same values as 2), 5 = the 5 queen measures
                                     //   of each side, weights per phase, 6 = the same with the
                                     //   middle game's weights everywhere (see territory_after)
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
    int  mpc = 2;                    // ["mpc"] O. Multi-ProbCut without regression: 0 off, 1 fail high only,
                                     //   2 both directions (see negamax)
    int  mpc_depth = 5;              // ["mpc depth"] only at nodes of at least this depth
    int  mpc_ratio = 40;             // ["mpc ratio"] shallow depth d' = max(1, depth · ratio / 100)
    int  mpc_margin = 4000;          // ["mpc margin"] margin = this + per ply · (depth − d')
    int  mpc_margin_per_ply = 500;   // ["mpc margin per ply"] (defaults: best of the study, Y2/Y5: +57 vs none at 1 s;
                                     //   the proxy cut after it +8 [−2, +19], before it −27)
    int  probcut = 0;                // ["probcut"] P. Stockfish's ProbCut generalized: number of moves tried
                                     //   (0 = off), each against beta + "probcut margin"
    int  probcut_depth = 5;          // ["probcut depth"] only at nodes of at least this depth (SF: > 4)
    int  probcut_reduction = 4;      // ["probcut reduction"] the moves are searched at depth − this (SF: 4)
    bool probcut_filter = true;      // ["probcut filter"] only if the static value after the move already
                                     //   reaches the bound (stands for Stockfish's qsearch test)
    bool sparse_h1 = true;           // ["sparse h1"] the network's 1024 → 64 layer as a sparse product (only
                                     //   the groups of 4 inputs with a non-zero value; same values, faster;
                                     //   S1: +61 Elo at 0.2 s, +35 at 1 s)
    std::string history_prior;       // ["history prior"] T: the prior's file (int16 move scores per phase,
                                     //   colour, from, to), "" = none
    int  history_prior_mode = 1;     // ["history prior mode"] 1: at each new search the history is pulled
                                     //   halfway toward the prior of the root's phase (instead of halved);
                                     //   2: the prior of the node's phase added to every history score;
                                     //   3: the moves whose prior is ≥ "history prior threshold" ordered
                                     //   before the history-ordered moves (by their prior)
                                     //   4: the "history prior top" moves with the highest prior, before the
                                     //   history-ordered moves (only the ranking of the prior counts)
    int  history_prior_threshold = 3000; // ["history prior threshold"] mode 3 (after the weight)
    int  history_prior_top = 1;      // ["history prior top"] mode 4: how many moves
    int  history_prior_weight = 100; // ["history prior weight"] percent of the prior's scores
    int  eval_grain = 1;             // ["eval grain"] network values rounded to a multiple of this (1 = exact)
    bool staged = true;              // ["staged"] table's move first, the others generated only if needed
    bool yolah_table = true;         // ["yolah table"] SearchTable instead of the reference's table
                                     //   (−15 ± 23 Elo at 0.2 s/move: the table is hardly loaded there)
    int  endgame_root = 0;           // ["endgame root"] try to prove the result at the root with at most
                                     //   this many free squares (0 = never)
    double endgame_root_time = 0.5;  // ["endgame root time"] share of the thinking time given to that proof
    int  endgame_tree = 0;           // ["endgame tree"] solve exactly the nodes with at most this many
                                     //   free squares (0 = never)
    uint64_t max_nodes = 0;          // ["max nodes"] stop a search once the main thread has visited this
                                     //   many nodes (0 = no limit): the same depth on any machine (data generation)
    int  record_depth = 0;           // ["record depth"] with a recorder set: record the PV nodes of the main
                                     //   thread whose value is exact and whose depth is at least this (0 = off)
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
        bool pv = false;                      // searched as a PV node ("tt pv", Yolah table only)
        bool fragile = false;                 // U5 (7): from a node whose moves were pruned (Yolah table only)
    };
    TTView tt_probe(uint64_t hash) const;
    void   tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta, int value, Move move,
                    bool pv = false, bool fragile = false);
    Move   tt_move(uint64_t hash) const;
    size_t nb_moves_at_full_depth;
    uint8_t late_move_reduction;
    const std::string nnue_q_parameters_filename;
    // The network, chosen by the weights file: the 193 NNUE (NNUE_Quantized)
    // or, for a "PERSP" file, the NNUE seen from the side to move with one
    // head per phase (NNUE_Persp, doc/nnue_roadmap.qmd step 1). The
    // accumulators below are sized for the larger of the two (2 views).
    NNUE_Quantized nnue;
    std::unique_ptr<NNUE_Persp> persp;
    using Accumulator = NNUE_Persp::Accumulator;
    void net_init(const Yolah& yolah, Accumulator& a);
    void net_play(uint8_t player, Move m, Accumulator& a, NNUE_Persp::RefreshCache* cache);
    void net_undo(uint8_t player, Move m, Accumulator& a, NNUE_Persp::RefreshCache* cache);
    float net_value(const Accumulator& a, const Yolah& yolah);
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
    // T. The history prior: prior[phase][colour][from][to] (empty = none).
    std::vector<int16_t> prior;
    int prior_of(int phase, uint8_t player, Move m) const {
        return prior[((size_t(phase) * 2 + player) * SQUARE_NB + m.from_sq()) * SQUARE_NB + m.to_sq()];
    }
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
        int16_t cont1[2][SQUARE_NB][SQUARE_NB]{}; // U4: by the opponent's last move's destination, our move's destination
        int16_t cont2[2][SQUARE_NB][SQUARE_NB]{}; // U4: by our previous move's destination
        int16_t corr[2][16384]{};                 // U4: correction history (pieces, side to move)
        int16_t rel[2][15][15]{};                 // U7: relative continuation
        int16_t holec[2][SQUARE_NB][SQUARE_NB]{}; // U7: opponent's last origin (now a hole) → our destination
        int16_t cfrom[2][SQUARE_NB][SQUARE_NB]{}; // U7: opponent's last destination → our origin
        int16_t cont2s[2][2][SQUARE_NB][SQUARE_NB]{}; // U7: 2-ply continuation split by "same piece"
        int16_t geo[2][256][8][3]{};              // U7: holes around the origin, direction, distance class
        int16_t hpv[2][4][SQUARE_NB][SQUARE_NB]{};// U7: PV best moves near the root
        int16_t lmrh[2][SQUARE_NB][SQUARE_NB]{};  // U7: reductions that were wrong, by context
        int16_t ccorr[2][SQUARE_NB][SQUARE_NB]{}; // U7: continuation correction of the static value
        int16_t cprec[2][SQUARE_NB][SQUARE_NB][SQUARE_NB]{};   // U10: precise continuation
        uint8_t cprec_n[2][SQUARE_NB][SQUARE_NB][SQUARE_NB]{}; // U10: its observations
        int16_t fnh[2][SQUARE_NB][SQUARE_NB]{};   // U10: real LMR misses by context
        // Adaptive aspiration windows: decaying sums of |v_d − v_{d−1}| and
        // of their count, over the iterations of this game's searches.
        double asp_sum = 0, asp_n = 0;
    };
    const size_t nb_threads;
    std::vector<std::unique_ptr<Worker>> workers;

    struct RootMove {
        Move     move;
        uint64_t nodes = 0;       // size of its subtree in the last iteration
        // Its last search (root log): fail-soft value, against the alpha it
        // was searched with, the reduction, the last stage of search_move.
        int      value = 0, alpha = 0;
        uint8_t  reduction = 0, stage = 0;
        uint8_t  rank = 0;        // its place in the last attempt
        uint8_t  rank_start = 0;  // its place at the start of the depth (the first attempt)
    };

    // Everything one thread needs during one search (L: one per thread).
    struct Search {
        int     id      = 0;      // thread number, 0 = the main thread
        Worker* w       = nullptr;
        uint64_t nb_evals = 0, nb_eval_hits = 0;
        // Aspiration statistics (search_bench): root attempts, failed ones,
        // fail highs right after a fail low, nodes spent in failed attempts.
        uint64_t asp_attempts = 0, asp_failed = 0, asp_rebounds = 0, asp_failed_nodes = 0;
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
        Accumulator acc;
        // With them (D): accs[p] is the accumulator of the position at ply p
        // of the current line, valid only if acc_ok[p]. played[p] (above)
        // tells how to get accs[p + 1] from accs[p].
        Accumulator accs[Yolah::MAX_NB_PLIES + 1];
        bool acc_ok[Yolah::MAX_NB_PLIES + 1]{};
        // A network with fc1 sets by phase: the refreshes of this thread start
        // from it (NNUE_Persp::RefreshCache), whichever of acc / accs is used.
        NNUE_Persp::RefreshCache refresh;
        bool proxy_off = false;                    // during a proxy cut verification: no proxy cut inside
        uint64_t nb_fly_tests = 0, nb_fly_prunes = 0, nb_fly_cuts = 0, nb_fly_nodes = 0, nb_region_cuts = 0;   // U: statistics (search_bench)
        int  fly_nest = 0;                         // U: fly replies in progress above this node
        int  fly_ply = -1;                         // U: the ply whose side has the extra power (negamax_fly), −1 none
        int  fly_jumps = 2;                        // U: the kind of extra power (fly_moves)
        int  no_tt = 0;                            // U: inside a subtree whose positions are not real (no-hole moves): no table
        int  restrict_ply = -1, restrict_k = 0;    // U (B): the ply whose moves are limited to the k best-ordered
        int  nohole_ply = -1;                      // U (D): the ply whose moves leave no hole
        bool verifying = false;                    // U (H): inside a verification search
        bool cutnode[Yolah::MAX_NB_PLIES + 2]{};   // U4: expected cut node (set by the parent in search_move)
        int  eval_stack[Yolah::MAX_NB_PLIES + 2]{};// U4: static value of the node at each ply (EVAL_NONE: none)
        Move excluded[Yolah::MAX_NB_PLIES + 2]{};  // U4: the move left out by the singular search
        int  ext_sum[Yolah::MAX_NB_PLIES + 2]{};   // U4: extensions along the line
        Move sib_ref[Yolah::MAX_NB_PLIES + 2][4]{};// U4: replies that cut after a sibling
        bool audit = false;                        // U5 (8): inside a root audit search
        uint64_t fn_counter = 0;                   // U10
        int  root_ply = 0;                         // U5: the root's ply
        uint64_t nb_checked = 0, nb_false = 0;     // U (H): decisions checked, wrong ones
        uint8_t last_stage = 0;                    // search_move: 1 reduced, 2 null window, 3 full re-search, 4 full
    };

    // Evaluation cache (D): one entry per slot, the newest wins. The low bits
    // of the hash choose the slot, its 47 high bits check the position (the
    // ones above the slot bits really do: 44 of them with the 2^20 slots of
    // the default, a wrong position about once in 1.8e13 probes). Each entry
    // is ONE 64-bit word, read and written atomically (L: the threads share
    // the cache; a torn entry — the key of one position with the value of
    // another — is impossible):
    //     bits 63..17: key (bits 63..17 of the hash)
    //     bits 16..1 : value (int16)
    //     bit  0     : used
    std::vector<std::atomic<uint64_t>> eval_cache;
    // The exact endgame solver (G), with its own table, kept between moves.
    EndgameSolver endgame;
    uint64_t nb_evals = 0, nb_eval_hits = 0;
    std::array<uint64_t, 4> asp_stats{};
    std::array<uint64_t, 4> fly_stats{};
    uint64_t region_cuts = 0;
    std::array<uint64_t, 2> validity{};   // U (H): decisions checked, wrong ones
    std::array<uint64_t, 2> audits{};     // U5 (8): audits, switches
    std::chrono::steady_clock::time_point search_start{};   // U5 (8)
    uint64_t search_us = 0;
    std::vector<int8_t> lmr_adjust;       // U5 (6): learnt adjustment per bucket (lmr_bucket)
    std::array<uint64_t, 3> prev_root{};  // U9: the previous root's black, white, empty bitboards
    int prev_root_ply = -1;
    static constexpr int LMR_BUCKETS = 8 * 5 * 4 * 4 * 3;
    Accumulator sym_acc;                  // U5 (9): the mirror's accumulator
    int  sym_eval(const Yolah& yolah);   // U: tests, prunes, cuts, nodes in the tests (search_bench)

    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    int  evaluate(const Yolah& yolah, Search& s, uint64_t hash);
    int  network_value(const Yolah& yolah, Search& s);
    int  raw_network_value(const Yolah& yolah, Search& s);
    void update_accumulator(const int16_t* in, int16_t* out, uint8_t player, Move m, NNUE_Persp::RefreshCache* cache) const;
    int  negamax(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);
    int  negamax_fly(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);   // U: the stronger reply
    int  fly_static(Yolah& yolah, Search&, uint64_t hash, int jumps, bool ordinary_only);
    int  negamax_nohole(Yolah& yolah, Search&, uint64_t hash, int alpha, int beta, int depth);   // U (D)
    uint64_t nohole_play(Yolah& yolah, Search& s, Move m, std::array<uint64_t, 6>& saved);
    void nohole_undo(Yolah& yolah, Search& s, const std::array<uint64_t, 6>& saved);
    bool verify_decision(Yolah& yolah, Search& s, uint64_t hash, Move m, int depth, int bound, bool prune);
    int  root_search(Yolah&, Search&, uint64_t hash, int alpha, int beta, int depth, Move&);
    int  late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const;
    int  search_move(Yolah&, Search&, uint64_t hash, Move m, size_t i, int reduction, int alpha, int beta, int depth);
    void score_moves(const Yolah&, const Search& s, Move tt_move, const Yolah::MoveList&, int* scores,
                     int depth = 0) const;
    int  territory_after(const Yolah&, Move) const;
    static bool is_legal(const Yolah&, Move);
    static Move pick_move(Yolah::MoveList&, int* scores, size_t i, size_t n);
    bool is_articulation_move(const Yolah&, Move) const;
    int  block_zone_score(const Yolah&, Move) const;
    void update_history(Search&, uint8_t player, Move m, int bonus);
    void iterative_deepening(Yolah, Search&, uint8_t max_depth);
    void print_pv(Yolah, uint64_t hash, int8_t depth);
    // The search debugger's helpers (defined only with YOLAH_SEARCH_TRACE, see search_trace.h).
    int  trace_return(int line, int value);
    int  trace_value(const Yolah&);
    void trace_child_evals(int line, const Yolah&);
    void trace_search_start(int line, const Yolah&);
    void trace_pv(int line, Yolah, uint64_t hash, int depth);

public:
    // T. Cutoff statistics for the history prior (search_bench cutstats):
    // for each move searched in the main loop of a node, per phase, colour,
    // from and to: how often it was tried and how often it cut.
    static constexpr int PRIOR_PHASES = 8;
    static int prior_phase(const Yolah& y) { return std::popcount(y.free_squares()) * PRIOR_PHASES / 57; }
    struct CutStats {
        uint32_t tried[PRIOR_PHASES][2][SQUARE_NB][SQUARE_NB]{};
        uint32_t cut[PRIOR_PHASES][2][SQUARE_NB][SQUARE_NB]{};
        void count(const Yolah& y, Move m, bool cutoff) {
            const int ph = prior_phase(y);
            const uint8_t p = y.current_player();
            tried[ph][p][m.from_sq()][m.to_sq()]++;
            cut[ph][p][m.from_sq()][m.to_sq()] += cutoff;
        }
    };
    CutStats* cut_stats = nullptr;   // when set, every search counts its cutoffs there
    // Data generation (test/selfplay_label_main.cpp): the PV nodes of the main
    // thread with an exact value (alpha < v < beta) searched to at least
    // "record depth", in the order they were finished. The value is for the
    // side to move, at the node's depth; the move is the node's best move.
    struct NodeRecord {
        uint64_t black, white, empty;          // empty: the holes
        uint16_t ply;                          // its parity: the side to move
        uint8_t  black_score, white_score;
        uint8_t  depth;
        int16_t  value;
        uint16_t move;                         // the best move found (Move::raw)
    };
    std::vector<NodeRecord>* recorder = nullptr;
    // Root log (search_bench run --rootlog): after each completed depth of the
    // main thread, one record per root move, in the order of the last attempt.
    // value ≤ alpha: upper bound (fail soft), value ≥ beta: lower bound,
    // between: exact. sigma: the mean |v_d − v_{d−1}| of the adaptive windows
    // before this depth (0 if none yet).
    struct RootRecord {
        int      tag;                          // root_log_tag (the bench position)
        uint8_t  depth, attempts, rank, rank_start, reduction, stage;
        uint16_t move;                         // Move::raw
        uint64_t nodes;
        int      value, alpha, beta;           // the move's value, the alpha it was searched with, the window's beta
        int      iter_value;                   // the value of the depth
        double   sigma;
    };
    std::vector<RootRecord>* root_log = nullptr;
    int root_log_tag = 0;
    // The search debugger's recorder (test/search_trace_main.cpp): only read by
    // a build with YOLAH_SEARCH_TRACE, where every step of the search is recorded.
    SearchTracer* tracer = nullptr;
    std::atomic_bool* stop_flag() { return &stop; }
    MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth, uint8_t late_move_reduction,
                         const std::string& nnue_q_parameters_filename, bool verbose = false, Options options = {},
                         size_t nb_threads = 1);
    Move play(Yolah) override;
    // Iterative deepening up to max_depth, stopped after `microseconds`
    // (0 = no time limit). For the benchmarks (test/search_bench_main.cpp).
    Result search(const Yolah&, uint8_t max_depth, uint64_t microseconds);
    // Leaf evaluations and how many came from the cache, since the start.
    std::pair<uint64_t, uint64_t> eval_stats() const { return {nb_evals, nb_eval_hits}; }
    // Summed over the searches since construction: attempts, failed, rebounds, nodes in failed attempts.
    std::array<uint64_t, 4> aspiration_stats() const { return asp_stats; }
    std::array<uint64_t, 4> fly_reply_stats() const { return fly_stats; }
    uint64_t region_bound_cuts() const { return region_cuts; }
    std::array<uint64_t, 2> validity_stats() const { return validity; }
    // U5 (9, search_bench): the network's value of a position and of its mirror image (side to move's view)
    std::pair<int, int> sym_values(const Yolah& yolah);
    // U5 (6, search_bench): where the LMR log goes
    std::ofstream* lmr_log_file = nullptr;
    uint64_t lmr_log_counter = 0;
    std::array<uint64_t, 2> audit_stats() const { return audits; }
    void clear_table(bool keep_ordering = false);   // forget everything: table, history, countermoves (reproducible
                                                    // benchmarks); keep_ordering: keep history and countermoves
    // The principal variation of the last search, read from the table (at most
    // max_len moves; it stops at a missing or illegal move). For the GUI (app/).
    std::vector<Move> principal_variation(Yolah, int max_len) const;
    std::string info() override;
    json config() override;
};

#endif
