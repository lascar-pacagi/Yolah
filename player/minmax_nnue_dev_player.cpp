#include "minmax_nnue_dev_player.h"
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "zobrist.h"
#include <utility>
#include <algorithm>
#include <cmath>
#include <bit>
#include <immintrin.h>
#include "magic.h"

using std::cout, std::endl;

MinMaxNNUE_DevPlayer::MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                                           uint8_t late_move_reduction, const std::string& nnue_q_parameters_filename,
                                           bool verbose, Options options, size_t nb_threads)
    : thinking_time(microseconds), tt_size_mb(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      nnue_q_parameters_filename(nnue_q_parameters_filename), options(options), verbose(verbose),
      nb_threads(std::max<size_t>(1, nb_threads)) {
    for (size_t i = 0; i < this->nb_threads; i++) workers.push_back(std::make_unique<Worker>());
    nnue.load(nnue_q_parameters_filename);
    if (options.yolah_table) yolah_table = std::make_unique<SearchTable>(tt_size_mb);
    else table = std::make_unique<TranspositionTable>(tt_size_mb);
    if (options.eval_cache_bits > 0) eval_cache = std::vector<std::atomic<uint64_t>>(size_t(1) << options.eval_cache_bits);
    // r(d, n) = base + ln(d)·ln(n) / divisor plies (n = move number, from 1).
    for (int d = 1; d < 64; d++) {
        for (int n = 1; n < Yolah::MAX_NB_MOVES; n++) {
            lmr_table[d][n] = int(1024 * (options.lmr_base + std::log(d) * std::log(n) / options.lmr_divisor));
        }
    }
}

MinMaxNNUE_DevPlayer::Result MinMaxNNUE_DevPlayer::search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds) {
    const auto start = std::chrono::steady_clock::now();
    stop = false;
    // The clock: sets `stop` after `microseconds`, unless the search is over
    // before (then the jthread's destructor wakes it up at once).
    std::mutex m;
    std::condition_variable_any cv;
    std::jthread clock;
    if (microseconds > 0) {
        clock = std::jthread([&](std::stop_token st) {
            std::unique_lock lock(m);
            if (!cv.wait_for(lock, st, std::chrono::microseconds(microseconds), [] { return false; })) {
                stop = true;
            }
        });
    }
    // G. Few free squares left: try to PROVE the result first (win / draw /
    // loss, see endgame_solver.h), with a share of the thinking time. A win or
    // a draw proven: its move is played, no search needed. A loss, or no proof
    // in time: the normal search below, with the time left — against a fallible
    // opponent, the move that "looks best" keeps more chances than any losing
    // move of the proof.
    if (options.endgame_root > 0 && std::popcount(yolah.free_squares()) <= options.endgame_root) {
        std::atomic_bool solver_stop = false;
        std::mutex sm;
        std::condition_variable_any scv;
        std::jthread solver_clock;
        if (microseconds > 0) {
            const auto budget = std::chrono::microseconds(uint64_t(microseconds * options.endgame_root_time));
            solver_clock = std::jthread([&, budget](std::stop_token st) {
                std::unique_lock lock(sm);
                if (!scv.wait_for(lock, st, budget, [] { return false; })) solver_stop = true;
            });
        }
        const auto proof = endgame.solve(yolah, true, &solver_stop);
        if (solver_clock.joinable()) {
            solver_clock.request_stop();
            solver_clock.join();
        }
        if (proof.complete && proof.value >= 0) {
            if (clock.joinable()) {
                clock.request_stop();
                clock.join();
            }
            Result r;
            r.move = proof.move;
            r.value = int16_t(proof.value > 0 ? WIN + proof.value : 0);
            r.depth = 63;
            r.nb_nodes = proof.nodes;
            r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            return r;
        }
    }
    if (table) table->new_search();
    else yolah_table->new_search(std::popcount(yolah.free_squares()));
    // History aging: what was learnt during the previous moves still helps
    // (the positions are close), but less and less.
    for (auto& w : workers)
        for (auto& by_from : w->history)
            for (auto& by_to : by_from)
                for (int16_t& h : by_to) h /= 2;
    // ── L. Lazy SMP ──
    // nb_threads threads run the SAME iterative deepening on the same root.
    // They share the transposition table (and the evaluation cache): what one
    // thread proves, the others find in the table and skip. They do not share
    // their move ordering (history, killers, root order): their trees differ,
    // so together they cover more of it than one thread would.
    //
    //     thread 0 (main) : depth 1 2 3 4 5 6 7 8 …   ← its result is played
    //     thread 1        : depth 1   3   5   7 …     (skips some depths,
    //     thread 2        : depth   2 3     6 7 …      see iterative_deepening)
    //            ╲   │   ╱
    //       shared transposition table
    //
    // No locks: a table entry written by one thread while another reads it can
    // be torn (half old, half new). Rare, and harmless enough: the key check
    // rejects most of them, and a table move is checked (is_legal / matched
    // against the generated moves) before being played.
    std::vector<std::unique_ptr<Search>> searches;
    for (size_t i = 0; i < nb_threads; i++) {
        auto s = std::make_unique<Search>();
        s->id = int(i);
        s->w = workers[i].get();
        // The root's accumulator: computed from scratch, the only one that is.
        if (options.lazy_accumulator) {
            nnue.init(yolah, s->accs[yolah.nb_plies()]);
            s->acc_ok[yolah.nb_plies()] = true;
        } else {
            nnue.init(yolah, s->acc);
        }
        searches.push_back(std::move(s));
    }
    {
        std::vector<std::jthread> helpers;
        for (size_t i = 1; i < nb_threads; i++) {
            helpers.emplace_back([&, i] { iterative_deepening(yolah, *searches[i], max_depth); });
        }
        iterative_deepening(yolah, *searches[0], max_depth);
        stop = true;            // the main thread is done (time or depth): the helpers stop too
    }                           // (joined here)
    if (clock.joinable()) {
        clock.request_stop();
        clock.join();
    }
    // The move played: the one of the thread that completed the deepest
    // iteration (the main thread on ties).
    const Search* best = searches[0].get();
    Result r;
    for (const auto& s : searches) {
        if (s->depth > best->depth && s->move != Move::none()) best = s.get();
        r.nb_nodes += s->nb_nodes;
        r.nb_hits += s->nb_hits;
        nb_evals += s->nb_evals;
        nb_eval_hits += s->nb_eval_hits;
    }
    r.move = best->move;
    r.value = best->value;
    r.depth = best->depth;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

void MinMaxNNUE_DevPlayer::clear_table() {
    if (table) table->clear(1);
    else yolah_table->clear();
    endgame.clear();
    // the move ordering statistics too: each benchmark position from scratch
    for (auto& w : workers) *w = Worker{};
    for (auto& e : eval_cache) e.store(0, std::memory_order_relaxed);
}

Move MinMaxNNUE_DevPlayer::play(Yolah yolah) {
    Result r = search(yolah, 63, thinking_time);
    if (verbose) {
        cout << "##########\n";
        cout << "depth  : " << int(r.depth) << '\n';
        cout << "value  : " << r.value << '\n';
        cout << "# nodes: " << r.nb_nodes << '\n';
        cout << "# hits : " << r.nb_hits << '\n';
        cout << "tt load: " << (table ? table->load() : yolah_table->load()) << '\n';
        print_pv(yolah, zobrist::hash(yolah), r.depth);
        cout << '\n';
    }
    return r.move;
}

std::string MinMaxNNUE_DevPlayer::info() {
    return "minmax nnue dev player (" + std::to_string(nb_threads) + " thread" + (nb_threads > 1 ? "s, lazy SMP" : "")
         + "; transposition table + late move reduction + killer"
         + (options.pvs ? " + PVS" : "")
         + (options.aspiration_window ? " + aspiration windows" : "")
         + (options.history ? " + history" : "")
         + (options.countermove ? " + countermove" : "")
         + (options.root_ordering ? " + root ordering" : "")
         + (options.lazy_accumulator ? " + lazy accumulators" : "")
         + (options.eval_cache_bits ? " + evaluation cache" : "")
         + (options.lmr ? " + logarithmic LMR" : "")
         + (options.rfp_depth ? " + reverse futility pruning" : "")
         + (options.null_move ? " + null move pruning" : "")
         + (options.lmp_depth ? " + late move pruning" : "")
         + (options.endgame_root || options.endgame_tree ? " + endgame solver" : "") + ")";
}

// ─── Late move reductions (E) ────────────────────────────────────────────────
// With a good move ordering, the best move is almost always among the first
// ones: the later a move comes, the less likely it is to matter, and the
// deeper the search, the more a full-depth search of it costs. So late moves
// are first searched at a reduced depth, and only searched again at full depth
// if that shallow search says they might beat alpha (see search_move).
//
// Reduction, in plies, of move number i (from 0) at a node of depth `depth`:
//     r = base + ln(depth)·ln(i + 1) / divisor          (lmr_table)
//         − 1   at a PV node (beta − alpha > 1): its value is the one we want
//         − 1   for a killer or a countermove: they cut elsewhere
//         − history / 8192  (−2 … +2): moves that usually cut are reduced less,
//                           moves that usually fail are reduced more
// The first move (the table's move) and the nodes of depth < 3 are never
// reduced, and the reduced search keeps at least one ply.
// Example (base 1.25, divisor 1.5, the defaults: in matches, 1.0 / 1.75 beat
// 0.75 / 2.25 and 1.25 / 1.5 beat 1.0 / 1.75), quiet move, neutral history:
//     depth 4, move 4: 1.25 + 1.39·1.39/1.5 = 2.5 → 2 plies
//     depth 10, move 20: 1.25 + 2.30·3.00/1.5 = 5.9 → 5 plies
// The reference reduces by a fixed late_move_reduction − 1 = 2 plies every move
// after the first nb_moves_at_full_depth = 2, at every depth (kept when "lmr"
// is false).
//
// Returns the number of plies to take off the normal depth − 1 (0 = none).
int MinMaxNNUE_DevPlayer::late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const {
    if (!options.lmr) {
        return i >= nb_moves_at_full_depth ? std::max(0, late_move_reduction - 1) : 0;
    }
    if (depth < 3 || i == 0) return 0;
    int r = lmr_table[std::min(depth, 63)][std::min<size_t>(i + 1, Yolah::MAX_NB_MOVES - 1)];
    if (pv_node) r -= 1024;
    if (special) r -= 1024;
    if (options.history) r -= hist * 1024 / 8192;
    return std::clamp(r / 1024, 0, depth - 2);
}

// ─── One move of a node ──────────────────────────────────────────────────────
// Plays move number i of the node, searches the child and returns its value
// for the side to move at the node. Callers check stopped() before using it:
// an interrupted child returns a meaningless 0.
//
// Three ways of searching a child, from the cheapest:
//   • a late move (reduction > 0, see late_move_reduction_of) is first
//     searched at a REDUCED depth, depth − 1 − reduction. If it does not beat
//     alpha, the move is believed bad and we stop there (LMR);
//   • with PVS, every move after the first one is then searched with a NULL
//     WINDOW (alpha, alpha + 1): the search only answers "is this move better
//     than alpha?", which cuts much more than a full window. The first move is
//     very often the best one (the transposition table's move), so the answer
//     is usually "no" and costs little;
//   • only when the answer is "yes" (alpha < v < beta) is the move searched
//     again with the full window, to get its exact value.
// Without PVS, every move that is not cut by LMR gets the full window (the
// reference's way).
int MinMaxNNUE_DevPlayer::search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i, int reduction,
                                      int alpha, int beta, int depth) {
    const uint8_t player = yolah.current_player();
    const uint64_t child = zobrist::update(hash, player, m);
    // H: start loading the child's table entry now; it is needed as soon as
    // the child's search starts.
    if (yolah_table) yolah_table->prefetch(child);
    s.played[yolah.nb_plies()] = m;       // for the countermove and the accumulator of the child
    if (options.lazy_accumulator) {
        s.acc_ok[yolah.nb_plies() + 1] = false;   // the child's accumulator: computed if needed
    } else {
        nnue.play(player, m, s.acc);
    }
    yolah.play(m);
    int v = alpha + 1;                    // "might beat alpha" until a search says otherwise
    if (reduction > 0) {                  // late move: reduced search first
        const int d = depth - 1 - reduction;
        v = options.pvs ? -negamax(yolah, s, child, -(alpha + 1), -alpha, d)
                        : -negamax(yolah, s, child, -beta, -alpha, d);
    }
    if (v > alpha && !stopped()) {
        if (options.pvs && i > 0) {
            v = -negamax(yolah, s, child, -(alpha + 1), -alpha, depth - 1);       // null window
            if (v > alpha && v < beta && !stopped()) {
                v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);          // exact value
            }
        } else {
            v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
        }
    }
    yolah.undo(m);
    if (!options.lazy_accumulator) nnue.undo(player, m, s.acc);
    return v;
}

// ─── Inner nodes ─────────────────────────────────────────────────────────────
// Fail-soft negamax: the returned value v is
//   • exact when alpha < v < beta,
//   • an upper bound of the true value when v ≤ alpha (every move failed low),
//   • a lower bound when v ≥ beta (a move was good enough to cut).
// Fail-soft (returning the best value found rather than alpha or beta) gives
// tighter bounds to the transposition table and to the aspiration windows.
int MinMaxNNUE_DevPlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth) {
    ++s.nb_nodes;
    if (stopped()) return 0;    // the caller ignores this value
    // 1. Finished game: the exact score (difference of the numbers of moves),
    //    pushed beyond any evaluation (|eval| ≤ WIN) so that a real win always
    //    beats a position that only looks good.
    if (yolah.game_over()) {
        int score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? WIN : -WIN);
    }
    // 2. Transposition table: a result of a search at least as deep as the
    //    one asked can answer at once, if its bound is good enough. Yolah has
    //    no cycles (every move leaves a hole), so a position's value never
    //    depends on the path to it: the table can be trusted everywhere.
    const TTView tt = tt_probe(hash);
    const Move tt_move = tt.move;
    if (tt.found) {
        s.nb_hits++;
        if (tt.depth >= depth) {
            if (tt.lower >= beta) return tt.lower;
            if (tt.upper <= alpha) return tt.upper;
            if (tt.lower == tt.upper) return tt.lower;
        }
    }
    // G. Very few free squares: the exact result from the endgame solver
    //    (win / draw / loss — only the sign matters for the outcome), instead
    //    of a search with the network. Stored in the table at the largest
    //    depth: it never needs to be searched again.
    // (Main thread only: the solver and its table are not shared.)
    if (options.endgame_tree > 0 && s.id == 0 && std::popcount(yolah.free_squares()) <= options.endgame_tree) {
        const auto proof = endgame.solve(yolah, true, &stop);
        if (!proof.complete) return 0;          // stopped: the caller ignores it
        const int v = proof.value > 0 ? WIN + proof.value : proof.value < 0 ? -WIN + proof.value : 0;
        tt_store(hash, yolah, 63, -INFINITE, INFINITE, v, proof.move);
        return v;
    }
    // 3. Horizon: the network's value for the side to move, scaled to ±WIN.
    //    FIX (A): not stored in the table. The table treats depth 0 as "empty
    //    slot" (TranspositionTable::probe), so the reference's leaf entries were
    //    never found again, and when a cluster was full they evicted a real
    //    entry of an inner node.
    if (depth <= 0) {
        return evaluate(yolah, s, hash);
    }
    // F. Pruning, at non-PV nodes only (null window: the question is only
    //    "is this node ≥ beta?", an approximate answer costs little; at PV
    //    nodes we want the exact value). It relies on the STATIC evaluation:
    //    the network's value of this node, before any search below it (cheap
    //    now: lazy accumulators and evaluation cache, see evaluate). Never
    //    near game results (|beta| ≥ WIN): a proven win must stay proven.
    const bool pv_node = beta - alpha > 1;
    const bool prunable = !pv_node && std::abs(beta) < WIN;
    int static_eval = 0;
    if (prunable && ((options.rfp_depth > 0 && depth <= options.rfp_depth) || (options.null_move && depth >= 3)
                     || (options.proxy_cut == 1 && depth >= options.proxy_depth))) {
        static_eval = evaluate(yolah, s, hash);
    }
    // F1. Reverse futility pruning ("static null move"): if the static value
    //     is above beta by a margin that a shallow search will hardly lose,
    //     trust it and cut. The margin grows with the depth: the deeper the
    //     search, the more the value may still change.
    if (prunable && depth <= options.rfp_depth && static_eval - options.rfp_margin * depth >= beta) {
        return static_eval;
    }
    // K. Staged move generation. The table's move is the best move of an
    //    earlier search of this position: it often cuts at once, and then the
    //    other moves are never needed. So it is searched FIRST, before the
    //    other moves are generated and scored; they are generated only if it
    //    does not cut (stage 2, at i = 1 in the loop below):
    //
    //        stage 1: table's move ──cut──→ return       (no generation at all)
    //                      │ no cut
    //        stage 2: generate all moves, put the table's move at index 0,
    //                 score the others (killers, countermove, history), and
    //                 pick them one at a time as before.
    //
    //    In pure alpha-beta the tree is exactly the same as without stages
    //    (checked: same nodes and values on the bench positions). With
    //    reductions it differs slightly: the other moves are scored AFTER the
    //    table's move was searched, so with killers and history updated by
    //    its subtree — fresher information (−4 % nodes, −6 % time).
    //    A table move can be illegal here (two positions with the same hash
    //    bits, or the entry of another position): is_legal checks it.
    Yolah::MoveList moves;
    const bool tt_first = options.staged && tt_move != Move::none() && is_legal(yolah, tt_move);
    bool generated = false;
    if (!tt_first) {
        yolah.moves(moves);
        generated = true;
    }
    // The pass rule (see EndgameSolver::search for the proof): a player who
    // must pass while the game is not over has lost — their final score
    // difference is ≤ −1, so the value is ≤ −(WIN + 1). An upper bound,
    // returned when it is enough to fail low. (A legal table move: not a pass.)
    if (options.pass_rule && !tt_first && moves[0] == Move::none() && -(WIN + 1) <= alpha) {
        return -(WIN + 1);
    }
    // F2. Null move pruning: let the opponent play twice (we pass). If even
    //     with that handicap a reduced search still says ≥ beta, the position
    //     is so good that a real move would cut too: cut now. In Yolah a pass
    //     costs a point (every move scores one), so the handicap is real —
    //     the idea's assumption "moving is better than passing" mostly holds.
    //     Exceptions (zugzwang: every move spoils our own region) exist but
    //     should be rare. Not twice in a row, and not when we must pass anyway.
    //     MEASURED: −81 Elo. In Yolah, zugzwangs are the rule rather than the
    //     exception (every move leaves a hole in one's own space): off by default.
    if (prunable && options.null_move && depth >= 3 && static_eval >= beta
        && (tt_first || moves[0] != Move::none()) && yolah.nb_plies() > 0
        && s.played[yolah.nb_plies() - 1] != Move::none()) {
        const int R = options.null_move_reduction + depth / 4 - 1;
        const uint8_t player = yolah.current_player();
        s.played[yolah.nb_plies()] = Move::none();
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else nnue.play(player, Move::none(), s.acc);
        yolah.play(Move::none());
        const int v = -negamax(yolah, s, zobrist::update(hash, player, Move::none()), -beta, -beta + 1, depth - 1 - R);
        yolah.undo(Move::none());
        if (!options.lazy_accumulator) nnue.undo(player, Move::none(), s.acc);
        if (stopped()) return 0;
        if (v >= beta) return v >= WIN ? beta : v;   // no unproven "win" from a null move
    }
    // 4. The moves, the most promising first. They are scored once, then
    //    picked one at a time (the best remaining one): most nodes cut after
    //    one or two moves, so a full sort would be wasted work.
    int scores[Yolah::MAX_NB_MOVES];
    size_t n = 1;                         // stage 1: only the table's move
    if (generated) {
        score_moves(yolah, s, tt_move, moves, scores, depth);
        n = moves.size();
    }
    // N. Proxy cut — the idea of the null move without its flaw in Yolah.
    //    The null move asks "even if I pass, am I still ≥ beta?", which
    //    assumes that passing is worse than the best move: false in a
    //    zugzwang, and Yolah is full of them (−81 Elo, F2). A REAL move m
    //    has no such problem: the best move is at least as good as any move,
    //
    //        value(best) ≥ value(m)   so   value(m) ≥ beta  ⇒  value(best) ≥ beta,
    //
    //    and in a zugzwang every move is bad: the test fails, no wrong cut.
    //    Like the null move, m is searched at a REDUCED depth (depth − 1 − R)
    //    to be cheap — the only approximation. Which m?
    //      1 = an ORDINARY move: the proxy_rank-th after the special ones
    //          (table's move, killers, countermove), by history. Demanding:
    //          if even a run-of-the-mill move reaches beta, the node is good.
    //      2 = ProbCut (Stockfish): the table's move (the likely best),
    //          against beta + probcut_margin instead — the margin makes the
    //          test demanding.
    //    Tried at non-PV nodes of depth ≥ proxy_depth whose static value is
    //    already ≥ beta (variant 1) — elsewhere it would rarely succeed and
    //    its cost would be wasted.
    if (options.proxy_cut > 0 && prunable && depth >= options.proxy_depth
        && (options.proxy_cut == 2 || static_eval >= beta)) {
        if (!generated) {                       // the test needs the move list
            yolah.moves(moves);
            generated = true;
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
        }
        Move m = Move::none();
        int bound = beta;
        if (options.proxy_cut == 1) {
            // the proxy_rank-th best-scored ordinary move
            bool taken[Yolah::MAX_NB_MOVES]{};
            for (int r = 0; r <= options.proxy_rank; r++) {
                int pick = -1;
                for (size_t j = 0; j < n; j++) {
                    if (!taken[j] && scores[j] < SCORE_COUNTER && (pick < 0 || scores[j] > scores[pick])) pick = int(j);
                }
                if (pick < 0) { m = Move::none(); break; }
                taken[pick] = true;
                m = moves[pick];
            }
        } else {
            m = tt_first ? tt_move : moves[0];
            if (!tt_first) {                    // no table move: the best-scored one
                size_t b = 0;
                for (size_t j = 1; j < n; j++) if (scores[j] > scores[b]) b = j;
                m = moves[b];
            }
            bound = std::min(beta + options.probcut_margin, WIN);
        }
        if (m != Move::none()) {
            const int R = options.proxy_reduction + depth / 4 - 1;
            const uint8_t player = yolah.current_player();
            const uint64_t child = zobrist::update(hash, player, m);
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else nnue.play(player, m, s.acc);
            yolah.play(m);
            const int v = -negamax(yolah, s, child, -bound, -bound + 1, depth - 1 - R);
            yolah.undo(m);
            if (!options.lazy_accumulator) nnue.undo(player, m, s.acc);
            if (stopped()) return 0;
            if (v >= bound) return v >= WIN ? beta : v;   // no unproven win from a reduced search
        }
    }
    const int alpha_orig = alpha;
    int best = -INFINITE;
    Move best_move = Move::none();
    bool pruned = false;
    for (size_t i = 0; i < n || !generated; i++) {
        if (!generated && i == 1) {
            // Stage 2: the table's move did not cut. Generate everything, put
            // the table's move where the selection would have put it (index
            // 0, swapped with what was there), score the others.
            yolah.moves(moves);
            generated = true;
            for (size_t j = 0; j < moves.size(); j++) {
                if (moves[j] == tt_move) {
                    std::swap(moves[0], moves[j]);
                    break;
                }
            }
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
            if (i >= n) break;
        }
        // F3. Late move pruning: at shallow non-PV nodes, after lmp_moves +
        //     depth² moves, the others (the worst ordered: bad history, never
        //     cut anywhere) are not searched at all — once a move that does not
        //     lose has been found. I: except the articulation moves, which are
        //     kept (moved to the front of what is left).
        if (!pruned && prunable && depth <= options.lmp_depth && i >= size_t(options.lmp_moves + depth * depth)
            && best > -WIN) {
            if (!options.articulation_lmr) break;
            size_t k = i;
            for (size_t j = i; j < n; j++) {
                if (is_articulation_move(yolah, moves[j])) {
                    std::swap(moves[k], moves[j]);
                    std::swap(scores[k], scores[j]);
                    k++;
                }
            }
            n = k;
            pruned = true;
            if (i >= n) break;
        }
        const Move m = generated ? pick_move(moves, scores, i, n) : tt_move;   // stage 1: no list yet
        // scores[i] is now m's score: special move (killer, countermove), articulation or history
        const bool tactical = options.articulation_lmr && is_articulation_move(yolah, m);
        const int r = tactical || !generated ? 0          // (the first move is never reduced anyway)
                    : late_move_reduction_of(depth, i, beta - alpha > 1, scores[i] >= SCORE_COUNTER,
                                             scores[i] >= SCORE_ARTICULATION ? 0 : scores[i]);
        const int v = search_move(yolah, s, hash, m, i, r, alpha, beta, depth);
        // FIX (A): once the clock has stopped the search, the values coming up
        // are meaningless: return before storing anything (the reference stored
        // them, and the table is kept for the next moves of the game).
        if (stopped()) return 0;
        if (v > best) {
            best = v;
            if (v > alpha) {
                best_move = m;
                if (v >= beta) {
                    // Beta cutoff: the opponent will not allow this position.
                    tt_store(hash, yolah, depth, alpha_orig, beta, v, m);
                    // Killer moves: quiet moves that cut at the same ply in a
                    // sibling node, tried early. FIX (A): only cutting moves
                    // (the reference also stored the best move of nodes without
                    // cutoff, often Move::none(), erasing a good killer).
                    const uint16_t ply = yolah.nb_plies();
                    if (s.killer1[ply] != m) {
                        s.killer2[ply] = s.killer1[ply];
                        s.killer1[ply] = m;
                    }
                    // History (C): the cutting move gets a bonus, the moves
                    // tried before it (moves[0..i-1], they did not cut) a
                    // malus of the same size. Deeper cutoffs say more about a
                    // move, hence a bonus growing with the depth.
                    if (options.history) {
                        const uint8_t player = yolah.current_player();
                        const int bonus = std::min(16 * depth * depth + 32 * depth, 2000);
                        update_history(s, player, m, bonus);
                        for (size_t k = 0; k < i; k++) update_history(s, player, moves[k], -bonus);
                    }
                    // Countermove (C): m refuted the opponent's last move.
                    if (options.countermove && ply > 0) {
                        const Move prev = s.played[ply - 1];
                        s.w->countermoves[prev.from_sq()][prev.to_sq()] = m;
                    }
                    return v;
                }
                alpha = v;
            }
        }
    }
    // No cutoff: exact value if a move raised alpha, otherwise only an upper
    // bound (all the moves failed low against the window we were given).
    tt_store(hash, yolah, depth, alpha_orig, beta, best, best_move);
    return best;
}

// ─── Articulation moves (I) ──────────────────────────────────────────────────
// The free squares form regions (8-connectivity: a queen also slides
// diagonally, so two free squares touching by a corner are connected). A move
// fills its destination (the origin, already occupied, becomes a hole): if
// the destination is an ARTICULATION POINT of its region, the move cuts the
// region in two — it creates territory. These are the "tactical" moves of
// Yolah, like captures in chess.
//
// Exact articulation points need a depth-first search of the whole region
// (Tarjan). The local test below is O(1): a square can only disconnect its
// region if its free neighbours fall into at least two groups that do not
// touch each other inside the 3×3 neighbourhood. (If they all touch, any path
// through the square can go around it.) So every true articulation point
// passes the test; a few other squares too (their groups meet again further
// away). The answer depends only on which of the ≤ 8 neighbours are free:
// articulation_lut[s][pext(free, neighbours of s)], built once.
//
// Diagrams: `.` free square, `#` hole or piece, `X` the destination of the
// move, `1` / `2` its free neighbours, numbered by group (neighbours of the
// same group touch each other, corners included).
//
//   (a) an articulation point: X is the only passage between the top and
//       the bottom region. Its free neighbours form two groups that do not
//       touch (1 and 2 are two rows apart): the test says yes, and filling X
//       does cut the region in two.
//
//         . . . . # . . .
//         . . . # 1 1 . .
//         # # # # X # # #
//         . . . 2 2 # . .
//         . . . . . # . .
//
//   (b) not an articulation point: the free neighbours of X all touch each
//       other around it (a single group 1), so any path through X can go
//       around it. The test says no.
//
//         . . . . .
//         . 1 1 # .
//         . 1 X # .
//         . 1 1 # .
//         . . . . .
//
//   (c) a false positive: two groups around X (1 above, 2 below: not
//       touching), but they meet again by the outer ring. The test says yes,
//       although filling X disconnects nothing. A detour longer than the 3×3
//       neighbourhood is invisible to a local test; the exact answer would
//       need a search of the whole region (Tarjan).
//
//         . . . . .
//         . # 1 # .
//         . # X # .
//         . # 2 # .
//         . . . . .
namespace {
    struct ArticulationTables {
        uint64_t neighbours[64];
        uint8_t lut[64][256];
        ArticulationTables() {
            for (int s = 0; s < 64; s++) {
                const int f = s & 7, r = s >> 3;
                int cells[8], n = 0;          // the neighbours of s, by increasing square (pext order)
                neighbours[s] = 0;
                for (int t = 0; t < 64; t++) {
                    const int df = (t & 7) - f, dr = (t >> 3) - r;
                    if (t != s && std::abs(df) <= 1 && std::abs(dr) <= 1) {
                        neighbours[s] |= uint64_t(1) << t;
                        cells[n++] = t;
                    }
                }
                for (int pattern = 0; pattern < (1 << n); pattern++) {
                    // groups of free neighbours, by union-find on king adjacency
                    int parent[8];
                    for (int a = 0; a < n; a++) parent[a] = a;
                    auto find = [&](int a) { while (parent[a] != a) a = parent[a]; return a; };
                    for (int a = 0; a < n; a++) {
                        for (int b = a + 1; b < n; b++) {
                            if (!((pattern >> a) & 1) || !((pattern >> b) & 1)) continue;
                            const int ta = cells[a], tb = cells[b];
                            if (std::abs((ta & 7) - (tb & 7)) <= 1 && std::abs((ta >> 3) - (tb >> 3)) <= 1) {
                                parent[find(a)] = find(b);
                            }
                        }
                    }
                    int groups = 0;
                    for (int a = 0; a < n; a++) groups += ((pattern >> a) & 1) && find(a) == a;
                    lut[s][pattern] = groups >= 2;
                }
            }
        }
    };
    const ArticulationTables articulation_tables;
}

bool MinMaxNNUE_DevPlayer::is_articulation_move(const Yolah& yolah, Move m) const {
    if (m == Move::none()) return false;
    const int to = m.to_sq();
    const uint64_t pattern = _pext_u64(yolah.free_squares(), articulation_tables.neighbours[to]);
    return articulation_tables.lut[to][pattern];
}

// ─── Transposition table access (H) ──────────────────────────────────────────
// Both tables seen the same way: a value interval [lower, upper] proven at
// some depth. The reference's table keeps one value and a bound type:
// exact → [v, v], lower bound → [v, +∞], upper bound → [−∞, v].
MinMaxNNUE_DevPlayer::TTView MinMaxNNUE_DevPlayer::tt_probe(uint64_t hash) const {
    TTView view;
    if (yolah_table) {
        if (const SearchTable::Entry* e = yolah_table->probe(hash)) {
            view = {true, e->move, e->depth, e->lower, e->upper};
        }
        return view;
    }
    bool found;
    const TranspositionTableEntry* e = table->probe(hash, found);
    if (found) {
        view.found = true;
        view.move = e->move();
        view.depth = e->depth();
        const int v = e->value();
        if (e->bound() & BOUND_LOWER) view.lower = v;
        if (e->bound() & BOUND_UPPER) view.upper = v;
    }
    return view;
}

// The result of a search of `depth` with the window (alpha, beta): fail-soft
// `value` (≤ alpha: upper bound, ≥ beta: lower bound, between: exact).
void MinMaxNNUE_DevPlayer::tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta, int value, Move move) {
    if (yolah_table) {
        yolah_table->store(hash, std::popcount(yolah.free_squares()), depth, alpha, beta, value, move);
        return;
    }
    const Bound b = value >= beta ? BOUND_LOWER : value > alpha ? BOUND_EXACT : BOUND_UPPER;
    table->update(hash, int16_t(value), b, uint8_t(depth), move);
}

Move MinMaxNNUE_DevPlayer::tt_move(uint64_t hash) const {
    return yolah_table ? yolah_table->get_move(hash) : table->get_move(hash);
}

// ─── Evaluation and lazy accumulators (D) ───────────────────────────────────
// The network's value for the side to move, scaled to ±WIN.
//
// The NNUE's first layer is a sum of weight rows, one per feature of the
// position (black pieces, white pieces, holes, side to move): the
// "accumulator". A move changes four features (the piece leaves `from`, arrives
// on `to`, `from` becomes a hole, the side to move flips), so the accumulator
// of a child is the parent's plus four rows — much cheaper than from scratch.
//
// The reference applies these four rows at EVERY move searched (nnue.play)
// and takes them off again after (nnue.undo), even for children that never
// need an evaluation: answered by the transposition table, finished games,
// inner nodes. Here, an accumulator is only computed when a leaf asks for it:
// we go up the current line to the last valid accumulator, then down again,
// one fused pass per ply (out = in − from + to + hole ± turn). The parent's
// accumulator is never modified, so there is nothing to undo.
//
// The result is bit for bit the reference's: int16 additions wrap around, so
// their order does not matter.
//
// Evaluation cache: a leaf reached again by another move order (very common:
// the same moves in a different order give the same position) takes its value
// from the cache, without network nor accumulators. The transposition table
// cannot hold these values (its depth-0 entries count as empty slots).
int MinMaxNNUE_DevPlayer::evaluate(const Yolah& yolah, Search& s, uint64_t hash) {
    s.nb_evals++;
    std::atomic<uint64_t>* e = nullptr;
    const uint64_t key = hash & 0xFFFFFFFF00000000ULL;
    if (!eval_cache.empty()) {
        e = &eval_cache[hash & (eval_cache.size() - 1)];
        const uint64_t word = e->load(std::memory_order_relaxed);
        if ((word & 1) && (word & 0xFFFFFFFF00000000ULL) == key) {
            s.nb_eval_hits++;
            return int16_t(uint16_t(word >> 16));
        }
    }
    const int v = network_value(yolah, s);
    if (e) e->store(key | (uint64_t(uint16_t(int16_t(v))) << 16) | 1, std::memory_order_relaxed);
    return v;
}

// M. Evaluation grain. Alpha-beta only compares values, and a comparison
// that ends in a tie with beta (v ≥ beta) is a cutoff: with a coarser
// evaluation, more siblings tie, more nodes cut, and aspiration windows and
// null-window searches settle faster. The price: differences smaller than the
// grain become invisible. (Old chess programs rounded their evaluation on
// purpose for this.) Values: ±30000 = tanh ±1; a grain of 1024 leaves about
// 60 levels.
//
//     value:   …  −1024     0    1024  2048 …       (grain 1024)
//     network:  −700 → −1024, 400 → 0, 600 → 1024
//
// Symmetric rounding to the nearest multiple; finished games (|v| > WIN) are
// never rounded — only the network's values pass here.
static int round_to_grain(int v, int grain) {
    if (grain <= 1) return v;
    const int q = v >= 0 ? (v + grain / 2) / grain : -((-v + grain / 2) / grain);
    return std::clamp(q * grain, -heuristic::MAX_VALUE, int(heuristic::MAX_VALUE));
}

int MinMaxNNUE_DevPlayer::network_value(const Yolah& yolah, Search& s) {
    return round_to_grain(raw_network_value(yolah, s), options.eval_grain);
}

int MinMaxNNUE_DevPlayer::raw_network_value(const Yolah& yolah, Search& s) {
    if (!options.lazy_accumulator) {
        return int(nnue.value(s.acc, yolah.current_player()) * WIN);
    }
    const int ply = yolah.nb_plies();
    int k = ply;
    while (!s.acc_ok[k]) k--;                 // the root's is always valid
    for (; k < ply; k++) {
        // Player at ply k: ply parity (a pass also counts as a ply).
        update_accumulator(s.accs[k].acc, s.accs[k + 1].acc, uint8_t(k & 1), s.played[k]);
        s.acc_ok[k + 1] = true;
    }
    return int(nnue.value(s.accs[ply], yolah.current_player()) * WIN);
}

// out = in + the change of features of move m by `player` (see NNUE_Quantized::play,
// whose arithmetic it reproduces, in one pass and without modifying `in`).
void MinMaxNNUE_DevPlayer::update_accumulator(const int16_t* __restrict in, int16_t* __restrict out,
                                              uint8_t player, Move m) const {
    constexpr int H = NNUE_Quantized::H1_SIZE;
    // Row of the "white to move" feature: added when black moves (white is to
    // move after), taken off when white moves.
    const int8_t* __restrict turn = nnue.input_to_h1 + (NNUE_Quantized::INPUT_SIZE - 1) * H;
    const int16_t sign = player == Yolah::BLACK ? 1 : -1;
    if (m == Move::none()) {                  // a pass: only the side to move changes
        for (int j = 0; j < H; j++) out[j] = int16_t(in[j] + sign * turn[j]);
        return;
    }
    // Feature rows: black pieces 0..63, white pieces 64..127, holes 128..191,
    // squares numbered 63 − square (the encoding of the training scripts).
    const int from = 63 - m.from_sq(), to = 63 - m.to_sq();
    const int pieces = player == Yolah::BLACK ? 0 : 64;
    const int8_t* __restrict w_from = nnue.input_to_h1 + (pieces + from) * H;
    const int8_t* __restrict w_to   = nnue.input_to_h1 + (pieces + to) * H;
    const int8_t* __restrict w_hole = nnue.input_to_h1 + (128 + from) * H;
    for (int j = 0; j < H; j++) {
        out[j] = int16_t(in[j] - w_from[j] + w_to[j] + w_hole[j] + sign * turn[j]);
    }
}

// ─── Root ────────────────────────────────────────────────────────────────────
// Like an inner node, but it returns the best move in `res`, and only moves
// whose search was COMPLETED (not interrupted by the clock) can become `res`:
// so even an interrupted iteration gives a usable move (see
// iterative_deepening).
//
// Root ordering (C): the root moves are kept from one iteration to the next
// (s.root_moves). After each search of the root, the best move goes first and
// the others are sorted by the size of their subtree: a move that took many
// nodes to refute is a move that came close, a good candidate to become the
// best at the next depth. (Their values cannot be used for this: with PVS,
// all moves but the best only have upper bounds from null windows.)
int MinMaxNNUE_DevPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth, Move& res) {
    res = Move::none();
    const int alpha_orig = alpha;
    int best = -INFINITE;
    if (!options.root_ordering) {
        // The reference's way: ordered like an inner node, at each iteration.
        s.root_moves.clear();
        Yolah::MoveList moves;
        yolah.moves(moves);
        int scores[Yolah::MAX_NB_MOVES];
        score_moves(yolah, s, tt_move(hash), moves, scores, 63);
        for (size_t i = 0; i < moves.size(); i++) s.root_moves.push_back({pick_move(moves, scores, i, moves.size())});
    }
    for (size_t i = 0; i < s.root_moves.size(); i++) {
        RootMove& rm = s.root_moves[i];
        const uint64_t nodes_before = s.nb_nodes;
        // The root is a PV node; no killers there, the history still speaks.
        const int r = options.articulation_lmr && is_articulation_move(yolah, rm.move) ? 0
                    : late_move_reduction_of(depth, i, true, false,
                                             s.w->history[yolah.current_player()][rm.move.from_sq()][rm.move.to_sq()]);
        const int v = search_move(yolah, s, hash, rm.move, i, r, alpha, beta, depth);
        rm.nodes = s.nb_nodes - nodes_before;
        if (stopped()) return best;
        if (v > best) {
            best = v;
            if (v > alpha) {
                res = rm.move;
                alpha = v;
                if (v >= beta) break;     // aspiration fail high: the caller widens the window
            }
        }
    }
    if (options.root_ordering) {
        std::stable_sort(s.root_moves.begin(), s.root_moves.end(),
                         [&](const RootMove& a, const RootMove& b) {
                             if ((a.move == res) != (b.move == res)) return a.move == res;
                             return a.nodes > b.nodes;
                         });
    }
    tt_store(hash, yolah, depth, alpha_orig, beta, best, res);
    return best;
}

// ─── Move ordering ───────────────────────────────────────────────────────────
// Scores, the highest searched first:
//   1. the table's move (the best move of an earlier search of this position),
//   2. the two killer moves of the ply (moves that cut in sibling nodes),
//   3. the countermove of the opponent's last move,
//   4. the others by history (C) — or, without history, in the generator's
//      order (the reference's way).
// All Yolah moves are "quiet" (no captures), so there is no static way to
// recognise a good move: the ordering only learns from the search itself.
void MinMaxNNUE_DevPlayer::score_moves(const Yolah& yolah, const Search& s, Move tt_move,
                                       const Yolah::MoveList& moves, int* scores, int depth) const {
    // J: territory ordering only where it pays (deep nodes: few, and a good
    // order there saves the most).
    const bool territory = options.territory_ordering > 0 && depth >= options.territory_depth;
    const uint16_t ply = yolah.nb_plies();
    const uint8_t player = yolah.current_player();
    Move counter = Move::none();
    if (options.countermove && ply > 0) {
        const Move prev = s.played[ply - 1];
        counter = s.w->countermoves[prev.from_sq()][prev.to_sq()];
    }
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        if (m == tt_move)                 scores[i] = SCORE_TT;
        else if (m == s.killer1[ply])     scores[i] = SCORE_KILLER1;
        else if (m == s.killer2[ply])     scores[i] = SCORE_KILLER2;
        else if (m == counter && counter != Move::none()) scores[i] = SCORE_COUNTER;
        else if (options.articulation_ordering && is_articulation_move(yolah, m))
            scores[i] = SCORE_ARTICULATION + (options.history ? s.w->history[player][m.from_sq()][m.to_sq()] : 0);
        else if (options.history)         scores[i] = s.w->history[player][m.from_sq()][m.to_sq()];
        else                              scores[i] = -int(i);
        if (territory && scores[i] < SCORE_ARTICULATION) {
            scores[i] += options.territory_weight * territory_after(yolah, m);
        }
    }
}

// A move of the side to move is legal if its piece is on the origin and the
// destination is free and reachable in a straight line through free squares
// (the queen attacks of the origin, with the occupied squares as blockers).
// Yolah::valid only checks the first two.
bool MinMaxNNUE_DevPlayer::is_legal(const Yolah& yolah, Move m) {
    if (m == Move::none()) return false;
    const uint64_t from = uint64_t(1) << m.from_sq(), to = uint64_t(1) << m.to_sq();
    if (!(yolah.bitboard(yolah.current_player()) & from)) return false;
    return attacks_bb(m.from_sq(), yolah.occupied_squares()) & yolah.free_squares() & to;
}

// ─── Territory after a move (J) ──────────────────────────────────────────────
// Who controls which free squares, after move m, for the player making it:
// (squares closer to their pieces) − (squares closer to the opponent's). Ties
// are neutral. Two distances (see the study nnue/move_features.py and the
// comparison with the convnet's choices: both predict its move 4–6× better
// than chance):
//
//   1. influence — heuristic::influence: both sides flood the free squares
//      one KING step at a time, simultaneously; a square reached by both at
//      the same step is neutral, and neutrality spreads to the free squares
//      next to it. (Amazons' "t2".)
//   2. queen distance — the minimum number of QUEEN moves (sliding through
//      free squares) to reach the square: a breadth-first search, level by
//      level, with the magic attack tables. A long open line is one move
//      away, not seven steps. (Amazons' "t1": the better one while the board
//      is open.)
//   3. mobility — not a territory but the immediate version: (number of
//      legal moves of the player) − (number of the opponent's), the squares
//      at queen distance 1 counted once per piece that reaches them. The
//      cheapest of the three (8 attack lookups).
//
// Example (B black, W white, # hole), the owner of each free square by queen
// distance (b, w, = for a tie):
//
//      4 | . . . W            4 | = w w W
//      3 | . # . .            3 | b # = w
//      2 | . . # .     →      2 | b = # w      territory 4 − 4 = 0
//      1 | B . . .            1 | B b b =
//          a b c d                a b c d
int MinMaxNNUE_DevPlayer::territory_after(const Yolah& yolah, Move m) const {
    if (m == Move::none()) return 0;
    const uint8_t player = yolah.current_player();
    const uint64_t from = uint64_t(1) << m.from_sq(), to = uint64_t(1) << m.to_sq();
    const uint64_t me = (yolah.bitboard(player) & ~from) | to;
    const uint64_t opp = yolah.bitboard(Yolah::other_player(player));
    const uint64_t free = yolah.free_squares() & ~to;     // the origin becomes a hole: not free
    const uint64_t occupied = ~free;
    if (options.territory_ordering == 3) {
        int n = 0;
        for (uint64_t b = me; b;)  n += std::popcount(attacks_bb(pop_lsb(b), occupied) & free);
        for (uint64_t b = opp; b;) n -= std::popcount(attacks_bb(pop_lsb(b), occupied) & free);
        return n;
    }
    if (options.territory_ordering == 1) {
        auto one_step = [&](uint64_t b) {
            return (shift<NORTH>(b) | shift<SOUTH>(b) | shift<EAST>(b) | shift<WEST>(b) |
                    shift<NORTH_EAST>(b) | shift<SOUTH_EAST>(b) | shift<NORTH_WEST>(b) | shift<SOUTH_WEST>(b)) & free;
        };
        uint64_t mi = me, oi = opp, mf = me, of = opp, neutral = 0;
        for (;;) {
            const uint64_t omi = mi, ooi = oi;
            mf = one_step(mf) & ~oi;
            of = one_step(of) & ~mi;
            neutral |= one_step(neutral) | (mf & of);
            mf &= ~neutral;
            of &= ~neutral;
            mi |= mf;
            oi |= of;
            if (mi == omi && oi == ooi) break;
        }
        return std::popcount(mi & free) - std::popcount(oi & free);
    }
    // Queen distance, both sides level by level: at level k, the squares first
    // reached by one side only are theirs, those reached by both are ties.
    auto expand = [&](uint64_t frontier) {
        uint64_t r = 0;
        while (frontier) r |= attacks_bb(pop_lsb(frontier), occupied);
        return r & free;
    };
    uint64_t seen_me = 0, seen_opp = 0, fm = me, fo = opp;
    int mine = 0, theirs = 0;
    while (fm | fo) {
        const uint64_t nm = expand(fm) & ~seen_me & ~seen_opp;   // not reached yet by anyone
        const uint64_t no = expand(fo) & ~seen_me & ~seen_opp;
        mine += std::popcount(nm & ~no);
        theirs += std::popcount(no & ~nm);
        seen_me |= nm;
        seen_opp |= no;
        fm = nm;
        fo = no;
    }
    return mine - theirs;
}

// Selection step: brings the best-scored move of moves[i..] to position i.
Move MinMaxNNUE_DevPlayer::pick_move(Yolah::MoveList& moves, int* scores, size_t i, size_t n) {
    size_t best = i;
    for (size_t j = i + 1; j < n; j++) {
        if (scores[j] > scores[best]) best = j;
    }
    std::swap(moves[i], moves[best]);
    std::swap(scores[i], scores[best]);
    return moves[i];
}

// "Gravity" update: h += bonus − h·|bonus| / HISTORY_MAX. The closer h is to
// ±HISTORY_MAX, the smaller its moves in that direction: h stays bounded, and
// recent cutoffs weigh more than old ones.
void MinMaxNNUE_DevPlayer::update_history(Search& s, uint8_t player, Move m, int bonus) {
    int16_t& h = s.w->history[player][m.from_sq()][m.to_sq()];
    h = int16_t(h + bonus - h * std::abs(bonus) / HISTORY_MAX);
}

void MinMaxNNUE_DevPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    const Move m = tt_move(hash);
    if (m == Move::none()) return;
    auto player = yolah.current_player();
    cout << m << ' ';
    yolah.play(m);
    print_pv(yolah, zobrist::update(hash, player, m), depth - 1);
}

// ─── Iterative deepening with aspiration windows ─────────────────────────────
// Depth 1, 2, 3… until the clock stops it. Each iteration fills the table
// with best moves that order the next one, so the deeper searches cost much
// less than they would from scratch.
//
// Aspiration windows (B): the value of depth d is usually close to the value
// of depth d − 1. Instead of the full window (−∞, +∞), depth d is searched in
// (v − δ, v + δ): a narrow window cuts more. If the value falls outside
// (fail low: v' ≤ v − δ, fail high: v' ≥ v + δ), the search is repeated with
// the window widened on that side, δ doubling each time. Not used for the
// first depths (unstable values) nor once a game result is in sight
// (|v| > WIN: exact scores, no point in a window).
//
// FIX (A): when the clock stops an iteration, its best move so far is kept if
// one was completely searched: the first move searched is the previous best
// (the table's move), so a different move found by the deeper search beat it.
void MinMaxNNUE_DevPlayer::iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth) {
    uint64_t hash = zobrist::hash(yolah);
    Move res = Move::none();
    uint8_t depth = 0;
    int value = 0;
    // The root moves, first ordered like an inner node (table's move, history…);
    // root_search reorders them after each iteration.
    {
        Yolah::MoveList moves;
        yolah.moves(moves);
        int scores[Yolah::MAX_NB_MOVES];
        score_moves(yolah, s, tt_move(hash), moves, scores, 63);
        s.root_moves.clear();
        for (size_t i = 0; i < moves.size(); i++) s.root_moves.push_back({pick_move(moves, scores, i, moves.size())});
    }
    // L. Helper threads skip some depths, each with its own pattern, so that
    // they are not all searching the same depth at the same time: thread i
    // skips depth d when ((d + SKIP_PHASE[i]) / SKIP_SIZE[i]) is odd (the
    // pattern of Stockfish's lazy SMP before 2018). The main thread (i = 0)
    // never skips.
    static constexpr int SKIP_SIZE[]  = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    static constexpr int SKIP_PHASE[] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
    const int k = s.id % 20;
    for (int d = 1; d <= max_depth && d < 64; d++) {
        if (s.id > 0 && ((d + SKIP_PHASE[k]) / SKIP_SIZE[k]) % 2 == 1 && d < max_depth) continue;
        int delta = options.aspiration_window;
        int alpha = -INFINITE, beta = INFINITE;
        if (delta > 0 && d >= 4 && std::abs(value) < WIN) {
            alpha = std::max(value - delta, -INFINITE);
            beta  = std::min(value + delta, INFINITE);
        }
        for (;;) {
            Move m;
            const int v = root_search(yolah, s, hash, alpha, beta, d, m);
            if (stopped()) {
                if (m != Move::none()) res = m;   // interrupted, but this move was fully searched
                goto done;
            }
            if (v <= alpha && alpha > -INFINITE) {          // fail low: widen downwards
                alpha = std::max(alpha - delta, -INFINITE);
                delta *= 2;
                continue;
            }
            if (v >= beta && beta < INFINITE) {             // fail high: widen upwards
                if (m != Move::none()) res = m;             // already known better than before
                beta = std::min(beta + delta, INFINITE);
                delta *= 2;
                continue;
            }
            res = m;
            depth = uint8_t(d);
            value = v;
            break;
        }
    }
done:
    s.depth = depth;
    s.value = int16_t(value);
    s.move = res;
}

json MinMaxNNUE_DevPlayer::config() {
    json j;
    j["name"] = "MinMaxNNUE_DevPlayer";
    j["microseconds"] = thinking_time;
    j["tt size"] = tt_size_mb;
    j["yolah table"] = options.yolah_table;
    j["staged"] = options.staged;
    j["eval grain"] = options.eval_grain;
    j["proxy cut"] = options.proxy_cut;
    j["proxy rank"] = options.proxy_rank;
    j["probcut margin"] = options.probcut_margin;
    j["proxy depth"] = options.proxy_depth;
    j["proxy reduction"] = options.proxy_reduction;
    j["nb threads"] = nb_threads;
    j["territory ordering"] = options.territory_ordering;
    j["territory depth"] = options.territory_depth;
    j["territory weight"] = options.territory_weight;
    j["articulation ordering"] = options.articulation_ordering;
    j["articulation lmr"] = options.articulation_lmr;
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    j["weights"] = nnue_q_parameters_filename;
    j["pvs"] = options.pvs;
    j["aspiration window"] = options.aspiration_window;
    j["history"] = options.history;
    j["countermove"] = options.countermove;
    j["root ordering"] = options.root_ordering;
    j["lazy accumulator"] = options.lazy_accumulator;
    j["eval cache"] = options.eval_cache_bits;
    j["lmr"] = options.lmr;
    j["lmr base"] = options.lmr_base;
    j["lmr divisor"] = options.lmr_divisor;
    j["rfp depth"] = options.rfp_depth;
    j["rfp margin"] = options.rfp_margin;
    j["null move"] = options.null_move;
    j["null move reduction"] = options.null_move_reduction;
    j["lmp depth"] = options.lmp_depth;
    j["lmp moves"] = options.lmp_moves;
    j["pass rule"] = options.pass_rule;
    j["endgame root"] = options.endgame_root;
    j["endgame root time"] = options.endgame_root_time;
    j["endgame tree"] = options.endgame_tree;
    j["verbose"] = verbose;
    return j;
}
