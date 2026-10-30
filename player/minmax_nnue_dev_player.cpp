#include "minmax_nnue_dev_player.h"
#include <limits>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "zobrist.h"
#include <utility>
#include <algorithm>
#include <limits>
#include <cmath>
#include <bit>
#include <fstream>
#include <iostream>
#include <immintrin.h>
#include "magic.h"

using std::cout, std::endl;

// The search debugger (player/search_trace.h): TR records an event with its
// source line, TRET a return value. Both are empty outside the search_trace
// executable, so the search of every other program is unchanged.
#ifdef YOLAH_SEARCH_TRACE
#include "search_trace.h"
#define TR(...) do { if (tracer) [[unlikely]] tracer->ev(__LINE__, SearchTracer::__VA_ARGS__); } while (0)
#define TRET(v) trace_return(__LINE__, (v))
#else
#define TR(...) do {} while (0)
#define TRET(v) (v)
#endif

MinMaxNNUE_DevPlayer::MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                                           uint8_t late_move_reduction, const std::string& nnue_q_parameters_filename,
                                           bool verbose, Options options, size_t nb_threads)
    : thinking_time(microseconds), tt_size_mb(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      nnue_q_parameters_filename(nnue_q_parameters_filename), options(options), verbose(verbose),
      nb_threads(std::max<size_t>(1, nb_threads)) {
    for (size_t i = 0; i < this->nb_threads; i++) workers.push_back(std::make_unique<Worker>());
    if (NNUE_Persp::is_persp_file(nnue_q_parameters_filename))
        persp = std::make_unique<NNUE_Persp>(nnue_q_parameters_filename);
    else
        nnue.load(nnue_q_parameters_filename);
    if (persp) persp->sparse = options.sparse_h1;
    else nnue.sparse = options.sparse_h1;
    if (!options.history_prior.empty()) {
        // T. int16 scores, phase-major (see prior_of), scaled by the weight.
        std::ifstream f(options.history_prior, std::ios::binary);
        prior.resize(size_t(PRIOR_PHASES) * 2 * SQUARE_NB * SQUARE_NB);
        if (!f.read(reinterpret_cast<char*>(prior.data()), prior.size() * sizeof(int16_t))) {
            std::cerr << "cannot read the history prior " << options.history_prior << '\n';
            std::exit(1);
        }
        for (int16_t& v : prior) v = int16_t(std::clamp(int(v) * options.history_prior_weight / 100, -HISTORY_MAX, HISTORY_MAX));
    }
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
    // T. With a prior (mode 1): pulled halfway toward the prior of the root's phase instead.
    if (!prior.empty() && options.history_prior_mode == 1) {
        const int ph = prior_phase(yolah);
        for (auto& w : workers)
            for (int p = 0; p < 2; p++)
                for (int from = 0; from < SQUARE_NB; from++)
                    for (int to = 0; to < SQUARE_NB; to++) {
                        int16_t& h = w->history[p][from][to];
                        h = int16_t((h + prior_of(ph, uint8_t(p), Move(Square(from), Square(to)))) / 2);
                    }
    } else {
        for (auto& w : workers)
            for (auto& by_from : w->history)
                for (auto& by_to : by_from)
                    for (int16_t& h : by_to) h /= 2;
    }
#ifdef YOLAH_SEARCH_TRACE
    if (tracer) trace_search_start(__LINE__, yolah);
#endif
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
            net_init(yolah, s->accs[yolah.nb_plies()]);
            s->acc_ok[yolah.nb_plies()] = true;
        } else {
            net_init(yolah, s->acc);
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

void MinMaxNNUE_DevPlayer::clear_table(bool keep_ordering) {
    if (table) table->clear(1);
    else yolah_table->clear();
    endgame.clear();
    // the move ordering statistics too: each benchmark position from scratch
    // (unless keep_ordering: as in a game, what the previous searches learnt)
    if (!keep_ordering) for (auto& w : workers) *w = Worker{};
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
         + (options.sparse_h1 ? " + sparse h1" : "")
         + (prior.empty() ? "" : " + history prior (mode " + std::to_string(options.history_prior_mode) + ", "
                                 + std::to_string(options.history_prior_weight) + " %)")
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
        net_play(player, m, s.acc);
    }
    yolah.play(m);
    int v = alpha + 1;                    // "might beat alpha" until a search says otherwise
    if (reduction > 0) {                  // late move: reduced search first
        const int d = depth - 1 - reduction;
        TR(CALL, 1, alpha, options.pvs ? alpha + 1 : beta, d);
        v = options.pvs ? -negamax(yolah, s, child, -(alpha + 1), -alpha, d)
                        : -negamax(yolah, s, child, -beta, -alpha, d);
        TR(CALL_RESULT, 1, v);
    }
    if (v > alpha && !stopped()) {
        if (options.pvs && i > 0) {
            TR(CALL, 2, alpha, alpha + 1, depth - 1);
            v = -negamax(yolah, s, child, -(alpha + 1), -alpha, depth - 1);       // null window
            TR(CALL_RESULT, 2, v);
            if (v > alpha && v < beta && !stopped()) {
                TR(CALL, 3, alpha, beta, depth - 1);
                v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);          // exact value
                TR(CALL_RESULT, 3, v);
            }
        } else {
            TR(CALL, 4, alpha, beta, depth - 1);
            v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
            TR(CALL_RESULT, 4, v);
        }
    }
    yolah.undo(m);
    if (!options.lazy_accumulator) net_undo(player, m, s.acc);
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
    TR(ENTER, 0, alpha, beta, depth, yolah.nb_plies(), s.played[yolah.nb_plies() - 1].raw(),
       int32_t(uint32_t(hash)), int32_t(uint32_t(hash >> 32)), trace_value(yolah));
#ifdef YOLAH_SEARCH_TRACE
    if (tracer && tracer->child_evals_depth > 0 && depth >= tracer->child_evals_depth) trace_child_evals(__LINE__, yolah);
#endif
    if (stopped()) return TRET(0);    // the caller ignores this value
    // 1. Finished game: the exact score (difference of the numbers of moves),
    //    pushed beyond any evaluation (|eval| ≤ WIN) so that a real win always
    //    beats a position that only looks good.
    if (yolah.game_over()) {
        int score = yolah.score(yolah.current_player());
        if (score == 0) return TRET(0);
        return TRET(score + (score > 0 ? WIN : -WIN));
    }
    // 2. Transposition table: a result of a search at least as deep as the
    //    one asked can answer at once, if its bound is good enough. Yolah has
    //    no cycles (every move leaves a hole), so a position's value never
    //    depends on the path to it: the table can be trusted everywhere.
    const TTView tt = tt_probe(hash);
    TR(TT_PROBE, tt.found, tt.depth, tt.lower, tt.upper, tt.move.raw());
    const Move tt_move = tt.move;
    if (tt.found) {
        s.nb_hits++;
        if (tt.depth >= depth) {
            if (tt.lower >= beta) return TRET(tt.lower);
            if (tt.upper <= alpha) return TRET(tt.upper);
            if (tt.lower == tt.upper) return TRET(tt.lower);
        }
    }
    // G. Very few free squares: the exact result from the endgame solver
    //    (win / draw / loss — only the sign matters for the outcome), instead
    //    of a search with the network. Stored in the table at the largest
    //    depth: it never needs to be searched again.
    // (Main thread only: the solver and its table are not shared.)
    if (options.endgame_tree > 0 && s.id == 0 && std::popcount(yolah.free_squares()) <= options.endgame_tree) {
        const auto proof = endgame.solve(yolah, true, &stop);
        if (!proof.complete) return TRET(0);          // stopped: the caller ignores it
        const int v = proof.value > 0 ? WIN + proof.value : proof.value < 0 ? -WIN + proof.value : 0;
        tt_store(hash, yolah, 63, -INFINITE, INFINITE, v, proof.move);
        return TRET(v);
    }
    // 3. Horizon: the network's value for the side to move, scaled to ±WIN.
    //    FIX (A): not stored in the table. The table treats depth 0 as "empty
    //    slot" (TranspositionTable::probe), so the reference's leaf entries were
    //    never found again, and when a cluster was full they evicted a real
    //    entry of an inner node.
    if (depth <= 0) {
        return TRET(evaluate(yolah, s, hash));
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
                     || (options.proxy_cut == 1 && depth >= options.proxy_depth)
                     || (options.mpc > 0 && depth >= options.mpc_depth))) {
        static_eval = evaluate(yolah, s, hash);
    }
    // F1. Reverse futility pruning ("static null move"): if the static value
    //     is above beta by a margin that a shallow search will hardly lose,
    //     trust it and cut. The margin grows with the depth: the deeper the
    //     search, the more the value may still change.
    if (prunable && depth <= options.rfp_depth)
        TR(RFP, static_eval, options.rfp_margin * depth, beta, static_eval - options.rfp_margin * depth >= beta);
    if (prunable && depth <= options.rfp_depth && static_eval - options.rfp_margin * depth >= beta) {
        return TRET(static_eval);
    }
    // O. Multi-ProbCut (Buro) without its regression, tuned by hand as
    //    Stockfish tunes its margins: search THIS position at a shallow depth
    //    d' and, if the result clears beta (alpha) by a margin, trust it for
    //    depth d. Buro fits v_d ≈ a·v_d' + b + N(0, σ²) per (d', d) and per
    //    phase and takes margin = t·σ; here the margin is
    //    mpc_margin + mpc_margin_per_ply · (d − d'), set by matches. As in
    //    Egaroucid, tried only when the static value is already on that side.
    constexpr int NO_CUT = std::numeric_limits<int>::min();
    auto mpc_cut = [&]() -> int {
        if (prunable && options.mpc > 0 && depth >= options.mpc_depth) {
            const int d2 = std::max(1, depth * options.mpc_ratio / 100);
            const int margin = options.mpc_margin + options.mpc_margin_per_ply * (depth - d2);
            if (static_eval >= beta && beta + margin < WIN) {
                const int bound = beta + margin;
                TR(MPC, 1, d2, margin, bound);
                const int v = negamax(yolah, s, hash, bound - 1, bound, d2);
                TR(MPC_RESULT, 1, v, v >= bound);
                if (stopped()) return 0;
                if (v >= bound) return beta;
            }
            if (options.mpc == 2 && static_eval <= alpha && alpha - margin > -WIN) {
                const int bound = alpha - margin;
                TR(MPC, 2, d2, margin, bound);
                const int v = negamax(yolah, s, hash, bound, bound + 1, d2);
                TR(MPC_RESULT, 2, v, v <= bound);
                if (stopped()) return 0;
                if (v <= bound) return alpha;
            }
        }
        return NO_CUT;
    };
    if (!options.proxy_first) {
        const int v = mpc_cut();
        if (v != NO_CUT) return TRET(v);
    }
    // N, lazy: a cheap witness (killer 1), probed before any move generation —
    // a proxy cut at about the cost of a null move. Same conditions as N below.
    bool proxy_done = false;
    if (options.proxy_cut == 1 && options.proxy_lazy > 0 && prunable && depth >= options.proxy_depth
        && !s.proxy_off && static_eval >= beta && beta + options.proxy_delta < WIN) {
        const Move k = s.killer1[yolah.nb_plies()];
        if (k != Move::none() && k != tt_move && is_legal(yolah, k)) {
            const int bound = beta + options.proxy_delta;
            const int d = depth - 1 - (options.proxy_reduction + depth / 4 - 1);
            const uint8_t player = yolah.current_player();
            const uint64_t child = zobrist::update(hash, player, k);
            s.played[yolah.nb_plies()] = k;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, k, s.acc);
            yolah.play(k);
            TR(PROXY, k.raw(), bound, d);
            const int v = -negamax(yolah, s, child, -bound, -bound + 1, d);
            TR(PROXY_RESULT, k.raw(), v, v >= bound);
            yolah.undo(k);
            if (!options.lazy_accumulator) net_undo(player, k, s.acc);
            if (stopped()) return TRET(0);
            if (v >= bound) return TRET(v >= WIN ? beta : v);
            proxy_done = true;               // one witness per node, as usual
        } else if (options.proxy_lazy == 2) {
            proxy_done = true;               // killer only: no witness here
        }
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
    TR(GEN, tt_first ? 1 : 2, tt_first ? 1 : int(moves.size()));
    // The pass rule (see EndgameSolver::search for the proof): a player who
    // must pass while the game is not over has lost — their final score
    // difference is ≤ −1, so the value is ≤ −(WIN + 1). An upper bound,
    // returned when it is enough to fail low. (A legal table move: not a pass.)
    if (options.pass_rule && !tt_first && moves[0] == Move::none() && -(WIN + 1) <= alpha) {
        return TRET(-(WIN + 1));
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
        else net_play(player, Move::none(), s.acc);
        yolah.play(Move::none());
        TR(CALL, 5, beta - 1, beta, depth - 1 - R);
        const int v = -negamax(yolah, s, zobrist::update(hash, player, Move::none()), -beta, -beta + 1, depth - 1 - R);
        TR(CALL_RESULT, 5, v);
        yolah.undo(Move::none());
        if (!options.lazy_accumulator) net_undo(player, Move::none(), s.acc);
        if (stopped()) return TRET(0);
        if (v >= beta) return TRET(v >= WIN ? beta : v);   // no unproven "win" from a null move
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
    // Q. "Proxy tail": what a FAILED proxy test says about the moves ordered
    //    after the witness (see options.proxy_tail).
    bool tail = false, tail_clear = false;
    Move witness = Move::none();
    bool scores_pending = false;             // witness 3: the moves are scored only if the loop needs them
    if (options.proxy_cut > 0 && prunable && depth >= options.proxy_depth && !s.proxy_off && !proxy_done
        && (options.proxy_cut == 2 || static_eval >= beta)) {
        if (!generated) {                       // the test needs the move list
            yolah.moves(moves);
            generated = true;
            TR(GEN, 2, int(moves.size()));
            if (options.proxy_cut == 1 && options.proxy_witness == 3) {
                scores_pending = true;
            } else {
                score_moves(yolah, s, tt_move, moves, scores, depth);
            }
            n = moves.size();
        }
        // Search move m at depth d against `bound` (null window); the value for us.
        auto probe = [&](Move m, int bound, int d) {
            const uint8_t player = yolah.current_player();
            const uint64_t child = zobrist::update(hash, player, m);
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc);
            yolah.play(m);
            TR(PROXY, m.raw(), bound, d);
            const int v = -negamax(yolah, s, child, -bound, -bound + 1, d);
            TR(PROXY_RESULT, m.raw(), v, v >= bound);
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc);
            return v;
        };
        const int R = options.proxy_reduction + depth / 4 - 1;
        const int d = depth - 1 - R;
        if (options.proxy_cut == 2) {
            // ProbCut (Stockfish): the table's move, against beta + probcut_margin
            Move m = tt_first ? tt_move : moves[0];
            if (!tt_first) {                    // no table move: the best-scored one
                size_t b = 0;
                for (size_t j = 1; j < n; j++) if (scores[j] > scores[b]) b = j;
                m = moves[b];
            }
            const int bound = std::min(beta + options.probcut_margin, WIN);
            if (m != Move::none()) {
                const int v = probe(m, bound, d);
                if (stopped()) return TRET(0);
                if (v >= bound) return TRET(v >= WIN ? beta : v);   // no unproven win from a reduced search
            }
        } else if (beta + options.proxy_delta < WIN) {
            const int bound = beta + options.proxy_delta;
            // The candidates: the ordinary moves (not the table's move, a killer
            // or the countermove), best-ordered first (ties: generation order),
            // from rank proxy_rank on. Witness 1/2: the killer / the countermove.
            int cand[Yolah::MAX_NB_MOVES];
            int nc = 0;
            if (options.proxy_witness == 3) {
                // by history only, among the non-special moves (no territory term)
                const uint16_t ply = yolah.nb_plies();
                const uint8_t player = yolah.current_player();
                Move counter = Move::none();
                if (options.countermove && ply > 0) {
                    const Move prev = s.played[ply - 1];
                    counter = s.w->countermoves[prev.from_sq()][prev.to_sq()];
                }
                for (size_t j = 0; j < n; j++) {
                    const Move m = moves[j];
                    if (m == tt_move || m == s.killer1[ply] || m == s.killer2[ply] || (m == counter && counter != Move::none()))
                        continue;
                    cand[nc++] = int(j);
                }
                std::stable_sort(cand, cand + nc, [&](int a, int b) {
                    return s.w->history[player][moves[a].from_sq()][moves[a].to_sq()]
                         > s.w->history[player][moves[b].from_sq()][moves[b].to_sq()];
                });
                const int r = std::min(nc, options.proxy_rank);
                std::copy(cand + r, cand + nc, cand);
                nc -= r;
            } else if (options.proxy_witness == 0) {
                for (size_t j = 0; j < n; j++) if (scores[j] < SCORE_COUNTER) cand[nc++] = int(j);
                std::stable_sort(cand, cand + nc, [&](int a, int b) { return scores[a] > scores[b]; });
                const int r = std::min(nc, options.proxy_rank);
                std::copy(cand + r, cand + nc, cand);
                nc -= r;
            } else {
                const int want = options.proxy_witness == 1 ? SCORE_KILLER1 : SCORE_COUNTER;
                for (size_t j = 0; j < n; j++) if (scores[j] == want && moves[j] != tt_move) cand[nc++] = int(j);
            }
            // weval: the first K candidates reordered by their static value after the move
            if (options.proxy_weval > 1 && nc > 1) {
                const int K = std::min(nc, options.proxy_weval);
                int val[Yolah::MAX_NB_MOVES];
                for (int k = 0; k < K; k++) {
                    const Move m = moves[cand[k]];
                    const uint8_t player = yolah.current_player();
                    const uint64_t child = zobrist::update(hash, player, m);
                    s.played[yolah.nb_plies()] = m;
                    if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
                    else net_play(player, m, s.acc);
                    yolah.play(m);
                    val[cand[k]] = -evaluate(yolah, s, child);
                    yolah.undo(m);
                    if (!options.lazy_accumulator) net_undo(player, m, s.acc);
                }
                std::stable_sort(cand, cand + K, [&](int a, int b) { return val[a] > val[b]; });
            }
            // The witnesses; diverse: each moves a piece not used yet when possible.
            const int want = std::max(options.proxy_multi_m, options.proxy_second >= 0 ? 2 : 1);
            int wit[Yolah::MAX_NB_MOVES];
            int nw = 0;
            bool used[Yolah::MAX_NB_MOVES]{};
            for (int pass = 0; pass < 2 && nw < want; pass++) {
                for (int k = 0; k < nc && nw < want; k++) {
                    if (used[k]) continue;
                    if (options.proxy_diverse && pass == 0) {
                        bool seen = false;
                        for (int i2 = 0; i2 < nw; i2++) seen |= moves[wit[i2]].from_sq() == moves[cand[k]].from_sq();
                        if (seen) continue;
                    }
                    used[k] = true;
                    wit[nw++] = cand[k];
                }
                if (!options.proxy_diverse) break;
            }
            // One witness: a shallow probe first (prefilter), then the probe.
            auto run = [&](int idx) {
                if (options.proxy_prefilter > 0 && d - options.proxy_prefilter >= 1) {
                    const int v0 = probe(moves[idx], bound, d - options.proxy_prefilter);
                    if (v0 < bound) return v0;
                }
                return probe(moves[idx], bound, d);
            };
            // A cut at a deep node, verified by a search of this node at the probe depth.
            auto verified = [&]() {
                if (options.proxy_verify <= 0 || depth < options.proxy_verify) return true;
                s.proxy_off = true;
                const int v = negamax(yolah, s, hash, bound - 1, bound, d);
                s.proxy_off = false;
                return v >= bound;
            };
            int cut_value = -INFINITE;
            int first_v = INFINITE;          // the first witness's (fail soft) value
            if (options.proxy_second >= 0 && options.proxy_multi_m == 1) {
                if (nw >= 1) {
                    int v = run(wit[0]);
                    first_v = v;
                    if (stopped()) return TRET(0);
                    if (v < bound && nw >= 2 && v >= bound - options.proxy_second) {
                        v = run(wit[1]);
                        if (stopped()) return TRET(0);
                    }
                    if (v >= bound && verified()) cut_value = v;
                }
            } else if (nw >= options.proxy_multi_c) {
                const int m = std::min(nw, options.proxy_multi_m);
                int cuts = 0, lowest = INFINITE;
                for (int k = 0; k < m; k++) {
                    const int v = run(wit[k]);
                    if (k == 0) first_v = v;
                    if (stopped()) return TRET(0);
                    if (v >= bound) {
                        lowest = std::min(lowest, v);
                        if (++cuts >= options.proxy_multi_c) {
                            if (verified()) cut_value = lowest;
                            break;
                        }
                    } else if (cuts + (m - 1 - k) < options.proxy_multi_c) {
                        break;
                    }
                }
            }
            if (stopped()) return TRET(0);
            if (cut_value >= bound) return TRET(cut_value >= WIN ? beta : cut_value);   // no unproven win
            if (options.proxy_tail > 0 && nw >= 1 && first_v < bound) {
                witness = moves[wit[0]];
                tail_clear = first_v < bound - options.proxy_tail_margin;
                tail = options.proxy_tail != 2 || tail_clear;
            }
        }
    }
    if (options.proxy_first) {
        const int v = mpc_cut();
        if (v != NO_CUT) return TRET(v);
    }
    // P. Stockfish's ProbCut, generalized to a game without captures: the
    //    best-ordered moves (table's move first), at most "probcut" of them,
    //    each searched at depth − probcut_reduction against beta + margin; the
    //    first that reaches it cuts. Stockfish first checks the move with a
    //    quiescence search; Yolah has none: the static value after the move
    //    plays that role ("probcut filter").
    if (options.probcut > 0 && prunable && depth >= options.probcut_depth && beta + options.probcut_margin < WIN) {
        if (!generated) {
            yolah.moves(moves);
            generated = true;
            TR(GEN, 2, int(moves.size()));
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
        }
        if (scores_pending) {
            score_moves(yolah, s, tt_move, moves, scores, depth);
            scores_pending = false;
        }
        const int bound = beta + options.probcut_margin;
        bool taken[Yolah::MAX_NB_MOVES]{};
        for (int k = 0; k < options.probcut; k++) {
            int pick = -1;
            for (size_t j = 0; j < n; j++) {
                if (!taken[j] && moves[j] != Move::none() && (pick < 0 || scores[j] > scores[pick])) pick = int(j);
            }
            if (pick < 0) break;
            taken[pick] = true;
            const Move m = moves[pick];
            const uint8_t player = yolah.current_player();
            const uint64_t child = zobrist::update(hash, player, m);
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc);
            yolah.play(m);
            int v = -INFINITE;
            if (!options.probcut_filter || -evaluate(yolah, s, child) >= bound) {
                TR(CALL, 6, bound - 1, bound, depth - options.probcut_reduction);
                v = -negamax(yolah, s, child, -bound, -bound + 1, depth - options.probcut_reduction);
                TR(CALL_RESULT, 6, v);
            }
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc);
            if (stopped()) return TRET(0);
            if (v >= bound) return TRET(v >= WIN ? beta : v);
        }
    }
    if (scores_pending) {
        score_moves(yolah, s, tt_move, moves, scores, depth);
    }
    const int alpha_orig = alpha;
    int best = -INFINITE;
    Move best_move = Move::none();
    bool pruned = false;
    int after_witness = -1;                  // Q: ordinary moves searched after the witness (-1: not yet)
    for (size_t i = 0; i < n || !generated; i++) {
        if (!generated && i == 1) {
            // Stage 2: the table's move did not cut. Generate everything, put
            // the table's move where the selection would have put it (index
            // 0, swapped with what was there), score the others.
            yolah.moves(moves);
            generated = true;
            TR(GEN, 2, int(moves.size()));
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
            TR(LMP, int(i), int(n));
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
        // Q. Proxy tail: an ordinary move ordered after the failed witness.
        int tail_r = 0;
        if (tail && after_witness >= 0 && scores[i] < SCORE_COUNTER && after_witness++ >= options.proxy_tail_keep) {
            const int k = after_witness - 1 - options.proxy_tail_keep;   // 0 for the first move concerned
            if (options.proxy_tail <= 2) {
                if (best > -WIN) break;      // pruned (once a move that does not lose was found)
            } else if (options.proxy_tail == 3) {
                tail_r = options.proxy_tail_reduction;
            } else {                         // 4: an LMR anchored on the witness that failed
                tail_r = std::min(options.proxy_tail_cap,
                                  options.proxy_tail_reduction + k / std::max(1, options.proxy_tail_step))
                       + (tail_clear ? 1 : 0);
            }
        }
        // scores[i] is now m's score: special move (killer, countermove), articulation or history
        const bool tactical = options.articulation_lmr && is_articulation_move(yolah, m);
        const int r = tactical || !generated ? 0          // (the first move is never reduced anyway)
                    : late_move_reduction_of(depth, i, beta - alpha > 1, scores[i] >= SCORE_COUNTER,
                                             scores[i] >= SCORE_ARTICULATION ? 0 : scores[i]);
        TR(MOVE, int(i), m.raw(), generated ? scores[i] : SCORE_TT, r, tail_r);
        const int v = search_move(yolah, s, hash, m, i, r + tail_r, alpha, beta, depth);
        TR(MOVE_DONE, int(i), m.raw(), v, best, alpha);
        if (tail && m == witness) after_witness = 0;
        // FIX (A): once the clock has stopped the search, the values coming up
        // are meaningless: return before storing anything (the reference stored
        // them, and the table is kept for the next moves of the game).
        if (stopped()) return TRET(0);
        if (cut_stats) [[unlikely]] cut_stats->count(yolah, m, v >= beta);   // T: statistics for the prior
        if (v > best) {
            best = v;
            if (v > alpha) {
                best_move = m;
                if (v >= beta) {
                    // Beta cutoff: the opponent will not allow this position.
                    TR(CUTOFF, int(i), m.raw(), v, beta);
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
                    TR(KILLER, int(ply), s.killer1[ply].raw(), s.killer2[ply].raw());
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
                        TR(COUNTER, prev.raw(), m.raw());
                    }
                    return TRET(v);
                }
                alpha = v;
            }
        }
    }
    // No cutoff: exact value if a move raised alpha, otherwise only an upper
    // bound (all the moves failed low against the window we were given).
    tt_store(hash, yolah, depth, alpha_orig, beta, best, best_move);
    return TRET(best);
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
#ifdef YOLAH_SEARCH_TRACE
    // The slot update() will write (same probe), before and after: kept,
    // replaced, or another position's entry evicted.
    if (tracer) {
        bool found;
        const TranspositionTableEntry* e = table->probe(hash, found);
        const uint32_t old_key = e->key32();
        const int old_depth = e->depth(), old_bound = e->bound(), old_value = e->value(), old_move = e->move().raw();
        uint64_t old_hash = 0;
        if (old_depth > 0) {
            auto it = tracer->hash_of_key.find(old_key);
            old_hash = it != tracer->hash_of_key.end() ? it->second : old_key;
        }
        table->update(hash, int16_t(value), b, uint8_t(depth), move);
        tracer->hash_of_key[uint32_t(hash)] = hash;
        TR(TT_STORE, depth, int(b), value, move.raw(), int32_t(uint32_t(old_hash)), int32_t(uint32_t(old_hash >> 32)),
           old_depth, old_bound, old_value, old_move, int(e->depth()), int(e->bound()), int(e->value()), e->move().raw());
        return;
    }
#endif
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
    constexpr uint64_t KEY_MASK = ~uint64_t(0) << 17;   // bits 63..17 (see eval_cache)
    const uint64_t key = hash & KEY_MASK;
    if (!eval_cache.empty()) {
        e = &eval_cache[hash & (eval_cache.size() - 1)];
        const uint64_t word = e->load(std::memory_order_relaxed);
        if ((word & 1) && (word & KEY_MASK) == key) {
            s.nb_eval_hits++;
            TR(EVAL, int16_t(uint16_t(word >> 1)), 1);
            return int16_t(uint16_t(word >> 1));
        }
    }
    const int v = network_value(yolah, s);
    if (e) e->store(key | (uint64_t(uint16_t(int16_t(v))) << 1) | 1, std::memory_order_relaxed);
    TR(EVAL, v, 0);
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

// Tempo ("tempo", off): a constant for the side to move. The value oscillates
// between odd and even depths (2026-10-30, search_bench depth 12, mean change
// for the root's side: −2400, +2350, −2160 from depth 1→2, 2→3, 3→4, still
// ±500 at 8→9). Tempo 1500–2000 (before the tanh) removes the mean
// oscillation at depths 1–4 but changes nothing else: the median change
// between depths does not fall (1326 → 1670), failed aspiration attempts
// neither (4896 → 5105), nodes ±2 %. The oscillation is a mean; what makes the
// windows fail is the spread from position to position.
int MinMaxNNUE_DevPlayer::network_value(const Yolah& yolah, Search& s) {
    int v = raw_network_value(yolah, s);
    if (options.tempo) {
        // Added before the tanh (to the logit), not to the value: near ±WIN a
        // constant would flatten the differences between won positions (and a
        // value of WIN would pass for a finished game). Near 0 both are the same.
        const double x = std::clamp(double(v) / WIN, -0.999999, 0.999999);
        v = std::clamp(int(std::lround(std::tanh(std::atanh(x) + double(options.tempo) / WIN) * WIN)), -WIN + 1, WIN - 1);
    }
    return round_to_grain(v, options.eval_grain);
}

int MinMaxNNUE_DevPlayer::raw_network_value(const Yolah& yolah, Search& s) {
    if (!options.lazy_accumulator) {
        return int(net_value(s.acc, yolah) * WIN);
    }
    const int ply = yolah.nb_plies();
    int k = ply;
    while (!s.acc_ok[k]) k--;                 // the root's is always valid
    for (; k < ply; k++) {
        // Player at ply k: ply parity (a pass also counts as a ply).
        update_accumulator(s.accs[k].acc, s.accs[k + 1].acc, uint8_t(k & 1), s.played[k]);
        s.acc_ok[k + 1] = true;
    }
    return int(net_value(s.accs[ply], yolah) * WIN);
}

// The network calls, for either network (see `persp` in the header).
void MinMaxNNUE_DevPlayer::net_init(const Yolah& yolah, Accumulator& a) {
    if (persp) persp->init(yolah, a.acc);
    else nnue.init(yolah, a.acc);
}

void MinMaxNNUE_DevPlayer::net_play(uint8_t player, Move m, Accumulator& a) {
    if (persp) persp->play(player, m, a.acc);
    else nnue.play(player, m, a.acc);
}

void MinMaxNNUE_DevPlayer::net_undo(uint8_t player, Move m, Accumulator& a) {
    if (persp) persp->undo(player, m, a.acc);
    else nnue.undo(player, m, a.acc);
}

float MinMaxNNUE_DevPlayer::net_value(const Accumulator& a, const Yolah& yolah) {
    if (persp) return persp->value(a.acc, yolah);
    return nnue.value(a.acc, yolah.current_player());
}

// out = in + the change of features of move m by `player` (see NNUE_Quantized::play,
// whose arithmetic it reproduces, in one pass and without modifying `in`).
void MinMaxNNUE_DevPlayer::update_accumulator(const int16_t* __restrict in, int16_t* __restrict out,
                                              uint8_t player, Move m) const {
    if (persp) {                              // two views, three rows each (see NNUE_Persp::update)
        persp->update(in, out, player, m);
        return;
    }
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
    TR(ENTER, 1, alpha, beta, depth, yolah.nb_plies(), 0, int32_t(uint32_t(hash)), int32_t(uint32_t(hash >> 32)),
       trace_value(yolah));
#ifdef YOLAH_SEARCH_TRACE
    if (tracer && tracer->child_evals_depth > 0) trace_child_evals(__LINE__, yolah);
#endif
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
        TR(MOVE, int(i), rm.move.raw(), s.w->history[yolah.current_player()][rm.move.from_sq()][rm.move.to_sq()], r, 0);
        const int v = search_move(yolah, s, hash, rm.move, i, r, alpha, beta, depth);
        TR(MOVE_DONE, int(i), rm.move.raw(), v, best, alpha);
        rm.nodes = s.nb_nodes - nodes_before;
        if (stopped()) return TRET(best);
        if (v > best) {
            best = v;
            if (v > alpha) {
                res = rm.move;
                alpha = v;
                if (v >= beta) {
                    TR(CUTOFF, int(i), rm.move.raw(), v, beta);
                    break;     // aspiration fail high: the caller widens the window
                }
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
#ifdef YOLAH_SEARCH_TRACE
    if (tracer) {
        std::vector<int32_t> body{int32_t(s.root_moves.size())};
        for (const RootMove& rm : s.root_moves) {
            body.push_back(rm.move.raw());
            body.push_back(int32_t(std::min<uint64_t>(rm.nodes, INT32_MAX)));
        }
        tracer->ev_list(__LINE__, SearchTracer::ROOT_ORDER, body);
    }
#endif
    tt_store(hash, yolah, depth, alpha_orig, beta, best, res);
    return TRET(best);
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
    // T. Mode 2: the prior of the node's phase added to the history.
    const bool add_prior = !prior.empty() && options.history_prior_mode == 2;
    const bool tier_prior = !prior.empty() && options.history_prior_mode == 3;
    const int ph = add_prior || tier_prior ? prior_phase(yolah) : 0;
    auto hist = [&](Move m) {
        return s.w->history[player][m.from_sq()][m.to_sq()] + (add_prior ? prior_of(ph, player, m) : 0);
    };
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        if (m == tt_move)                 scores[i] = SCORE_TT;
        else if (m == s.killer1[ply])     scores[i] = SCORE_KILLER1;
        else if (m == s.killer2[ply])     scores[i] = SCORE_KILLER2;
        else if (m == counter && counter != Move::none()) scores[i] = SCORE_COUNTER;
        else if (options.articulation_ordering && is_articulation_move(yolah, m))
            scores[i] = SCORE_ARTICULATION + (options.history ? hist(m) : 0);
        else if (tier_prior && prior_of(ph, player, m) >= options.history_prior_threshold)
            scores[i] = SCORE_ARTICULATION + prior_of(ph, player, m);   // T, mode 3: above the history
        else if (options.history)         scores[i] = hist(m);
        else                              scores[i] = -int(i);
        if (territory && scores[i] < SCORE_ARTICULATION) {
            scores[i] += options.territory_weight * territory_after(yolah, m);
        }
    }
    // T, mode 4: the "history prior top" moves of the node with the highest
    // prior (the moves the players choose most in this phase), among the
    // ordinary ones, placed right after the special moves, before the history.
    if (!prior.empty() && options.history_prior_mode == 4) {
        const int ph = prior_phase(yolah);
        for (int k = 0; k < options.history_prior_top; k++) {
            int pick = -1, best = std::numeric_limits<int>::min();
            for (size_t i = 0; i < moves.size(); i++) {
                if (scores[i] >= SCORE_ARTICULATION || moves[i] == Move::none()) continue;
                const int p = prior_of(ph, player, moves[i]);
                if (p > best) { best = p; pick = int(i); }
            }
            if (pick < 0) break;
            scores[pick] = SCORE_ARTICULATION + options.history_prior_top - k;
        }
    }
#ifdef YOLAH_SEARCH_TRACE
    if (tracer) {   // the territory term again (territory_after has no side effect)
        std::vector<int32_t> body{int32_t(moves.size())};
        for (size_t i = 0; i < moves.size(); i++) {
            body.push_back(moves[i].raw());
            body.push_back(scores[i]);
            body.push_back(territory && scores[i] < SCORE_ARTICULATION ? territory_after(yolah, moves[i]) : 0);
        }
        tracer->ev_list(__LINE__, SearchTracer::SCORES, body);
    }
#endif
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

// ─── Fast influence (the user's version) ─────────────────────────────────────
// The same territory as heuristic::influence, computed with less work:
//
//   • expand_king(b): b and its 8 neighbours in 9 operations — first the row
//     (b, one square east, one square west; the masks stop wrapping from the
//     h-file to the a-file and back), then that row one rank up and one down:
//
//         . . .        x x x        h | h<<8 | h>>8
//         . b .   →    x x x   ←    h = b | east(b) | west(b)
//         . . .        x x x
//
//   • the fronts only hold the squares reached at THIS step (both influences
//     are excluded): each square is visited once, and the loop stops as soon
//     as no front can advance. (Without excluding one's own influence the
//     front would contain itself forever — expand_king(b) contains b — and the
//     loop would never end in most positions.)
//   • the neutral squares are only computed once there are some.
//
// Checked: the same score as heuristic::influence on 3280 positions.
static inline uint64_t expand_king(uint64_t b) {
    const uint64_t h = b | ((b << 1) & 0xFEFEFEFEFEFEFEFEULL) | ((b >> 1) & 0x7F7F7F7F7F7F7F7FULL);
    return h | (h << 8) | (h >> 8);
}

static int influence_fast(uint64_t me, uint64_t opp, uint64_t free) {
    uint64_t mi = me, oi = opp, mf = me, of = opp, neutral = 0;
    while (mf | of) {
        uint64_t nm = expand_king(mf) & free & ~oi & ~mi;    // new squares only
        uint64_t no = expand_king(of) & free & ~mi & ~oi;
        const uint64_t collision = nm & no;                  // reached by both at the same step
        if (neutral | collision) {
            neutral |= collision | (expand_king(neutral) & free);
            nm &= ~neutral;
            no &= ~neutral;
        }
        mi |= nm;
        oi |= no;
        mf = nm;
        of = no;
    }
    return std::popcount(mi & free) - std::popcount(oi & free);
}

// ─── Queen distance by occluded fill (modes 4–6) ────────────────────────────
// A whole level of squares slides in each of the 8 directions at once
// (Kogge-Stone: 3 doubling steps), whatever its number of squares; checked
// equal to the square-by-square search with the magic tables
// (test/territory_study_main.cpp, CHECK_QUEEN=1).
template <Direction D> static inline uint64_t queen_slide(uint64_t gen, uint64_t free) {
    uint64_t pro = free;
    gen |= pro & shift<D>(gen);
    pro &= shift<D>(pro);
    gen |= pro & shift<D>(shift<D>(gen));
    pro &= shift<D>(shift<D>(pro));
    gen |= pro & shift<D>(shift<D>(shift<D>(shift<D>(gen))));
    return shift<D>(gen) & free;
}

static inline uint64_t queen_reach(uint64_t from, uint64_t free) {
    return queen_slide<NORTH>(from, free) | queen_slide<SOUTH>(from, free) | queen_slide<EAST>(from, free) |
           queen_slide<WEST>(from, free) | queen_slide<NORTH_EAST>(from, free) | queen_slide<NORTH_WEST>(from, free) |
           queen_slide<SOUTH_EAST>(from, free) | queen_slide<SOUTH_WEST>(from, free);
}

// The 5 queen measures of each side (territory_study's Q1, Q2, Q3, QFIRST, QW):
// squares within 1, 2, 3 moves, squares reached strictly first, 64 · Σ 2^−d.
static void queen_measures(uint64_t me, uint64_t opp, uint64_t free, int qm[5], int qo[5]) {
    uint64_t seen_m = 0, seen_o = 0, fm = me, fo = opp;
    for (int i = 0; i < 5; i++) qm[i] = qo[i] = 0;
    for (int k = 0; fm | fo; k++) {
        const uint64_t nm = fm ? queen_reach(fm, free) & ~seen_m : 0;
        const uint64_t no = fo ? queen_reach(fo, free) & ~seen_o : 0;
        qm[3] += std::popcount(nm & ~no & ~seen_o);
        qo[3] += std::popcount(no & ~nm & ~seen_m);
        if (k < 6) { qm[4] += std::popcount(nm) << (5 - k); qo[4] += std::popcount(no) << (5 - k); }
        seen_m |= nm;
        seen_o |= no;
        if (k < 3) { qm[k] = std::popcount(seen_m); qo[k] = std::popcount(seen_o); }
        fm = nm;
        fo = no;
    }
    for (int k = 1; k < 3; k++) { qm[k] = std::max(qm[k], qm[k - 1]); qo[k] = std::max(qo[k], qo[k - 1]); }
}

// Mode 5–6 weights (× 1/16), for the player who made the move and for the
// opponent: a least-squares fit of the points still to score on the 10
// measures (nnue/territory_study.py data, 24.5 M positions of data/games),
// divided by the influence's own coefficient, so that the score is in
// "influence squares" and "territory weight" keeps its meaning.
static constexpr int QUEEN_W[3][2][5] = {
    { {  -6,  39, -36,   6,  3 }, { -348, -209, -78,   3, 21 } },   // opening, 40–56 free
    { {  13,  12,   8,  18, -1 }, {  -10,  -12,  -6, -18,  1 } },   // middle, 20–39
    { {   0,  -1,  -2,  17,  0 }, {    3,    2,   2, -16,  0 } },   // endgame, 0–19
};

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
    if (options.territory_ordering == 4) {
        int qm[5], qo[5];
        queen_measures(me, opp, free, qm, qo);
        return qm[3] - qo[3];
    }
    if (options.territory_ordering >= 5) {
        int qm[5], qo[5];
        queen_measures(me, opp, free, qm, qo);
        const int nfree = std::popcount(free);
        const int ph = options.territory_ordering == 6 ? 1 : nfree >= 40 ? 0 : nfree >= 20 ? 1 : 2;
        int sc = 0;
        for (int i = 0; i < 5; i++) sc += QUEEN_W[ph][0][i] * qm[i] + QUEEN_W[ph][1][i] * qo[i];
        return sc / 16;
    }
    if (options.territory_ordering == 1 && options.fast_influence) {
        return influence_fast(me, opp, free);
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
    [[maybe_unused]] const int before = h;
    h = int16_t(h + bonus - h * std::abs(bonus) / HISTORY_MAX);
    TR(HISTORY, player, m.raw(), before, h, bonus);
}

void MinMaxNNUE_DevPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    const Move m = tt_move(hash);
    if (m == Move::none() || !is_legal(yolah, m)) return;   // a collision can bring another position's move
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
// Adaptive windows ("aspiration k" > 0; 2026-10-30): with ±300, half the
// iterations failed (the median change of value between two depths is ~630,
// p90 ~4600) and 45 % of the nodes were in failed attempts. δ is k times the
// mean |v_d − v_{d−1}| this thread has seen (decaying by 0.9 per iteration,
// kept from one move of the game to the next), at least "aspiration window".
// After a failure the window is re-centred on the fail-soft value v (the
// result is at most / at least v, often far beyond the old bound): fail low
// α = v − δ and β = (α + β) / 2, fail high β = v + δ; δ grows by "aspiration
// grow". search_bench depth 12 (g046, 1 thread): k 1.25–2 −22 % nodes.
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
        const bool adaptive = options.aspiration_k > 0;
        if (adaptive && s.w->asp_n > 0) delta = std::max(delta, int(options.aspiration_k * s.w->asp_sum / s.w->asp_n));
        int alpha = -INFINITE, beta = INFINITE;
        if (delta > 0 && d >= 4 && std::abs(value) < WIN) {
            alpha = std::max(value - delta, -INFINITE);
            beta  = std::min(value + delta, INFINITE);
        }
        [[maybe_unused]] int attempt = 0;
        for (;;) {
            Move m;
            TR(ITER, d, alpha, beta, attempt++);
            const int v = root_search(yolah, s, hash, alpha, beta, d, m);
            if (stopped()) {
                TR(ITER_RESULT, d, v, m.raw(), 3);
                if (m != Move::none()) res = m;   // interrupted, but this move was fully searched
                goto done;
            }
            if (v <= alpha && alpha > -INFINITE) {          // fail low: widen downwards
                TR(ITER_RESULT, d, v, m.raw(), 1);
                if (adaptive) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(v - delta, -INFINITE);
                    delta = int(delta * options.aspiration_grow);
                } else {
                    alpha = std::max(alpha - delta, -INFINITE);
                    delta *= 2;
                }
                continue;
            }
            if (v >= beta && beta < INFINITE) {             // fail high: widen upwards
                TR(ITER_RESULT, d, v, m.raw(), 2);
                if (m != Move::none()) res = m;             // already known better than before
                if (adaptive) {
                    beta = std::min(v + delta, INFINITE);
                    delta = int(delta * options.aspiration_grow);
                } else {
                    beta = std::min(beta + delta, INFINITE);
                    delta *= 2;
                }
                continue;
            }
            TR(ITER_RESULT, d, v, m.raw(), 0);
#ifdef YOLAH_SEARCH_TRACE
            if (tracer) trace_pv(__LINE__, yolah, hash, d);
#endif
            res = m;
            if (d >= 4 && depth > 0 && std::abs(v) < WIN && std::abs(value) < WIN) {
                s.w->asp_sum = 0.9 * s.w->asp_sum + std::abs(v - value);
                s.w->asp_n   = 0.9 * s.w->asp_n + 1;
            }
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
    j["sparse h1"] = options.sparse_h1;
    j["history prior"] = options.history_prior;
    j["history prior mode"] = options.history_prior_mode;
    j["history prior weight"] = options.history_prior_weight;
    j["history prior threshold"] = options.history_prior_threshold;
    j["history prior top"] = options.history_prior_top;
    j["proxy cut"] = options.proxy_cut;
    j["proxy rank"] = options.proxy_rank;
    j["probcut margin"] = options.probcut_margin;
    j["proxy depth"] = options.proxy_depth;
    j["proxy reduction"] = options.proxy_reduction;
    j["proxy delta"] = options.proxy_delta;
    j["proxy witness"] = options.proxy_witness;
    j["proxy multi c"] = options.proxy_multi_c;
    j["proxy multi m"] = options.proxy_multi_m;
    j["proxy diverse"] = options.proxy_diverse;
    j["proxy second"] = options.proxy_second;
    j["proxy prefilter"] = options.proxy_prefilter;
    j["proxy weval"] = options.proxy_weval;
    j["proxy verify"] = options.proxy_verify;
    j["proxy lazy"] = options.proxy_lazy;
    j["proxy first"] = options.proxy_first;
    j["proxy tail"] = options.proxy_tail;
    j["proxy tail keep"] = options.proxy_tail_keep;
    j["proxy tail margin"] = options.proxy_tail_margin;
    j["proxy tail reduction"] = options.proxy_tail_reduction;
    j["proxy tail step"] = options.proxy_tail_step;
    j["proxy tail cap"] = options.proxy_tail_cap;
    j["mpc"] = options.mpc;
    j["mpc depth"] = options.mpc_depth;
    j["mpc ratio"] = options.mpc_ratio;
    j["mpc margin"] = options.mpc_margin;
    j["mpc margin per ply"] = options.mpc_margin_per_ply;
    j["probcut"] = options.probcut;
    j["probcut depth"] = options.probcut_depth;
    j["probcut reduction"] = options.probcut_reduction;
    j["probcut filter"] = options.probcut_filter;
    j["nb threads"] = nb_threads;
    j["territory ordering"] = options.territory_ordering;
    j["territory depth"] = options.territory_depth;
    j["territory weight"] = options.territory_weight;
    j["fast influence"] = options.fast_influence;
    j["articulation ordering"] = options.articulation_ordering;
    j["articulation lmr"] = options.articulation_lmr;
    j["nb moves at full depth"] = nb_moves_at_full_depth;
    j["late move reduction"] = late_move_reduction;
    j["weights"] = nnue_q_parameters_filename;
    j["pvs"] = options.pvs;
    j["aspiration window"] = options.aspiration_window;
    j["aspiration k"] = options.aspiration_k;
    j["aspiration grow"] = options.aspiration_grow;
    j["tempo"] = options.tempo;
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

#ifdef YOLAH_SEARCH_TRACE
// ─── The search debugger's helpers (search_trace executable only) ───────────
int MinMaxNNUE_DevPlayer::trace_return(int line, int value) {
    if (tracer) tracer->ev(line, SearchTracer::RETURN, value);
    return value;
}

// The network's value of a position, from a fresh accumulator: no cache, no
// counter, no lazy accumulator touched (the search does not see it). The same
// value as evaluate() (the accumulator sums are exact, see raw_network_value).
int MinMaxNNUE_DevPlayer::trace_value(const Yolah& yolah) {
    if (!tracer || yolah.game_over()) return 0;
    Accumulator a;
    net_init(yolah, a);
    return round_to_grain(int(net_value(a, yolah) * WIN), options.eval_grain);
}

// The network's value of every child, for the side to move here.
void MinMaxNNUE_DevPlayer::trace_child_evals(int line, const Yolah& yolah) {
    if (yolah.game_over()) return;
    Yolah y = yolah;
    Yolah::MoveList moves;
    y.moves(moves);
    std::vector<int32_t> body{int32_t(moves.size())};
    for (Move m : moves) {
        y.play(m);
        int v;
        if (y.game_over()) {
            const int score = y.score(y.current_player());
            v = score == 0 ? 0 : score + (score > 0 ? WIN : -WIN);
        } else {
            v = trace_value(y);
        }
        y.undo(m);
        body.push_back(m.raw());
        body.push_back(-v);
    }
    tracer->ev_list(line, SearchTracer::CHILD_EVALS, body);
}

// SEARCH: the root's ply and number of moves, then the main thread's history
// (2 × 64 × 64, after the aging of this search) and its countermoves (count,
// then (opponent move, countermove) pairs).
void MinMaxNNUE_DevPlayer::trace_search_start(int line, const Yolah& yolah) {
    Yolah::MoveList moves;
    yolah.moves(moves);
    std::vector<int32_t> body{yolah.nb_plies(), int32_t(moves.size())};
    const Worker& w = *workers[0];
    for (int p = 0; p < 2; p++)
        for (int from = 0; from < SQUARE_NB; from++)
            for (int to = 0; to < SQUARE_NB; to++) body.push_back(w.history[p][from][to]);
    std::vector<int32_t> counters;
    for (int from = 0; from < SQUARE_NB; from++)
        for (int to = 0; to < SQUARE_NB; to++)
            if (w.countermoves[from][to] != Move::none()) {
                counters.push_back(Move(Square(from), Square(to)).raw());
                counters.push_back(w.countermoves[from][to].raw());
            }
    body.push_back(int32_t(counters.size() / 2));
    body.insert(body.end(), counters.begin(), counters.end());
    tracer->ev_list(line, SearchTracer::SEARCH, body);
}

// PV: the table's moves from the root (as print_pv), at most `depth` of them.
void MinMaxNNUE_DevPlayer::trace_pv(int line, Yolah yolah, uint64_t hash, int depth) {
    std::vector<int32_t> body{0};
    for (int k = 0; k < depth && !yolah.game_over(); k++) {
        const Move m = tt_move(hash);
        if (m == Move::none() || !is_legal(yolah, m)) break;
        body.push_back(m.raw());
        const uint8_t player = yolah.current_player();
        yolah.play(m);
        hash = zobrist::update(hash, player, m);
    }
    body[0] = int32_t(body.size() - 1);
    tracer->ev_list(line, SearchTracer::PV, body);
}
#endif
