#include "minmax_nnue_dev_player.h"
#include <limits>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "zobrist.h"
#include <utility>
#include <algorithm>
#include <span>
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
    if (options.yolah_table) {
        yolah_table = std::make_unique<SearchTable>(tt_size_mb);
        yolah_table->tiebreak = options.table_tiebreak;
    }
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
    search_start = start;   // U5 (8)
    search_us = microseconds;
    if (!options.lmr_table.empty() && lmr_adjust.empty()) {   // U5 (6)
        lmr_adjust.assign(LMR_BUCKETS, 0);
        std::ifstream in(options.lmr_table);
        int b, d;
        while (in >> b >> d) if (b >= 0 && b < LMR_BUCKETS) lmr_adjust[b] = int8_t(d);
    }
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
    if (options.cont_age)   // U7 fix: the context tables age like the history
        for (auto& w : workers) {
            auto halve = [](auto& arr) {   // any int16 array, flat
                int16_t* p = reinterpret_cast<int16_t*>(&arr);
                for (size_t k = 0; k < sizeof(arr) / sizeof(int16_t); k++) p[k] /= 2;
            };
            halve(w->cont1); halve(w->cont2); halve(w->holec); halve(w->cfrom); halve(w->lmrh); halve(w->ccorr);
            halve(w->rel); halve(w->cont2s); halve(w->hpv); halve(w->geo); halve(w->cprec); halve(w->fnh);
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
    //     thread 2        : depth 1     4 5     8 …    see iterative_deepening)
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
    // U9: the moves that led to the root, by comparison with the previous root
    // (two plies before: our move, then the opponent's), for the context tables.
    if (options.root_context) {
        const int ply = yolah.nb_plies();
        Move p1 = Move::none(), p2 = Move::none();
        if (prev_root_ply >= 0 && ply == prev_root_ply + 2) {
            const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
            auto moved = [&](uint64_t before, uint64_t after) {
                const uint64_t from = before & ~after, to = after & ~before;
                return std::popcount(from) == 1 && std::popcount(to) == 1 ? Move(Square(std::countr_zero(from)), Square(std::countr_zero(to))) : Move::none();
            };
            p2 = moved(prev_root[us], yolah.bitboard(us));
            p1 = moved(prev_root[them], yolah.bitboard(them));
        }
        for (auto& sp : searches) {
            if (ply >= 1) sp->played[ply - 1] = p1;
            if (ply >= 2) sp->played[ply - 2] = p2;
        }
        prev_root = {yolah.bitboard(Yolah::BLACK), yolah.bitboard(Yolah::WHITE), yolah.empty_bitboard()};
        prev_root_ply = ply;
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
        asp_stats[0] += s->asp_attempts; asp_stats[1] += s->asp_failed;
        asp_stats[2] += s->asp_rebounds; asp_stats[3] += s->asp_failed_nodes;
        region_cuts += s->nb_region_cuts;
        validity[0] += s->nb_checked; validity[1] += s->nb_false;
        fly_stats[0] += s->nb_fly_tests; fly_stats[1] += s->nb_fly_prunes; fly_stats[2] += s->nb_fly_cuts; fly_stats[3] += s->nb_fly_nodes;
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
         + (options.fly_reply ? " + fly reply" : "")
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
    // U4: the child's expected node type (PV child: unused; a non-PV child of a
    // PV node or of an ALL node: CUT; of a CUT node: ALL for the first move)
    s.cutnode[yolah.nb_plies() + 1] = beta - alpha > 1 ? true : s.cutnode[yolah.nb_plies()] ? i > 0 : true;
    // H: start loading the child's table entry now; it is needed as soon as
    // the child's search starts.
    if (yolah_table) yolah_table->prefetch(child);
    s.played[yolah.nb_plies()] = m;       // for the countermove and the accumulator of the child
    if (options.lazy_accumulator) {
        s.acc_ok[yolah.nb_plies() + 1] = false;   // the child's accumulator: computed if needed
    } else {
        net_play(player, m, s.acc, &s.refresh);
    }
    yolah.play(m);
    int v = alpha + 1;                    // "might beat alpha" until a search says otherwise
    s.last_stage = 1;
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
            s.last_stage = 2;
            TR(CALL_RESULT, 2, v);
            // ("pvs research high": also when the null window says ≥ beta at a PV
            // node: that lower bound comes from a non-PV search, with its pruning)
            if (v > alpha && (v < beta || (options.pvs_research_high && beta - alpha > 1)) && !stopped()) {
                TR(CALL, 3, alpha, beta, depth - 1);
                v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);          // exact value
                s.last_stage = 3;
                TR(CALL_RESULT, 3, v);
            }
        } else {
            TR(CALL, 4, alpha, beta, depth - 1);
            v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
            s.last_stage = 4;
            TR(CALL_RESULT, 4, v);
        }
    }
    yolah.undo(m);
    if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
    return v;
}

// ─── U. The stronger reply ───────────────────────────────────────────────────
// The moves of `player` with extra power: its ordinary moves plus jumps along
// the 8 lines of each piece, landing on a free square, by "jumps":
//   1 every free square of the line past the obstacles
//   2 only the first free square past the first run of obstacles
//   3 the same, but only over holes (a run containing a piece cannot be jumped)
//   4 the same, but only over a single obstacle square
//   5 "micro-fly": over a single obstacle square, landing in the same region
//     (8-connected squares that are not holes) as the piece: no new reserve
// ordinary[k]: moves[k] is an ordinary move. Returns the number of moves.
int fly_moves(const Yolah& yolah, uint8_t player, int jumps, Move* moves, bool* ordinary) {
    const uint64_t free = yolah.free_squares(), occupied = yolah.occupied_squares();
    const uint64_t pieces = yolah.bitboard(Yolah::BLACK) | yolah.bitboard(Yolah::WHITE);
    static constexpr int DF[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int n = 0;
    uint64_t bb = yolah.bitboard(player);
    auto region_of = [&](Square sq) {   // 8-connected squares that are not holes, from sq
        constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t open = ~yolah.empty_bitboard();
        uint64_t r = square_bb(sq), prev = 0;
        while (r != prev) {
            prev = r;
            const uint64_t row = r | ((r << 1) & NOT_A) | ((r >> 1) & NOT_H);
            r = (row | (row << 8) | (row >> 8)) & open;
        }
        return r;
    };
    while (bb) {
        const Square from = pop_lsb(bb);
        const uint64_t normal = attacks_bb(from, occupied) & free;
        uint64_t targets = normal;
        const uint64_t region = jumps == 5 ? region_of(from) : ~uint64_t(0);
        if (jumps == 1) targets = attacks_bb(from, 0) & free;
        else
            for (int d = 0; d < 8; d++) {
                int f = int(file_of(from)) + DF[d], r = int(rank_of(from)) + DR[d];
                auto inside = [&] { return f >= 0 && f < 8 && r >= 0 && r < 8; };
                auto sq = [&] { return square_bb(make_square(File(f), Rank(r))); };
                while (inside() && (sq() & free)) { f += DF[d]; r += DR[d]; }      // the ordinary slide
                int run = 0;
                bool piece = false;
                while (inside() && !(sq() & free)) { piece |= (sq() & pieces) != 0; run++; f += DF[d]; r += DR[d]; }   // the obstacles
                if (!inside() || run == 0) continue;
                if (jumps == 3 && piece) continue;
                if ((jumps == 4 || jumps == 5) && run > 1) continue;
                if (jumps == 5 && !(sq() & region)) continue;
                targets |= sq();
            }
        while (targets) {
            const Square to = pop_lsb(targets);
            ordinary[n] = (normal & square_bb(to)) != 0;
            moves[n++] = Move(from, to);
        }
    }
    return n;
}

// The node of the side with extra power (the fly reply's opponent, or our
// next move for the optimistic test): its moves are fly_moves(s.fly_jumps).
// Its children are ordinary positions (searched and stored in the table as
// usual); this node itself is not a real position: no table, no killers, no
// history. Fail-soft.
int MinMaxNNUE_DevPlayer::negamax_fly(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth) {
    ++s.nb_nodes;
    if (stopped()) return 0;
    const uint8_t player = yolah.current_player();
    Move moves[4 * 64];
    bool ordinary[4 * 64];
    int scores[4 * 64];
    const int n = fly_moves(yolah, player, s.fly_jumps, moves, ordinary);
    // No ordinary move: no extra power either, the side passes as the rules say
    // (forcing a jump instead of a pass could hurt it in a zugzwang, and the
    // bound would no longer hold).
    if (std::none_of(ordinary, ordinary + n, [](bool b) { return b; })) {
        const int saved = s.fly_ply;
        s.fly_ply = -1;
        const int v = negamax(yolah, s, hash, alpha, beta, depth);
        s.fly_ply = saved;
        return v;
    }
    for (int k = 0; k < n; k++)
        scores[k] = s.w->history[player][moves[k].from_sq()][moves[k].to_sq()]
                  + (options.fly_reply_order == 0 && ordinary[k] ? 1 << 20 : options.fly_reply_order == 1 && !ordinary[k] ? 1 << 20 : 0);
    int best = -INFINITE;
    for (int i = 0; i < n; i++) {
        int k = i;
        for (int j = i + 1; j < n; j++) if (scores[j] > scores[k]) k = j;
        std::swap(moves[i], moves[k]);
        std::swap(scores[i], scores[k]);
        const Move m = moves[i];
        const uint64_t child = zobrist::update(hash, player, m);
        s.played[yolah.nb_plies()] = m;
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else net_play(player, m, s.acc, &s.refresh);
        yolah.play(m);
        const int v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
        if (stopped()) return 0;
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) break;
    }
    return best;
}

// One ply with extra power, statically: the best network value the side to
// move can reach with fly_moves(jumps) (its own point of view). With
// `ordinary_only`, its ordinary moves only. (pivots 6 and 7, and the gap of 4)
int MinMaxNNUE_DevPlayer::fly_static(Yolah& yolah, Search& s, uint64_t hash, int jumps, bool ordinary_only) {
    const uint8_t player = yolah.current_player();
    Move moves[4 * 64];
    bool ordinary[4 * 64];
    const int n = fly_moves(yolah, player, jumps, moves, ordinary);
    int best = -INFINITE;
    for (int k = 0; k < n; k++) {
        if (ordinary_only && !ordinary[k]) continue;
        const Move m = moves[k];
        s.played[yolah.nb_plies()] = m;
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else net_play(player, m, s.acc, &s.refresh);
        yolah.play(m);
        const int v = yolah.game_over() ? (yolah.score(yolah.current_player()) > 0 ? WIN : yolah.score(yolah.current_player()) < 0 ? -WIN : 0)
                                        : evaluate(yolah, s, zobrist::update(hash, player, m));
        yolah.undo(m);
        if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
        best = std::max(best, -v);
    }
    return best == -INFINITE ? 0 : best;
}

// Pivot 5: bounds of the final score difference (side to move's view) from the
// regions of free squares (8-connected): a piece can only ever reach the free
// squares of the regions next to it (sound), and is assumed to use all the
// squares that only it can reach (heuristic: a region can be lost by blocking
// oneself). low = diff + private(us) − reach(them), high = diff + reach(us) − private(them).
static uint64_t region_reach(uint64_t pieces, uint64_t free) {
    auto grow = [](uint64_t b) {
        constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t e = (b << 1) & NOT_A, w = (b >> 1) & NOT_H;
        const uint64_t row = b | e | w;
        return row | (row << 8) | (row >> 8);
    };
    uint64_t r = grow(pieces) & free, prev = 0;
    while (r != prev) { prev = r; r = grow(r) & free; }
    return r;
}
// G (sound): a legal harvest of the squares `excl` (free squares only `pieces`
// can ever reach): moves only through and onto those squares, greedy (the
// destination with the fewest free neighbours first). It can be played
// whatever the opponent does, so our future moves are at least its length.
static int greedy_harvest(uint64_t pieces, uint64_t excl) {
    constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
    auto neighbours = [&](uint64_t b) {
        const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
        return (row | (row << 8) | (row >> 8)) & ~b;
    };
    int count = 0;
    for (;;) {
        Square best_from = SQ_NONE, best_to = SQ_NONE;
        int best_n = 99;
        uint64_t pb = pieces;
        while (pb) {
            const Square from = pop_lsb(pb);
            uint64_t t = attacks_bb(from, ~excl) & excl;
            while (t) {
                const Square to = pop_lsb(t);
                const int nb = std::popcount(neighbours(square_bb(to)) & excl);
                if (nb < best_n) { best_n = nb; best_from = from; best_to = to; }
            }
        }
        if (best_from == SQ_NONE) return count;
        pieces ^= square_bb(best_from) | square_bb(best_to);
        excl &= ~square_bb(best_to);
        count++;
    }
}
// G: the bounds as score differences (low, high), side to move's view (search_bench checks them).
std::pair<int, int> region_score_bounds(const Yolah& yolah) {
    const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
    const uint64_t free = yolah.free_squares();
    const uint64_t ru = region_reach(yolah.bitboard(us), free), rt = region_reach(yolah.bitboard(them), free);
    const int lu = (ru & ~rt) ? greedy_harvest(yolah.bitboard(us), ru & ~rt) : 0;
    const int lt = (rt & ~ru) ? greedy_harvest(yolah.bitboard(them), rt & ~ru) : 0;
    return {yolah.score(us) + lu - std::popcount(rt), yolah.score(us) + std::popcount(ru) - lt};
}
// G: sound bounds of the final value (the engine's scale: WIN + difference).
std::pair<int, int> region_values(const Yolah& yolah, int win) {
    const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
    const uint64_t free = yolah.free_squares();
    const uint64_t ru = region_reach(yolah.bitboard(us), free), rt = region_reach(yolah.bitboard(them), free);
    const int diff = yolah.score(us);
    const int lu = (ru & ~rt) ? greedy_harvest(yolah.bitboard(us), ru & ~rt) : 0;
    const int lt = (rt & ~ru) ? greedy_harvest(yolah.bitboard(them), rt & ~ru) : 0;
    const int low = diff + lu - std::popcount(rt), high = diff + std::popcount(ru) - lt;
    auto value = [&](int d) { return d > 0 ? win + d : d < 0 ? d - win : 0; };
    return {value(low), value(high)};
}
std::pair<int, int> region_bounds(const Yolah& yolah) {
    const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
    const uint64_t free = yolah.free_squares();
    const uint64_t ru = region_reach(yolah.bitboard(us), free), rt = region_reach(yolah.bitboard(them), free);
    const int diff = yolah.score(us);
    const int low = diff + std::popcount(ru & ~rt) - std::popcount(rt);
    const int high = diff + std::popcount(ru) - std::popcount(rt & ~ru);
    return {low, high};
}

// ─── U7 helpers ──────────────────────────────────────────────────────────────
// C4: the context of a move for the geometric history: the holes around its
// origin (8 bits), its direction (8) and distance class (1, 2-3, 4+).
static inline void geo_index(const Yolah& y, Move m, int& mask, int& dir, int& dist) {
    static constexpr int DF[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int f = int(file_of(m.from_sq())), r = int(rank_of(m.from_sq()));
    mask = 0;
    for (int d = 0; d < 8; d++) {
        const int ff = f + DF[d], rr = r + DR[d];
        if (ff < 0 || ff > 7 || rr < 0 || rr > 7 || (y.empty_bitboard() & square_bb(make_square(File(ff), Rank(rr))))) mask |= 1 << d;
    }
    const int df = int(file_of(m.to_sq())) - f, dr = int(rank_of(m.to_sq())) - r;
    const int sf = (df > 0) - (df < 0), sr = (dr > 0) - (dr < 0);
    dir = 0;
    for (int d = 0; d < 8; d++) if (DF[d] == sf && DR[d] == sr) dir = d;
    const int len = std::max(std::abs(df), std::abs(dr));
    dist = len <= 1 ? 0 : len <= 3 ? 1 : 2;
}

// ─── U5 helpers ──────────────────────────────────────────────────────────────
// 6: the bucket of a late move for the learnt LMR (8 phases × 5 depths × 4 ranks
// × 4 history levels × 3 counts of the piece's moves already tried).
int lmr_bucket(int free, int depth, size_t i, int hist, int tried) {
    const int fb = std::min(free / 8, 7), db = std::min(depth, 14) / 3, rb = i < 4 ? 0 : i < 8 ? 1 : i < 16 ? 2 : 3;
    const int hb = hist < -2000 ? 0 : hist < 0 ? 1 : hist < 2000 ? 2 : 3, tb = std::min(tried, 2);
    return (((fb * 5 + db) * 4 + rb) * 4 + hb) * 3 + tb;
}
// 9: the network on the position turned by 180° (the rules are symmetric, and
// this symmetry keeps the starting position: a1 ↔ h8, d5 ↔ e4; a top-bottom
// mirror swaps the two sides' starting corners and leaves the network's domain).
static uint64_t rotate180(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return __builtin_bswap64(x);
}
int MinMaxNNUE_DevPlayer::sym_eval(const Yolah& yolah) {
    Yolah y;
    const auto [bs, ws] = yolah.score();
    y.set_state(rotate180(yolah.bitboard(Yolah::BLACK)), rotate180(yolah.bitboard(Yolah::WHITE)),
                rotate180(yolah.empty_bitboard()), bs, ws, yolah.nb_plies());
    net_init(y, sym_acc);
    return int(net_value(sym_acc, y) * WIN);
}
std::pair<int, int> MinMaxNNUE_DevPlayer::sym_values(const Yolah& yolah) {
    net_init(yolah, sym_acc);
    const int e = int(net_value(sym_acc, yolah) * WIN);
    return {e, sym_eval(yolah)};
}

// ─── U4 helpers ──────────────────────────────────────────────────────────────
// 1: the longest harvest of `pieces` on the free squares `region` (moves only
// through and onto them), exactly, by a depth-first search with a memo; −1
// beyond `budget` nodes.
static int harvest_exact_rec(uint64_t pieces, uint64_t region, std::unordered_map<uint64_t, int>& memo, int64_t& budget) {
    if (--budget < 0) return -1;
    const uint64_t key = (pieces * 0x9E3779B97F4A7C15ULL) ^ (region + 0x632BE59BD9B4E019ULL) ^ (region >> 17);
    if (auto it = memo.find(key); it != memo.end()) return it->second;
    int best = 0;
    uint64_t pb = pieces;
    while (pb) {
        const Square from = pop_lsb(pb);
        uint64_t t = attacks_bb(from, ~region) & region;
        while (t) {
            const Square to = pop_lsb(t);
            if (best >= std::popcount(region)) break;   // cannot do better than every square
            const int v = harvest_exact_rec(pieces ^ square_bb(from) ^ square_bb(to), region & ~square_bb(to), memo, budget);
            if (v < 0) return -1;
            best = std::max(best, 1 + v);
        }
    }
    memo.emplace(key, best);
    return best;
}
static int harvest_exact(uint64_t pieces, uint64_t region, int budget) {
    std::unordered_map<uint64_t, int> memo;
    int64_t b = budget;
    return harvest_exact_rec(pieces, region, memo, b);
}
// 1: the exact final difference (side to move) once the sides are separated;
// false if they are not, or beyond the budget (search_bench checks it).
bool split_score(const Yolah& yolah, int budget, int& diff) {
    const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
    const uint64_t free = yolah.free_squares();
    const uint64_t ru = region_reach(yolah.bitboard(us), free), rt = region_reach(yolah.bitboard(them), free);
    if (ru & rt) return false;
    const int lu = harvest_exact(yolah.bitboard(us), ru, budget), lt = harvest_exact(yolah.bitboard(them), rt, budget);
    if (lu < 0 || lt < 0) return false;
    diff = yolah.score(us) + lu - lt;
    return true;
}
// 6: the correction history's slot: the pieces of both sides.
static inline int corr_key(const Yolah& yolah) {
    uint64_t x = yolah.bitboard(Yolah::BLACK) * 0x9E3779B97F4A7C15ULL ^ yolah.bitboard(Yolah::WHITE) * 0xC2B2AE3D27D4EB4FULL;
    x ^= x >> 29;
    return int(x & 16383);
}
// 5b: free squares the opponent can reach through squares that are not holes,
// pieces included as stepping stones (`open`: the squares that are not holes).
static uint64_t reach_open(uint64_t pieces, uint64_t open) {
    constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t r = pieces & open, prev = 0;
    while (r != prev) {
        prev = r;
        const uint64_t row = r | ((r << 1) & NOT_A) | ((r >> 1) & NOT_H);
        r = (row | (row << 8) | (row >> 8)) & open;
    }
    return r;
}
constexpr int EVAL_NONE = std::numeric_limits<int>::min();

// ─── U (C, D): moves that leave no hole ──────────────────────────────────────
// The piece moves and scores, but its square stays free. Not a real position
// (the scores no longer match the holes): the subtree uses no table (s.no_tt),
// and its accumulator is computed from scratch (the incremental update assumes
// a hole). Lazy accumulators only.
uint64_t MinMaxNNUE_DevPlayer::nohole_play(Yolah& yolah, Search& s, Move m, std::array<uint64_t, 6>& saved) {
    const auto [bs, ws] = yolah.score();
    saved = {yolah.bitboard(Yolah::BLACK), yolah.bitboard(Yolah::WHITE), yolah.empty_bitboard(), bs, ws, yolah.nb_plies()};
    const uint64_t mv = square_bb(m.from_sq()) | square_bb(m.to_sq());
    const bool black = yolah.current_player() == Yolah::BLACK;
    yolah.set_state(saved[0] ^ (black ? mv : 0), saved[1] ^ (black ? 0 : mv), saved[2],
                    uint16_t(bs + (black ? 1 : 0)), uint16_t(ws + (black ? 0 : 1)), uint16_t(saved[5] + 1));
    s.played[saved[5]] = m;
    net_init(yolah, s.accs[saved[5] + 1]);
    s.acc_ok[saved[5] + 1] = true;
    return zobrist::hash(yolah);
}
void MinMaxNNUE_DevPlayer::nohole_undo(Yolah& yolah, Search& s, const std::array<uint64_t, 6>& saved) {
    yolah.set_state(saved[0], saved[1], saved[2], uint16_t(saved[3]), uint16_t(saved[4]), uint16_t(saved[5]));
    s.acc_ok[saved[5] + 1] = false;
}
// D: the node whose moves leave no hole (the proxy witness's opponent).
int MinMaxNNUE_DevPlayer::negamax_nohole(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth) {
    ++s.nb_nodes;
    if (stopped()) return 0;
    Yolah::MoveList moves;
    yolah.moves(moves);
    if (moves[0] == Move::none()) {   // it must pass: an ordinary node
        const int saved = s.nohole_ply;
        s.nohole_ply = -1;
        const int v = negamax(yolah, s, hash, alpha, beta, depth);
        s.nohole_ply = saved;
        return v;
    }
    const uint8_t player = yolah.current_player();
    std::stable_sort(moves.begin(), moves.end(), [&](Move a, Move b) {
        return s.w->history[player][a.from_sq()][a.to_sq()] > s.w->history[player][b.from_sq()][b.to_sq()];
    });
    int best = -INFINITE;
    for (Move m : moves) {
        std::array<uint64_t, 6> saved;
        const uint64_t h = nohole_play(yolah, s, m, saved);
        s.no_tt++;
        const int v = -negamax(yolah, s, h, -beta, -alpha, depth - 1);
        s.no_tt--;
        nohole_undo(yolah, s, saved);
        if (stopped()) return 0;
        best = std::max(best, v);
        alpha = std::max(alpha, v);
        if (alpha >= beta) break;
    }
    return best;
}
// H: a test's decision checked by a normal search (no test inside): a pruned
// move m (prune: wrong if m beats `bound` = alpha), a cut by m (wrong if m stays
// below `bound` = beta); m = none: the whole node (prune: wrong if it beats
// alpha). Statistics only: it changes the table and the history.
bool MinMaxNNUE_DevPlayer::verify_decision(Yolah& yolah, Search& s, uint64_t hash, Move m, int depth, int bound, bool prune) {
    s.verifying = true;
    s.fly_nest++;
    int v;
    if (m == Move::none()) v = prune ? negamax(yolah, s, hash, bound, bound + 1, depth) : negamax(yolah, s, hash, bound - 1, bound, depth);
    else {
        const uint8_t player = yolah.current_player();
        s.played[yolah.nb_plies()] = m;
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else net_play(player, m, s.acc, &s.refresh);
        yolah.play(m);
        const uint64_t child = zobrist::update(hash, player, m);
        v = prune ? -negamax(yolah, s, child, -(bound + 1), -bound, depth - 1) : -negamax(yolah, s, child, -bound, -bound + 1, depth - 1);
        yolah.undo(m);
        if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
    }
    s.fly_nest--;
    s.verifying = false;
    const bool wrong = prune ? v > bound : v < bound;
    s.nb_checked++;
    s.nb_false += wrong;
    return wrong;
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
    if (options.max_nodes > 0 && s.id == 0 && s.nb_nodes >= options.max_nodes) stop = true;
    TR(ENTER, 0, alpha, beta, depth, yolah.nb_plies(), s.played[yolah.nb_plies() - 1].raw(),
       int32_t(uint32_t(hash)), int32_t(uint32_t(hash >> 32)), trace_value(yolah));
#ifdef YOLAH_SEARCH_TRACE
    if (tracer && tracer->child_evals_depth > 0 && depth >= tracer->child_evals_depth) trace_child_evals(__LINE__, yolah);
#endif
    if (stopped()) return TRET(0);    // the caller ignores this value
    // U, pivot 1: the node whose side has the extra power (the optimistic test)
    if (yolah.nb_plies() == s.fly_ply) return TRET(negamax_fly(yolah, s, hash, alpha, beta, depth));
    if (yolah.nb_plies() == s.nohole_ply) return TRET(negamax_nohole(yolah, s, hash, alpha, beta, depth));
    // between a test's root and its node with extra power or restricted moves, and
    // below a no-hole move, the values are not real: no table
    const bool fly_above = s.fly_ply > int(yolah.nb_plies()) || s.no_tt > 0 || s.restrict_ply >= int(yolah.nb_plies())
                           || s.excluded[yolah.nb_plies()] != Move::none();   // U4: the singular search (same hash)
    const uint16_t ply_now = yolah.nb_plies();
    s.eval_stack[ply_now] = EVAL_NONE;
    // U9: with "killer special", the killers count as special moves even when ordered as a bonus
    auto killer_sp = [&](Move mv) { return options.killer_special && mv != Move::none() && (mv == s.killer1[ply_now] || mv == s.killer2[ply_now]); };
    if (options.sibling_ref) for (auto& r : s.sib_ref[ply_now + 1]) r = Move::none();
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
    if (tt.found && !fly_above) {
        s.nb_hits++;
        // ("tt cut pv" off: at PV nodes only an exact value — unless "tt cut pv
        // exact" is off too — or a proven result cuts; the bounds mostly come
        // from non-PV searches, with their pruning. "tt depth asym": a lower
        // bound must come from a strictly deeper search.)
        const int tt_need = depth + (options.tt_fragile > 0 && tt.fragile
                                     && (options.tt_fragile == 2 || int(ply_now) - s.root_ply < options.tt_fragile_ply) ? 1 : 0);
        if (tt.depth >= tt_need && !(s.audit && int(ply_now) - s.root_ply < 3) && (options.tt_cut_pv || beta - alpha == 1
                                  || (options.tt_cut_pv_exact && tt.lower == tt.upper)
                                  || tt.lower > WIN || tt.upper < -WIN)) {
            if (tt.lower >= beta && (!options.tt_depth_asym || tt.depth > depth)) {
                if (options.tt_bonus > 0 && tt_move != Move::none() && options.history && is_legal(yolah, tt_move)) {   // U10
                    const uint8_t player = yolah.current_player();
                    const int b = std::min(16 * depth * depth + 32 * depth, 2000) * options.tt_bonus / 100;
                    update_history(s, player, tt_move, b);
                    if (options.cont_hist > 0 && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {
                        int16_t& h = s.w->cont1[player][s.played[ply_now - 1].to_sq()][tt_move.to_sq()];
                        h = int16_t(h + b - h * b / HISTORY_MAX);
                    }
                }
                return TRET(tt.lower);
            }
            if (tt.upper <= alpha) return TRET(tt.upper);
            if (tt.lower == tt.upper) return TRET(tt.lower);
        }
    }
    // U (G): sound bounds of the final value from the regions of free squares.
    if (options.region_bounds == 2 && std::popcount(yolah.free_squares()) <= options.region_max_free) {
        const auto [lv, uv] = region_values(yolah, WIN);
        if (lv >= beta) { s.nb_region_cuts++; return TRET(lv); }
        if (uv <= alpha) { s.nb_region_cuts++; return TRET(uv); }
    }
    // U4 (1): no region reached by both sides: the exact value from each side's longest harvest.
    if (options.split_solve > 0 && std::popcount(yolah.free_squares()) <= options.split_max_free) {
        const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
        const uint64_t free = yolah.free_squares();
        const uint64_t ru = region_reach(yolah.bitboard(us), free), rt = region_reach(yolah.bitboard(them), free);
        if ((ru & rt) == 0) {
            const int lu = harvest_exact(yolah.bitboard(us), ru, options.split_budget);
            const int lt = lu < 0 ? -1 : harvest_exact(yolah.bitboard(them), rt, options.split_budget);
            if (lu >= 0 && lt >= 0) {
                const int d = yolah.score(us) + lu - lt;
                const int v = d > 0 ? WIN + d : d < 0 ? d - WIN : 0;
                s.nb_region_cuts++;
                if (!fly_above) tt_store(hash, yolah, 63, -INFINITE, INFINITE, v, Move::none());
                return TRET(v);
            }
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
        int leaf = evaluate(yolah, s, hash);
        if (options.corr_hist == 2)   // U4 (6): the corrected value at the leaves too
            leaf = std::clamp(leaf + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        if (options.qs == 0 || depth <= -options.qs_depth || s.verifying) return TRET(leaf);
        // U4 (3): quiescence
        Yolah::MoveList qm;
        yolah.moves(qm);
        const uint8_t player = yolah.current_player(), opp = Yolah::other_player(player);
        if (options.qs == 2) {   // mobility: few moves, all played, no stand pat
            if (qm[0] != Move::none() && int(qm.size()) > options.qs_moves) return TRET(leaf);
            int best = -INFINITE;
            for (Move m : qm) {
                const uint64_t child = zobrist::update(hash, player, m);
                s.played[ply_now] = m;
                if (options.lazy_accumulator) s.acc_ok[ply_now + 1] = false;
                else net_play(player, m, s.acc, &s.refresh);
                yolah.play(m);
                const int v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
                yolah.undo(m);
                if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
                if (stopped()) return TRET(0);
                best = std::max(best, v);
                alpha = std::max(alpha, v);
                if (alpha >= beta) break;
            }
            return TRET(best);
        }
        // 1: tension moves with stand pat
        if (leaf >= beta || qm[0] == Move::none()) return TRET(leaf);
        int best = leaf;
        alpha = std::max(alpha, leaf);
        const uint64_t free = yolah.free_squares();
        uint64_t weak_ours = 0;
        for (uint64_t b = yolah.bitboard(player); b;) { const Square sq = pop_lsb(b); if (std::popcount(around(square_bb(sq)) & free) <= 2) weak_ours |= square_bb(sq); }
        for (Move m : qm) {
            bool tension = (weak_ours & square_bb(m.from_sq())) != 0;
            if (!tension) {
                const uint64_t f2 = free & ~square_bb(m.to_sq());
                for (uint64_t b = yolah.bitboard(opp); b && !tension;) {
                    const uint64_t q = square_bb(pop_lsb(b));
                    tension = std::popcount(around(q) & f2) <= 1 && std::popcount(around(q) & free) > 1;
                }
            }
            if (!tension) continue;
            const uint64_t child = zobrist::update(hash, player, m);
            s.played[ply_now] = m;
            if (options.lazy_accumulator) s.acc_ok[ply_now + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            const int v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            if (stopped()) return TRET(0);
            best = std::max(best, v);
            alpha = std::max(alpha, v);
            if (alpha >= beta) break;
        }
        return TRET(best);
    }
    // F. Pruning, at non-PV nodes only (null window: the question is only
    //    "is this node ≥ beta?", an approximate answer costs little; at PV
    //    nodes we want the exact value). It relies on the STATIC evaluation:
    //    the network's value of this node, before any search below it (cheap
    //    now: lazy accumulators and evaluation cache, see evaluate). Never
    //    near game results (|beta| ≥ WIN): a proven win must stay proven.
    const bool pv_node = beta - alpha > 1;
    const bool prunable = !pv_node && std::abs(beta) < WIN;
    // "tt pv": a position searched as a PV node before keeps PV treatment
    // (no reverse futility pruning, PV reductions), even with a null window.
    const bool tt_pv = options.tt_pv && yolah_table && (pv_node || (tt.found && tt.pv));
    int static_eval = 0;
    if (prunable && ((options.rfp_depth > 0 && depth <= options.rfp_depth) || (options.null_move && depth >= 3)
                     || (options.proxy_cut == 1 && depth >= options.proxy_depth)
                     || (options.mpc > 0 && depth >= options.mpc_depth)
                     || (options.pass_razor > 0 && depth >= options.pass_razor_depth)
                     || (options.nohole_razor > 0 && depth >= options.nohole_razor_depth)
                     || options.improving || options.corr_hist > 0)) {
        static_eval = evaluate(yolah, s, hash);
        if (options.corr_hist > 0)   // U4 (6)
            static_eval = std::clamp(static_eval + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        if (options.corr_cont > 0 && ply_now >= 2 && s.played[ply_now - 1] != Move::none() && s.played[ply_now - 2] != Move::none())   // U7 (C7)
            static_eval = std::clamp(static_eval + s.w->ccorr[yolah.current_player()][s.played[ply_now - 2].to_sq()][s.played[ply_now - 1].to_sq()],
                                     -WIN + 1, WIN - 1);
        s.eval_stack[ply_now] = static_eval;
    }
    // U4 (6): improving — better than two plies ago (unknown: assumed improving)
    const bool improving = !options.improving || ply_now < 2 || s.eval_stack[ply_now] == EVAL_NONE
                           || s.eval_stack[ply_now - 2] == EVAL_NONE || s.eval_stack[ply_now] > s.eval_stack[ply_now - 2];
    const int rfp_d = depth - (options.improving && improving ? 1 : 0);
    const bool cut_node = !pv_node && s.cutnode[ply_now];
    // U5 (9): near a cut, the network on the mirrored position; a disagreement
    // means an unreliable static value: no static pruning here, LMR one ply less.
    bool sym_unsure = false;
    if (options.sym > 0 && prunable && s.id == 0 && depth >= options.sym_depth && s.eval_stack[ply_now] != EVAL_NONE
        && std::abs(static_eval - beta) < options.sym_window && !fly_above)
        sym_unsure = std::abs(evaluate(yolah, s, hash) - sym_eval(yolah)) > options.sym_thr;
    // F1. Reverse futility pruning ("static null move"): if the static value
    //     is above beta by a margin that a shallow search will hardly lose,
    //     trust it and cut. The margin grows with the depth: the deeper the
    //     search, the more the value may still change.
    if (prunable && !tt_pv && depth <= options.rfp_depth)
        TR(RFP, static_eval, options.rfp_margin * depth, beta, static_eval - options.rfp_margin * rfp_d >= beta);
    if (prunable && options.razor_margin > 0 && depth <= 2 && s.eval_stack[ply_now] != EVAL_NONE
        && static_eval + options.razor_margin * depth <= alpha)   // U7 (B7): razoring
        return TRET(static_eval);
    if (prunable && !tt_pv && !sym_unsure && depth <= options.rfp_depth && static_eval - options.rfp_margin * rfp_d >= beta) {
        return TRET(static_eval);
    }
    // U, pivot 5: bounds of the final score from the regions of free squares.
    if (prunable && options.region_bounds == 1 && std::popcount(yolah.free_squares()) <= options.region_max_free) {
        const auto [low, high] = region_bounds(yolah);
        if (low >= options.region_margin) { s.nb_region_cuts++; return TRET(beta); }
        if (high <= -options.region_margin) { s.nb_region_cuts++; return TRET(alpha); }
    }
    // U (E): a free pass to fail low — if even passing stays <= alpha − margin.
    if (prunable && options.pass_razor > 0 && depth >= options.pass_razor_depth && !s.verifying && s.fly_nest == 0
        && s.no_tt == 0 && static_eval <= alpha - options.pass_razor_guard && alpha - options.pass_razor_margin > -WIN
        && yolah.nb_plies() > 0 && s.played[yolah.nb_plies() - 1] != Move::none()) {
        const int thr = alpha - options.pass_razor_margin;
        const int d = std::max(0, depth - 1 - (options.pass_razor_reduction + depth / 4 - 1));
        const uint8_t player = yolah.current_player();
        s.nb_fly_tests++;
        s.played[yolah.nb_plies()] = Move::none();
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else net_play(player, Move::none(), s.acc, &s.refresh);
        yolah.play(Move::none());
        s.fly_nest++;
        const int v = -negamax(yolah, s, zobrist::update(hash, player, Move::none()), -(thr + 1), -thr, d);
        s.fly_nest--;
        yolah.undo(Move::none());
        if (!options.lazy_accumulator) net_undo(player, Move::none(), s.acc, &s.refresh);
        if (stopped()) return TRET(0);
        if (v <= thr) {
            s.nb_fly_prunes++;
            if (options.verify_tests) verify_decision(yolah, s, hash, Move::none(), depth, alpha, true);
            return TRET(alpha);
        }
    }
    // U (C): the table's move (else killer 1) played without leaving a hole; if even so <= alpha, fail low.
    if (prunable && options.nohole_razor > 0 && options.lazy_accumulator && depth >= options.nohole_razor_depth
        && !s.verifying && s.fly_nest == 0 && s.no_tt == 0 && static_eval <= alpha - options.nohole_razor_guard) {
        Move m = tt_move != Move::none() && is_legal(yolah, tt_move) ? tt_move : Move::none();
        if (m == Move::none() && s.killer1[yolah.nb_plies()] != Move::none() && is_legal(yolah, s.killer1[yolah.nb_plies()]))
            m = s.killer1[yolah.nb_plies()];
        if (m != Move::none()) {
            const int d = std::max(0, depth - 1 - (options.nohole_razor_reduction + depth / 4 - 1));
            s.nb_fly_tests++;
            std::array<uint64_t, 6> saved;
            const uint64_t h = nohole_play(yolah, s, m, saved);
            s.no_tt++;
            s.fly_nest++;
            const int v = -negamax(yolah, s, h, -(alpha + 1), -alpha, d);
            s.fly_nest--;
            s.no_tt--;
            nohole_undo(yolah, s, saved);
            if (stopped()) return TRET(0);
            if (v <= alpha) {
                s.nb_fly_prunes++;
                if (options.verify_tests) verify_decision(yolah, s, hash, Move::none(), depth, alpha, true);
                return TRET(alpha);
            }
        }
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
        if (prunable && !sym_unsure && options.mpc > 0 && depth >= options.mpc_depth) {
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
            else net_play(player, k, s.acc, &s.refresh);
            yolah.play(k);
            TR(PROXY, k.raw(), bound, d);
            const int v = -negamax(yolah, s, child, -bound, -bound + 1, d);
            TR(PROXY_RESULT, k.raw(), v, v >= bound);
            yolah.undo(k);
            if (!options.lazy_accumulator) net_undo(player, k, s.acc, &s.refresh);
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
    // U4 (9): internal iterative reduction — no table move: this node was never
    // searched (or lost), its ordering is poor: one ply less.
    if (options.iir > 0 && depth >= options.iir_depth && tt_move == Move::none() && s.excluded[ply_now] == Move::none()
        && (options.iir == 1 || pv_node || cut_node))
        depth--;
    // U4 (8): singular extension / multi-cut on the table move.
    int se_ext = 0;
    if (options.se > 0 && depth >= options.se_depth && tt_move != Move::none() && s.excluded[ply_now] == Move::none()
        && tt.found && tt.depth >= depth - 3 && tt.lower > -WIN && tt.lower < WIN && is_legal(yolah, tt_move)
        && s.ext_sum[ply_now] < (options.ext_max > 0 ? options.ext_max : std::max(1, int(s.depth) / 2)) && !s.verifying) {
        const int s_beta = tt.lower - options.se_margin * depth;
        s.excluded[ply_now] = tt_move;
        const int v = negamax(yolah, s, hash, s_beta - 1, s_beta, (depth - 1) / 2);
        s.excluded[ply_now] = Move::none();
        if (stopped()) return TRET(0);
        if (v < s_beta) {
            se_ext = 1 + (options.se_double && v < s_beta - options.se_margin * depth ? 1 : 0);
            s.nb_fly_tests++;
        } else if (options.multicut && v >= beta && std::abs(v) < WIN) {
            s.nb_fly_cuts++;
            return TRET(v);
        }
    }
    const bool tt_first = options.staged && tt_move != Move::none() && is_legal(yolah, tt_move) && s.excluded[ply_now] != tt_move;
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
        else net_play(player, Move::none(), s.acc, &s.refresh);
        yolah.play(Move::none());
        TR(CALL, 5, beta - 1, beta, depth - 1 - R);
        const int v = -negamax(yolah, s, zobrist::update(hash, player, Move::none()), -beta, -beta + 1, depth - 1 - R);
        TR(CALL_RESULT, 5, v);
        yolah.undo(Move::none());
        if (!options.lazy_accumulator) net_undo(player, Move::none(), s.acc, &s.refresh);
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
    // The proxy test as a function: NO_CUT, or what the node returns (already
    // traced). "proxy after tt": with a table move (staged), it waits until
    // that move has been searched without cutting (stage 2 of the loop).
    auto proxy_test = [&]() -> int {
        if (options.proxy_cut > 0 && prunable && !sym_unsure && depth >= options.proxy_depth && !s.proxy_off && !proxy_done
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
                else net_play(player, m, s.acc, &s.refresh);
                yolah.play(m);
                TR(PROXY, m.raw(), bound, d);
                int v;
                if (options.proxy_fly && s.fly_nest == 0) {   // U, pivot 3: against the stronger reply
                    s.fly_nest++;
                    s.fly_jumps = options.proxy_fly_jumps;
                    v = -negamax_fly(yolah, s, child, -bound, -bound + 1, d);
                    s.fly_nest--;
                } else if (options.proxy_nohole && options.lazy_accumulator && s.no_tt == 0 && s.fly_nest == 0) {   // U (D)
                    s.fly_nest++;
                    s.nohole_ply = yolah.nb_plies();
                    v = -negamax(yolah, s, child, -bound, -bound + 1, d);
                    s.nohole_ply = -1;
                    s.fly_nest--;
                } else if (options.proxy_restrict > 0 && s.fly_nest == 0) {   // U (B): our next turn limited to k moves
                    s.fly_nest++;
                    s.restrict_ply = yolah.nb_plies() + 1;
                    s.restrict_k = options.proxy_restrict;
                    v = -negamax(yolah, s, child, -bound, -bound + 1, d + options.proxy_restrict_deeper);
                    s.restrict_ply = -1;
                    s.fly_nest--;
                } else v = -negamax(yolah, s, child, -bound, -bound + 1, d);
                TR(PROXY_RESULT, m.raw(), v, v >= bound);
                yolah.undo(m);
                if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
                if (options.verify_tests && !s.verifying && v >= bound && !stopped()) verify_decision(yolah, s, hash, m, depth, bound, false);
                if (options.proxy_bonus > 0 && v >= bound && !stopped() && options.history) {   // U10: the witness that cuts
                    const int b = std::min(16 * depth * depth + 32 * depth, 2000) * options.proxy_bonus / 100;
                    update_history(s, player, m, b);
                    if (options.cont_hist > 0 && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {
                        int16_t& h = s.w->cont1[player][s.played[ply_now - 1].to_sq()][m.to_sq()];
                        h = int16_t(h + b - h * b / HISTORY_MAX);
                    }
                }
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
                    for (size_t j = 0; j < n; j++) if (scores[j] < SCORE_COUNTER && !killer_sp(moves[j])) cand[nc++] = int(j);
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
                        else net_play(player, m, s.acc, &s.refresh);
                        yolah.play(m);
                        val[cand[k]] = -evaluate(yolah, s, child);
                        yolah.undo(m);
                        if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
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
        return NO_CUT;
    };
    bool proxy_deferred = options.proxy_after_tt && tt_first && !generated;
    if (!proxy_deferred) {
        const int v = proxy_test();
        if (v != NO_CUT) return v;
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
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            int v = -INFINITE;
            if (!options.probcut_filter || -evaluate(yolah, s, child) >= bound) {
                TR(CALL, 6, bound - 1, bound, depth - options.probcut_reduction);
                v = -negamax(yolah, s, child, -bound, -bound + 1, depth - options.probcut_reduction);
                TR(CALL_RESULT, 6, v);
            }
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            if (stopped()) return TRET(0);
            if (v >= bound) return TRET(v >= WIN ? beta : v);
        }
    }
    if (proxy_deferred && generated) {      // ProbCut generated the moves: no stage 2
        proxy_deferred = false;
        const int v = proxy_test();
        if (v != NO_CUT) return v;
    }
    if (scores_pending) {
        score_moves(yolah, s, tt_move, moves, scores, depth);
    }
    // U, pivot 7: the ordinary moves ordered by our static value after the
    // opponent's best reply with extra power.
    auto fly_reorder = [&]() {
        if (!(options.fly_order > 0 && depth >= options.fly_order_depth && s.fly_nest == 0)) return;
        const uint8_t player = yolah.current_player();
        for (size_t k = 0; k < n; k++) {
            if (scores[k] >= SCORE_COUNTER || moves[k] == Move::none()) continue;
            const Move m = moves[k];
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            const int v = yolah.game_over() ? 0 : -fly_static(yolah, s, zobrist::update(hash, player, m), options.fly_order_jumps, false);
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            scores[k] = (scores[k] >= SCORE_ARTICULATION ? SCORE_ARTICULATION : 0) + v;
        }
    };
    if (generated) fly_reorder();
    // U, pivot 4: openness of the position, from the opponent's best static
    // reply (after a pass) with and without extra power.
    int fly_lmr_delta = 0;
    if (options.fly_lmr > 0 && prunable && depth >= options.fly_lmr_depth && s.fly_nest == 0 && yolah.nb_plies() > 0) {
        const uint8_t player = yolah.current_player();
        s.played[yolah.nb_plies()] = Move::none();
        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
        else net_play(player, Move::none(), s.acc, &s.refresh);
        yolah.play(Move::none());
        const uint64_t h2 = zobrist::update(hash, player, Move::none());
        const int v_norm = fly_static(yolah, s, h2, options.fly_lmr_jumps, true);
        const int v_fly = fly_static(yolah, s, h2, options.fly_lmr_jumps, false);
        yolah.undo(Move::none());
        if (!options.lazy_accumulator) net_undo(player, Move::none(), s.acc, &s.refresh);
        const int gap = v_fly - v_norm;
        fly_lmr_delta = gap < options.fly_lmr_closed ? 1 : gap > options.fly_lmr_open ? -1 : 0;
    }
    const int alpha_orig = alpha;
    int best = -INFINITE;
    Move best_move = Move::none();
    // U4 (6): the correction history learns (best − static value) where the
    // search's result says something about the static value.
    auto corr_update = [&](int bestv) {
        if ((options.corr_hist == 0 && options.corr_cont == 0) || fly_above || s.eval_stack[ply_now] == EVAL_NONE || std::abs(bestv) >= WIN || stopped()) return;
        if (bestv >= beta && bestv <= static_eval) return;          // a lower bound below the static value: says nothing
        if (bestv <= alpha_orig && bestv >= static_eval) return;    // an upper bound above it: says nothing
        const int b = std::clamp((bestv - static_eval) * depth / std::max(1, options.corr_scale), -2000, 2000);
        auto learn = [&](int16_t& e, int cap) { e = int16_t(std::clamp(e + b - e * std::abs(b) / cap, -cap, cap)); };
        if (options.corr_hist > 0) learn(s.w->corr[yolah.current_player()][corr_key(yolah)], 8000);
        if (options.corr_cont > 0 && ply_now >= 2 && s.played[ply_now - 1] != Move::none() && s.played[ply_now - 2] != Move::none())
            learn(s.w->ccorr[yolah.current_player()][s.played[ply_now - 2].to_sq()][s.played[ply_now - 1].to_sq()], options.corr_cont);
    };
    bool pruned = false;
    int fly_first = -INFINITE;
    bool fly_pruned = false;                 // U: a move of this node was pruned by the fly reply
    bool node_fragile = false;               // U5 (7): moves of this node pruned (LMP, proxy tail)               // U: the first move's value against the stronger reply ("fly reply relative")
    int after_witness = -1;                  // Q: ordinary moves searched after the witness (-1: not yet)
    const bool restricted = int(yolah.nb_plies()) == s.restrict_ply;   // U (B)
    // U6: the continuation history's term of a move, for the LMR ("cont lmr")
    auto cont_term = [&](Move mv) {
        if (!options.cont_lmr || options.cont_hist == 0 || mv == Move::none()) return 0;
        const uint8_t pl = yolah.current_player();
        const Move q1 = ply_now > 0 ? s.played[ply_now - 1] : Move::none(), q2 = ply_now > 1 ? s.played[ply_now - 2] : Move::none();
        int t = 0;
        if (q1 != Move::none()) t += s.w->cont1[pl][q1.to_sq()][mv.to_sq()] * options.cont_weight / 100;
        if (options.cont_hist > 1 && q2 != Move::none()) t += s.w->cont2[pl][q2.to_sq()][mv.to_sq()] * options.cont2_weight / 100;
        return t;
    };
    // U5 (1): the sealed pockets of the side to move: pieces that can only move
    // into a region of free squares only they reach. Such a pocket is a reserve
    // of tempos: only its maximal harvest matters, so one move that keeps it is
    // enough (any order of the pocket's moves can be played at any time).
    uint64_t reserve_pieces[4]{};
    Move reserve_keep[4]{};
    int n_reserves = 0;
    bool reserve_done = false;
    auto compute_reserves = [&]() {
        const uint8_t us = yolah.current_player();
        const uint64_t free = yolah.free_squares(), mine = yolah.bitboard(us), occupied = yolah.occupied_squares();
        uint64_t priv = region_reach(mine, free) & ~region_reach(yolah.bitboard(Yolah::other_player(us)), free);
        while (priv && n_reserves < 4) {
            const uint64_t seed = priv & -priv;
            const uint64_t comp = region_reach(seed, priv) | seed;
            priv &= ~comp;
            if (std::popcount(comp) > options.reserve_max) continue;
            uint64_t pieces = 0;
            bool sealed = true;
            for (uint64_t b = mine; b;) {
                const uint64_t q = square_bb(pop_lsb(b));
                const uint64_t nb = around(q) & free;
                if (nb & comp) { pieces |= q; if (nb & ~comp) sealed = false; }
            }
            if (!sealed || !pieces) continue;
            const int h = harvest_exact(pieces, comp, options.split_budget);
            if (h <= 0) continue;
            Move keep = Move::none();
            for (uint64_t b = pieces; b && keep == Move::none();) {
                const Square from = pop_lsb(b);
                for (uint64_t t = attacks_bb(from, occupied) & comp; t;) {
                    const Square to = pop_lsb(t);
                    if (harvest_exact(pieces ^ square_bb(from) ^ square_bb(to), comp & ~square_bb(to), options.split_budget) == h - 1) {
                        keep = Move(from, to);
                        break;
                    }
                }
            }
            if (keep == Move::none()) continue;
            reserve_pieces[n_reserves] = pieces;
            reserve_keep[n_reserves++] = keep;
        }
    };
    bool searched[Yolah::MAX_NB_MOVES]{};   // U7: the moves really searched (for the maluses)
    // U4: per-node state of the move loop
    uint8_t tried_from[SQUARE_NB]{};          // 4: moves of each piece already searched
    uint64_t pocket_done = 0;                 // 2: private regions already given a move
    uint64_t u4_ru = 0, u4_rt = 0;            // 2, 5: the regions reached by us / them (computed once)
    bool u4_regions = false;
    int opp_mob = -1;                         // 10: the opponent's mobility at this node
    auto regions = [&]() {
        if (!u4_regions) {
            const uint64_t free = yolah.free_squares();
            u4_ru = region_reach(yolah.bitboard(yolah.current_player()), free);
            u4_rt = region_reach(yolah.bitboard(Yolah::other_player(yolah.current_player())), free);
            u4_regions = true;
        }
    };
    for (size_t i = 0; i < n || !generated; i++) {
        if (restricted && i >= size_t(s.restrict_k)) break;
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
            fly_reorder();
            if (proxy_deferred) {             // "proxy after tt": the table's move did not cut
                proxy_deferred = false;
                const int v = proxy_test();
                if (v != NO_CUT) return v;
            }
            if (i >= n) break;
        }
        // F3. Late move pruning: at shallow non-PV nodes, after lmp_moves +
        //     depth² moves, the others (the worst ordered: bad history, never
        //     cut anywhere) are not searched at all — once a move that does not
        //     lose has been found. I: except the articulation moves, which are
        //     kept (moved to the front of what is left).
        if (!pruned && prunable && !s.audit && depth <= options.lmp_depth && i >= size_t(options.lmp_moves + depth * depth)
            && best > -WIN) {
            TR(LMP, int(i), int(n));
            node_fragile = true;
            if (!options.articulation_lmr && options.lmp_protect == 0) break;
            const Move q1 = ply_now > 0 ? s.played[ply_now - 1] : Move::none();
            auto keep = [&](Move mv) {   // articulation moves; U7 (A4): strong contextual refutations
                if (options.articulation_lmr && is_articulation_move(yolah, mv)) return true;
                return options.lmp_protect > 0 && q1 != Move::none() && mv != Move::none()
                       && s.w->cont1[yolah.current_player()][q1.to_sq()][mv.to_sq()] >= options.lmp_protect;
            };
            size_t k = i;
            for (size_t j = i; j < n; j++) {
                if (keep(moves[j])) {
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
        if (m == s.excluded[ply_now] && m != Move::none()) { TR(SKIP, 7, int(i), m.raw()); continue; }   // U4 (8): the singular search
        // U5 (1): sealed pockets — one move kept per pocket (exact).
        if (options.reserve > 0 && generated && m != Move::none() && std::popcount(yolah.free_squares()) <= options.reserve_max_free) {
            if (!reserve_done) { compute_reserves(); reserve_done = true; }
            bool drop = false;
            for (int k = 0; k < n_reserves; k++)
                if ((reserve_pieces[k] & square_bb(m.from_sq())) && m != reserve_keep[k]) drop = true;
            if (drop) { s.nb_region_cuts++; TR(SKIP, 1, int(i), m.raw()); continue; }
        }
        // U4 (2): one move per private region (only we can reach it) at non-PV nodes.
        if (options.pocket > 0 && prunable && depth >= options.pocket_depth && i > 0 && m != Move::none()) {
            regions();
            const uint64_t priv = u4_ru & ~u4_rt;
            if (priv & square_bb(m.to_sq())) {
                const uint64_t comp = region_reach(square_bb(m.to_sq()), priv) | square_bb(m.to_sq());
                if (comp & pocket_done) { s.nb_fly_prunes++; fly_pruned = true; TR(SKIP, 2, int(i), m.raw()); continue; }
                pocket_done |= comp;
            }
        }
        // U7 (A4): history pruning; (B6): futility pruning per move.
        if (generated && prunable && i >= 1 && best > -WIN && scores[i] < SCORE_COUNTER && !killer_sp(m) && m != Move::none() && depth <= 3) {
            if (options.hist_prune > 0 && i >= 2 && scores[i] < SCORE_ARTICULATION && scores[i] < -options.hist_prune * depth) { node_fragile = true; TR(SKIP, 3, int(i), m.raw()); continue; }
            if (options.fp_margin > 0 && s.eval_stack[ply_now] != EVAL_NONE && static_eval + 1000 + options.fp_margin * depth <= alpha) { node_fragile = true; TR(SKIP, 4, int(i), m.raw()); continue; }
        }
        // Q. Proxy tail: an ordinary move ordered after the failed witness.
        int tail_r = 0;
        if (tail && after_witness >= 0 && scores[i] < SCORE_COUNTER && !killer_sp(m) && after_witness++ >= options.proxy_tail_keep) {
            const int k = after_witness - 1 - options.proxy_tail_keep;   // 0 for the first move concerned
            if (options.proxy_tail <= 2) {
                if (best > -WIN) { node_fragile = true; break; }      // pruned (once a move that does not lose was found)
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
                    : late_move_reduction_of(depth, i, beta - alpha > 1 || tt_pv, scores[i] >= SCORE_COUNTER || killer_sp(m),
                                             scores[i] >= SCORE_ARTICULATION ? 0
                                             : options.lmr_territory || !options.history ? scores[i]
                                             : options.lmr_hist_full ? scores[i]
                                             : s.w->history[yolah.current_player()][m.from_sq()][m.to_sq()] + cont_term(m));
        // U. Fly reply: m first searched at a reduced depth against a stronger
        //    reply (the opponent may also jump over obstacles). Non-PV nodes
        //    only (null window: beta = alpha + 1), never in a branch where a
        //    fly reply is already active.
        const bool fly_ref = options.fly_reply_relative && i == 0;   // relative: the first move gives the threshold
        if (options.fly_reply > 0 && prunable && (generated || fly_ref) && (i >= size_t(options.fly_reply_from) || fly_ref)
            && depth >= options.fly_reply_depth && s.fly_nest == 0 && m != Move::none()) {
            const int R = options.fly_reply_reduction + depth / std::max(1, options.fly_reply_div);
            const int d = std::max(0, depth - 1 - R);
            const uint8_t player = yolah.current_player();
            s.nb_fly_tests++;
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            if (options.fly_reply_filter && !fly_ref) {   // U, pivot 6: a static ply with extra power first
                const int sv = yolah.game_over() ? 0 : -fly_static(yolah, s, zobrist::update(hash, player, m), options.fly_reply_jumps, false);
                if (sv < beta + options.fly_reply_filter_margin) {
                    yolah.undo(m);
                    if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
                    goto fly_reply_done;
                }
                s.played[yolah.nb_plies() - 1] = m;
                if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies()] = false;
            }
            s.fly_nest++;
            s.fly_jumps = options.fly_reply_jumps;
            {
            const size_t nodes0 = s.nb_nodes;
            // window: alpha, or (relative) the first move's value − margin; the first move itself: a wide window
            const int thr = options.fly_reply_relative ? fly_first - options.fly_reply_margin : alpha - options.fly_reply_margin;
            const int vf = fly_ref ? -negamax_fly(yolah, s, zobrist::update(hash, player, m), -INFINITE, INFINITE, d)
                                   : -negamax_fly(yolah, s, zobrist::update(hash, player, m), -(thr + 1), -thr, d);
            s.nb_fly_nodes += s.nb_nodes - nodes0;
            s.fly_nest--;
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            if (stopped()) return TRET(0);
            if (fly_ref) fly_first = vf;
            else if (vf <= thr && (options.fly_reply & 1) && !(options.fly_reply_relative && fly_first == -INFINITE)) {   // pruned
                s.nb_fly_prunes++;
                if (options.verify_tests && !s.verifying) verify_decision(yolah, s, hash, m, depth, alpha, true);
                fly_pruned = true;
                TR(SKIP, 5, int(i), m.raw());
                continue;
            }
            if (!fly_ref && vf > alpha && vf < WIN && (options.fly_reply & 2) && thr == alpha) {   // good even against the stronger reply: cut
                s.nb_fly_cuts++;
                if (options.verify_tests && !s.verifying) verify_decision(yolah, s, hash, m, depth, beta, false);
                return TRET(vf);
            }
            }
        }
    fly_reply_done:
        // U, pivot 1: the optimistic test, the right direction to prune: m
        //    searched at a reduced depth with OUR next move given the extra
        //    power (an upper bound of m's value at that depth); if even so it
        //    stays <= alpha, m is pruned.
        if (options.fly_prune > 0 && prunable && generated && i >= size_t(options.fly_prune_from)
            && depth >= options.fly_prune_depth && s.fly_nest == 0 && m != Move::none()) {
            const int R = options.fly_prune_reduction + depth / std::max(1, options.fly_prune_div);
            const int d = std::max(1, depth - 1 - R);   // at least one ply: the opponent's reply, then our move with extra power
            const uint8_t player = yolah.current_player();
            s.nb_fly_tests++;
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            s.fly_nest++;
            s.fly_jumps = options.fly_prune_jumps;
            const int saved_ply = s.fly_ply;
            s.fly_ply = yolah.nb_plies() + 1;
            const size_t nodes0 = s.nb_nodes;
            const int vo = -negamax(yolah, s, zobrist::update(hash, player, m), -(alpha + 1), -alpha, d);
            s.nb_fly_nodes += s.nb_nodes - nodes0;
            s.fly_ply = saved_ply;
            s.fly_nest--;
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            if (stopped()) return TRET(0);
            if (vo <= alpha) {
                s.nb_fly_prunes++;
                if (options.verify_tests && !s.verifying) verify_decision(yolah, s, hash, m, depth, alpha, true);
                fly_pruned = true;
                TR(SKIP, 5, int(i), m.raw());
                continue;
            }
        }
        // U (A): refutation by a restricted reply — the opponent limited to its
        //    likeliest replies; if one keeps us <= alpha, m is pruned.
        if (options.refute > 0 && prunable && generated && i >= size_t(options.refute_from) && depth >= options.refute_depth
            && s.fly_nest == 0 && s.no_tt == 0 && !s.verifying && m != Move::none()) {
            const uint8_t player = yolah.current_player(), opp = Yolah::other_player(player);
            const uint64_t child = zobrist::update(hash, player, m);
            s.played[yolah.nb_plies()] = m;
            if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            bool refuted = false;
            if (!yolah.game_over()) {
                Yolah::MoveList replies;
                yolah.moves(replies);
                if (replies[0] != Move::none()) {
                    Move cand[4];
                    int nc = 0;
                    auto add = [&](Move r) {
                        if (r == Move::none() || nc >= options.refute_replies) return;
                        for (int k = 0; k < nc; k++) if (cand[k] == r) return;
                        if (std::find(replies.begin(), replies.end(), r) == replies.end()) return;
                        cand[nc++] = r;
                    };
                    add(tt_probe(child).move);
                    add(s.w->countermoves[m.from_sq()][m.to_sq()]);
                    while (nc < options.refute_replies && nc < int(replies.size())) {   // then by history
                        Move bm = Move::none();
                        int bh = std::numeric_limits<int>::min();
                        for (Move r : replies) {
                            bool used = false;
                            for (int k = 0; k < nc; k++) used |= cand[k] == r;
                            if (!used && s.w->history[opp][r.from_sq()][r.to_sq()] > bh) { bh = s.w->history[opp][r.from_sq()][r.to_sq()]; bm = r; }
                        }
                        if (bm == Move::none()) break;
                        cand[nc++] = bm;
                    }
                    const int d2 = depth - 2 - options.refute_reduction;
                    s.nb_fly_tests++;
                    s.fly_nest++;
                    const size_t nodes0 = s.nb_nodes;
                    for (int k = 0; k < nc && !refuted && !stopped(); k++) {
                        const Move r = cand[k];
                        s.played[yolah.nb_plies()] = r;
                        if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
                        else net_play(opp, r, s.acc, &s.refresh);
                        yolah.play(r);
                        const int w = negamax(yolah, s, zobrist::update(child, opp, r), alpha, alpha + 1, d2);
                        yolah.undo(r);
                        if (!options.lazy_accumulator) net_undo(opp, r, s.acc, &s.refresh);
                        refuted = w <= alpha;
                    }
                    s.nb_fly_nodes += s.nb_nodes - nodes0;
                    s.fly_nest--;
                }
            }
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            if (stopped()) return TRET(0);
            if (refuted) {
                TR(SKIP, 6, int(i), m.raw());
                s.nb_fly_prunes++;
                if (options.refute_reduction > 0) fly_pruned = true;   // exact at reduction 0: the node may store its bound
                if (options.verify_tests && !s.verifying) verify_decision(yolah, s, hash, m, depth, alpha, true);
                continue;
            }
        }
        TR(MOVE, int(i), m.raw(), generated ? scores[i] : SCORE_TT, r, tail_r);
        int r_fly = r > 0 ? std::max(0, r + fly_lmr_delta) : 0;   // U, pivot 4
        // U4: reductions
        if (r_fly > 0 && (s.audit || sym_unsure)) r_fly = std::max(0, r_fly - 1);   // U5 (8, 9)
        if (r_fly > 0 && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {   // U7 (B8, C5)
            const Move q1 = s.played[ply_now - 1];
            if (options.intercept_lmr && ((attacks_bb(q1.to_sq(), yolah.occupied_squares()) | around(square_bb(q1.to_sq()))) & square_bb(m.to_sq())))
                r_fly = std::max(0, r_fly - 1);
            if (options.lmrh > 0 && s.w->lmrh[yolah.current_player()][q1.to_sq()][m.to_sq()] >= options.lmrh) r_fly = std::max(0, r_fly - 1);
            if (options.fn_sample > 0 && r_fly > 0 && s.w->fnh[yolah.current_player()][q1.to_sq()][m.to_sq()] >= options.fn_thr) r_fly = std::max(0, r_fly - 1);   // U10
        }
        if (r_fly > 0 && !lmr_adjust.empty())                                       // U5 (6)
            r_fly = std::max(0, r_fly + lmr_adjust[lmr_bucket(std::popcount(yolah.free_squares()), depth, i,
                                                              s.w->history[yolah.current_player()][m.from_sq()][m.to_sq()], tried_from[m.from_sq()])]);
        if (r_fly > 0) {
            if (cut_node) r_fly += options.cutnode_lmr;                                   // 9
            if (options.improving && !improving) r_fly += 1;                              // 6
            if (options.piece_lmr > 0 && tried_from[m.from_sq()] >= options.piece_lmr) r_fly += 1;   // 4
            if (options.family_lmr && tried_from[m.from_sq()] == 0) r_fly = std::max(0, r_fly - 1); // 4b
            if (options.block_lmr && r_fly > 0 && depth >= options.block_depth && block_zone_score(yolah, m) >= options.block_min)   // U12
                r_fly = std::max(0, r_fly - 1);
            if (options.gate_lmr > 0 && depth >= options.gate_depth) {                    // 5
                regions();
                const uint64_t free2 = yolah.free_squares() & ~square_bb(m.to_sq());
                const uint64_t mine2 = yolah.bitboard(yolah.current_player()) ^ square_bb(m.from_sq()) ^ square_bb(m.to_sq());
                const uint64_t ru2 = region_reach(mine2, free2), rt2 = region_reach(yolah.bitboard(Yolah::other_player(yolah.current_player())), free2);
                if (std::popcount(ru2 & rt2) <= std::popcount(u4_ru & u4_rt) - options.gate_min) r_fly = 0;
            }
            if (options.opp_lmr > 0 && r_fly > 0) {                                      // 10
                const uint8_t opp = Yolah::other_player(yolah.current_player());
                auto mobility = [&]() { const auto [a, b, c, d] = yolah.moves_bb(opp); return std::popcount(a) + std::popcount(b) + std::popcount(c) + std::popcount(d); };
                if (opp_mob < 0) opp_mob = mobility();
                yolah.play(m);
                const int after = mobility();
                yolah.undo(m);
                if (opp_mob - after >= options.opp_lmr_k) r_fly = std::max(0, r_fly - 1);
                else if (opp_mob == after && !(around(yolah.bitboard(opp)) & square_bb(m.to_sq()))) r_fly += 1;
            }
        }
        // U4: extensions (8: the singular table move; 5b: a destroyed square that cuts the opponent off)
        int ext = 0;
        const int ext_cap = options.ext_max > 0 ? options.ext_max : std::max(1, int(s.depth) / 2);
        if (se_ext > 0 && m == tt_move) ext = se_ext;
        else if (options.gate_ext > 0 && i < 4 && depth >= 6 && s.ext_sum[ply_now] < std::min(ext_cap, 2) && m != Move::none()) {
            const uint64_t open = ~yolah.empty_bitboard(), open2 = open & ~square_bb(m.from_sq());
            const uint64_t opp = yolah.bitboard(Yolah::other_player(yolah.current_player()));
            const uint64_t free = yolah.free_squares();
            if (std::popcount(reach_open(opp, open) & ~reach_open(opp, open2) & free) >= options.gate_ext_min) ext = 1;
        }
        if (s.ext_sum[ply_now] + ext > ext_cap) ext = 0;
        s.ext_sum[ply_now + 1] = s.ext_sum[ply_now] + ext;
        if (ext > 0) r_fly = 0;
        const int r_used = std::min(r_fly + tail_r, std::max(0, depth - 2));
        TR(REDUCE, int(i), m.raw(), r, r_used, ext);
        if (i < Yolah::MAX_NB_MOVES) searched[i] = true;
        const int v = search_move(yolah, s, hash, m, i, r_used, alpha, beta, depth + ext);
        // U7 (C5): a reduction that was wrong (the reduced search beat alpha and the full one confirmed it)
        if (options.lmrh > 0 && r_used > 0 && v > alpha && !stopped() && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {
            int16_t& h = s.w->lmrh[yolah.current_player()][s.played[ply_now - 1].to_sq()][m.to_sq()];
            const int b = std::min(16 * depth * depth + 32 * depth, 2000);
            h = int16_t(h + b - h * b / HISTORY_MAX);
        }
        // U10: the real LMR misses — a sample of the reduced fail-lows re-searched at full depth
        if (options.fn_sample > 0 && s.id == 0 && !s.verifying && r_used > 0 && v <= alpha && !stopped()
            && ply_now > 0 && s.played[ply_now - 1] != Move::none() && ++s.fn_counter % uint64_t(options.fn_sample) == 0) {
            const uint8_t player = yolah.current_player();
            s.verifying = true;
            s.played[ply_now] = m;
            if (options.lazy_accumulator) s.acc_ok[ply_now + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            const int vf = -negamax(yolah, s, zobrist::update(hash, player, m), -(alpha + 1), -alpha, depth - 1);
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            s.verifying = false;
            if (!stopped()) {
                int16_t& h = s.w->fnh[player][s.played[ply_now - 1].to_sq()][m.to_sq()];
                const int b = (vf > alpha ? 1 : -1) * std::min(16 * depth * depth + 32 * depth, 2000);
                h = int16_t(h + b - h * std::abs(b) / HISTORY_MAX);
            }
        }
        // U5 (6): a sample of the reduced fail-lows re-searched at full depth, logged with their features
        if (options.lmr_log > 0 && lmr_log_file && s.id == 0 && !s.verifying && r_used > 0 && v <= alpha && !stopped()
            && ++lmr_log_counter % uint64_t(options.lmr_log) == 0) {
            const uint8_t player = yolah.current_player();
            const int hist = s.w->history[player][m.from_sq()][m.to_sq()];
            s.verifying = true;
            s.played[ply_now] = m;
            if (options.lazy_accumulator) s.acc_ok[ply_now + 1] = false;
            else net_play(player, m, s.acc, &s.refresh);
            yolah.play(m);
            const int vf = -negamax(yolah, s, zobrist::update(hash, player, m), -(alpha + 1), -alpha, depth - 1);
            yolah.undo(m);
            if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
            s.verifying = false;
            if (!stopped()) {
                const int fr = std::popcount(yolah.free_squares());
                *lmr_log_file << lmr_bucket(fr, depth, i, hist, tried_from[m.from_sq()]) << ' ' << fr << ' ' << depth << ' ' << i << ' '
                              << hist << ' ' << int(tried_from[m.from_sq()]) << ' ' << r_used << ' ' << (vf > alpha ? 1 : 0) << '\n';
            }
        }
        tried_from[m.from_sq()]++;
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
                    if (!fly_above) tt_store(hash, yolah, depth, alpha_orig, beta, v, m, tt_pv);
                    corr_update(v);
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
                        int bonus = std::min(16 * depth * depth + 32 * depth, 2000);
                        if (options.bonus_margin > 0) bonus *= 1 + std::clamp((v - beta) / options.bonus_margin, 0, 2);   // U7 (10)
                        const int malus = bonus * options.malus_pct / 100;                                              // U7 (8)
                        const size_t k0 = options.malus_cap > 0 && i > size_t(options.malus_cap) ? i - size_t(options.malus_cap) : 0;
                        update_history(s, player, m, bonus);
                        for (size_t k = k0; k < i; k++)
                            if (options.cont_fix < 2 || searched[k]) update_history(s, player, moves[k], -malus);
                        // U4 (7), U7: the context tables
                        const Move p1 = ply > 0 ? s.played[ply - 1] : Move::none(), p2 = ply > 1 ? s.played[ply - 2] : Move::none();
                        auto upd = [&](int16_t& h, int b0) { const int b = b0 * options.cont_bonus / 100; h = int16_t(h + b - h * std::abs(b) / HISTORY_MAX); };
                        // U10 "malus per key": each table penalises each key once, never the rewarded key
                        std::vector<int> seen[8];
                        int rk[8];
                        auto keys_of = [&](Move mv, int* k) {
                            k[0] = mv.to_sq();
                            k[1] = p1 != Move::none() ? (int(file_of(mv.to_sq())) - int(file_of(p1.to_sq())) + 7) * 15 + int(rank_of(mv.to_sq())) - int(rank_of(p1.to_sq())) + 7 : 0;
                            k[2] = mv.to_sq();
                            k[3] = mv.from_sq();
                            k[4] = (p2 != Move::none() && mv.from_sq() == p2.to_sq()) * 64 + mv.to_sq();
                            int mk = 0, dr = 0, ds = 0;
                            if (options.geo_weight) geo_index(yolah, mv, mk, dr, ds);
                            k[5] = (mk * 8 + dr) * 3 + ds;
                            k[6] = mv.from_sq() * 64 + mv.to_sq();
                            k[7] = 0;
                        };
                        auto ok = [&](int t, int key, bool penal) {
                            if (!penal || !options.malus_per_key) return true;
                            if (key == rk[t]) return false;
                            if (std::find(seen[t].begin(), seen[t].end(), key) != seen[t].end()) return false;
                            seen[t].push_back(key);
                            return true;
                        };
                        auto ctx_update = [&](Move mv, int b) {
                            const bool penal = b < 0;
                            int k[8];
                            keys_of(mv, k);
                            if (p1 != Move::none()) {
                                if (options.cont_hist > 0 && ok(0, k[0], penal)) upd(s.w->cont1[player][p1.to_sq()][mv.to_sq()], b);
                                if (options.rel_weight && ok(1, k[1], penal)) upd(s.w->rel[player][int(file_of(mv.to_sq())) - int(file_of(p1.to_sq())) + 7][int(rank_of(mv.to_sq())) - int(rank_of(p1.to_sq())) + 7], b);
                                if (options.hole_weight && ok(2, k[2], penal)) upd(s.w->holec[player][p1.from_sq()][mv.to_sq()], b);
                                if (options.from_weight && ok(3, k[3], penal)) upd(s.w->cfrom[player][p1.to_sq()][mv.from_sq()], b);
                                if (options.prec_weight && ok(6, k[6], penal)) {
                                    upd(s.w->cprec[player][p1.to_sq()][mv.from_sq()][mv.to_sq()], b);
                                    uint8_t& n = s.w->cprec_n[player][p1.to_sq()][mv.from_sq()][mv.to_sq()];
                                    if (n < 255) n++;
                                }
                            }
                            if (options.cont_hist > 1 && p2 != Move::none() && ok(4, k[4], penal)) {
                                if (options.cont2_same) upd(s.w->cont2s[player][mv.from_sq() == p2.to_sq()][p2.to_sq()][mv.to_sq()], b);
                                else upd(s.w->cont2[player][p2.to_sq()][mv.to_sq()], b);
                            }
                            if (options.geo_weight && ok(5, k[5], penal)) { int mk, dr, ds; geo_index(yolah, mv, mk, dr, ds); upd(s.w->geo[player][mk][dr][ds], b); }
                        };
                        keys_of(m, rk);
                        ctx_update(m, bonus);
                        uint64_t done = 0;
                        for (size_t k = k0; k < i; k++) {
                            if (moves[k] == Move::none()) continue;
                            if (options.cont_fix >= 1) {   // U7 fix: only searched moves, never the cutting destination, once per destination
                                if (!searched[k] || moves[k].to_sq() == m.to_sq() || (done & square_bb(moves[k].to_sq()))) continue;
                                done |= square_bb(moves[k].to_sq());
                            }
                            ctx_update(moves[k], -malus);
                        }
                    }
                    if (options.sibling_ref) {   // U4 (5c): local to the parent
                        auto& sr = s.sib_ref[ply];
                        if (sr[0] != m) { sr[3] = sr[2]; sr[2] = sr[1]; sr[1] = sr[0]; sr[0] = m; }
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
    // U: with moves pruned by the fly reply, that bound is a guess: not stored
    // ("fly reply store" off), and fail-hard (alpha, not a lower guess).
    if (fly_pruned && !options.fly_reply_store) return TRET(std::max(best, alpha_orig));
    if (!fly_above) tt_store(hash, yolah, depth, alpha_orig, beta, best, best_move, tt_pv, node_fragile && options.tt_fragile > 0);
    corr_update(best);
    if (options.hpv_weight > 0 && pv_node && best > alpha_orig && best < beta && best_move != Move::none()
        && int(ply_now) - s.root_ply < 4 && !fly_above) {   // U7 (C6): the PV's best moves near the root
        int16_t& h = s.w->hpv[yolah.current_player()][int(ply_now) - s.root_ply][best_move.from_sq()][best_move.to_sq()];
        const int b = std::min(8 * depth * depth + 16 * depth, 1000);
        h = int16_t(h + b - h * b / HISTORY_MAX);
    }
    if (recorder && options.record_depth > 0 && s.id == 0 && depth >= options.record_depth
        && alpha_orig < best && best < beta && beta - alpha_orig > 1 && !stopped()) {
        const auto [bs, ws] = yolah.score();
        recorder->push_back({yolah.bitboard(Yolah::BLACK), yolah.bitboard(Yolah::WHITE), yolah.empty_bitboard(),
                             yolah.nb_plies(), uint8_t(bs), uint8_t(ws), uint8_t(depth),
                             int16_t(std::clamp(best, -32767, 32767)), best_move.raw()});
    }
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

// ─── U12. Blocking an opponent piece's ray toward a zone ─────────────────────
namespace {
    struct BlockTables {
        int8_t dir_between[64][64];     // 0..7 if aligned on a queen line, -1 otherwise
        uint64_t ray_from[64][8];       // squares strictly after sq in direction d
        uint64_t cone[2][64][8];        // 0: 90-degree cone (|cross| <= dot, dot > 0), 1: half-plane (dot > 0)
        BlockTables() {
            static constexpr int DF[8] = {1, -1, 0, 0, 1, 1, -1, -1};
            static constexpr int DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};
            std::memset(dir_between, -1, sizeof(dir_between));
            std::memset(ray_from, 0, sizeof(ray_from));
            std::memset(cone, 0, sizeof(cone));
            for (int s = 0; s < 64; s++) {
                const int sf = s & 7, sr = s >> 3;
                for (int d = 0; d < 8; d++) {
                    const int df = DF[d], dr = DR[d];
                    for (int f = sf + df, r = sr + dr; f >= 0 && f < 8 && r >= 0 && r < 8; f += df, r += dr) {
                        const int t = (r << 3) | f;
                        dir_between[s][t] = int8_t(d);
                        ray_from[s][d] |= uint64_t(1) << t;
                    }
                    for (int t = 0; t < 64; t++) {
                        const int tf = (t & 7) - sf, tr = (t >> 3) - sr;
                        const int dot = tf * df + tr * dr;
                        const int cross = std::abs(tf * dr - tr * df);
                        if (dot > 0) {
                            cone[1][s][d] |= uint64_t(1) << t;
                            if (cross <= dot) cone[0][s][d] |= uint64_t(1) << t;
                        }
                    }
                }
            }
        }
    };
    const BlockTables block_tables;
    inline uint64_t expand_king_bb(uint64_t b) {
        const uint64_t h = b | ((b << 1) & 0xFEFEFEFEFEFEFEFEULL) | ((b >> 1) & 0x7F7F7F7F7F7F7F7FULL);
        return h | (h << 8) | (h >> 8);
    }
    template <Direction D> inline uint64_t queen_slide_bb(uint64_t gen, uint64_t free) {
        uint64_t pro = free;
        gen |= pro & shift<D>(gen);
        pro &= shift<D>(pro);
        gen |= pro & shift<D>(shift<D>(gen));
        pro &= shift<D>(shift<D>(pro));
        gen |= pro & shift<D>(shift<D>(shift<D>(shift<D>(gen))));
        return shift<D>(gen) & free;
    }
    inline uint64_t queen_reach_bb(uint64_t from, uint64_t free) {
        return queen_slide_bb<NORTH>(from, free) | queen_slide_bb<SOUTH>(from, free) | queen_slide_bb<EAST>(from, free) |
               queen_slide_bb<WEST>(from, free) | queen_slide_bb<NORTH_EAST>(from, free) | queen_slide_bb<NORTH_WEST>(from, free) |
               queen_slide_bb<SOUTH_EAST>(from, free) | queen_slide_bb<SOUTH_WEST>(from, free);
    }
    inline int influence_fast_bb(uint64_t me, uint64_t opp, uint64_t free) {
        uint64_t mi = me, oi = opp, mf = me, of = opp, neutral = 0;
        while (mf | of) {
            uint64_t nm = expand_king_bb(mf) & free & ~oi & ~mi;
            uint64_t no = expand_king_bb(of) & free & ~mi & ~oi;
            const uint64_t collision = nm & no;
            if (neutral | collision) {
                neutral |= collision | (expand_king_bb(neutral) & free);
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
    inline int apply_block_inf(int bz, int delta_inf, int mode) {
        if (mode <= 0 || delta_inf >= 0) return bz;
        if (mode == 1) return 0;
        if (mode == 2) return std::max(0, bz + delta_inf);
        if (mode == 3) return bz + delta_inf;
        if (mode == 4) return std::max(0, bz + 2 * delta_inf);
        return bz + 2 * delta_inf;
    }
}

int MinMaxNNUE_DevPlayer::block_zone_score(const Yolah& yolah, Move m) const {
    if (m == Move::none()) return 0;
    const Square from = m.from_sq(), to = m.to_sq();
    const uint64_t to_bb = square_bb(to);
    const uint64_t free = yolah.free_squares(), occupied = yolah.occupied_squares();
    const uint64_t free_after = free & ~to_bb;
    const int cone_idx = options.block_cone == 2 ? 1 : 0;
    auto side_block = [&](uint64_t pieces_bb, Square skip_sq) {
        Square sqs[4];
        uint64_t atks[4]{}, rem_atk[4]{};
        int np = 0;
        while (pieces_bb && np < 4) {
            const Square w = pop_lsb(pieces_bb);
            if (w == skip_sq) continue;
            sqs[np] = w;
            atks[np] = attacks_bb(w, occupied) & free;
            const int d = block_tables.dir_between[w][to];
            rem_atk[np] = (d >= 0 && (atks[np] & to_bb)) ? (atks[np] & ~(block_tables.ray_from[to][d] | to_bb)) : atks[np];
            np++;
        }
        int total_zone = 0, nb_blocked = 0;
        uint64_t union_zone = 0;
        for (int p = 0; p < np; p++) {
            if (!(atks[p] & to_bb)) continue;
            const int dir = block_tables.dir_between[sqs[p]][to];
            if (dir < 0) continue;
            uint64_t team_other = 0;
            if (options.block_team > 0) {
                for (int q = 0; q < np; q++) if (q != p) team_other |= rem_atk[q];
            }
            const uint64_t ray_beyond = block_tables.ray_from[to][dir];
            const uint64_t shadow_ray = atks[p] & ray_beyond;
            const uint64_t atk_after  = rem_atk[p];
            const uint64_t start_other = options.block_team == 2 ? (atk_after | team_other) : atk_after;
            const uint64_t mask = free_after & block_tables.cone[cone_idx][to][dir];
            uint64_t zone = 0;
            if (options.block_steps < 0) {
                const uint64_t via_ray   = (shadow_ray | queen_reach_bb(shadow_ray | to_bb, free_after)) & mask;
                const uint64_t via_other = start_other | queen_reach_bb(start_other, free_after);
                zone = via_ray & ~via_other;
            } else {
                zone = shadow_ray;
                uint64_t front = shadow_ray | to_bb;
                uint64_t other = start_other, ofront = start_other;
                for (int step = 0; step < options.block_steps; step++) {
                    front  = expand_king_bb(front)  & mask       & ~zone;
                    ofront = expand_king_bb(ofront) & free_after & ~other;
                    zone  |= front;
                    other |= ofront;
                    if (!(front | ofront)) break;
                }
                zone &= ~other;
            }
            if (options.block_team == 1) zone &= ~team_other;
            const int sz = options.block_team == 3
                         ? (2 * std::popcount(zone & ~team_other) + std::popcount(zone & team_other)) / 2
                         : std::popcount(zone);
            if (sz > 0) {
                total_zone += sz;
                union_zone |= zone;
                nb_blocked++;
            }
        }
        if (nb_blocked == 0) return 0;
        const int base = options.block_union ? std::popcount(union_zone) : total_zone;
        return base * (100 + options.block_multi * (nb_blocked - 1)) / 100 + options.block_per_piece * (nb_blocked - 1);
    };
    const uint8_t us = yolah.current_player(), them = Yolah::other_player(us);
    const int opp_bz = side_block(yolah.bitboard(them), SQ_NONE);
    int res = opp_bz;
    if (options.block_self != 0) {
        if (options.block_self > 0 && opp_bz == 0) return 0;
        const int self_bz = side_block(yolah.bitboard(us), from);
        const int net = opp_bz - self_bz * std::abs(options.block_self) / 100;
        res = options.block_self > 0 ? std::max(0, net) : net;
    }
    if (options.block_inf > 0 && res > 0) {
        const uint64_t me = yolah.bitboard(us), opp = yolah.bitboard(them);
        const int inf_before = influence_fast_bb(me, opp, free);
        const int inf_after  = influence_fast_bb((me & ~square_bb(from)) | to_bb, opp, free_after);
        res = apply_block_inf(res, inf_after - inf_before, options.block_inf);
    }
    return res;
}

// ─── Transposition table access (H) ──────────────────────────────────────────
// Both tables seen the same way: a value interval [lower, upper] proven at
// some depth. The reference's table keeps one value and a bound type:
// exact → [v, v], lower bound → [v, +∞], upper bound → [−∞, v].
MinMaxNNUE_DevPlayer::TTView MinMaxNNUE_DevPlayer::tt_probe(uint64_t hash) const {
    TTView view;
    if (yolah_table) {
        if (const SearchTable::Entry* e = yolah_table->probe(hash)) {
            view = {true, e->move, e->depth, e->lower, e->upper, e->pv != 0, (e->padding & 1) != 0};
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
void MinMaxNNUE_DevPlayer::tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta, int value, Move move,
                                    bool pv, bool fragile) {
    if (yolah_table) {
#ifdef YOLAH_SEARCH_TRACE
        if (tracer) {
            const SearchTable::Entry* e = yolah_table->slot(hash);
            auto ykey = [](uint64_t key32, uint64_t key16) { return key32 << 16 | key16; };
            const SearchTable::Entry old = *e;
            uint64_t old_hash = 0;
            if (old.key32 != 0 && yolah_table->probe(hash) == e) old_hash = hash;   // this position's entry
            else if (old.key32 != 0) {
                auto it = tracer->hash_of_ykey.find(ykey(old.key32, old.key16));
                old_hash = it != tracer->hash_of_ykey.end() ? it->second : ykey(old.key32, old.key16);
            }
            yolah_table->store(hash, std::popcount(yolah.free_squares()), depth, alpha, beta, value, move, pv, fragile);
            tracer->hash_of_ykey[ykey(e->key32, e->key16)] = hash;
            TR(YT_STORE, depth, alpha, beta, value, move.raw(), int32_t(uint32_t(old_hash)), int32_t(uint32_t(old_hash >> 32)),
               old.key32 ? int(old.depth) : -1, int(old.lower), int(old.upper), old.move.raw(),
               int(e->depth), int(e->lower), int(e->upper), e->move.raw());
            return;
        }
#endif
        yolah_table->store(hash, std::popcount(yolah.free_squares()), depth, alpha, beta, value, move, pv, fragile);
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
        update_accumulator(s.accs[k].acc, s.accs[k + 1].acc, uint8_t(k & 1), s.played[k], &s.refresh);
        s.acc_ok[k + 1] = true;
    }
    return int(net_value(s.accs[ply], yolah) * WIN);
}

// The network calls, for either network (see `persp` in the header).
void MinMaxNNUE_DevPlayer::net_init(const Yolah& yolah, Accumulator& a) {
    if (persp) persp->init(yolah, a.acc);
    else nnue.init(yolah, a.acc);
}

void MinMaxNNUE_DevPlayer::net_play(uint8_t player, Move m, Accumulator& a, NNUE_Persp::RefreshCache* cache) {
    if (persp) persp->play(player, m, a.acc, cache);
    else nnue.play(player, m, a.acc);
}

void MinMaxNNUE_DevPlayer::net_undo(uint8_t player, Move m, Accumulator& a, NNUE_Persp::RefreshCache* cache) {
    if (persp) persp->undo(player, m, a.acc, cache);
    else nnue.undo(player, m, a.acc);
}

float MinMaxNNUE_DevPlayer::net_value(const Accumulator& a, const Yolah& yolah) {
    if (persp) return persp->value(a.acc, yolah);
    return nnue.value(a.acc, yolah.current_player());
}

// out = in + the change of features of move m by `player` (see NNUE_Quantized::play,
// whose arithmetic it reproduces, in one pass and without modifying `in`).
void MinMaxNNUE_DevPlayer::update_accumulator(const int16_t* __restrict in, int16_t* __restrict out,
                                              uint8_t player, Move m, NNUE_Persp::RefreshCache* cache) const {
    if (persp) {                              // two views, three rows each (see NNUE_Persp::update)
        persp->update(in, out, player, m, cache);
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
    std::fill(std::begin(s.ext_sum), std::end(s.ext_sum), 0);   // U4: extensions counted from the root
    s.root_ply = yolah.nb_plies();
    s.cutnode[yolah.nb_plies()] = false;
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
    // "root close full": the moves closest to the best in the last search (their
    // values are read before this search overwrites them)
    bool close[Yolah::MAX_NB_MOVES]{};
    if (options.root_close_full > 0 && s.root_moves.size() > 1) {
        std::vector<size_t> idx;
        for (size_t i = 1; i < s.root_moves.size(); i++) idx.push_back(i);
        const size_t k = std::min(idx.size(), size_t(options.root_close_full));
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](size_t a, size_t b) { return s.root_moves[a].value > s.root_moves[b].value; });
        for (size_t j = 0; j < k; j++) close[idx[j]] = true;
    }
    for (size_t i = 0; i < s.root_moves.size(); i++) {
        RootMove& rm = s.root_moves[i];
        const uint64_t nodes_before = s.nb_nodes;
        // The root is a PV node; no killers there, the history still speaks.
        const int r = close[i] ? 0
                    : options.articulation_lmr && is_articulation_move(yolah, rm.move) ? 0
                    : late_move_reduction_of(depth, i, true, false,
                                             s.w->history[yolah.current_player()][rm.move.from_sq()][rm.move.to_sq()]
                                             + (options.root_cont_lmr > 0 && yolah.nb_plies() > 0 && s.played[yolah.nb_plies() - 1] != Move::none()
                                                ? s.w->cont1[yolah.current_player()][s.played[yolah.nb_plies() - 1].to_sq()][rm.move.to_sq()] * options.root_cont_lmr / 100 : 0));
        TR(MOVE, int(i), rm.move.raw(), s.w->history[yolah.current_player()][rm.move.from_sq()][rm.move.to_sq()], r, 0);
        const int v = search_move(yolah, s, hash, rm.move, i, r, alpha, beta, depth);
        TR(MOVE_DONE, int(i), rm.move.raw(), v, best, alpha);
        rm.value = v; rm.alpha = alpha; rm.reduction = uint8_t(r); rm.stage = s.last_stage; rm.rank = uint8_t(i);
        if (options.root_fix && !options.root_nodes_last) rm.nodes += s.nb_nodes - nodes_before;   // summed over the attempts of this depth
        else rm.nodes = s.nb_nodes - nodes_before;
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
    if (options.root_ordering && options.root_fix && res == Move::none()) {
        // "root fix", fail low: no move beat alpha, nothing learnt about the
        // order — the previous best stays first (sorting by nodes alone used to
        // put a quickly refuted secondary move there; FIX (A) relies on it).
    } else if (options.root_ordering && options.root_fix && best >= beta) {
        // fail high: the moves after the cut were not searched at this depth;
        // only the new best comes to the front.
        auto it = std::find_if(s.root_moves.begin(), s.root_moves.end(), [&](const RootMove& rm) { return rm.move == res; });
        std::rotate(s.root_moves.begin(), it, it + 1);
    } else if (options.root_ordering && options.root_order_gap) {
        // "root order gap": the sum of the two ranks, ties by nodes
        const size_t n = s.root_moves.size();
        std::vector<size_t> by_nodes(n), by_value(n), score(n, 0);
        for (size_t i = 0; i < n; i++) by_nodes[i] = by_value[i] = i;
        std::stable_sort(by_nodes.begin(), by_nodes.end(), [&](size_t a, size_t b) { return s.root_moves[a].nodes > s.root_moves[b].nodes; });
        std::stable_sort(by_value.begin(), by_value.end(), [&](size_t a, size_t b) { return s.root_moves[a].value > s.root_moves[b].value; });
        for (size_t r = 0; r < n; r++) { score[by_nodes[r]] += r; score[by_value[r]] += r; }
        std::vector<std::pair<size_t, RootMove>> keyed;
        for (size_t i = 0; i < n; i++) keyed.push_back({s.root_moves[i].move == res ? 0 : 1 + score[i], s.root_moves[i]});
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first < b.first : a.second.nodes > b.second.nodes;
        });
        for (size_t i = 0; i < n; i++) s.root_moves[i] = keyed[i].second;
    } else if (options.root_ordering) {
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
    tt_store(hash, yolah, depth, alpha_orig, beta, best, res, true);
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
    const bool use_block = options.block_weight > 0 && depth >= options.block_depth;
    const uint16_t ply = yolah.nb_plies();
    const uint8_t player = yolah.current_player();
    // U12: precompute the pieces' attacks once for the node when block ordering is on
    Square opp_sq[4], my_sq[4];
    uint64_t opp_atk[4]{}, opp_atk_all = 0, my_atk[4]{}, my_atk_all = 0;
    int n_opp = 0, n_my = 0;
    const uint64_t node_free = yolah.free_squares();
    if (use_block) {
        const uint64_t occupied = yolah.occupied_squares();
        for (uint64_t b = yolah.bitboard(Yolah::other_player(player)); b && n_opp < 4;) {
            const Square w = pop_lsb(b);
            opp_sq[n_opp] = w;
            opp_atk[n_opp] = attacks_bb(w, occupied) & node_free;
            opp_atk_all |= opp_atk[n_opp++];
        }
        if (options.block_self != 0) {
            for (uint64_t b = yolah.bitboard(player); b && n_my < 4;) {
                const Square u = pop_lsb(b);
                my_sq[n_my] = u;
                my_atk[n_my] = attacks_bb(u, occupied) & node_free;
                my_atk_all |= my_atk[n_my++];
            }
        }
    }
    auto side_fast = [&](const Square* sqs, const uint64_t* atks, int np, uint64_t all_atk, Square to, uint64_t to_bb, Square skip_sq) -> int {
        if (!(all_atk & to_bb)) return 0;
        const uint64_t free_after = node_free & ~to_bb;
        const int cone_idx = options.block_cone == 2 ? 1 : 0;
        uint64_t rem_atk[4]{};
        if (options.block_team > 0) {
            for (int q = 0; q < np; q++) {
                if (sqs[q] == skip_sq) continue;
                const int dq = block_tables.dir_between[sqs[q]][to];
                rem_atk[q] = (dq >= 0 && (atks[q] & to_bb)) ? (atks[q] & ~(block_tables.ray_from[to][dq] | to_bb)) : atks[q];
            }
        }
        int total_zone = 0, nb_blocked = 0;
        uint64_t union_zone = 0;
        for (int p = 0; p < np; p++) {
            if (sqs[p] == skip_sq || !(atks[p] & to_bb)) continue;
            const int dir = block_tables.dir_between[sqs[p]][to];
            if (dir < 0) continue;
            const uint64_t ray_beyond = block_tables.ray_from[to][dir];
            const uint64_t shadow_ray = atks[p] & ray_beyond;
            const uint64_t atk_after  = atks[p] & ~(ray_beyond | to_bb);
            uint64_t team_other = 0;
            if (options.block_team > 0) {
                for (int q = 0; q < np; q++) if (q != p && sqs[q] != skip_sq) team_other |= rem_atk[q];
            }
            const uint64_t start_other = options.block_team == 2 ? (atk_after | team_other) : atk_after;
            const uint64_t mask = free_after & block_tables.cone[cone_idx][to][dir];
            uint64_t zone = 0;
            if (options.block_steps < 0) {
                const uint64_t via_ray   = (shadow_ray | queen_reach_bb(shadow_ray | to_bb, free_after)) & mask;
                const uint64_t via_other = start_other | queen_reach_bb(start_other, free_after);
                zone = via_ray & ~via_other;
            } else {
                zone = shadow_ray;
                uint64_t front = shadow_ray | to_bb;
                uint64_t other = start_other, ofront = start_other;
                for (int step = 0; step < options.block_steps; step++) {
                    front  = expand_king_bb(front)  & mask       & ~zone;
                    ofront = expand_king_bb(ofront) & free_after & ~other;
                    zone  |= front;
                    other |= ofront;
                    if (!(front | ofront)) break;
                }
                zone &= ~other;
            }
            if (options.block_team == 1) zone &= ~team_other;
            const int sz = options.block_team == 3
                         ? (2 * std::popcount(zone & ~team_other) + std::popcount(zone & team_other)) / 2
                         : std::popcount(zone);
            if (sz > 0) {
                total_zone += sz;
                union_zone |= zone;
                nb_blocked++;
            }
        }
        if (nb_blocked == 0) return 0;
        const int base = options.block_union ? std::popcount(union_zone) : total_zone;
        return base * (100 + options.block_multi * (nb_blocked - 1)) / 100 + options.block_per_piece * (nb_blocked - 1);
    };
    int inf_before = EVAL_NONE;
    auto block_fast = [&](Move m, int known_inf_after = EVAL_NONE) -> int {
        if (m == Move::none()) return 0;
        const Square from = m.from_sq(), to = m.to_sq();
        const uint64_t to_bb = square_bb(to);
        const int opp_bz = side_fast(opp_sq, opp_atk, n_opp, opp_atk_all, to, to_bb, SQ_NONE);
        int res = opp_bz;
        if (options.block_self != 0) {
            if (options.block_self > 0 && opp_bz == 0) return 0;
            const int self_bz = side_fast(my_sq, my_atk, n_my, my_atk_all, to, to_bb, from);
            const int net = opp_bz - self_bz * std::abs(options.block_self) / 100;
            res = options.block_self > 0 ? std::max(0, net) : net;
        }
        if (options.block_inf > 0 && res > 0) {
            const uint64_t me = yolah.bitboard(player), opp = yolah.bitboard(Yolah::other_player(player));
            if (inf_before == EVAL_NONE) inf_before = influence_fast_bb(me, opp, node_free);
            const int inf_after = known_inf_after != EVAL_NONE ? known_inf_after
                                : influence_fast_bb((me & ~square_bb(from)) | to_bb, opp, node_free & ~to_bb);
            res = apply_block_inf(res, inf_after - inf_before, options.block_inf);
        }
        return res;
    };
    Move counter = Move::none();
    if (options.countermove && ply > 0) {
        const Move prev = s.played[ply - 1];
        counter = s.w->countermoves[prev.from_sq()][prev.to_sq()];
    }
    // T. Mode 2: the prior of the node's phase added to the history.
    const bool add_prior = !prior.empty() && options.history_prior_mode == 2;
    const bool tier_prior = !prior.empty() && options.history_prior_mode == 3;
    const int ph = add_prior || tier_prior ? prior_phase(yolah) : 0;
    const Move p1 = ply > 0 ? s.played[ply - 1] : Move::none(), p2 = ply > 1 ? s.played[ply - 2] : Move::none();
    auto hist = [&](Move m) {
        int h = s.w->history[player][m.from_sq()][m.to_sq()] + (add_prior ? prior_of(ph, player, m) : 0);
        if (options.cont_hist > 0 && p1 != Move::none() && m != Move::none()) h += s.w->cont1[player][p1.to_sq()][m.to_sq()] * options.cont_weight / 100;   // U4 (7)
        if (options.cont_hist > 1 && p2 != Move::none() && m != Move::none() && !options.cont2_same) h += s.w->cont2[player][p2.to_sq()][m.to_sq()] * options.cont2_weight / 100;
        if (m == Move::none()) return h;
        // U7: more context terms
        if (options.cont2_same && options.cont_hist > 1 && p2 != Move::none())
            h += s.w->cont2s[player][m.from_sq() == p2.to_sq()][p2.to_sq()][m.to_sq()] * options.cont2_weight / 100;
        if (p1 != Move::none()) {
            if (options.rel_weight)
                h += s.w->rel[player][int(file_of(m.to_sq())) - int(file_of(p1.to_sq())) + 7][int(rank_of(m.to_sq())) - int(rank_of(p1.to_sq())) + 7] * options.rel_weight / 100;
            if (options.hole_weight) h += s.w->holec[player][p1.from_sq()][m.to_sq()] * options.hole_weight / 100;
            if (options.from_weight) h += s.w->cfrom[player][p1.to_sq()][m.from_sq()] * options.from_weight / 100;
            if (options.prec_weight) {   // U10
                const int n = s.w->cprec_n[player][p1.to_sq()][m.from_sq()][m.to_sq()];
                h += s.w->cprec[player][p1.to_sq()][m.from_sq()][m.to_sq()] * options.prec_weight / 100 * (options.prec_conf ? std::min(n, 8) : 8) / 8;
            }
            if (options.intercept && ((attacks_bb(p1.to_sq(), yolah.occupied_squares()) | around(square_bb(p1.to_sq()))) & square_bb(m.to_sq())))
                h += options.intercept;
        }
        if (options.geo_weight) { int mk, dr, ds; geo_index(yolah, m, mk, dr, ds); h += s.w->geo[player][mk][dr][ds] * options.geo_weight / 100; }
        if (options.hpv_weight && int(ply) - s.root_ply < 4)
            h += s.w->hpv[player][int(ply) - s.root_ply][m.from_sq()][m.to_sq()] * options.hpv_weight / 100;
        return h;
    };
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        const bool kil = m == s.killer1[ply] || m == s.killer2[ply];
        const bool kil_first = kil && (options.killer_mode == 0
                                       || (options.killer_mode == 2 && p1 != Move::none() && s.w->cont1[player][p1.to_sq()][m.to_sq()] > 0));
        if (m == tt_move)                 scores[i] = SCORE_TT;
        else if (kil_first && m == s.killer1[ply]) scores[i] = SCORE_KILLER1;
        else if (kil_first)               scores[i] = SCORE_KILLER2;
        else if (kil && options.killer_mode == 1) scores[i] = std::min(hist(m) + options.killer_bonus, SCORE_ARTICULATION - 1);
        else if (options.killer2ply && ply >= 2 && m != Move::none() && (m == s.killer1[ply - 2] || m == s.killer2[ply - 2]))
            scores[i] = SCORE_COUNTER - 5;
        else if (m == counter && counter != Move::none()) scores[i] = SCORE_COUNTER;
        else if (options.sibling_ref && m != Move::none() && std::find(std::begin(s.sib_ref[ply]), std::end(s.sib_ref[ply]), m) != std::end(s.sib_ref[ply]))
            scores[i] = SCORE_COUNTER - 1 - int(std::find(std::begin(s.sib_ref[ply]), std::end(s.sib_ref[ply]), m) - std::begin(s.sib_ref[ply]));   // U4 (5c)
        else if (options.articulation_ordering && is_articulation_move(yolah, m))
            scores[i] = SCORE_ARTICULATION + (options.history ? hist(m) : 0);
        else if (use_block && options.block_mode == 1) {
            const int bz = block_fast(m);
            if (bz >= options.block_min) scores[i] = SCORE_ARTICULATION + options.block_weight * bz + (options.history ? hist(m) : 0);
            else                         scores[i] = options.history ? hist(m) : -int(i);
        }
        else if (tier_prior && prior_of(ph, player, m) >= options.history_prior_threshold)
            scores[i] = SCORE_ARTICULATION + prior_of(ph, player, m);   // T, mode 3: above the history
        else if (options.history)         scores[i] = hist(m);
        else                              scores[i] = -int(i);
        if (scores[i] < SCORE_ARTICULATION) {
            const int terr = territory ? territory_after(yolah, m) : EVAL_NONE;
            if (use_block && options.block_mode == 0) {
                scores[i] += options.block_weight * block_fast(m, options.territory_ordering == 1 ? terr : EVAL_NONE);
            }
            if (territory) {
                scores[i] += options.territory_weight * terr;
            }
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

std::vector<Move> MinMaxNNUE_DevPlayer::principal_variation(Yolah yolah, int max_len) const {
    std::vector<Move> pv;
    uint64_t hash = zobrist::hash(yolah);
    while (int(pv.size()) < max_len && !yolah.game_over()) {
        const Move m = tt_move(hash);
        if (m == Move::none() || !is_legal(yolah, m)) break;
        pv.push_back(m);
        hash = zobrist::update(hash, yolah.current_player(), m);
        yolah.play(m);
    }
    return pv;
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
    int value_prev = 0;          // the value of the depth before (aspiration center 2)
    bool have_prev = false;
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
        if (adaptive && options.aspiration_last > 0 && have_prev && std::abs(value) < WIN && std::abs(value_prev) < WIN)
            delta = std::max(delta, int(options.aspiration_last * std::abs(value - value_prev)));
        int alpha = -INFINITE, beta = INFINITE;
        if (options.root_fix) for (RootMove& rm : s.root_moves) rm.nodes = 0;
        for (size_t i = 0; i < s.root_moves.size(); i++) s.root_moves[i].rank_start = uint8_t(i);
        const double sigma = s.w->asp_n > 0 ? s.w->asp_sum / s.w->asp_n : 0;
        // ("aspiration center 2": between the last two depths, against the odd/even swing)
        const int center = options.aspiration_center2 && have_prev && std::abs(value_prev) < WIN ? (value + value_prev) / 2 : value;
        if (delta > 0 && d >= 4 && std::abs(value) < WIN) {
            alpha = std::max(center - delta, -INFINITE);
            beta  = std::min(center + delta, INFINITE);
        }
        [[maybe_unused]] int attempt = 0;
        int nb_attempts = 0;
        int last_outcome = 0;
        for (;;) {
            Move m;
            nb_attempts++;
            TR(ITER, d, alpha, beta, attempt++, delta);
            const uint64_t nodes_before = s.nb_nodes;
            s.asp_attempts++;
            const int v = root_search(yolah, s, hash, alpha, beta, d, m);
            if (!stopped() && ((v <= alpha && alpha > -INFINITE) || (v >= beta && beta < INFINITE))) {
                s.asp_failed++;
                s.asp_failed_nodes += s.nb_nodes - nodes_before;
                const int outcome = v <= alpha ? 1 : 2;
                if (outcome == 2 && last_outcome == 1) s.asp_rebounds++;
                last_outcome = outcome;
            }
            if (stopped()) {
                TR(ITER_RESULT, d, v, m.raw(), 3);
                if (m != Move::none()) res = m;   // interrupted, but this move was fully searched
                goto done;
            }
            if (v <= alpha && alpha > -INFINITE) {          // fail low: widen downwards
                TR(ITER_RESULT, d, v, m.raw(), 1);
                if (adaptive) {
                    if (!options.aspiration_keep_beta) beta = (alpha + beta) / 2;
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
            if (root_log && s.id == 0) {
                for (const RootMove& rm : s.root_moves) {
                    root_log->push_back({root_log_tag, uint8_t(d), uint8_t(nb_attempts), rm.rank, rm.rank_start, rm.reduction,
                                         rm.stage, rm.move.raw(), rm.nodes, rm.value, rm.alpha, beta, v, sigma});
                }
            }
            if (d >= options.aspiration_stats_depth && depth > 0 && std::abs(v) < WIN && std::abs(value) < WIN) {
                s.w->asp_sum = 0.9 * s.w->asp_sum + std::abs(v - value);
                s.w->asp_n   = 0.9 * s.w->asp_n + 1;
            }
            have_prev = depth > 0;
            value_prev = value;
            depth = uint8_t(d);
            value = v;
            break;
        }
        // U5 (8): root audit — the strongest challenger searched with another
        // policy (no LMP, LMR one ply less, no table cut near the root), against
        // the best move under the same policy; switched if it is better.
        if (options.root_audit > 0 && s.id == 0 && d >= options.root_audit_depth && s.root_moves.size() >= 2 && !stopped()
            && std::abs(value) < WIN && (search_us == 0 || std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - search_start).count()
                                                          < search_us * (100 - options.root_audit_frac) / 100.0)) {
            const Move best = res;
            const Move ch = s.root_moves[0].move == best ? s.root_moves[1].move : s.root_moves[0].move;
            auto audited = [&](Move m, int a, int b) {
                const uint8_t player = yolah.current_player();
                s.audit = true;
                s.played[yolah.nb_plies()] = m;
                if (options.lazy_accumulator) s.acc_ok[yolah.nb_plies() + 1] = false;
                else net_play(player, m, s.acc, &s.refresh);
                yolah.play(m);
                const int r = -negamax(yolah, s, zobrist::update(hash, player, m), -b, -a, d - 1);
                yolah.undo(m);
                if (!options.lazy_accumulator) net_undo(player, m, s.acc, &s.refresh);
                s.audit = false;
                return r;
            };
            audits[0]++;
            const int vc = audited(ch, value, value + 1);
            if (!stopped() && vc > value) {
                const int vb = audited(best, vc - 1, vc);
                if (!stopped() && vb < vc) {
                    audits[1]++;
                    res = ch;
                    for (size_t k = 0; k < s.root_moves.size(); k++)
                        if (s.root_moves[k].move == ch) { std::rotate(s.root_moves.begin(), s.root_moves.begin() + k, s.root_moves.begin() + k + 1); break; }
                }
            }
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
    j["root fix"] = options.root_fix;
    j["root nodes last"] = options.root_nodes_last;
    j["root order gap"] = options.root_order_gap;
    j["root close full"] = options.root_close_full;
    j["aspiration keep beta"] = options.aspiration_keep_beta;
    j["aspiration stats depth"] = options.aspiration_stats_depth;
    j["aspiration center 2"] = options.aspiration_center2;
    j["aspiration last"] = options.aspiration_last;
    j["pvs research high"] = options.pvs_research_high;
    j["tt cut pv"] = options.tt_cut_pv;
    j["tt cut pv exact"] = options.tt_cut_pv_exact;
    j["tt pv"] = options.tt_pv;
    j["tt depth asym"] = options.tt_depth_asym;
    j["table tiebreak"] = options.table_tiebreak;
    j["proxy after tt"] = options.proxy_after_tt;
    j["lmr territory"] = options.lmr_territory;
    j["history"] = options.history;
    j["countermove"] = options.countermove;
    j["root ordering"] = options.root_ordering;
    j["lazy accumulator"] = options.lazy_accumulator;
    j["eval cache"] = options.eval_cache_bits;
    j["max nodes"] = options.max_nodes;
    j["record depth"] = options.record_depth;
    j["lmr"] = options.lmr;
    j["lmr base"] = options.lmr_base;
    j["lmr divisor"] = options.lmr_divisor;
    j["rfp depth"] = options.rfp_depth;
    j["rfp margin"] = options.rfp_margin;
    j["null move"] = options.null_move;
    j["null move reduction"] = options.null_move_reduction;
    j["fly reply"] = options.fly_reply;
    j["fly reply jumps"] = options.fly_reply_jumps;
    j["fly reply depth"] = options.fly_reply_depth;
    j["fly reply reduction"] = options.fly_reply_reduction;
    j["fly reply div"] = options.fly_reply_div;
    j["fly reply from"] = options.fly_reply_from;
    j["fly reply order"] = options.fly_reply_order;
    j["fly reply margin"] = options.fly_reply_margin;
    j["fly reply store"] = options.fly_reply_store;
    j["fly reply filter"] = options.fly_reply_filter;
    j["fly reply filter margin"] = options.fly_reply_filter_margin;
    j["fly prune"] = options.fly_prune;
    j["fly prune jumps"] = options.fly_prune_jumps;
    j["fly prune depth"] = options.fly_prune_depth;
    j["fly prune reduction"] = options.fly_prune_reduction;
    j["fly prune div"] = options.fly_prune_div;
    j["fly prune from"] = options.fly_prune_from;
    j["proxy fly"] = options.proxy_fly;
    j["proxy fly jumps"] = options.proxy_fly_jumps;
    j["fly lmr"] = options.fly_lmr;
    j["fly lmr depth"] = options.fly_lmr_depth;
    j["fly lmr closed"] = options.fly_lmr_closed;
    j["fly lmr open"] = options.fly_lmr_open;
    j["fly lmr jumps"] = options.fly_lmr_jumps;
    j["region bounds"] = options.region_bounds;
    j["region margin"] = options.region_margin;
    j["region max free"] = options.region_max_free;
    j["fly order"] = options.fly_order;
    j["fly order depth"] = options.fly_order_depth;
    j["fly order jumps"] = options.fly_order_jumps;
    j["refute"] = options.refute;
    j["refute replies"] = options.refute_replies;
    j["refute depth"] = options.refute_depth;
    j["refute from"] = options.refute_from;
    j["refute reduction"] = options.refute_reduction;
    j["proxy restrict"] = options.proxy_restrict;
    j["proxy restrict deeper"] = options.proxy_restrict_deeper;
    j["pass razor"] = options.pass_razor;
    j["pass razor depth"] = options.pass_razor_depth;
    j["pass razor margin"] = options.pass_razor_margin;
    j["pass razor reduction"] = options.pass_razor_reduction;
    j["pass razor guard"] = options.pass_razor_guard;
    j["nohole razor"] = options.nohole_razor;
    j["nohole razor depth"] = options.nohole_razor_depth;
    j["nohole razor guard"] = options.nohole_razor_guard;
    j["nohole razor reduction"] = options.nohole_razor_reduction;
    j["proxy nohole"] = options.proxy_nohole;
    j["verify tests"] = options.verify_tests;
    j["iir"] = options.iir;
    j["iir depth"] = options.iir_depth;
    j["cutnode lmr"] = options.cutnode_lmr;
    j["improving"] = options.improving;
    j["corr hist"] = options.corr_hist;
    j["corr scale"] = options.corr_scale;
    j["se"] = options.se;
    j["se depth"] = options.se_depth;
    j["se margin"] = options.se_margin;
    j["se double"] = options.se_double;
    j["multicut"] = options.multicut;
    j["ext max"] = options.ext_max;
    j["cont hist"] = options.cont_hist;
    j["cont weight"] = options.cont_weight;
    j["cont2 weight"] = options.cont2_weight;
    j["cont bonus"] = options.cont_bonus;
    j["cont lmr"] = options.cont_lmr;
    j["cont age"] = options.cont_age;
    j["cont fix"] = options.cont_fix;
    j["malus pct"] = options.malus_pct;
    j["malus cap"] = options.malus_cap;
    j["bonus margin"] = options.bonus_margin;
    j["rel weight"] = options.rel_weight;
    j["hole weight"] = options.hole_weight;
    j["from weight"] = options.from_weight;
    j["cont2 same"] = options.cont2_same;
    j["geo weight"] = options.geo_weight;
    j["hpv weight"] = options.hpv_weight;
    j["intercept"] = options.intercept;
    j["intercept lmr"] = options.intercept_lmr;
    j["killer2ply"] = options.killer2ply;
    j["killer mode"] = options.killer_mode;
    j["killer bonus"] = options.killer_bonus;
    j["hist prune"] = options.hist_prune;
    j["lmp protect"] = options.lmp_protect;
    j["fp margin"] = options.fp_margin;
    j["razor margin"] = options.razor_margin;
    j["lmrh"] = options.lmrh;
    j["corr cont"] = options.corr_cont;
    j["root context"] = options.root_context;
    j["killer special"] = options.killer_special;
    j["lmr hist full"] = options.lmr_hist_full;
    j["reserve max free"] = options.reserve_max_free;
    j["malus per key"] = options.malus_per_key;
    j["root cont lmr"] = options.root_cont_lmr;
    j["prec weight"] = options.prec_weight;
    j["prec conf"] = options.prec_conf;
    j["fn sample"] = options.fn_sample;
    j["fn thr"] = options.fn_thr;
    j["tt bonus"] = options.tt_bonus;
    j["proxy bonus"] = options.proxy_bonus;
    j["block weight"] = options.block_weight;
    j["block mode"] = options.block_mode;
    j["block min"] = options.block_min;
    j["block steps"] = options.block_steps;
    j["block cone"] = options.block_cone;
    j["block multi"] = options.block_multi;
    j["block union"] = options.block_union;
    j["block per piece"] = options.block_per_piece;
    j["block team"] = options.block_team;
    j["block self"] = options.block_self;
    j["block inf"] = options.block_inf;
    j["block depth"] = options.block_depth;
    j["block lmr"] = options.block_lmr;
    j["piece lmr"] = options.piece_lmr;
    j["family lmr"] = options.family_lmr;
    j["gate lmr"] = options.gate_lmr;
    j["gate min"] = options.gate_min;
    j["gate depth"] = options.gate_depth;
    j["gate ext"] = options.gate_ext;
    j["gate ext min"] = options.gate_ext_min;
    j["opp lmr"] = options.opp_lmr;
    j["opp lmr k"] = options.opp_lmr_k;
    j["split solve"] = options.split_solve;
    j["split max free"] = options.split_max_free;
    j["split budget"] = options.split_budget;
    j["pocket"] = options.pocket;
    j["pocket depth"] = options.pocket_depth;
    j["qs"] = options.qs;
    j["qs depth"] = options.qs_depth;
    j["qs moves"] = options.qs_moves;
    j["sibling ref"] = options.sibling_ref;
    j["tt fragile"] = options.tt_fragile;
    j["tt fragile ply"] = options.tt_fragile_ply;
    j["root audit"] = options.root_audit;
    j["root audit depth"] = options.root_audit_depth;
    j["root audit frac"] = options.root_audit_frac;
    j["sym"] = options.sym;
    j["sym depth"] = options.sym_depth;
    j["sym window"] = options.sym_window;
    j["sym thr"] = options.sym_thr;
    j["reserve"] = options.reserve;
    j["reserve max"] = options.reserve_max;
    j["lmr log"] = options.lmr_log;
    j["lmr table"] = options.lmr_table;
    j["fly reply relative"] = options.fly_reply_relative;
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
