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

using std::cout, std::endl;

MinMaxNNUE_DevPlayer::MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                                           uint8_t late_move_reduction, const std::string& nnue_q_parameters_filename,
                                           bool verbose, Options options)
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      nnue_q_parameters_filename(nnue_q_parameters_filename), options(options), verbose(verbose) {
    nnue.load(nnue_q_parameters_filename);
    if (options.eval_cache_bits > 0) eval_cache.resize(size_t(1) << options.eval_cache_bits);
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
    table.new_search();
    // History aging: what was learnt during the previous moves still helps
    // (the positions are close), but less and less.
    for (auto& by_from : history)
        for (auto& by_to : by_from)
            for (int16_t& h : by_to) h /= 2;
    Search s;
    // The root's accumulator: computed from scratch, the only one that is.
    if (options.lazy_accumulator) {
        nnue.init(yolah, s.accs[yolah.nb_plies()]);
        s.acc_ok[yolah.nb_plies()] = true;
    } else {
        nnue.init(yolah, s.acc);
    }
    iterative_deepening(yolah, s, max_depth);
    if (clock.joinable()) {
        clock.request_stop();
        clock.join();
    }
    Result r;
    r.move = s.move;
    r.value = s.value;
    r.depth = s.depth;
    r.nb_nodes = s.nb_nodes;
    r.nb_hits = s.nb_hits;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

void MinMaxNNUE_DevPlayer::clear_table() {
    table.clear(1);
    endgame.clear();
    // the move ordering statistics too: each benchmark position from scratch
    std::fill(&history[0][0][0], &history[0][0][0] + sizeof(history) / sizeof(int16_t), int16_t(0));
    std::fill(&countermoves[0][0], &countermoves[0][0] + SQUARE_NB * SQUARE_NB, Move::none());
    std::fill(eval_cache.begin(), eval_cache.end(), EvalEntry{});
}

Move MinMaxNNUE_DevPlayer::play(Yolah yolah) {
    Result r = search(yolah, 63, thinking_time);
    if (verbose) {
        cout << "##########\n";
        cout << "depth  : " << int(r.depth) << '\n';
        cout << "value  : " << r.value << '\n';
        cout << "# nodes: " << r.nb_nodes << '\n';
        cout << "# hits : " << r.nb_hits << '\n';
        cout << "tt load: " << table.load() << '\n';
        print_pv(yolah, zobrist::hash(yolah), r.depth);
        cout << '\n';
    }
    return r.move;
}

std::string MinMaxNNUE_DevPlayer::info() {
    return std::string("minmax nnue dev player (one thread; transposition table + late move reduction + killer")
         + (options.pvs ? " + PVS" : "")
         + (options.aspiration_window ? " + aspiration windows" : "")
         + (options.history ? " + history" : "")
         + (options.countermove ? " + countermove" : "")
         + (options.root_ordering ? " + root ordering" : "")
         + (options.lazy_accumulator ? " + lazy accumulators" : "")
         + (options.eval_cache_bits ? " + evaluation cache" : "")
         + (options.lmr ? " + logarithmic LMR" : "")
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
// Example (base 1.0, divisor 1.75, the defaults: they beat 0.75 / 2.25 in a
// match), quiet move with a neutral history:
//     depth 4, move 4: 1.0 + 1.39·1.39/1.75 = 2.1 → 2 plies
//     depth 10, move 20: 1.0 + 2.30·3.00/1.75 = 4.9 → 4 plies
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
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    Move tt_move = Move::none();
    if (found) {
        s.nb_hits++;
        tt_move = entry->move();
        if (entry->depth() >= depth) {
            const int v = entry->value();
            const Bound b = entry->bound();
            if (b == BOUND_EXACT || (b == BOUND_LOWER && v >= beta) || (b == BOUND_UPPER && v <= alpha)) {
                return v;
            }
        }
    }
    // G. Very few free squares: the exact result from the endgame solver
    //    (win / draw / loss — only the sign matters for the outcome), instead
    //    of a search with the network. Stored in the table at the largest
    //    depth: it never needs to be searched again.
    if (options.endgame_tree > 0 && std::popcount(yolah.free_squares()) <= options.endgame_tree) {
        const auto proof = endgame.solve(yolah, true, &stop);
        if (!proof.complete) return 0;          // stopped: the caller ignores it
        const int v = proof.value > 0 ? WIN + proof.value : proof.value < 0 ? -WIN + proof.value : 0;
        table.update(hash, int16_t(v), BOUND_EXACT, 63, proof.move);
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
    // 4. The moves, the most promising first. They are scored once, then
    //    picked one at a time (the best remaining one): most nodes cut after
    //    one or two moves, so a full sort would be wasted work.
    Yolah::MoveList moves;
    yolah.moves(moves);
    int scores[Yolah::MAX_NB_MOVES];
    score_moves(yolah, s, tt_move, moves, scores);
    const int alpha_orig = alpha;
    int best = -INFINITE;
    Move best_move = Move::none();
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = pick_move(moves, scores, i);
        // scores[i] is now m's score: special move (killer, countermove) or history
        const int r = late_move_reduction_of(depth, i, beta - alpha > 1, scores[i] >= SCORE_COUNTER,
                                             scores[i] >= SCORE_COUNTER ? 0 : scores[i]);
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
                    table.update(hash, int16_t(v), BOUND_LOWER, depth, m);
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
                        update_history(player, m, bonus);
                        for (size_t k = 0; k < i; k++) update_history(player, moves[k], -bonus);
                    }
                    // Countermove (C): m refuted the opponent's last move.
                    if (options.countermove && ply > 0) {
                        const Move prev = s.played[ply - 1];
                        countermoves[prev.from_sq()][prev.to_sq()] = m;
                    }
                    return v;
                }
                alpha = v;
            }
        }
    }
    // No cutoff: exact value if a move raised alpha, otherwise only an upper
    // bound (all the moves failed low against the window we were given).
    table.update(hash, int16_t(best), best > alpha_orig ? BOUND_EXACT : BOUND_UPPER, depth, best_move);
    return best;
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
    nb_evals++;
    EvalEntry* e = nullptr;
    if (!eval_cache.empty()) {
        e = &eval_cache[hash & (eval_cache.size() - 1)];
        if (e->used && e->key == uint32_t(hash >> 32)) {
            nb_eval_hits++;
            return e->value;
        }
    }
    const int v = network_value(yolah, s);
    if (e) *e = {uint32_t(hash >> 32), int16_t(v), true};
    return v;
}

int MinMaxNNUE_DevPlayer::network_value(const Yolah& yolah, Search& s) {
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
        score_moves(yolah, s, table.get_move(hash), moves, scores);
        for (size_t i = 0; i < moves.size(); i++) s.root_moves.push_back({pick_move(moves, scores, i)});
    }
    for (size_t i = 0; i < s.root_moves.size(); i++) {
        RootMove& rm = s.root_moves[i];
        const uint64_t nodes_before = s.nb_nodes;
        // The root is a PV node; no killers there, the history still speaks.
        const int r = late_move_reduction_of(depth, i, true, false,
                                             history[yolah.current_player()][rm.move.from_sq()][rm.move.to_sq()]);
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
    const Bound b = best >= beta ? BOUND_LOWER : best > alpha_orig ? BOUND_EXACT : BOUND_UPPER;
    table.update(hash, int16_t(best), b, depth, res);
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
                                       const Yolah::MoveList& moves, int* scores) const {
    const uint16_t ply = yolah.nb_plies();
    const uint8_t player = yolah.current_player();
    Move counter = Move::none();
    if (options.countermove && ply > 0) {
        const Move prev = s.played[ply - 1];
        counter = countermoves[prev.from_sq()][prev.to_sq()];
    }
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        if (m == tt_move)                 scores[i] = SCORE_TT;
        else if (m == s.killer1[ply])     scores[i] = SCORE_KILLER1;
        else if (m == s.killer2[ply])     scores[i] = SCORE_KILLER2;
        else if (m == counter && counter != Move::none()) scores[i] = SCORE_COUNTER;
        else if (options.history)         scores[i] = history[player][m.from_sq()][m.to_sq()];
        else                              scores[i] = -int(i);
    }
}

// Selection step: brings the best-scored move of moves[i..] to position i.
Move MinMaxNNUE_DevPlayer::pick_move(Yolah::MoveList& moves, int* scores, size_t i) {
    size_t best = i;
    for (size_t j = i + 1; j < moves.size(); j++) {
        if (scores[j] > scores[best]) best = j;
    }
    std::swap(moves[i], moves[best]);
    std::swap(scores[i], scores[best]);
    return moves[i];
}

// "Gravity" update: h += bonus − h·|bonus| / HISTORY_MAX. The closer h is to
// ±HISTORY_MAX, the smaller its moves in that direction: h stays bounded, and
// recent cutoffs weigh more than old ones.
void MinMaxNNUE_DevPlayer::update_history(uint8_t player, Move m, int bonus) {
    int16_t& h = history[player][m.from_sq()][m.to_sq()];
    h = int16_t(h + bonus - h * std::abs(bonus) / HISTORY_MAX);
}

void MinMaxNNUE_DevPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth == 0) return;
    bool found;
    TranspositionTableEntry* entry = table.probe(hash, found);
    if (!found) return;
    auto player = yolah.current_player();
    cout << entry->move() << ' ';
    yolah.play(entry->move());
    print_pv(yolah, zobrist::update(hash, player, entry->move()), depth - 1);
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
        score_moves(yolah, s, table.get_move(hash), moves, scores);
        s.root_moves.clear();
        for (size_t i = 0; i < moves.size(); i++) s.root_moves.push_back({pick_move(moves, scores, i)});
    }
    for (int d = 1; d <= max_depth && d < 64; d++) {
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
    j["tt size"] = table.size();
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
    j["endgame root"] = options.endgame_root;
    j["endgame root time"] = options.endgame_root_time;
    j["endgame tree"] = options.endgame_tree;
    j["verbose"] = verbose;
    return j;
}
