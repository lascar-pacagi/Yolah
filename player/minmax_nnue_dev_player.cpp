#include "minmax_nnue_dev_player.h"
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "zobrist.h"
#include <utility>

using std::cout, std::endl;

MinMaxNNUE_DevPlayer::MinMaxNNUE_DevPlayer(uint64_t microseconds, size_t tt_size_mb, size_t nb_moves_at_full_depth,
                                           uint8_t late_move_reduction, const std::string& nnue_q_parameters_filename,
                                           bool verbose, Options options)
    : thinking_time(microseconds), table(tt_size_mb),
      nb_moves_at_full_depth(nb_moves_at_full_depth), late_move_reduction(late_move_reduction),
      nnue_q_parameters_filename(nnue_q_parameters_filename), options(options), verbose(verbose) {
    nnue.load(nnue_q_parameters_filename);
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
    table.new_search();
    Search s;
    nnue.init(yolah, s.acc);
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
         + (options.aspiration_window ? " + aspiration windows" : "") + ")";
}

// ─── One move of a node ──────────────────────────────────────────────────────
// Plays move number i of the node, searches the child and returns its value
// for the side to move at the node. Callers check stopped() before using it:
// an interrupted child returns a meaningless 0.
//
// Three ways of searching a child, from the cheapest:
//   • a late move (i ≥ nb_moves_at_full_depth) is first searched at a REDUCED
//     depth (depth − late_move_reduction instead of depth − 1). If it does not
//     beat alpha, the move is believed bad and we stop there (LMR);
//   • with PVS, every move after the first one is then searched with a NULL
//     WINDOW (alpha, alpha + 1): the search only answers "is this move better
//     than alpha?", which cuts much more than a full window. The first move is
//     very often the best one (the transposition table's move), so the answer
//     is usually "no" and costs little;
//   • only when the answer is "yes" (alpha < v < beta) is the move searched
//     again with the full window, to get its exact value.
// Without PVS, every move that is not cut by LMR gets the full window (the
// reference's way).
int MinMaxNNUE_DevPlayer::search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i,
                                      int alpha, int beta, int depth) {
    const uint8_t player = yolah.current_player();
    const uint64_t child = zobrist::update(hash, player, m);
    nnue.play(player, m, s.acc);
    yolah.play(m);
    int v = alpha + 1;                    // "might beat alpha" until a search says otherwise
    if (i >= nb_moves_at_full_depth) {    // late move: reduced search first
        v = options.pvs ? -negamax(yolah, s, child, -(alpha + 1), -alpha, depth - late_move_reduction)
                        : -negamax(yolah, s, child, -beta, -alpha, depth - late_move_reduction);
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
    nnue.undo(player, m, s.acc);
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
    // 3. Horizon: the network's value for the side to move, scaled to ±WIN.
    //    FIX (A): not stored in the table. The table treats depth 0 as "empty
    //    slot" (TranspositionTable::probe), so the reference's leaf entries were
    //    never found again, and when a cluster was full they evicted a real
    //    entry of an inner node.
    if (depth <= 0) {
        return int(nnue.value(s.acc, yolah.current_player()) * WIN);
    }
    // 4. The moves, the most promising first.
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, s, tt_move, moves);
    const int alpha_orig = alpha;
    int best = -INFINITE;
    Move best_move = Move::none();
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        const int v = search_move(yolah, s, hash, m, i, alpha, beta, depth);
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

// ─── Root ────────────────────────────────────────────────────────────────────
// Like an inner node, but it returns the best move in `res`, and only moves
// whose search was COMPLETED (not interrupted by the clock) can become `res`:
// so even an interrupted iteration gives a usable move (see
// iterative_deepening).
int MinMaxNNUE_DevPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth, Move& res) {
    res = Move::none();
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, s, table.get_move(hash), moves);
    const int alpha_orig = alpha;
    int best = -INFINITE;
    for (size_t i = 0; i < moves.size(); i++) {
        const Move m = moves[i];
        const int v = search_move(yolah, s, hash, m, i, alpha, beta, depth);
        if (stopped()) return best;
        if (v > best) {
            best = v;
            if (v > alpha) {
                res = m;
                alpha = v;
                if (v >= beta) break;     // aspiration fail high: the caller widens the window
            }
        }
    }
    const Bound b = best >= beta ? BOUND_LOWER : best > alpha_orig ? BOUND_EXACT : BOUND_UPPER;
    table.update(hash, int16_t(best), b, depth, res);
    return best;
}

// Order: the table's move (the best move of an earlier, shallower search of
// this position), the two killer moves of the ply, then the others in the
// generator's order.
void MinMaxNNUE_DevPlayer::sort_moves(Yolah& yolah, const Search& s, Move tt_move, Yolah::MoveList& moves) {
    Move tmp[Yolah::MAX_NB_MOVES];
    size_t nb_moves = moves.size();
    Move killer_move1 = s.killer1[yolah.nb_plies()];
    Move killer_move2 = s.killer2[yolah.nb_plies()];
    Move b = Move::none();
    Move k1 = Move::none();
    Move k2 = Move::none();
    size_t n = 0;
    for (size_t i = 0; i < nb_moves; i++) {
        Move m = moves[i];
        if (m == tt_move) b = m;
        else if (m == killer_move1) k1 = m;
        else if (m == killer_move2) k2 = m;
        else tmp[n++] = m;
    }
    size_t i = 0;
    if (b != Move::none())  moves[i++] = b;
    if (k1 != Move::none()) moves[i++] = k1;
    if (k2 != Move::none()) moves[i++] = k2;
    for (size_t j = 0; j < n; j++) {
        moves[i++] = tmp[j];
    }
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
    j["verbose"] = verbose;
    return j;
}
