#include "endgame_solver.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>

EndgameSolver::EndgameSolver(Options options) : options(options) {
    if (options.tt_bits > 0) table.resize(size_t(1) << options.tt_bits);
}

void EndgameSolver::clear() {
    std::fill(table.begin(), table.end(), Entry{});
    nb_nodes = 0;
}

// Stores what a search with the window (alpha, beta) learnt: a fail-soft value
// v is an upper bound if v ≤ alpha, a lower bound if v ≥ beta, exact in
// between. The bounds of the same position are merged (both stay true), so the
// entry only gets tighter. A different position in the slot is replaced.
void EndgameSolver::store(uint64_t hash, int alpha, int beta, int value, Move move) {
    if (table.empty()) return;
    Entry& e = table[hash & (table.size() - 1)];
    if (e.key != hash) e = Entry{hash};
    if (value > alpha) e.lower = int8_t(std::max<int>(e.lower, value));
    if (value < beta)  e.upper = int8_t(std::min<int>(e.upper, value));
    if (move != Move::none()) e.move = move;
}

// Puts the table's move first and, with Ordering::Fastest, sorts the others
// by the number of replies they leave the opponent, fewest first. Why it works
// in an exact search: a move that leaves few replies is often a strong one
// (the opponent is cramped, maybe soon blocked), and the subtree after it is
// small, so even when it is not the best it is cheap to refute. Counting the
// replies costs one move generation per move: worth it with many free
// squares, not near the very end (see brute_force_free).
// Returns the number of moves.
int EndgameSolver::order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const {
    const int n = int(moves.size());
    int first = 0;
    if (options.ordering != Ordering::None && tt_move != Move::none()) {
        for (int i = 0; i < n; i++) {
            if (moves[i] == tt_move) {
                std::swap(moves[0], moves[i]);
                first = 1;
                break;
            }
        }
    }
    if (options.ordering == Ordering::Fastest && n - first > 1) {
        int replies[Yolah::MAX_NB_MOVES];
        for (int i = first; i < n; i++) {
            yolah.play(moves[i]);
            const auto [b0, b1, b2, b3] = yolah.moves_bb(yolah.current_player());
            replies[i] = std::popcount(b0) + std::popcount(b1) + std::popcount(b2) + std::popcount(b3);
            yolah.undo(moves[i]);
        }
        // insertion sort: a few dozen moves at most
        for (int i = first + 1; i < n; i++) {
            const Move m = moves[i];
            const int r = replies[i];
            int j = i - 1;
            while (j >= first && replies[j] > r) {
                moves[j + 1] = moves[j];
                replies[j + 1] = replies[j];
                j--;
            }
            moves[j + 1] = m;
            replies[j + 1] = r;
        }
    }
    return n;
}

// Fail-soft alpha-beta on the remaining value (see the header).
int EndgameSolver::search(Yolah& yolah, uint64_t hash, int alpha, int beta) {
    ++nb_nodes;
    if (stopped()) return 0;
    if (yolah.game_over()) return 0;
    const int nb_free = std::popcount(yolah.free_squares());
    const bool brute = nb_free <= options.brute_force_free || table.empty();
    // The table: bounds proven by earlier searches of this position.
    Move tt_move = Move::none();
    if (!brute) {
        const Entry& e = table[hash & (table.size() - 1)];
        if (e.key == hash) {
            if (e.lower >= beta)  return e.lower;
            if (e.upper <= alpha) return e.upper;
            if (e.lower == e.upper) return e.lower;
            alpha = std::max<int>(alpha, e.lower);   // both bounds are true:
            beta  = std::min<int>(beta, e.upper);    // the window can shrink
            tt_move = e.move;
        }
    }
    const int alpha_orig = alpha;
    Yolah::MoveList moves;
    yolah.moves(moves);
    if (!brute) order_moves(yolah, tt_move, moves);
    const uint8_t player = yolah.current_player();
    int best = -127;
    Move best_move = Move::none();
    for (Move m : moves) {
        // a move earns its player one point, a pass none
        const int gain = m != Move::none();
        const uint64_t child = zobrist::update(hash, player, m);
        yolah.play(m);
        // v = gain − child value; v in (alpha, beta) ⇔ child in (gain − beta, gain − alpha)
        const int v = gain - search(yolah, child, gain - beta, gain - alpha);
        yolah.undo(m);
        if (stopped()) return 0;
        if (v > best) {
            best = v;
            best_move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;
            }
        }
    }
    if (!brute) store(hash, alpha_orig, beta, best, best_move);
    return best;
}

// Root: the same loop, keeping the best move. Values are converted between the
// final score difference (what the caller wants) and the remaining part (what
// the search computes): final = current difference + remaining.
EndgameSolver::Result EndgameSolver::solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop_flag) {
    stop = stop_flag;
    const uint64_t nodes_before = nb_nodes;
    Yolah yolah = position;
    Result res;
    const int current = yolah.score(yolah.current_player());
    if (yolah.game_over()) {
        res.value = current;
        res.complete = true;
        return res;
    }
    // Window on the remaining value. WLD: final in (−1, 1) ⇔ remaining in
    // (−1 − current, 1 − current): the search only has to tell < 0, 0, > 0.
    int alpha = wld_only ? -1 - current : -127;
    const int beta = wld_only ? 1 - current : 127;
    const uint64_t hash = zobrist::hash(yolah);
    Yolah::MoveList moves;
    yolah.moves(moves);
    Move tt_move = Move::none();
    if (!table.empty()) {
        const Entry& e = table[hash & (table.size() - 1)];
        if (e.key == hash) tt_move = e.move;
    }
    order_moves(yolah, tt_move, moves);
    const uint8_t player = yolah.current_player();
    int best = -127;
    for (Move m : moves) {
        const int gain = m != Move::none();
        yolah.play(m);
        const int v = gain - search(yolah, zobrist::update(hash, player, m), gain - beta, gain - alpha);
        yolah.undo(m);
        if (stopped()) {
            res.nodes = nb_nodes - nodes_before;
            return res;                         // complete = false
        }
        if (v > best) {
            best = v;
            res.move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;           // WLD: a win is enough
            }
        }
    }
    res.value = current + best;
    res.complete = true;
    res.nodes = nb_nodes - nodes_before;
    return res;
}
