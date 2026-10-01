#ifndef ENDGAME_SOLVER_H
#define ENDGAME_SOLVER_H
#include "game.h"
#include <atomic>
#include <cstdint>
#include <vector>

// Exact endgame search for Yolah.
//
// Every move turns one free square into an occupied one (the destination) and
// one occupied square into a hole (the origin): the number of free squares
// drops by one at each move, so a position with k free squares ends in at most
// k moves (plus passes). When k is small enough, the game can be searched to
// the very end: the values are then exact final scores, no evaluation is
// needed (no NNUE, no accumulators), there are no reductions, and the result
// is a proof.
//
// Values. A player's score is the number of moves they made. The solver works
// on the REMAINING score: the number of moves the side to move will still
// make, minus the opponent's, with perfect play. The final score difference
// is the current one plus this remaining part. Working on the remaining part
// makes the value depend only on the board and the side to move (what the
// Zobrist hash covers), not on the scores so far: the transposition table can
// share it between positions reached with different scores.
//
//     remaining(position) = max over moves m of  1 − remaining(child)   (a move: +1 point)
//                                              or  − remaining(child)   (a pass)
//     remaining(finished game) = 0
//

// (Outside the class: a nested struct with default member initializers cannot
// be a default argument of the class.)
enum class EndgameOrdering {
    None,       // the generator's order (pure brute force)
    TT,         // the transposition table's move first, then the generator's order
    Fastest     // the table's move, then "fastest first": the moves that leave the
                // opponent the fewest replies (as in Othello endgame solvers)
};
struct EndgameSolverOptions {
    EndgameOrdering ordering = EndgameOrdering::Fastest;
    int brute_force_free = 6;   // at most this many free squares: no table, no ordering
    int tt_bits = 21;           // 2^bits table entries (16 bytes each)
    bool pass_rule = true;      // a player who must pass has lost (see EndgameSolver::search)
};

class EndgameSolver {
public:
    using Ordering = EndgameOrdering;
    using Options = EndgameSolverOptions;
    struct Result {
        Move move = Move::none();
        int  value = 0;             // final score difference for the side to move (exact or WLD sign)
        bool complete = false;      // false: interrupted by the stop flag, move/value meaningless
        uint64_t nodes = 0;
    };

    explicit EndgameSolver(Options options = Options());
    // Best move and exact final score difference (side to move − opponent).
    // wld_only: only win / draw / loss — the value's sign is exact, its
    // magnitude is not (much cheaper: the search only asks "≥ 1?" and "≤ −1?").
    Result solve(const Yolah& yolah, bool wld_only, const std::atomic_bool* stop = nullptr);
    void clear();
    uint64_t nodes() const { return nb_nodes; }

private:
    struct Entry {
        uint64_t key = 0;
        int8_t   lower = -127;      // the remaining value is ≥ lower
        int8_t   upper = 127;       //                     and ≤ upper
        Move     move = Move::none();
    };
    Options options;
    std::vector<Entry> table;
    uint64_t nb_nodes = 0;
    const std::atomic_bool* stop = nullptr;

    bool stopped() const { return stop && stop->load(std::memory_order_relaxed); }
    int  search(Yolah& yolah, uint64_t hash, int alpha, int beta);
    int  order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const;
    void store(uint64_t hash, int alpha, int beta, int value, Move move);
};

#endif
