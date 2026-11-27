#ifndef SOLVER_PLAYER_H
#define SOLVER_PLAYER_H
#include "player.h"
#include "endgame_solver.h"
#include <memory>

// The exact endgame solver as a player: in a position with at most "max free"
// free squares, it searches the game to its very end (EndgameSolver, exact
// final score) and plays a move that maximises its final lead. Maximising the
// final lead is the best defence against every goal of the form "win by at
// least N", whatever N: a puzzle's opponent that cannot be beaten by luck.
// When the position has more free squares, or the proof does not finish within
// "microseconds", it plays the move of its "fallback" player (any player
// configuration, e.g. the dev minimax), or a legal move if it has none.
//
//   { "name": "SolverPlayer", "microseconds": 5000000, "max free": 36,
//     "tt bits": 22, "nb threads": 8, "fallback": { ...a player configuration... },
//     "goal": "points" | "collect" | "first", "targets": "<64-bit mask>" }
//
// "goal" and "targets" (the app's target puzzles, EndgameGoal): the solver then
// plays for the targets; without a proof it takes a target when it can (First:
// that wins), else the fallback plays (for points).
//
// The solver's "pass rule" (a player who must pass has lost) is off: it only
// holds for positions reached by normal play, and a puzzle's starting bonus
// can let a blocked player win. The transposition table is kept from move to
// move: its values depend on the board only, so the next solves reuse them.
class SolverPlayer : public Player {
    json cfg;
    uint64_t microseconds;
    int max_free;
    EndgameSolver::Options options;
    EndgameSolver solver;
    std::unique_ptr<Player> fallback;
    std::string last;   // what the last move came from (info)
public:
    explicit SolverPlayer(const json& j);
    Move play(Yolah yolah) override;
    std::string info() override { return last; }
    json config() override { return cfg; }
};

#endif
