#ifndef ENDGAME_SOLVER_H
#define ENDGAME_SOLVER_H
#include "game.h"
#include "solo_region.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
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
// Threads (Options::threads > 1): the work is shared at the nodes (split,
// "young brothers wait") and the table is shared without a lock (Entry). The
// values and the proofs are the same; the nodes are more (the moves searched in
// parallel after one that wins). Measured on proofs of 24 to 28 free squares:
// about 3 to 5 times faster with 8 threads, no better beyond. With 1 thread the
// search is the sequential one, node for node (test/final_solver_check).
//
// Compared (12 positions of 24 to 26 free squares, win/draw/loss, one bench run;
// the sequential search: 18.6 s, 203 M nodes):
//   root only shared, 8 threads (EndgameSolverLazy, root_split)   38.1 s   (2 061 M nodes)
//   Lazy SMP, 4 / 8 threads (EndgameSolverLazy)            12.0 / 14.7 s   (  616 / 1 375 M)
//   this one (YBWC), 4 / 8 threads                           5.9 /  5.3 s   (  286 /   440 M)
//   YBWC + ETC + short searches, 1 / 4 / 8 (EndgameSolverPro) 17.7 / 7.5 / 7.0 s
// The work sharing wins clearly; ETC and the ordering by short searches gain
// 5 % alone and lose with threads (their short searches are wasted work too).
//

// (Outside the class: a nested struct with default member initializers cannot
// be a default argument of the class.)
enum class EndgameOrdering {
    None,       // the generator's order (pure brute force)
    TT,         // the transposition table's move first, then the generator's order
    Fastest     // the table's move, then "fastest first": the moves that leave the
                // opponent the fewest replies (as in Othello endgame solvers)
};
// What the game is scored on (the app's target puzzles):
//   Points   a move earns its player one point (the game of Yolah);
//   Collect  a move that lands on a target square earns one target, other moves
//            nothing (a target is taken once: its square is then occupied, then
//            a hole); the value is the difference of targets taken;
//   First    the first move that lands on a target wins (+1 for its player, −1
//            for the other); nobody reaching one: 0 (a draw).
// The remaining value still depends on the board and the side to move only
// (the targets are fixed for the solver's lifetime), so the table works as is.
enum class EndgameGoal { Points, Collect, First };
struct EndgameSolverOptions {
    EndgameOrdering ordering = EndgameOrdering::Fastest;
    int brute_force_free = 6;   // at most this many free squares: no table, no ordering
    int tt_bits = 21;           // 2^bits table entries (16 bytes each)
    bool pass_rule = true;      // a player who must pass has lost (see EndgameSolver::search); Points only
    EndgameGoal goal = EndgameGoal::Points;
    uint64_t targets = 0;       // the target squares (Collect, First)
    // Threads of a solve (1: the plain sequential search, node for node). With
    // more, the work is shared at the nodes ("young brothers wait", see
    // EndgameSolver::split): at a node with at least split_min_free free
    // squares, once its first move is searched and has not cut, the other moves
    // are searched in parallel by the threads that are idle.
    int threads = 1;
    int split_min_free = 12;
    // Private regions (player/solo_region.h; Points goal only): a region only
    // one side can reach is a reserve of moves whose size the cluster solver
    // knows exactly. Two exact shortcuts: a position with no contested region
    // left is a leaf (remaining = reserve of the side to move − the opponent's),
    // and in each private cluster of the side to move one move is searched, one
    // that keeps the largest reserve (the others lead to equivalent positions
    // or to smaller reserves). Off by default: the search is then node for node
    // the one of the commented copies.
    bool regions = false;
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
    // current: the value so far for the side to move, when it is not the score
    // difference of the position (Collect: the targets taken so far, side to move −
    // opponent, minus any handicap; First: 0). The result's value includes it.
    Result solve(const Yolah& yolah, bool wld_only, const std::atomic_bool* stop = nullptr, const int* current = nullptr);
    // A node budget for the next solves (0 = none): a solve that reaches it stops
    // like with the stop flag (complete = false). Used by the MCTS leaves
    // (player/leaf_minimax.h, "proven only"), where a clock per solve costs too much.
    void set_node_limit(uint64_t n) { node_limit = n; }
    void clear();
    uint64_t nodes() const { return nb_nodes; }

private:
    // A table entry, shared by the threads without a lock: `data` packs the
    // bounds and the move (with a "valid" bit), `check` = key XOR data. A reader
    // that gets the halves of two different writes sees check ≠ key ^ data and
    // ignores the entry, so a torn entry can never pass for a true bound (the
    // proofs stay exact). An empty slot (0, 0) has no valid bit.
    struct Entry {
        std::atomic<uint64_t> check{0}, data{0};
    };
    static constexpr uint64_t VALID = uint64_t(1) << 63;
    static uint64_t pack(int lower, int upper, Move m) {
        return VALID | uint64_t(uint8_t(int8_t(lower))) | uint64_t(uint8_t(int8_t(upper))) << 8 | uint64_t(m.raw()) << 16;
    }
    // The bounds of the position `hash` (false: not in the table).
    bool probe(uint64_t hash, int& lower, int& upper, Move& move) const {
        const Entry& e = table[hash & (table_size - 1)];
        const uint64_t d = e.data.load(std::memory_order_relaxed), c = e.check.load(std::memory_order_relaxed);
        if (!(d & VALID) || (c ^ d) != hash) return false;
        lower = int8_t(d & 0xFF);
        upper = int8_t(d >> 8 & 0xFF);
        move = Move(uint16_t(d >> 16));
        return true;
    }

    // A node whose remaining moves are shared between threads (split). Guarded
    // by `m`, except `cutoff`, read by every thread searching below it.
    struct SplitPoint {
        SplitPoint* parent = nullptr;   // the split point the master was working for
        Yolah pos;                      // the node's position, its hash
        uint64_t hash = 0;
        // The moves (a plain array: a copied Yolah::MoveList would keep its end
        // pointer into the original list, on the master's stack).
        Move moves[Yolah::MAX_NB_MOVES];
        size_t nb_moves = 0;
        size_t next = 0;                // the next move to hand out
        int alpha = 0, beta = 0, best = -127;
        Move best_move = Move::none();
        std::atomic<int> workers{0};    // threads searching one of its moves (the master included)
        std::mutex m;
        // A move reached beta: the others are stopped. Read at every node by every
        // thread below: on a cache line of its own, away from the lock and the
        // counters the threads write (each write would evict it from all caches).
        alignas(64) std::atomic<bool> cutoff{false};
        char pad[63];
        // Still has moves to hand out (under m).
        bool open() const { return !cutoff.load(std::memory_order_relaxed) && next < nb_moves; }
    };
    // One thread of a solve: its node count, and the split point it works for.
    struct alignas(64) Worker {   // a cache line each: `nodes` is written at every node
        uint64_t nodes = 0;
        SplitPoint* sp = nullptr;
        solo::Solver* solo = nullptr;   // the cluster solver of this thread (options.regions)
    };
    // One cluster solver per thread, kept from solve to solve (its memory grows).
    std::vector<std::unique_ptr<solo::Solver>> solos;
    // options.regions: the leaf value (true, in `value`) or the moves filtered.
    bool regions_step(Worker& w, const Yolah& yolah, Yolah::MoveList& moves, int& value);

    Options options;
    std::unique_ptr<Entry[]> table;
    size_t table_size = 0;
    uint64_t nb_nodes = 0;
    const std::atomic_bool* stop = nullptr;
    uint64_t node_limit = 0;   // the budget of a solve, per thread
    // Some thread has reached the budget: all stop (a helper that stopped alone
    // would drop its move without a word, and its master conclude without it).
    mutable std::atomic<bool> limit_hit{false};
    bool pass_rule_now = false;   // the pass rule for this solve (see solve())
    uint64_t split_gen = 0;       // split points opened so far (under sp_mutex): wakes the idle helpers
    // The split points open for helpers, and the idle threads (a split is only
    // worth its cost when some thread can join it).
    std::mutex sp_mutex;
    std::vector<SplitPoint*> active;
    std::condition_variable sp_cv;
    std::atomic<int> idle{0};
    std::atomic<int> nb_active{0};   // active.size(), read without the lock

    std::atomic<bool> solving{false};

    // What a move earns its player: Points 1 (a pass 0); Collect 1 on a target;
    // First: FIRST_WIN on a target (the game ends there), else 0.
    static constexpr int FIRST_WIN = 1000;
    int move_gain(Move m) const {
        if (options.goal == EndgameGoal::Points) return m != Move::none();
        const bool hit = m != Move::none() && (options.targets >> m.to_sq() & 1);
        if (options.goal == EndgameGoal::Collect) return hit;
        return hit ? FIRST_WIN : 0;
    }
    // The search must stop: the stop flag, the node budget, or a cutoff at one of
    // the split points this thread works under (its result is not wanted).
    bool stopped(const Worker& w) const {
        if (stop && stop->load(std::memory_order_relaxed)) return true;
        if (node_limit && (limit_hit.load(std::memory_order_relaxed) || w.nodes >= node_limit)) {
            limit_hit.store(true, std::memory_order_relaxed);
            return true;
        }
        for (const SplitPoint* p = w.sp; p; p = p->parent)
            if (p->cutoff.load(std::memory_order_relaxed)) return true;
        return false;
    }
    int  search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta);
    // The value of move m at the node (yolah, hash) for the side to move, with
    // the window (alpha, beta): gain − the child's value.
    int  move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta);
    // The moves from `first` on, shared with the idle threads; updates alpha,
    // best, best_move (the node's loop goes on with them).
    void split(Worker& w, Yolah& yolah, uint64_t hash, const Yolah::MoveList& moves, size_t first,
               int& alpha, int beta, int& best, Move& best_move);
    void work_on(Worker& w, SplitPoint& sp);   // takes moves of sp until none is left
    bool help(Worker& w, const SplitPoint* under);   // joins an open split point (below `under` if given)
    void helper_loop(Worker& w);                // an extra thread of a solve
    int  order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const;
    void store(uint64_t hash, int alpha, int beta, int value, Move move);
};

#endif
