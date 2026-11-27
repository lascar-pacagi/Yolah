#ifndef ENDGAME_SOLVER_PRO_H
#define ENDGAME_SOLVER_PRO_H
#include "endgame_solver.h"

// EndgameSolverPro: the exact solver of endgame_solver.h (same values, same
// options, same parallel search: read that one first, it is simpler), with
// two techniques of the Othello endgame solvers (Edax) on top:
//
//   • ETC, "enhanced transposition cutoff": at a node with at least
//     etc_min_free free squares, before searching anything, the table is
//     probed for every child; a child already proven bad enough for the
//     opponent (gain − its upper bound ≥ beta) cuts the node at once. The
//     sequential search only finds such a cutoff when it reaches that move.
//   • Ordering by short searches: at a node with at least probe_min_free free
//     squares, each move first gets a short search (probe_nodes nodes, the same
//     window). A move proven to reach beta cuts at once; the moves proven to
//     fail go last; the others keep the "fastest first" order. What the probes
//     prove is stored in the table, so it is not lost. A good order matters
//     twice with threads: fewer nodes, and less work wasted in parallel after
//     the move that wins.
//
// Measured (see endgame_solver.h): 5 % faster than EndgameSolver with one
// thread, slower with 4 or 8 threads (7.5 / 7.0 s against 5.9 / 5.3 s): the
// short searches are wasted work too. Kept as an experiment; not used by the app.
struct EndgameSolverProOptions : EndgameSolverOptions {
    int etc_min_free = 10;
    int probe_min_free = 20;
    uint64_t probe_nodes = 3000;
};

class EndgameSolverPro {
public:
    using Ordering = EndgameOrdering;
    using Options = EndgameSolverProOptions;
    struct Result {
        Move move = Move::none();
        int  value = 0;             // final score difference for the side to move (exact or WLD sign)
        bool complete = false;      // false: interrupted by the stop flag, move/value meaningless
        uint64_t nodes = 0;
    };

    explicit EndgameSolverPro(Options options = Options());
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
        uint64_t limit = 0;       // a short search's end (nodes), 0: none
        bool probing = false;     // inside a short search: no other short search, no split
        SplitPoint* sp = nullptr;
    };

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
        if (w.limit && w.nodes >= w.limit) return true;
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
    // ETC and the short searches (see the top of this file), on the node's
    // ordered moves: true when one move already reaches beta (its value and
    // move in cut_value, cut_move); else the moves are reordered.
    bool prepare_moves(Worker& w, Yolah& yolah, uint64_t hash, Yolah::MoveList& moves, int nb_free,
                       int alpha, int beta, int& cut_value, Move& cut_move);
    void store(uint64_t hash, int alpha, int beta, int value, Move move);
};

#endif
