#ifndef ENDGAME_SOLVER_LAZY_H
#define ENDGAME_SOLVER_LAZY_H
#include "endgame_solver.h"

// EndgameSolverLazy: the exact solver of endgame_solver.h (same values, same
// options, read that one first), made parallel the simple way, "Lazy SMP" (as
// the dev minimax player): every thread runs the WHOLE search of the root, all
// sharing one table; nothing else is shared. The first thread to finish gives
// the answer and stops the others. They do not all search the same moves at the
// same moment because their move orders differ a little (at the nodes with at
// least lazy_min_free free squares, thread k rotates the moves after the first
// by k), and each finds in the table what the others have proved.
// Measured (see endgame_solver.h): Lazy SMP 1.6× faster than the sequential
// search with 4 threads, 1.3× with 8 (the threads search the same subtrees);
// root_split 2× SLOWER (one root move takes most of the time, and the others
// are searched without the window the first gives). EndgameSolver's work
// sharing: 3.2× and 3.5×. Kept as an experiment; not used by the app.
//
// root_split: the simplest parallel search of all, instead of Lazy SMP: the
// threads share out the moves of the ROOT only (each takes the next one and
// searches it alone, to the end, with the shared table); a move that reaches
// beta stops the others. Nothing is shared below the root: with few root moves,
// or one that takes most of the time, the other threads have nothing to do.
struct EndgameSolverLazyOptions : EndgameSolverOptions {
    int lazy_min_free = 14;
    bool root_split = false;
};

class EndgameSolverLazy {
public:
    using Options = EndgameSolverLazyOptions;
    using Result = EndgameSolver::Result;
    explicit EndgameSolverLazy(Options options = Options());
    Result solve(const Yolah& yolah, bool wld_only, const std::atomic_bool* stop = nullptr, const int* current = nullptr);
    void clear();
    uint64_t nodes() const { return nb_nodes; }

private:
    // The shared table, as EndgameSolver's (key XOR data: a torn entry is ignored).
    struct Entry { std::atomic<uint64_t> check{0}, data{0}; };
    static constexpr uint64_t VALID = uint64_t(1) << 63;
    static uint64_t pack(int lower, int upper, Move m) {
        return VALID | uint64_t(uint8_t(int8_t(lower))) | uint64_t(uint8_t(int8_t(upper))) << 8 | uint64_t(m.raw()) << 16;
    }
    bool probe(uint64_t hash, int& lower, int& upper, Move& move) const {
        const Entry& e = table[hash & (table_size - 1)];
        const uint64_t d = e.data.load(std::memory_order_relaxed), c = e.check.load(std::memory_order_relaxed);
        if (!(d & VALID) || (c ^ d) != hash) return false;
        lower = int8_t(d & 0xFF);
        upper = int8_t(d >> 8 & 0xFF);
        move = Move(uint16_t(d >> 16));
        return true;
    }
    struct alignas(64) Worker {   // a cache line each
        int id = 0;
        uint64_t nodes = 0;
    };
    Options options;
    std::unique_ptr<Entry[]> table;
    size_t table_size = 0;
    uint64_t nb_nodes = 0;
    const std::atomic_bool* stop = nullptr;
    bool pass_rule_now = false;   // the pass rule for this solve (as EndgameSolver::solve)
    alignas(64) std::atomic<bool> done{false};   // one thread has finished: the others stop

    static constexpr int FIRST_WIN = 1000;
    int move_gain(Move m) const {
        if (options.goal == EndgameGoal::Points) return m != Move::none();
        const bool hit = m != Move::none() && (options.targets >> m.to_sq() & 1);
        if (options.goal == EndgameGoal::Collect) return hit;
        return hit ? FIRST_WIN : 0;
    }
    bool stopped() const { return (stop && stop->load(std::memory_order_relaxed)) || done.load(std::memory_order_relaxed); }
    int  search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta);
    int  move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta);
    void order_moves(const Worker& w, Yolah& yolah, Move tt_move, Yolah::MoveList& moves, int nb_free) const;
    void store(uint64_t hash, int alpha, int beta, int value, Move move);
};

#endif
