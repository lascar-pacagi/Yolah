#include "endgame_solver_lazy.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>
#include <thread>

EndgameSolverLazy::EndgameSolverLazy(Options options) : options(options) {
    if (options.tt_bits > 0) {
        table_size = size_t(1) << options.tt_bits;
        table = std::make_unique<Entry[]>(table_size);
    }
}

void EndgameSolverLazy::clear() {
    for (size_t i = 0; i < table_size; i++) { table[i].check.store(0); table[i].data.store(0); }
    nb_nodes = 0;
}

void EndgameSolverLazy::store(uint64_t hash, int alpha, int beta, int value, Move move) {
    if (table_size == 0) return;
    int lower = -127, upper = 127;
    Move old = Move::none();
    probe(hash, lower, upper, old);
    if (value > alpha) lower = std::max(lower, value);
    if (value < beta)  upper = std::min(upper, value);
    if (move == Move::none()) move = old;
    Entry& e = table[hash & (table_size - 1)];
    const uint64_t d = pack(lower, upper, move);
    e.data.store(d, std::memory_order_relaxed);
    e.check.store(hash ^ d, std::memory_order_relaxed);
}

void EndgameSolverLazy::order_moves(const Worker& w, Yolah& yolah, Move tt_move, Yolah::MoveList& moves, int nb_free) const {
    const int n = int(moves.size());
    int first = 0;
    if (options.ordering != EndgameOrdering::None && tt_move != Move::none()) {
        for (int i = 0; i < n; i++) {
            if (moves[i] == tt_move) {
                std::swap(moves[0], moves[i]);
                first = 1;
                break;
            }
        }
    }
    if (options.ordering == EndgameOrdering::Fastest && n - first > 1) {
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
    // Lazy SMP: thread k > 0 rotates the moves after the first by k, at the
    // big nodes, so that the threads start on different subtrees.
    if (w.id > 0 && nb_free >= options.lazy_min_free && n - 1 > 1) std::rotate(moves.begin() + 1, moves.begin() + 1 + (w.id % (n - 1)), moves.end());
}

// The free squares that `pieces` can still reach, through free squares
// (8-connected flood fill: a slide is a chain of steps over free squares).
static uint64_t reach(uint64_t pieces, uint64_t free) {
    auto grow = [](uint64_t b) {
        constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
        return row | (row << 8) | (row >> 8);
    };
    uint64_t r = grow(pieces) & free, prev = 0;
    while (r != prev) {
        prev = r;
        r = grow(r) & free;
    }
    return r;
}

// The same search as EndgameSolver's (see there), one per thread.
int EndgameSolverLazy::search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta) {
    ++w.nodes;
    if (stopped()) return 0;
    if (yolah.game_over()) return 0;
    const int nb_free = std::popcount(yolah.free_squares());
    const bool brute = nb_free <= options.brute_force_free || table_size == 0;
    Move tt_move = Move::none();
    if (!brute) {
        int lower, upper;
        Move m;
        if (probe(hash, lower, upper, m)) {
            if (lower >= beta)  return lower;
            if (upper <= alpha) return upper;
            if (lower == upper) return lower;
            alpha = std::max(alpha, lower);
            beta  = std::min(beta, upper);
            tt_move = m;
        }
    }
    const int alpha_orig = alpha;
    Yolah::MoveList moves;
    yolah.moves(moves);
    // The pass rule. A blocked player stays blocked (a free square never comes
    // back: a piece leaves a hole, and only moves to free squares), so after a
    // pass the opponent plays alone until the end. And when a player must pass
    // while the game is not over, they have lost:
    //   • white to move: black has made one move more and white will make no
    //     other: black wins;
    //   • black to move: the scores are equal and white can still move (the
    //     game is not over), so white will make at least one move more.
    // (The only draw, both blocked at black's turn, is a finished game.) So the
    // final difference of the side to move is ≤ −1, i.e. remaining ≤ −1 −
    // current: an upper bound, enough to fail low whenever it is ≤ alpha —
    // always the case in a win/draw/loss search. No need to search the
    // opponent's solo moves, which is what the exact score would require.
    // (Tested on the move list, generated anyway: free.)
    if (options.goal == EndgameGoal::Points && pass_rule_now && moves[0] == Move::none()) {
        const int upper = -1 - yolah.score(yolah.current_player());
        if (upper <= alpha) return upper;
    }
    // Without the pass rule (any scores, e.g. a puzzle's starting bonus), the
    // same facts give exact bounds when one side is blocked for good (a blocked
    // player stays blocked) and the other plays alone to the end:
    //   • the side to move must pass: the opponent will make at least one more
    //     move (the game is not over) and at most one per free square it can
    //     reach (each move fills one): remaining ∈ [−reach(opponent), −1];
    //   • the opponent is blocked (no free square next to its pieces): we will
    //     make at least one move and at most one per free square we can reach:
    //     remaining ∈ [1, reach(us)].
    // Without them, the solver would search every order of the lone player's
    // moves to find their exact number.
    if (options.goal != EndgameGoal::Points) {
        // Targets: only the free squares next to a side's pieces can still be landed on.
        // No target left there for either side: nothing more to win (exact 0).
        // Collect, one side blocked for good: the other takes at most the targets it
        // can reach, maybe none: remaining ∈ [−targets(opponent), 0] when we must
        // pass, ∈ [0, targets(us)] when the opponent is blocked.
        const uint64_t free = yolah.free_squares();
        const uint8_t us = yolah.current_player();
        const uint64_t r_us = reach(yolah.bitboard(us), free), r_them = reach(yolah.bitboard(Yolah::other_player(us)), free);
        if (((r_us | r_them) & options.targets) == 0) return 0;
        if (options.goal == EndgameGoal::Collect) {
            if (moves[0] == Move::none()) {
                if (0 <= alpha) return 0;
                const int lower = -std::popcount(r_them & options.targets);
                if (lower >= beta) return lower;
            } else if (r_them == 0) {
                if (0 >= beta) return 0;
                const int upper = std::popcount(r_us & options.targets);
                if (upper <= alpha) return upper;
            }
        }
    } else if (!pass_rule_now) {
        const uint64_t free = yolah.free_squares();
        const uint8_t us = yolah.current_player();
        if (moves[0] == Move::none()) {
            if (-1 <= alpha) return -1;
            const int lower = -std::popcount(reach(yolah.bitboard(Yolah::other_player(us)), free));
            if (lower >= beta) return lower;
        } else if (reach(yolah.bitboard(Yolah::other_player(us)), free) == 0) {
            if (1 >= beta) return 1;
            const int upper = std::popcount(reach(yolah.bitboard(us), free));
            if (upper <= alpha) return upper;
        }
    }
    if (!brute) order_moves(w, yolah, tt_move, moves, nb_free);
    int best = -127;
    Move best_move = Move::none();
    for (Move m : moves) {
        const int v = move_value(w, yolah, hash, m, alpha, beta);
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

int EndgameSolverLazy::move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta) {
    const int gain = move_gain(m);
    if (gain == FIRST_WIN) return 1;
    const uint64_t child = zobrist::update(hash, yolah.current_player(), m);
    yolah.play(m);
    const int v = gain - search(w, yolah, child, gain - beta, gain - alpha);
    yolah.undo(m);
    return v;
}

// Every thread searches the root; the first to finish writes the result and
// sets `done`, which stops the others (their partial results are dropped).
EndgameSolverLazy::Result EndgameSolverLazy::solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop_flag, const int* current_value) {
    stop = stop_flag;
    done.store(false);
    Result res;
    const int current = current_value ? *current_value : options.goal == EndgameGoal::Points ? position.score(position.current_player()) : 0;
    {   // the pass rule only with the scores of a normal game (see EndgameSolver::solve)
        const auto [b, wh] = position.score();
        pass_rule_now = options.pass_rule && !current_value && int(b) - int(wh) == (position.current_player() == Yolah::WHITE ? 1 : 0);
    }
    if (position.game_over()) { res.value = current; res.complete = true; return res; }
    const int alpha0 = wld_only ? -1 - current : -127;
    const int beta = wld_only ? 1 - current : 127;
    const uint64_t hash = zobrist::hash(position);
    std::vector<Worker> workers(std::max(1, options.threads));
    std::mutex m;
    auto run = [&](Worker& w) {
        Yolah yolah = position;
        Yolah::MoveList moves;
        yolah.moves(moves);
        Move tt_move = Move::none();
        int lower, upper;
        Move tm;
        if (table_size && probe(hash, lower, upper, tm)) tt_move = tm;
        order_moves(w, yolah, tt_move, moves, std::popcount(yolah.free_squares()));
        int alpha = alpha0, best = -127;
        Move best_move = Move::none();
        for (Move mv : moves) {
            const int v = move_value(w, yolah, hash, mv, alpha, beta);
            if (stopped()) return;
            if (v > best) {
                best = v;
                best_move = mv;
                if (v > alpha) { alpha = v; if (v >= beta) break; }
            }
        }
        std::lock_guard lk(m);
        if (done.load()) return;
        res.move = best_move;
        res.value = current + best;
        res.complete = true;
        done.store(true);
    };
    // root_split: the root's moves handed out one by one; the best so far and the
    // window under the lock.
    Yolah::MoveList root_moves;
    Yolah root = position;
    root.moves(root_moves);
    {
        Move tt_move = Move::none();
        int lower, upper;
        Move tm;
        if (table_size && probe(hash, lower, upper, tm)) tt_move = tm;
        Worker w0;   // id 0: no rotation
        order_moves(w0, root, tt_move, root_moves, 0);
    }
    std::atomic<size_t> next{0};
    int shared_alpha = alpha0, shared_best = -127;
    Move shared_move = Move::none();
    auto run_root = [&](Worker& w) {
        Yolah yolah = position;
        for (size_t i; (i = next.fetch_add(1)) < root_moves.size();) {
            int a;
            { std::lock_guard lk(m); a = shared_alpha; }
            const int v = move_value(w, yolah, hash, root_moves[i], a, beta);
            if (stopped()) return;
            std::lock_guard lk(m);
            if (v > shared_best) {
                shared_best = v;
                shared_move = root_moves[i];
                if (v > shared_alpha) { shared_alpha = v; if (v >= beta) done.store(true); }   // a win: the others stop
            }
        }
    };
    std::vector<std::thread> threads;
    for (size_t k = 1; k < workers.size(); k++) {
        workers[k].id = int(k);
        if (options.root_split) threads.emplace_back(run_root, std::ref(workers[k]));
        else threads.emplace_back(run, std::ref(workers[k]));
    }
    if (options.root_split) run_root(workers[0]);
    else run(workers[0]);
    for (auto& t : threads) t.join();
    if (options.root_split && !(stop && stop->load())) {   // complete unless stopped from outside
        res.move = shared_move;
        res.value = current + shared_best;
        res.complete = true;
    }
    for (const Worker& x : workers) res.nodes += x.nodes;
    nb_nodes += res.nodes;
    return res;
}
