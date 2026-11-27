#include "endgame_solver_pro.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <thread>

EndgameSolverPro::EndgameSolverPro(Options options) : options(options) {
    if (options.tt_bits > 0) {
        table_size = size_t(1) << options.tt_bits;
        table = std::make_unique<Entry[]>(table_size);
    }
}

void EndgameSolverPro::clear() {
    for (size_t i = 0; i < table_size; i++) {
        table[i].check.store(0, std::memory_order_relaxed);
        table[i].data.store(0, std::memory_order_relaxed);
    }
    nb_nodes = 0;
}

// Stores what a search with the window (alpha, beta) learnt: a fail-soft value
// v is an upper bound if v ≤ alpha, a lower bound if v ≥ beta, exact in
// between. The bounds of the same position are merged (both stay true), so the
// entry only gets tighter. A different position in the slot is replaced.
// (With several threads, two stores of the same slot can interleave: the
// entry then holds one of them, or is torn and ignored by probe. Either way
// every bound the table gives is true; at worst a merge is lost.)
void EndgameSolverPro::store(uint64_t hash, int alpha, int beta, int value, Move move) {
    if (table_size == 0) return;
    int lower = -127, upper = 127;
    Move old = Move::none();
    probe(hash, lower, upper, old);   // a different position, or none: the defaults
    if (value > alpha) lower = std::max(lower, value);
    if (value < beta)  upper = std::min(upper, value);
    if (move == Move::none()) move = old;
    Entry& e = table[hash & (table_size - 1)];
    const uint64_t d = pack(lower, upper, move);
    e.data.store(d, std::memory_order_relaxed);
    e.check.store(hash ^ d, std::memory_order_relaxed);
}

// Puts the table's move first and, with Ordering::Fastest, sorts the others
// by the number of replies they leave the opponent, fewest first. Why it works
// in an exact search: a move that leaves few replies is often a strong one
// (the opponent is cramped, maybe soon blocked), and the subtree after it is
// small, so even when it is not the best it is cheap to refute. Counting the
// replies costs one move generation per move: worth it with many free
// squares, not near the very end (see brute_force_free).
// Returns the number of moves.
int EndgameSolverPro::order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const {
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

// Fail-soft alpha-beta on the remaining value (see the header).
int EndgameSolverPro::search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta) {
    ++w.nodes;
    if (stopped(w)) return 0;
    if (yolah.game_over()) return 0;
    const int nb_free = std::popcount(yolah.free_squares());
    const bool brute = nb_free <= options.brute_force_free || table_size == 0;
    // The table: bounds proven by earlier searches of this position.
    Move tt_move = Move::none();
    if (!brute) {
        int lower, upper;
        Move m;
        if (probe(hash, lower, upper, m)) {
            if (lower >= beta)  return lower;
            if (upper <= alpha) return upper;
            if (lower == upper) return lower;
            alpha = std::max(alpha, lower);   // both bounds are true:
            beta  = std::min(beta, upper);    // the window can shrink
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
    if (!brute) {
        order_moves(yolah, tt_move, moves);
        int cut_value;
        Move cut_move;
        if (prepare_moves(w, yolah, hash, moves, nb_free, alpha, beta, cut_value, cut_move)) {
            if (stopped(w)) return 0;
            store(hash, alpha_orig, beta, cut_value, cut_move);
            return cut_value;
        }
        if (stopped(w)) return 0;
    }
    int best = -127;
    Move best_move = Move::none();
    for (size_t i = 0; i < moves.size(); i++) {
        // Young brothers wait: the first move alone (it often cuts, and its value
        // narrows the window for the others), then, if it has not cut and some
        // thread is idle, the others shared (split). (Measured: waiting for more
        // moves at the nodes expected to cut, or splitting only deeper, is slower;
        // the extra nodes of the parallel search are those searched after a move
        // that wins, mostly unavoidable.)
        if (i == 1 && options.threads > 1 && !w.probing && nb_free >= options.split_min_free && idle.load(std::memory_order_relaxed) > 0) {
            split(w, yolah, hash, moves, i, alpha, beta, best, best_move);
            if (stopped(w)) return 0;
            break;
        }
        const Move m = moves[i];
        const int v = move_value(w, yolah, hash, m, alpha, beta);
        if (stopped(w)) return 0;
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

bool EndgameSolverPro::prepare_moves(Worker& w, Yolah& yolah, uint64_t hash, Yolah::MoveList& moves, int nb_free,
                                     int alpha, int beta, int& cut_value, Move& cut_move) {
    const size_t n = moves.size();
    if (n < 2) return false;
    // ETC: a child the table already proves bad enough for the opponent.
    if (table_size > 0 && nb_free >= options.etc_min_free) {   // (the root calls it even without a table)
        const uint8_t us = yolah.current_player();
        for (size_t i = 0; i < n; i++) {
            const Move m = moves[i];
            const int gain = move_gain(m);
            int v;
            if (gain == FIRST_WIN) v = 1;
            else {
                int lower, upper;
                Move mm;
                if (!probe(zobrist::update(hash, us, m), lower, upper, mm)) continue;
                v = gain - upper;   // the child is worth at most `upper` to the opponent
            }
            if (v >= beta) { cut_value = v; cut_move = m; return true; }
        }
    }
    // Short searches: each move with a small node budget, same window.
    if (nb_free < options.probe_min_free || options.probe_nodes == 0 || w.probing) return false;
    int rank[Yolah::MAX_NB_MOVES];   // 0: proven above alpha, 1: unknown, 2: proven ≤ alpha
    w.probing = true;
    const uint64_t saved = w.limit;
    for (size_t i = 0; i < n; i++) {
        w.limit = w.nodes + options.probe_nodes;
        const int v = move_value(w, yolah, hash, moves[i], alpha, beta);
        const bool cut_short = w.nodes >= w.limit;
        w.limit = saved;
        if (stopped(w)) { w.probing = false; return false; }   // the real stop: the caller sees it too
        if (cut_short) rank[i] = 1;
        else if (v >= beta) { w.probing = false; cut_value = v; cut_move = moves[i]; return true; }
        else rank[i] = v > alpha ? 0 : 2;
    }
    w.probing = false;
    // Stable: within a rank, the "fastest first" order is kept.
    Move sorted[Yolah::MAX_NB_MOVES];
    size_t k = 0;
    for (int r = 0; r <= 2; r++)
        for (size_t i = 0; i < n; i++)
            if (rank[i] == r) sorted[k++] = moves[i];
    std::copy(sorted, sorted + n, moves.begin());
    return false;
}

int EndgameSolverPro::move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta) {
    // a move earns its player one point, a pass none (Points); see move_gain
    const int gain = move_gain(m);
    if (gain == FIRST_WIN) return 1;   // First: on a target, the game is won
    const uint64_t child = zobrist::update(hash, yolah.current_player(), m);
    yolah.play(m);
    // v = gain − child value; v in (alpha, beta) ⇔ child in (gain − beta, gain − alpha)
    const int v = gain - search(w, yolah, child, gain - beta, gain - alpha);
    yolah.undo(m);
    return v;
}

// ── Sharing the work (options.threads > 1) ──────────────────────────────────
// "Young brothers wait" (YBWC), as in parallel chess and Othello solvers: a
// node's moves after the first are handed out one by one, to the thread that
// split the node (its master) and to the idle threads, which each search a
// move on their own copy of the position and report its value. The node's
// alpha rises as the values come in (later moves get the better window); a
// value ≥ beta sets `cutoff`, which stops every thread below the split point
// (stopped() walks the chain of split points). While its helpers finish, the
// master does not wait idle: it helps at the split points opened below its own
// (by its helpers), the only ones whose end it is waiting for.
void EndgameSolverPro::split(Worker& w, Yolah& yolah, uint64_t hash, const Yolah::MoveList& moves, size_t first,
                          int& alpha, int beta, int& best, Move& best_move) {
    SplitPoint sp;
    sp.parent = w.sp;
    sp.pos = yolah;
    sp.hash = hash;
    sp.nb_moves = moves.size();
    std::copy(moves.begin(), moves.end(), sp.moves);
    sp.next = first;
    sp.alpha = alpha;
    sp.beta = beta;
    sp.best = best;
    sp.best_move = best_move;
    sp.workers = 1;   // the master
    {
        std::lock_guard lk(sp_mutex);
        active.push_back(&sp);
        nb_active.store(int(active.size()), std::memory_order_relaxed);
        split_gen++;
    }
    sp_cv.notify_all();
    w.sp = &sp;
    work_on(w, sp);
    {
        // Closed: nothing more is handed out. work_on also returns when the search
        // above is stopped with moves left; no thread may join then, since the
        // split point dies with this call once its helpers have left.
        std::lock_guard lk(sp.m);
        sp.next = sp.nb_moves;
        sp.workers--;
    }
    // The helpers still on a move: help below (only when other split points
    // exist: no lock taken for nothing), else let them run.
    while (sp.workers.load(std::memory_order_acquire) > 0)
        if (nb_active.load(std::memory_order_relaxed) <= 1 || !help(w, &sp)) std::this_thread::yield();
    {
        std::lock_guard lk(sp_mutex);
        active.erase(std::find(active.begin(), active.end(), &sp));
        nb_active.store(int(active.size()), std::memory_order_relaxed);
    }
    w.sp = sp.parent;
    alpha = sp.alpha;
    best = sp.best;
    best_move = sp.best_move;
}

void EndgameSolverPro::work_on(Worker& w, SplitPoint& sp) {
    for (;;) {
        size_t i;
        int a;
        {
            std::lock_guard lk(sp.m);
            if (!sp.open()) return;
            i = sp.next++;
            a = sp.alpha;
        }
        Yolah y = sp.pos;
        const int v = move_value(w, y, sp.hash, sp.moves[i], a, sp.beta);
        if (stopped(w)) return;   // the stop flag, or a cutoff here or above: the value is not wanted
        std::lock_guard lk(sp.m);
        if (v > sp.best) {
            sp.best = v;
            sp.best_move = sp.moves[i];
            if (v > sp.alpha) {
                sp.alpha = v;
                if (v >= sp.beta) sp.cutoff.store(true, std::memory_order_relaxed);
            }
        }
    }
}

// Joins an open split point (the shallowest: the most work), below `under`
// when given (a master waiting for its helpers). Joining under sp_mutex: a
// split point leaves `active` (and its master's stack) only once closed and
// left by all, and a closed one never opens again.
bool EndgameSolverPro::help(Worker& w, const SplitPoint* under) {
    SplitPoint* sp = nullptr;
    {
        std::lock_guard lk(sp_mutex);
        for (SplitPoint* c : active) {
            if (under) {
                const SplitPoint* p = c->parent;
                while (p && p != under) p = p->parent;
                if (!p) continue;
            }
            std::lock_guard lk2(c->m);
            if (c->open()) { c->workers++; sp = c; break; }
        }
    }
    if (!sp) return false;
    SplitPoint* saved = w.sp;
    w.sp = sp;
    work_on(w, *sp);
    // The last access: once the count is 0, the master may return and the split
    // point (on its stack) is gone, so no lock of sp->m around it.
    sp->workers.fetch_sub(1, std::memory_order_release);
    w.sp = saved;
    return true;
}

void EndgameSolverPro::helper_loop(Worker& w) {
    while (solving.load(std::memory_order_acquire)) {
        uint64_t gen;
        {
            std::lock_guard lk(sp_mutex);
            gen = split_gen;
        }
        if (help(w, nullptr)) continue;
        idle.fetch_add(1, std::memory_order_relaxed);
        {
            // Asleep until a new split point or the end of the solve (a split made
            // between help() and here is seen through split_gen: no lost wake-up).
            std::unique_lock lk(sp_mutex);
            sp_cv.wait_for(lk, std::chrono::microseconds(200), [&] { return split_gen != gen || !solving.load(std::memory_order_relaxed); });
        }
        idle.fetch_sub(1, std::memory_order_relaxed);
    }
}

// Root: the same loop, keeping the best move. Values are converted between the
// final score difference (what the caller wants) and the remaining part (what
// the search computes): final = current difference + remaining.
EndgameSolverPro::Result EndgameSolverPro::solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop_flag, const int* current_value) {
    stop = stop_flag;
    limit_hit.store(false, std::memory_order_relaxed);
    Yolah yolah = position;
    Result res;
    // The value so far: given by the caller, or the score difference of the position
    // (Points). With the target goals the scores count moves, not targets: 0.
    const int current = current_value ? *current_value : options.goal == EndgameGoal::Points ? yolah.score(yolah.current_player()) : 0;
    // The pass rule holds only for the scores of a normal game (black one point ahead
    // when white is to move, equal otherwise) and only when the caller adds nothing:
    // anything else (a puzzle's bonus, a value passed in) and it would give a false
    // bound; the bounds without it are exact whatever the scores.
    {
        const auto [b, wh] = yolah.score();
        const bool normal = int(b) - int(wh) == (yolah.current_player() == Yolah::WHITE ? 1 : 0);
        pass_rule_now = options.pass_rule && !current_value && normal;
    }
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
    if (table_size) {
        int lower, upper;
        Move m;
        if (probe(hash, lower, upper, m)) tt_move = m;
    }
    order_moves(yolah, tt_move, moves);
    // The helpers of this solve: they wait for split points until it ends.
    std::vector<Worker> workers(std::max(1, options.threads));
    std::vector<std::thread> helpers;
    solving.store(true, std::memory_order_release);
    for (size_t k = 1; k < workers.size(); k++) helpers.emplace_back([this, &workers, k] { helper_loop(workers[k]); });
    Worker& w = workers[0];
    int best = -127;
    bool complete = true, root_cut = false;
    // The root too: ETC and short searches; a move that reaches beta is the answer.
    {
        int cut_value;
        Move cut_move;
        if (prepare_moves(w, yolah, hash, moves, std::popcount(yolah.free_squares()), alpha, beta, cut_value, cut_move)) {
            best = cut_value;
            res.move = cut_move;
            root_cut = true;   // nothing more to search
        }
        if (stopped(w)) complete = false;
    }
    for (size_t i = 0; complete && !root_cut && i < moves.size(); i++) {
        if (i == 1 && options.threads > 1) {   // the root too: its first move alone, then shared
            split(w, yolah, hash, moves, 1, alpha, beta, best, res.move);
            if (stopped(w)) complete = false;
            break;
        }
        const Move m = moves[i];
        const int v = move_value(w, yolah, hash, m, alpha, beta);
        if (stopped(w)) { complete = false; break; }
        if (v > best) {
            best = v;
            res.move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;           // WLD: a win is enough
            }
        }
    }
    {
        std::lock_guard lk(sp_mutex);   // under the lock: a helper about to sleep sees it
        solving.store(false, std::memory_order_release);
    }
    sp_cv.notify_all();
    for (auto& h : helpers) h.join();
    for (const Worker& x : workers) res.nodes += x.nodes;
    nb_nodes += res.nodes;
    if (!complete) {
        res.move = Move::none();
        return res;                             // complete = false
    }
    res.value = current + best;
    res.complete = true;
    return res;
}
