// final_ybwc_check — doc/final_ybwc.cpp is the same code as the editor's parallel solver.
//
// On random positions (6..13 free squares, scores 0..6 per side, random side to
// move, random targets, the three goals), compares:
//   • final_ybwc::Solver with ONE thread and EndgameSolver configured as the
//     editor uses it (no pass rule, "fastest" ordering, 2^22 entries): same
//     values, same best moves, same node counts, exact and win/draw/loss;
//   • final_ybwc::Solver with 8 threads and a plain negamax to the end of the
//     game: same exact values, same signs, and its move reaches the value;
//   • final_ybwc::verify (4 threads) with the plain negamax: verdict and number
//     of key moves.
// The positions have few free squares, so the 8-thread searches split rarely;
// the parallel search on bigger positions is checked against the sequential one
// by the second part with 16..20 free squares (fewer positions).
// Built by test/final_ybwc_check.sh. Prints the counts and the mismatches.
#include "endgame_solver.h"
#include "final_ybwc.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <cstdio>
#include <random>

using final_ybwc::Goal;

// The value for the side to move: gain of each move as the goal says.
static int brute(Yolah& y, Goal g, uint64_t t) {
    if (y.game_over()) return 0;
    Yolah::MoveList ms;
    y.moves(ms);
    int best = -1000;
    for (Move m : ms) {
        const bool hit = m != Move::none() && (t >> m.to_sq() & 1);
        int v;
        if (g == Goal::First && hit) v = 1;
        else {
            const int gain = g == Goal::Points ? (m != Move::none()) : g == Goal::Collect ? int(hit) : 0;
            y.play(m);
            v = gain - brute(y, g, t);
            y.undo(m);
        }
        best = std::max(best, v);
    }
    return best;
}
static int current_of(const Yolah& y, Goal g) { return g == Goal::Points ? y.score(y.current_player()) : 0; }

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    const int count = argc > 1 ? std::atoi(argv[1]) : 1000;
    std::mt19937_64 rng(13);
    int n = 0, bad_ref = 0, bad_brute = 0, bad_move = 0, bad_verify = 0, bad_big = 0;
    auto random_position = [&](int lo, int span, Yolah& y, uint64_t& targets) {
        for (;;) {
            int sq[64];
            for (int i = 0; i < 64; i++) sq[i] = i;
            std::shuffle(sq, sq + 64, rng);
            const int nfree = lo + int(rng() % span);
            uint64_t b = 0, w = 0, e = 0;
            for (int i = 0; i < 4; i++) b |= 1ULL << sq[i];
            for (int i = 4; i < 8; i++) w |= 1ULL << sq[i];
            for (int i = 8 + nfree; i < 64; i++) e |= 1ULL << sq[i];
            targets = 0;
            for (int i = 8; i < 8 + std::min(nfree, 3); i++) targets |= 1ULL << sq[i];
            y.set_state(b, w, e, uint16_t(rng() % 7), uint16_t(rng() % 7), uint16_t(rng() % 2));
            if (!y.game_over()) return;
        }
    };
    while (n < count) {
        Yolah y;
        uint64_t targets;
        random_position(6, 8, y, targets);
        const Goal g = Goal(n % 3);
        const int truth = brute(y, g, targets) + current_of(y, g);
        EndgameSolver::Options ro;
        ro.pass_rule = false;
        ro.tt_bits = 22;
        ro.goal = EndgameGoal(int(g));
        ro.targets = targets;
        EndgameSolver ref(ro);
        final_ybwc::Options o1;
        o1.goal = g;
        o1.targets = targets;
        final_ybwc::Solver one(o1);
        final_ybwc::Options o8 = o1;
        o8.threads = 8;
        final_ybwc::Solver eight(o8);
        const int zero = 0;
        const int* cur = g == Goal::Points ? nullptr : &zero;
        for (bool wld : {true, false}) {
            const auto r1 = ref.solve(y, wld, nullptr, cur);
            const auto r2 = one.solve(y, wld, nullptr, cur);
            if (r1.value != r2.value || r1.move != r2.move || r1.nodes != r2.nodes)
                if (bad_ref++ < 5) std::printf("ref/doc mismatch (goal %d wld %d): value %d/%d nodes %llu/%llu\n", int(g), wld, r1.value, r2.value,
                                               (unsigned long long)r1.nodes, (unsigned long long)r2.nodes);
            const auto r8 = eight.solve(y, wld, nullptr, cur);
            const auto sgn = [](int v) { return (v > 0) - (v < 0); };
            const bool ok = r8.complete && (wld ? sgn(r8.value) == sgn(truth) : r8.value == truth);
            if (!ok && bad_brute++ < 5) std::printf("8 threads/brute mismatch (goal %d wld %d): %d vs %d\n", int(g), wld, r8.value, truth);
            if (!wld) {   // the move announced reaches the value
                const Move m = r8.move;
                const bool hit = m != Move::none() && (targets >> m.to_sq() & 1);
                int v;
                if (g == Goal::First && hit) v = 1;
                else {
                    Yolah c = y;
                    c.play(m);
                    const int gain = g == Goal::Points ? (m != Move::none()) : g == Goal::Collect ? int(hit) : 0;
                    v = g == Goal::Points ? -(brute(c, g, targets) + c.score(c.current_player())) : gain - brute(c, g, targets);
                }
                if (v != r8.value && bad_move++ < 5) std::printf("move mismatch (goal %d): %d vs %d\n", int(g), v, r8.value);
            }
        }
        // verify: the player's side and the goal's margin at random
        final_ybwc::Puzzle p;
        p.start = y;
        p.human = uint8_t(rng() % 2);
        p.goal = g;
        p.targets = targets;
        p.margin = g == Goal::First ? 1 : 1 + int(rng() % 2);
        const auto c = final_ybwc::verify(p, 4);
        const int sign = y.current_player() == p.human ? 1 : -1;
        const int lead = sign * (brute(y, g, targets) + current_of(y, g));
        const bool want = lead >= p.margin;
        bool ok = c.verdict == (want ? final_ybwc::Verdict::Feasible : final_ybwc::Verdict::Impossible);
        if (ok && c.keys_counted) {
            Yolah::MoveList ms;
            y.moves(ms);
            int keys = 0;
            for (Move m : ms) {
                const bool hit = m != Move::none() && (targets >> m.to_sq() & 1);
                if (g == Goal::First && hit) { keys++; continue; }
                Yolah ch = y;
                ch.play(m);
                const int gain = g == Goal::Collect ? int(hit) : 0;
                const int child = g == Goal::Points ? -(brute(ch, g, targets) + ch.score(ch.current_player())) : gain - brute(ch, g, targets);
                if (child >= p.margin) keys++;
            }
            ok = keys == int(c.keys.size());
        }
        if (!ok && bad_verify++ < 5) std::printf("verify mismatch (goal %d margin %d): lead %d\n", int(g), p.margin, lead);
        n++;
    }
    // Bigger positions: the 8-thread search really splits; compared with one thread.
    const int big = std::max(10, count / 20);
    for (int k = 0; k < big; k++) {
        Yolah y;
        uint64_t targets;
        random_position(16, 5, y, targets);
        const Goal g = Goal(k % 3);
        final_ybwc::Options o;
        o.goal = g;
        o.targets = targets;
        final_ybwc::Solver one(o);
        o.threads = 8;
        final_ybwc::Solver eight(o);
        const int zero = 0;
        const int* cur = g == Goal::Points ? nullptr : &zero;
        for (bool wld : {true, false}) {
            const auto a = one.solve(y, wld, nullptr, cur), b = eight.solve(y, wld, nullptr, cur);
            const bool ok = b.complete && (wld ? ((a.value > 0) - (a.value < 0)) == ((b.value > 0) - (b.value < 0)) : a.value == b.value);
            if (!ok && bad_big++ < 5) std::printf("big: 1/8 threads mismatch (goal %d wld %d): %d vs %d\n", int(g), wld, a.value, b.value);
        }
    }
    std::printf("%d positions: ref/doc mismatches %d, 8 threads/brute %d, moves %d, verify %d; %d big positions: 1/8 threads %d\n",
                n, bad_ref, bad_brute, bad_move, bad_verify, big, bad_big);
    return bad_ref || bad_brute || bad_move || bad_verify || bad_big;
}
