// final_solver_check — doc/final_solver.cpp is the same code as the editor's solver.
//
// On random positions (6..13 free squares, scores 0..6 per side, random side to
// move), compares:
//   • final_solver::Solver with EndgameSolver configured as the editor uses it
//     (no pass rule, "fastest" ordering, 2^22 entries): same values, same best
//     moves, same node counts, exact and win/draw/loss;
//   • both with a plain negamax to the end of the game (no table, no ordering,
//     no bounds): same exact values;
//   • final_solver::verify with the plain negamax, for goals "win by 1..3".
// Built by test/final_solver_check.sh. Prints the counts and the mismatches.
#include "endgame_solver.h"
#include "final_solver.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <cstdio>
#include <random>

static int brute(Yolah& y) {   // final score difference for the side to move
    if (y.game_over()) return y.score(y.current_player());
    Yolah::MoveList ms;
    y.moves(ms);
    int best = -1000;
    for (Move m : ms) {
        y.play(m);
        best = std::max(best, -brute(y));
        y.undo(m);
    }
    return best;
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    const int count = argc > 1 ? std::atoi(argv[1]) : 2000;
    std::mt19937_64 rng(11);
    EndgameSolver::Options o;
    o.pass_rule = false;
    o.tt_bits = 22;
    EndgameSolver ref(o);
    final_solver::Solver doc(22);
    int n = 0, bad_ref = 0, bad_brute = 0, bad_verify = 0;
    while (n < count) {
        int sq[64];
        for (int i = 0; i < 64; i++) sq[i] = i;
        std::shuffle(sq, sq + 64, rng);
        const int nfree = 6 + int(rng() % 8);
        uint64_t b = 0, w = 0, e = 0;
        for (int i = 0; i < 4; i++) b |= 1ULL << sq[i];
        for (int i = 4; i < 8; i++) w |= 1ULL << sq[i];
        for (int i = 8 + nfree; i < 64; i++) e |= 1ULL << sq[i];
        Yolah y;
        y.set_state(b, w, e, uint16_t(rng() % 7), uint16_t(rng() % 7), uint16_t(rng() % 2));
        if (y.game_over()) continue;
        const int truth = brute(y);
        for (bool wld : {true, false}) {
            const auto r1 = ref.solve(y, wld);
            const auto r2 = doc.solve(y, wld);
            if (r1.value != r2.value || r1.move != r2.move || r1.nodes != r2.nodes) {
                if (bad_ref++ < 5) std::printf("ref/doc mismatch (wld %d): value %d/%d nodes %llu/%llu\n", wld, r1.value, r2.value,
                                               (unsigned long long)r1.nodes, (unsigned long long)r2.nodes);
            }
            const bool ok = wld ? ((r2.value > 0) - (r2.value < 0)) == ((truth > 0) - (truth < 0)) : r2.value == truth;
            if (!ok && bad_brute++ < 5) std::printf("doc/brute mismatch (wld %d): %d vs %d\n", wld, r2.value, truth);
        }
        // verify: the human side and the goal at random; the truth from the plain negamax
        const uint8_t human = uint8_t(rng() % 2);
        const int margin = 1 + int(rng() % 3);
        const auto c = final_solver::verify(y, human, margin);
        const int lead = y.current_player() == human ? truth : -truth;
        const auto want = lead >= margin ? final_solver::Verdict::Feasible : final_solver::Verdict::Impossible;
        bool ok = c.verdict == want;
        if (ok && c.keys_counted) {   // each key move must keep the goal, every other move must lose it
            Yolah::MoveList ms;
            y.moves(ms);
            int keys = 0;
            for (Move m : ms) {
                Yolah ch = y;
                ch.play(m);
                if (-brute(ch) >= margin) keys++;
            }
            ok = keys == int(c.keys.size());
        }
        if (!ok && bad_verify++ < 5) std::printf("verify mismatch: margin %d lead %d\n", margin, lead);
        n++;
    }
    std::printf("%d positions: ref/doc mismatches %d, doc/brute mismatches %d, verify mismatches %d\n", n, bad_ref, bad_brute, bad_verify);
    return bad_ref || bad_brute || bad_verify;
}
