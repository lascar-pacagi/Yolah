// final_retro_check — doc/final_retro.cpp is the same code as player/solo_region.cpp.
//
//   • on the positions of real games (data/games): the same split (contested
//     squares, private clusters), the same canonical forms and the same values;
//   • final_retro::build_table (up to 7 free squares, 2 pieces) against the real
//     solver's values, state by state;
//   • on the all-private positions of those games, the reserves' difference
//     against the remaining value of the exact endgame solver (the idea itself).
// Built by test/final_retro_check.sh.
#include "endgame_solver.h"
#include "final_retro.h"
#include "solo_region.h"
#include "magic.h"
#include "zobrist.h"
#include <bit>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    magic::init();
    zobrist::init();
    const int max_games = argc > 1 ? atoi(argv[1]) : 1000;
    std::ifstream f("data/games/games_2025_12_09_09_53_30.421779150_0.txt", std::ios::binary);
    std::vector<uint8_t> e((std::istreambuf_iterator<char>(f)), {});
    solo::Solver real;
    final_retro::Solver doc;
    EndgameSolver::Options o;
    o.pass_rule = false;
    o.tt_bits = 20;
    EndgameSolver exact(o);
    long long positions = 0, bad_split = 0, bad_canon = 0, bad_value = 0, all_private = 0, bad_exact = 0;
    size_t n = 0;
    int games = 0;
    while (n < e.size() && games < max_games) {
        const int nb = e[n];
        const uint8_t* mv = &e[n + 2];
        Yolah y;
        for (int i = 0; i <= nb; i++) {
            const solo::Split a = solo::split(y);
            const final_retro::Split b = final_retro::split(y);
            positions++;
            bool same = a.contested == b.contested;
            int value[2] = {0, 0};
            for (int side = 0; side < 2 && same; side++) {
                same = a.priv[side].size() == b.priv[side].size();
                for (size_t c = 0; same && c < a.priv[side].size(); c++) {
                    const solo::Cluster& ca = a.priv[side][c];
                    const final_retro::Cluster& cb = b.priv[side][c];
                    same = ca.free == cb.free && ca.pieces == cb.pieces;
                    const solo::Key ka = solo::canonical(ca.free, ca.pieces);
                    const final_retro::Key kb = final_retro::canonical(cb.free, cb.pieces);
                    if (ka.free != kb.free || ka.pieces != kb.pieces) bad_canon++;
                    const int va = real.value(ca.free, ca.pieces), vb = doc.value(cb.free, cb.pieces);
                    if (va != vb) bad_value++;
                    value[side] += vb;
                }
            }
            if (!same) bad_split++;
            if (same && !y.game_over() && b.contested == 0 && std::popcount(y.free_squares()) <= 22) {
                all_private++;
                const auto r = exact.solve(y, false);
                const int us = y.current_player();
                if (r.value - y.score(us) != value[us] - value[1 - us]) bad_exact++;
            }
            if (i == nb) break;
            Move m(Square(mv[2 * i]), Square(mv[2 * i + 1]));
            if (mv[2 * i] == mv[2 * i + 1]) m = Move::none();
            y.play(m);
        }
        n += 2 + 2 * nb + 2;
        games++;
    }
    // The table of this file against the real solver.
    const final_retro::Table t = final_retro::build_table(7, 2);
    long long bad_table = 0;
    for (size_t i = 0; i < t.keys.size(); i++)
        if (real.value(t.keys[i].free, t.keys[i].pieces) != t.values[i]) bad_table++;
    std::printf("%d games, %lld positions: split mismatches %lld, canonical %lld, values %lld; %lld all-private positions, "
                "exact solver mismatches %lld; table (7 free, 2 pieces): %zu states, %lld mismatches\n",
                games, positions, bad_split, bad_canon, bad_value, all_private, bad_exact, t.keys.size(), bad_table);
    return bad_split || bad_canon || bad_value || bad_exact || bad_table;
}
