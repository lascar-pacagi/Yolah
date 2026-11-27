// solo_region_check — the private-cluster values of player/solo_region.cpp.
//
// Replays real games (data/games); at every position, splits the free squares
// into contested regions and private clusters, and values the clusters
// (solo::Solver, one table for all). Checks: in a position where nothing is
// contested (and no piece links a private region to a contested one), the exact
// endgame solver's remaining value must equal the private values of the side to
// move minus the opponent's. Prints the coverage and the size of the table.
#include "endgame_solver.h"
#include "solo_region.h"
#include "magic.h"
#include "zobrist.h"
#include <bit>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    magic::init();
    zobrist::init();
    const char* path = argc > 1 ? argv[1] : "data/games/games_2025_12_09_09_53_30.421779150_0.txt";
    const int max_games = argc > 2 ? atoi(argv[2]) : 2000;
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> e((std::istreambuf_iterator<char>(f)), {});
    solo::Solver solver;
    EndgameSolver::Options o;
    o.pass_rule = false;
    o.tt_bits = 20;
    EndgameSolver exact(o);
    size_t n = 0;
    int games = 0;
    long long positions = 0, clusters = 0, all_private = 0, checked = 0, wrong = 0, max_cluster = 0;
    long long by_size[65] = {};
    double t_solo = 0;
    while (n < e.size() && games < max_games) {
        const int nb = e[n];
        const uint8_t* mv = &e[n + 2];
        Yolah y;
        for (int i = 0; i <= nb; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            const solo::Split s = solo::split(y);
            int value[2] = {0, 0};
            uint64_t covered = 0;
            for (int side = 0; side < 2; side++)
                for (const auto& c : s.priv[side]) {
                    value[side] += solver.value(c.free, c.pieces);
                    covered |= c.free;
                    clusters++;
                    const int sz = std::popcount(c.free);
                    by_size[sz]++;
                    max_cluster = std::max<long long>(max_cluster, sz);
                }
            t_solo += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            positions++;
            // Everything private: the exact solver must agree.
            if (!y.game_over() && s.contested == 0 && covered == (y.free_squares() & ~0ULL) ) {
                // regions touched by no piece are not covered: they count for nobody
                all_private++;
                if (std::popcount(y.free_squares()) <= 24) {
                    const auto r = exact.solve(y, false);
                    const int remaining = r.value - y.score(y.current_player());
                    const int us = y.current_player();
                    checked++;
                    if (remaining != value[us] - value[1 - us]) {
                        if (wrong++ < 5) std::printf("mismatch: solver %d, clusters %d - %d\n", remaining, value[us], value[1 - us]);
                    }
                }
            }
            if (i == nb) break;
            Move m(Square(mv[2 * i]), Square(mv[2 * i + 1]));
            if (mv[2 * i] == mv[2 * i + 1]) m = Move::none();
            y.play(m);
        }
        n += 2 + 2 * nb + 2;
        games++;
    }
    std::printf("%d games, %lld positions: %lld private clusters (largest %lld squares), %lld all-private positions, "
                "%lld checked against the exact solver, %lld wrong\n", games, positions, clusters, max_cluster, all_private, checked, wrong);
    std::printf("table: %zu canonical clusters, %llu solved; time %.2f s (%.1f µs per position)\n", solver.size(),
                (unsigned long long)solver.nodes(), t_solo, 1e6 * t_solo / positions);
    std::printf("cluster sizes:");
    for (int s = 1; s <= max_cluster; s++) if (by_size[s]) std::printf(" %d:%lld", s, by_size[s]);
    std::printf("\n");
    return wrong != 0;
}
