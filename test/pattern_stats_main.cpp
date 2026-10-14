// pattern_stats_main.cpp — which local patterns around a piece go with won
// games, which with lost ones? (study of the user's idea: frequent patterns in
// the positions of data/games)
//
//   ./pattern_stats GAMES_DIR OUT.bin [--stride 20] [--threads 16]
//
// For every --stride-th position of every game (after the random opening
// moves; the first position taken depends on the game, so that all plies are
// covered), and for each of its 8 pieces: the 3×3 block centred on the
// piece, seen from the piece's owner. Each of the 8 neighbours is
//     1 free   2 hole   3 mine (the owner's)   4 his   5 off the board
// The outcome is the game's final result for the owner (+1 win, 0 draw,
// −1 loss), minus the mean result of the owner in the same situation (colour,
// side to move or not, number of free squares): what is left is what the
// position adds, not that Black wins 61 % of the games.
//
// A pattern may also contain ? (0: any state) in any neighbour: every
// occurrence counts for the 2^8 = 256 patterns obtained by replacing some of
// its neighbours with ?. A pattern is a number in base 6, neighbour k being
// digit k (neighbours: (+1,−1) (+1,0) (+1,+1) (0,−1) (0,+1) (−1,−1) (−1,0)
// (−1,+1) as (rank, file) steps, rank +1 = north), so 6^8 = 1 679 616 patterns.
// The 8 symmetric versions of a pattern (rotations, reflections) are one
// pattern, stored at the smallest of their numbers; an occurrence counts at
// most once for each (it can match two orientations of the same pattern).
//
// OUT.bin, for each of 3 phases (free squares 40–56, 20–39, 0–19) and each
// pattern: uint32 occurrences, then for each phase and pattern: double sum of
// the adjusted outcomes; then for each phase: uint64 occurrences, double sum
// of the squared adjusted outcomes (for the spread). nnue/pattern_stats.py
// merges the symmetric patterns and ranks them.
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

constexpr int NB = 8, NP = 1679616, PHASES = 3;   // 6^8
static int phase_of(int free) { return free >= 40 ? 0 : free >= 20 ? 1 : 2; }

struct Occurrence {
    int8_t owner_outcome;       // final result for the piece's owner
    uint8_t colour, stm, free;
    uint8_t digits[NB];         // 1..5
};

// For each position taken: the 8 occurrences (one per piece).
template <class F> static uint64_t for_each_game(const string& path, size_t stride, F&& f) {
    std::ifstream in(path, std::ios::binary);
    vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
    static constexpr int DR[NB] = {1, 1, 1, 0, 0, -1, -1, -1}, DC[NB] = {-1, 0, 1, -1, 1, -1, 0, 1};
    vector<Occurrence> occ;
    size_t game = 0;
    uint64_t used = 0;
    for (size_t i = 0; i + 1 < e.size(); game++) {
        const int n = e[i], nb_random = e[i + 1];
        if (i + 2 * n + 4 > e.size()) break;
        const int black_score = e[i + 2 + 2 * n], white_score = e[i + 3 + 2 * n];
        const int black_result = black_score > white_score ? 1 : black_score < white_score ? -1 : 0;
        Yolah y;
        Yolah::MoveList moves;
        occ.clear();
        bool ok = true;
        for (int k = 0; k <= n; k++) {
            if (k >= nb_random && (k + game) % stride == 0 && !y.game_over()) {
                const uint64_t free = y.free_squares();
                const uint64_t holes = y.empty_bitboard();
                for (int c = 0; c < 2; c++) {
                    const uint64_t mine = y.bitboard(c), his = y.bitboard(1 - c);
                    for (uint64_t b = mine; b; b &= b - 1) {
                        const int sq = std::countr_zero(b), r = sq / 8, f = sq % 8;
                        Occurrence o;
                        o.owner_outcome = int8_t(c == 0 ? black_result : -black_result);
                        o.colour = uint8_t(c);
                        o.stm = uint8_t(y.current_player() == c);
                        o.free = uint8_t(std::popcount(free));
                        for (int d = 0; d < NB; d++) {
                            const int rr = r + DR[d], ff = f + DC[d];
                            if (rr < 0 || rr > 7 || ff < 0 || ff > 7) { o.digits[d] = 5; continue; }
                            const uint64_t s = uint64_t(1) << (rr * 8 + ff);
                            o.digits[d] = (free & s) ? 1 : (holes & s) ? 2 : (mine & s) ? 3 : (his & s) ? 4 : 0;
                        }
                        occ.push_back(o);
                    }
                }
            }
            if (k == n) break;
            const Move m(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k]));
            y.moves(moves);
            if (std::find(moves.begin(), moves.end(), m) == moves.end()) { ok = false; break; }
            y.play(m);
        }
        if (ok) for (const Occurrence& o : occ) f(o);
        used += ok && !occ.empty();
        i += 2 * n + 4;
    }
    return used;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { std::cerr << "usage: pattern_stats GAMES_DIR OUT.bin [--stride S] [--threads T]\n"; return 1; }
    magic::init();
    zobrist::init();
    size_t stride = 20;
    int threads = 16;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (string(argv[i]) == "--stride") stride = std::stoul(argv[i + 1]);
        else if (string(argv[i]) == "--threads") threads = std::stoi(argv[i + 1]);
    }
    vector<string> files;
    for (const auto& d : fs::directory_iterator(argv[1])) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());
    auto run = [&](auto&& body) {
        std::atomic<size_t> next{0};
        vector<std::thread> pool;
        for (int t = 0; t < threads; t++) pool.emplace_back([&] { body(next); });
        for (auto& t : pool) t.join();
    };
    std::mutex mtx;

    // 1. The baseline: the owner's mean result per (colour, side to move or not, free squares).
    double base_sum[2][2][65]{}, base_n[2][2][65]{};
    run([&](std::atomic<size_t>& next) {
        double s[2][2][65]{}, c[2][2][65]{};
        for (size_t j; (j = next++) < files.size();)
            for_each_game(files[j], stride, [&](const Occurrence& o) {
                s[o.colour][o.stm][o.free] += o.owner_outcome;
                c[o.colour][o.stm][o.free] += 1;
            });
        std::lock_guard lk(mtx);
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int f = 0; f < 65; f++) {
            base_sum[a][b][f] += s[a][b][f]; base_n[a][b][f] += c[a][b][f];
        }
    });
    double base[2][2][65]{};
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int f = 0; f < 65; f++)
        base[a][b][f] = base_n[a][b][f] ? base_sum[a][b][f] / base_n[a][b][f] : 0;

    // 2. The patterns.
    vector<uint32_t> count(size_t(PHASES) * NP);
    vector<double> sum(size_t(PHASES) * NP);
    uint64_t total[PHASES]{}, games = 0;
    double total_sq[PHASES]{};
    int pow6[NB];
    for (int d = 0, p = 1; d < NB; d++, p *= 6) pow6[d] = p;
    // canon[p]: the smallest number among the 8 symmetric versions of pattern p.
    static constexpr int DR[NB] = {1, 1, 1, 0, 0, -1, -1, -1}, DC[NB] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int perms[8][NB];
    for (int k = 0, t = 0; k < 4; k++)
        for (int flip = 0; flip < 2; flip++, t++)
            for (int d = 0; d < NB; d++) {
                int r = DR[d], c = flip ? -DC[d] : DC[d];
                for (int q = 0; q < k; q++) { const int r0 = r; r = c; c = -r0; }
                for (int e = 0; e < NB; e++) if (DR[e] == r && DC[e] == c) perms[t][d] = e;
            }
    vector<int> canon(NP);
    for (int p = 0; p < NP; p++) {
        int best = NP;
        for (int t = 0; t < 8; t++) {
            int q = 0;
            for (int d = 0; d < NB; d++) q += (p / pow6[d] % 6) * pow6[perms[t][d]];
            best = std::min(best, q);
        }
        canon[p] = best;
    }
    run([&](std::atomic<size_t>& next) {
        vector<uint32_t> c(size_t(PHASES) * NP);
        vector<double> s(size_t(PHASES) * NP);
        uint64_t t[PHASES]{};
        double tsq[PHASES]{};
        int idx[256];
        uint64_t g = 0;
        for (size_t j; (j = next++) < files.size();)
            g += for_each_game(files[j], stride, [&](const Occurrence& o) {
                const double adj = o.owner_outcome - base[o.colour][o.stm][o.free];
                const int ph = phase_of(o.free);
                t[ph]++;
                tsq[ph] += adj * adj;
                idx[0] = 0;
                for (int mask = 1; mask < 256; mask++) {
                    const int d = std::countr_zero(unsigned(mask));
                    idx[mask] = idx[mask & (mask - 1)] + o.digits[d] * pow6[d];
                }
                for (int mask = 0; mask < 256; mask++) idx[mask] = canon[idx[mask]];
                std::sort(idx, idx + 256);
                const size_t off = size_t(ph) * NP;
                for (int mask = 0; mask < 256; mask++) {
                    if (mask && idx[mask] == idx[mask - 1]) continue;
                    c[off + idx[mask]]++;
                    s[off + idx[mask]] += adj;
                }
            });
        std::lock_guard lk(mtx);
        for (size_t k = 0; k < c.size(); k++) { count[k] += c[k]; sum[k] += s[k]; }
        for (int p = 0; p < PHASES; p++) { total[p] += t[p]; total_sq[p] += tsq[p]; }
        games += g;
    });
    std::ofstream out(argv[2], std::ios::binary);
    out.write((const char*)count.data(), count.size() * sizeof(uint32_t));
    out.write((const char*)sum.data(), sum.size() * sizeof(double));
    out.write((const char*)total, sizeof(total));
    out.write((const char*)total_sq, sizeof(total_sq));
    out.write((const char*)&games, sizeof(games));
    std::printf("%zu files, every %zu-th position: %lu / %lu / %lu piece occurrences (free 40–56 / 20–39 / 0–19)\n",
                files.size(), stride, (unsigned long)total[0], (unsigned long)total[1], (unsigned long)total[2]);
    auto mean_of = [&](int colour) {
        double s = 0, n = 0;
        for (int b = 0; b < 2; b++) for (int f = 0; f < 65; f++) { s += base_sum[colour][b][f]; n += base_n[colour][b][f]; }
        return s / n;
    };
    std::printf("owner's mean result: black %+.3f, white %+.3f\n", mean_of(0), mean_of(1));
    return 0;
}
