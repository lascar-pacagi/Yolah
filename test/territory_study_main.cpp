// territory_study_main.cpp — which measure of space predicts the rest of the
// game? (study of the user's two remarks: blocking the opponent's lines while
// keeping one's own seems strong; a flood fill can be misleading when a piece
// must choose between two blocks of free squares)
//
//   ./territory_study GAMES_DIR OUT.bin [--stride 50] [--threads 16] [--weights NNUE.quantized.txt]
//   python3 ../nnue/territory_study.py OUT.bin
//
// For every --stride-th position of every game of data/games (after the random
// opening moves; the first position taken depends on the game so that all
// plies are covered), seen from the side to move ("me") and the other side
// ("him"), with for each side:
//   mob       number of legal moves (queen moves to the first obstacle)
//   dirs      open directions: (piece, direction) pairs whose first square is
//             free — a move exists in that direction
//   trapped   pieces without any free neighbour
//   ff        flood fill: free squares reachable by king steps from one of
//             the side's pieces (heuristic::floodfill)
//   priv      the part of ff the other side cannot reach
//   corr      corrected flood fill: the free squares split into blocks
//             (connected components); a piece can only ever enter ONE of the
//             blocks that touch it, because the square it leaves becomes a
//             hole. corr = the largest total size of blocks the side's pieces
//             can occupy, each piece choosing one block (several pieces may
//             share one). ff counts every block touching any piece.
//   corrpriv  the same, counting only the blocks the other side cannot reach
//   infl      the engine's influence (heuristic::influence, king-step Voronoi
//             with neutral squares)
// NNUE (with --weights, a 193 network): its value for the side to move × 10000,
// to measure what each quantity adds to what the network already sees.
// And the targets: the result of the game for me (+1 / 0 / −1) and the points
// I score from here to the end minus his (final margin − current margin).
//
// OUT.bin: one record per position, NF int16 (see the Field enum).
#include "game.h"
#include "heuristic.h"
#include "magic.h"
#include "nnue_quantized.h"
#include "zobrist.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
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

enum Field { FREE, COLOUR, RESULT, FUTURE, CURRENT,
             MOB, DIRS, TRAPPED, FF, PRIV, CORR, CORRPRIV, INFL,                   // me
             H_MOB, H_DIRS, H_TRAPPED, H_FF, H_PRIV, H_CORR, H_CORRPRIV, H_INFL,   // him
             NNUE, NF };

struct Side { int mob, dirs, trapped, ff, priv, corr, corrpriv; };

// Blocks of free squares: connected components by king steps.
static int components(uint64_t free, uint64_t comps[64]) {
    int n = 0;
    while (free) {
        uint64_t c = free & -free, prev = 0;
        while (c != prev) { prev = c; c |= shift_all_directions(c) & free; }
        comps[n++] = c;
        free &= ~c;
    }
    return n;
}

// The largest total size of the blocks covered when each piece picks one of
// the blocks touching it (or none); allowed = the blocks it may pick.
static int best_cover(const vector<uint64_t>& options, size_t k, uint64_t covered,
                      const uint64_t* comps, const int* size) {
    if (k == options.size()) {
        int s = 0;
        for (uint64_t c = covered; c; c &= c - 1) s += size[std::countr_zero(c)];
        return s;
    }
    int best = best_cover(options, k + 1, covered, comps, size);
    for (uint64_t o = options[k] & ~covered; o; o &= o - 1)
        best = std::max(best, best_cover(options, k + 1, covered | (o & -o), comps, size));
    return best;
}

static Side side(const Yolah& y, uint8_t c, uint64_t free, const uint64_t* comps, const int* size, int nc) {
    Side s{};
    Yolah::MoveList ml;
    y.moves(c, ml);
    for (Move m : ml) s.mob += m != Move::none();
    const uint64_t mine = y.bitboard(c), his = y.bitboard(1 - c);
    const uint64_t near_his = shift_all_directions(his);
    uint64_t touch_mine = 0, touch_his = 0;   // block masks (bit i = block i)
    vector<uint64_t> all, priv;
    for (uint64_t b = mine; b; b &= b - 1) {
        const int sq = std::countr_zero(b);
        const uint64_t around = shift_all_directions(uint64_t(1) << sq);
        const int nfree = std::popcount(around & free);
        s.dirs += nfree;
        s.trapped += nfree == 0;
        uint64_t o = 0, p = 0;
        for (int i = 0; i < nc; i++)
            if (comps[i] & around) { o |= uint64_t(1) << i; if (!(comps[i] & near_his)) p |= uint64_t(1) << i; }
        all.push_back(o); priv.push_back(p);
        touch_mine |= o;
    }
    for (int i = 0; i < nc; i++) if (comps[i] & near_his) touch_his |= uint64_t(1) << i;
    for (uint64_t t = touch_mine; t; t &= t - 1) {
        const int i = std::countr_zero(t);
        s.ff += size[i];
        if (!(touch_his >> i & 1)) s.priv += size[i];
    }
    s.corr = best_cover(all, 0, 0, comps, size);
    s.corrpriv = best_cover(priv, 0, 0, comps, size);
    return s;
}

template <class F> static uint64_t for_each_game(const string& path, size_t stride, NNUE_Quantized* net, F&& f) {
    std::ifstream in(path, std::ios::binary);
    vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
    size_t game = 0;
    uint64_t used = 0;
    vector<std::array<int16_t, NF>> recs;
    for (size_t i = 0; i + 1 < e.size(); game++) {
        const int n = e[i], nb_random = e[i + 1];
        if (i + 2 * n + 4 > e.size()) break;
        const int black_score = e[i + 2 + 2 * n], white_score = e[i + 3 + 2 * n];
        Yolah y;
        Yolah::MoveList moves;
        recs.clear();
        bool ok = true;
        for (int k = 0; k <= n; k++) {
            if (k >= nb_random && (k + game) % stride == 0 && !y.game_over()) {
                const uint64_t free = y.free_squares();
                uint64_t comps[64];
                int size[64];
                const int nc = components(free, comps);
                for (int j = 0; j < nc; j++) size[j] = std::popcount(comps[j]);
                const uint8_t me = y.current_player(), him = 1 - me;
                const Side a = side(y, me, free, comps, size, nc), b = side(y, him, free, comps, size, nc);
                const auto [bi, wi] = heuristic::influence(y);
                const int final_margin = (black_score - white_score) * (me == Yolah::BLACK ? 1 : -1);
                const int current = y.score(me);
                std::array<int16_t, NF> r{};
                r[FREE] = int16_t(std::popcount(free));
                r[COLOUR] = me;
                r[RESULT] = int16_t(final_margin > 0 ? 1 : final_margin < 0 ? -1 : 0);
                r[FUTURE] = int16_t(final_margin - current);
                r[CURRENT] = int16_t(current);
                r[MOB] = a.mob; r[DIRS] = a.dirs; r[TRAPPED] = a.trapped; r[FF] = a.ff; r[PRIV] = a.priv;
                r[CORR] = a.corr; r[CORRPRIV] = a.corrpriv;
                r[H_MOB] = b.mob; r[H_DIRS] = b.dirs; r[H_TRAPPED] = b.trapped; r[H_FF] = b.ff; r[H_PRIV] = b.priv;
                r[H_CORR] = b.corr; r[H_CORRPRIV] = b.corrpriv;
                // influence includes the pieces themselves: count free squares only
                r[INFL] = int16_t(std::popcount((me == Yolah::BLACK ? bi : wi) & free));
                r[H_INFL] = int16_t(std::popcount((me == Yolah::BLACK ? wi : bi) & free));
                if (net) {
                    NNUE_Quantized::Accumulator acc;
                    net->init(y, acc);
                    r[NNUE] = int16_t(std::lround(10000 * net->value(acc, me)));
                }
                recs.push_back(r);
            }
            if (k == n) break;
            const Move m(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k]));
            y.moves(moves);
            if (std::find(moves.begin(), moves.end(), m) == moves.end()) { ok = false; break; }
            y.play(m);
        }
        if (ok) for (const auto& r : recs) f(r);
        used += ok && !recs.empty();
        i += 2 * n + 4;
    }
    return used;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { std::cerr << "usage: territory_study GAMES_DIR OUT.bin [--stride S] [--threads T]\n"; return 1; }
    magic::init();
    zobrist::init();
    size_t stride = 50;
    int threads = 16;
    string weights;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (string(argv[i]) == "--stride") stride = std::stoul(argv[i + 1]);
        else if (string(argv[i]) == "--weights") weights = argv[i + 1];
        else if (string(argv[i]) == "--threads") threads = std::stoi(argv[i + 1]);
    }
    vector<string> files;
    for (const auto& d : fs::directory_iterator(argv[1])) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());
    std::ofstream out(argv[2], std::ios::binary);
    std::mutex mtx;
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> games{0}, positions{0};
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&] {
            vector<std::array<int16_t, NF>> buf;
            std::unique_ptr<NNUE_Quantized> net;
            if (!weights.empty()) { net = std::make_unique<NNUE_Quantized>(); net->load(weights); }
            for (size_t j; (j = next++) < files.size();) {
                buf.clear();
                games += for_each_game(files[j], stride, net.get(), [&](const std::array<int16_t, NF>& r) { buf.push_back(r); });
                positions += buf.size();
                std::lock_guard lk(mtx);
                out.write((const char*)buf.data(), buf.size() * sizeof(buf[0]));
            }
        });
    for (auto& t : pool) t.join();
    std::printf("%zu files, %lu games, %lu positions (every %zu-th), %d int16 per record → %s\n", files.size(),
                (unsigned long)games, (unsigned long)positions, stride, int(NF), argv[2]);
    return 0;
}
