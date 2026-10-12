// private_moves_main.cpp — how many moves could be cut by the private regions?
// (study of "pieces that do not need to move")
//
//   ./private_moves CACHE_DIR [--stride S]
//
// In Yolah the origin of a move becomes a HOLE: borders never reopen. The
// free squares split into 8-connected components (a queen slides step by step
// through free squares, so a piece can only ever reach the components it
// touches). A component touched only by the side to move's pieces is PRIVATE,
// for good: the opponent can never enter it. A piece of the side to move is
//     enclosed   if every component it touches is private: its moves change
//                nothing for the opponent nor for the contested regions, and
//                can be postponed — all but ONE (a waiting move, for zugzwangs)
//     mixed      if it touches both private and contested components (moving
//                into its private region gives up its access to the contested one)
//     contested  if it touches only contested components
// Per range of free squares, over the positions of the cache: the moves of
// the side to move by kind, the branching factor once the moves of the
// enclosed pieces are replaced by one, and how often the board is fully
// separated (no contested component: the value is then exact).
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include <bit>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::string, std::vector;

static uint64_t bits(const uint8_t* row) {
    uint64_t b = 0;
    for (int i = 0; i < 64; i++) b = (b << 1) | (row[i] & 1);
    return b;
}

static uint64_t flood(uint64_t seed, uint64_t allowed) {
    uint64_t f = seed, prev = 0;
    while (f != prev) { prev = f; f |= shift_all_directions(f) & allowed; }
    return f;
}

struct Stats {
    uint64_t positions = 0, with_enclosed = 0, separated = 0, no_contested_piece = 0;
    uint64_t moves = 0, enclosed = 0, mixed_private = 0, mixed_contested = 0, contested = 0, after_cut = 0;
    double ratio_sum = 0;          // mean over positions of (moves after the cut) / moves
};

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "usage: private_moves CACHE_DIR [--stride S]\n"; return 1; }
    magic::init();
    zobrist::init();
    const string cache = argv[1];
    size_t stride = 1;
    for (int i = 2; i + 1 < argc; i++) if (string(argv[i]) == "--stride") stride = std::stoul(argv[i + 1]);
    std::ifstream in(cache + "/inputs.u8", std::ios::binary | std::ios::ate);
    const size_t n_all = size_t(in.tellg()) / 193;
    in.seekg(0);
    constexpr int BIN = 4;
    vector<Stats> st(64 / BIN + 1);
    vector<uint8_t> row(193);
    for (size_t i = 0; i < n_all; i++) {
        in.read((char*)row.data(), 193);
        if (i % stride) continue;
        const uint64_t black = bits(&row[0]), white = bits(&row[64]), empty = bits(&row[128]);
        const int turn = row[192], holes = std::popcount(empty);
        Yolah y;
        y.set_state(black, white, empty, 0, 0, uint16_t(holes + ((holes & 1) != turn)));
        if (y.game_over()) continue;
        Yolah::MoveList moves;
        y.moves(moves);
        if (moves.size() == 0 || moves[0] == Move::none()) continue;     // must pass
        const uint64_t free = y.free_squares();
        const uint64_t mine = turn == 0 ? black : white, his = turn == 0 ? white : black;
        // The components and the private ones (touched by none of his pieces).
        uint64_t private_sq = 0, contested_sq = 0;
        for (uint64_t rest = free; rest;) {
            const uint64_t comp = flood(rest & -rest, free);
            rest &= ~comp;
            const uint64_t touch = shift_all_directions(comp);
            if (touch & his) { if (touch & mine) contested_sq |= comp; }   // his only: irrelevant here
            else private_sq |= comp;
        }
        Stats& s = st[std::popcount(free) / BIN];
        s.positions++;
        uint64_t n_enclosed = 0, n_other = 0;
        bool contested_piece = false;
        for (const Move& m : moves) {
            const uint64_t from = square_bb(m.from_sq()), to = square_bb(m.to_sq());
            const uint64_t adj = shift_all_directions(from) & free;
            const bool touches_private = adj & private_sq, touches_contested = adj & contested_sq;
            if (!touches_contested) { s.enclosed++; n_enclosed++; continue; }
            contested_piece = true;
            n_other++;
            if (touches_private) { if (to & private_sq) s.mixed_private++; else s.mixed_contested++; }
            else s.contested++;
        }
        const uint64_t after = n_other + (n_enclosed ? 1 : 0);
        s.moves += moves.size();
        s.after_cut += after;
        s.ratio_sum += double(after) / moves.size();
        s.with_enclosed += n_enclosed > 0;
        s.separated += contested_sq == 0;
        s.no_contested_piece += !contested_piece;
    }
    std::printf("positions of %s (every %zu-th), side to move's moves:\n", cache.c_str(), stride);
    std::printf("%6s %9s %7s %7s %9s %9s %9s %9s %9s %9s\n", "free", "positions", "moves", "after", "cut %",
                "enclosed", "mixed→pr", "pos.encl", "separated", "all encl");
    std::printf("%6s %9s %7s %7s %9s %9s %9s %9s %9s %9s\n", "", "", "mean", "mean", "(mean)", "% moves",
                "% moves", "% pos", "% pos", "% pos");
    for (size_t b = 0; b < st.size(); b++) {
        const Stats& s = st[b];
        if (!s.positions) continue;
        const double p = s.positions;
        std::printf("%2zu–%-3zu %9lu %7.1f %7.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", b * BIN, b * BIN + BIN - 1,
                    (unsigned long)s.positions, s.moves / p, s.after_cut / p, 100 * (1 - s.ratio_sum / p),
                    100.0 * s.enclosed / s.moves, 100.0 * s.mixed_private / s.moves, 100.0 * s.with_enclosed / p,
                    100.0 * s.separated / p, 100.0 * s.no_contested_piece / p);
    }
    return 0;
}
