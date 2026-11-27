// solo_retro — builds the table of private regions by retrograde analysis.
//
// Every single-region private cluster that fits on the board, with up to
// MAX_PIECES pieces of one side and up to MAX_FREE free squares, canonical
// (translation and the 8 symmetries), with its value: the largest number of
// moves the pieces can still make in it (player/solo_region.h).
//
// Retrograde, layer by layer: the layer n holds the states with n free squares.
// A move fills one free square, so the value of a state of layer n only needs
// states of the layers below (value = max over moves of 1 + the values of the
// clusters it leaves): layer n is computed once layers 1..n−1 are in the table.
// (A move can leave a cluster of two regions joined by a piece, not a state of
// the table: solo::Solver computes it, and keeps it, on the spot.)
//
//   solo_retro [MAX_FREE=9] [MAX_PIECES=2] [out file]   → table, and checks
//
// The shapes of layer n are grown from those of layer n − 1 (one more square
// next to the shape, at any of the four placements near the corner where the
// shape fits on the board), and deduplicated by their canonical form.
// Checks: a sample of the table against a fresh solver without table (same
// values). The file: count (u64), then per state free (u64), pieces (u64),
// value (i8), sorted by (free, pieces).
#include "solo_region.h"
#include "magic.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <unordered_set>
#include <vector>

namespace {
uint64_t grow(uint64_t b) {
    constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
    const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
    return row | (row << 8) | (row >> 8);
}
int width(uint64_t b) {
    uint64_t c = b; c |= c >> 32; c |= c >> 16; c |= c >> 8;
    c &= 0xFF;
    return c ? 64 - std::countl_zero(c) - std::countr_zero(c) : 0;
}
int height(uint64_t b) { return b ? (63 - std::countl_zero(b)) / 8 - std::countr_zero(b) / 8 + 1 : 0; }
// The placements of a corner shape one file and/or one rank away from the corner,
// where it still fits on the board: room for the squares around it.
std::vector<uint64_t> placements(uint64_t s) {
    std::vector<uint64_t> v{s};
    const bool right = width(s) < 8, up = height(s) < 8;
    if (right) v.push_back(s << 1);
    if (up) v.push_back(s << 8);
    if (right && up) v.push_back(s << 9);
    return v;
}
struct KH { size_t operator()(const solo::Key& k) const { return solo::KeyHash{}(k); } };
}

int main(int argc, char** argv) {
    magic::init();
    const int MAX_FREE = argc > 1 ? atoi(argv[1]) : 9;
    const int MAX_PIECES = argc > 2 ? atoi(argv[2]) : 2;
    const char* out = argc > 3 ? argv[3] : nullptr;
    solo::Solver solver;
    solver.keep_full = false;   // only the exceptions (value < free squares) are kept
    std::vector<solo::Key> table_keys;   // the exceptions, layer after layer
    long long nb_states = 0;
    std::vector<uint64_t> shapes = {1};   // layer 1: one square
    const auto t0 = std::chrono::steady_clock::now();
    for (int n = 1; n <= MAX_FREE; n++) {
        // The states of layer n: each shape, at each placement, with 1..MAX_PIECES
        // pieces on the board squares around it.
        std::unordered_set<solo::Key, KH> states;
        for (uint64_t sh : shapes)
            for (uint64_t f : placements(sh)) {
                std::vector<uint64_t> ring;
                for (uint64_t r = grow(f) & ~f; r; r &= r - 1) ring.push_back(r & -r);
                const int m = int(ring.size());
                for (int a = 0; a < m; a++) {
                    states.insert(solo::canonical(f, ring[a]));
                    if (MAX_PIECES >= 2)
                        for (int b = a + 1; b < m; b++) {
                            states.insert(solo::canonical(f, ring[a] | ring[b]));
                            if (MAX_PIECES >= 3)
                                for (int c = b + 1; c < m; c++) states.insert(solo::canonical(f, ring[a] | ring[b] | ring[c]));
                        }
                }
            }
        // Their values: every move leads to layers below, already solved.
        std::vector<solo::Key> layer(states.begin(), states.end());
        std::sort(layer.begin(), layer.end(), [](const solo::Key& x, const solo::Key& y) { return x.free != y.free ? x.free < y.free : x.pieces < y.pieces; });
        int vmax = 0;
        long long hist[64] = {};
        for (const solo::Key& k : layer) {
            const int v = solver.value(k.free, k.pieces);
            vmax = std::max(vmax, v);
            hist[v]++;
            if (v < n) table_keys.push_back(k);
        }
        nb_states += layer.size();
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("layer %2d: %8zu shapes, %10zu states, values ≤ %d (", n, shapes.size(), layer.size(), vmax);
        for (int v = 0; v <= vmax; v++) std::printf("%s%lld", v ? " " : "", hist[v]);
        std::printf("), exceptions so far %zu, %.1f s\n", table_keys.size(), el);
        std::fflush(stdout);
        if (n == MAX_FREE) break;
        // The shapes of layer n + 1.
        std::unordered_set<uint64_t> next;
        for (uint64_t sh : shapes)
            for (uint64_t f : placements(sh))
                for (uint64_t add = grow(f) & ~f; add; add &= add - 1) next.insert(solo::canonical(f | (add & -add), 0).free);
        shapes.assign(next.begin(), next.end());
    }
    // Check: a sample against a fresh solver (no table, no shared memory).
    std::mt19937_64 rng(1);
    int wrong = 0, checked = 0;
    for (int i = 0; i < 3000 && !table_keys.empty(); i++) {
        const solo::Key& k = table_keys[rng() % table_keys.size()];
        solo::Solver fresh;
        checked++;
        if (fresh.value(k.free, k.pieces) != solver.memo.at(k)) wrong++;
    }
    // and states of the last layers taken at random among all (full ones included)
    for (int i = 0; i < 2000 && !table_keys.empty(); i++) {
        const solo::Key& k = table_keys[rng() % table_keys.size()];
        // a neighbour state: the same shape with its pieces moved to another square around it
        uint64_t ring = grow(k.free) & ~k.free & ~k.pieces;
        if (!ring) continue;
        int skip = int(rng() % std::popcount(ring));
        while (skip--) ring &= ring - 1;
        const uint64_t p2 = (k.pieces & (k.pieces - 1)) | (ring & -ring);
        solo::Solver fresh, again;
        again.keep_full = false;
        checked++;
        if (fresh.value(k.free, p2) != solver.value(k.free, p2)) wrong++;
    }
    std::printf("%lld states, %zu exceptions (value below the free squares); check: %d states against a fresh solver, %d wrong\n",
                nb_states, table_keys.size(), checked, wrong);
    if (out) {
        std::ofstream f(out, std::ios::binary);
        const uint64_t count = table_keys.size();
        f.write(reinterpret_cast<const char*>(&count), 8);
        for (const solo::Key& k : table_keys) {
            const int8_t v = solver.memo.at(k);
            f.write(reinterpret_cast<const char*>(&k.free), 8);
            f.write(reinterpret_cast<const char*>(&k.pieces), 8);
            f.write(reinterpret_cast<const char*>(&v), 1);
        }
        std::printf("wrote %s: %llu states\n", out, (unsigned long long)count);
    }
    return wrong != 0;
}
