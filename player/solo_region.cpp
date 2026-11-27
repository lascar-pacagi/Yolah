#include "solo_region.h"
#include "magic.h"
#include <algorithm>
#include <bit>

namespace solo {

namespace {
constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
// b and its 8 neighbours.
inline uint64_t grow(uint64_t b) {
    const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
    return row | (row << 8) | (row >> 8);
}
// The 8 symmetries of the board, from three involutions.
inline uint64_t flip_vertical(uint64_t b) { return __builtin_bswap64(b); }   // rank 1 ↔ rank 8
inline uint64_t mirror_horizontal(uint64_t b) {                              // file a ↔ file h
    constexpr uint64_t k1 = 0x5555555555555555ULL, k2 = 0x3333333333333333ULL, k4 = 0x0f0f0f0f0f0f0f0fULL;
    b = ((b >> 1) & k1) | ((b & k1) << 1);
    b = ((b >> 2) & k2) | ((b & k2) << 2);
    b = ((b >> 4) & k4) | ((b & k4) << 4);
    return b;
}
inline uint64_t flip_diagonal(uint64_t b) {                                  // a1-h8 diagonal: file ↔ rank
    uint64_t t;
    constexpr uint64_t k1 = 0x5500550055005500ULL, k2 = 0x3333000033330000ULL, k4 = 0x0f0f0f0f00000000ULL;
    t = k4 & (b ^ (b << 28)); b ^= t ^ (t >> 28);
    t = k2 & (b ^ (b << 14)); b ^= t ^ (t >> 14);
    t = k1 & (b ^ (b << 7));  b ^= t ^ (t >> 7);
    return b;
}
// Translated so that the shape touches rank 1 and file a.
inline void to_corner(uint64_t& f, uint64_t& p) {
    const uint64_t all = f | p;
    if (!all) return;
    const int rank = std::countr_zero(all) / 8;
    f >>= 8 * rank;
    p >>= 8 * rank;
    uint64_t cols = (f | p);
    cols |= cols >> 32; cols |= cols >> 16; cols |= cols >> 8;   // the files used, in the low byte
    const int file = std::countr_zero(cols & 0xFF);
    // Shifting right by `file` moves every square `file` files to the left; no
    // square crosses to the previous rank since no square is left of `file`.
    f >>= file;
    p >>= file;
}
}  // namespace

Key canonical(uint64_t free, uint64_t pieces) {
    Key best{~uint64_t(0), ~uint64_t(0)};
    uint64_t f = free, p = pieces;
    for (int d = 0; d < 2; d++) {
        for (int v = 0; v < 2; v++) {
            for (int h = 0; h < 2; h++) {
                uint64_t ff = f, pp = p;
                to_corner(ff, pp);
                if (ff < best.free || (ff == best.free && pp < best.pieces)) best = {ff, pp};
                f = mirror_horizontal(f);
                p = mirror_horizontal(p);
            }
            f = flip_vertical(f);
            p = flip_vertical(p);
        }
        f = flip_diagonal(f);
        p = flip_diagonal(p);
    }
    return best;
}

int regions(uint64_t free, uint64_t out[64]) {
    int n = 0;
    while (free) {
        uint64_t r = free & -free, prev = 0;
        while (r != prev) { prev = r; r = grow(r) & free; }
        out[n++] = r;
        free &= ~r;
    }
    return n;
}

int clusters(uint64_t free, uint64_t pieces, Cluster out[64]) {
    uint64_t reg[64];
    const int nr = regions(free, reg);
    // Union-find over the regions: a piece touching several joins them.
    int parent[64];
    for (int i = 0; i < nr; i++) parent[i] = i;
    auto find = [&](int i) { while (parent[i] != i) i = parent[i] = parent[parent[i]]; return i; };
    uint64_t touch[64];
    for (int i = 0; i < nr; i++) touch[i] = grow(reg[i]);
    for (uint64_t ps = pieces; ps; ps &= ps - 1) {
        const uint64_t bit = ps & -ps;
        int first = -1;
        for (int i = 0; i < nr; i++)
            if (touch[i] & bit) {
                if (first < 0) first = i;
                else parent[find(i)] = find(first);
            }
    }
    int n = 0, index[64];
    for (int i = 0; i < nr; i++) index[i] = -1;
    for (int i = 0; i < nr; i++) {
        const int r = find(i);
        if (index[r] < 0) { index[r] = n; out[n++] = Cluster{}; }
        out[index[r]].free |= reg[i];
    }
    // The pieces of each cluster (those touching it); clusters without a piece are dropped.
    int m = 0;
    for (int c = 0; c < n; c++) {
        out[c].pieces = grow(out[c].free) & pieces;
        if (out[c].pieces) out[m++] = out[c];
    }
    return m;
}

Split split(const Yolah& y) {
    Split s;
    uint64_t reg[64];
    const uint64_t free = y.free_squares();
    const int nr = regions(free, reg);
    uint64_t priv_free[2] = {0, 0};
    for (int i = 0; i < nr; i++) {
        const uint64_t t = grow(reg[i]);
        const bool b = t & y.bitboard(Yolah::BLACK), w = t & y.bitboard(Yolah::WHITE);
        if (b && w) s.contested |= reg[i];
        else if (b) priv_free[0] |= reg[i];
        else if (w) priv_free[1] |= reg[i];
    }
    for (int side = 0; side < 2; side++) {
        // A piece of this side touching a private region and a contested one links them:
        // the private region is then not independent (the piece may go either way). Such
        // regions are left in no cluster here.
        Cluster c[64];
        const int n = clusters(priv_free[side], y.bitboard(side), c);
        for (int i = 0; i < n; i++)
            if (!(grow(s.contested) & c[i].pieces)) s.priv[side].push_back(c[i]);
    }
    return s;
}

int Solver::value(uint64_t free, uint64_t pieces) {
    Cluster c[64];
    const int n = clusters(free, pieces, c);
    int v = 0;
    for (int i = 0; i < n; i++) v += cluster_value(c[i].free, c[i].pieces);
    return v;
}

int Solver::cluster_value(uint64_t free, uint64_t pieces) {
    Raw& r = raw[(free * 0x9E3779B97F4A7C15ull ^ pieces * 0xC2B2AE3D27D4EB4Full) >> (64 - RAW_BITS)];
    if (r.value >= 0 && r.free == free && r.pieces == pieces) return r.value;
    const int v = [&]() -> int {
    const Key k = canonical(free, pieces);
    if (auto it = memo.find(k); it != memo.end()) return it->second;
    ++nb_nodes;
    if (over || (budget_left >= 0 && --budget_left < 0)) { over = true; return 0; }
    const uint64_t occupied = ~free;
    const int most = std::popcount(free);   // one move per free square at most
    int best = 0;
    for (uint64_t ps = pieces; ps && best < most && !over; ps &= ps - 1) {
        const Square from = Square(std::countr_zero(ps));
        const uint64_t from_bit = ps & -ps;
        for (uint64_t to = attacks_bb(from, occupied) & free; to && best < most && !over; to &= to - 1) {
            const uint64_t to_bit = to & -to;
            // The destination is no longer free; the origin becomes a hole.
            best = std::max(best, 1 + value(free & ~to_bit, (pieces & ~from_bit) | to_bit));
        }
    }
    if (over) return 0;   // unfinished: kept nowhere
    if (keep_full || best < most) memo.emplace(k, int8_t(best));
    return best;
    }();
    if (over) return 0;
    r = {free, pieces, int8_t(v)};
    return v;
}

}  // namespace solo
