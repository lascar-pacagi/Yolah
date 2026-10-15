#ifndef SPACE_FEATURES_H
#define SPACE_FEATURES_H
#include <algorithm>
#include <bit>
#include <cstdint>
#include "magic.h"
#include "types.h"

// Space features: extra inputs of the second layer of the NNUE (NetPersp with
// a "space mask", nnue/nnue_persp.py), chosen by test/territory_study_main.cpp
// (what each measure adds to the network's own value: the corrected flood
// fill and the queen distance, then the zones reached late and the mobility).
//
// 8 features per side, the side to move first ("us", ids 0–7), then the other
// side ("them", ids 8–15); each an integer clamped to [0, 64], given to the
// network as value / 64 — in C++ the int8 at scale 64 IS the integer, so the
// quantized network sees exactly what the float one was trained on.
//   0 CORR    corrected flood fill: the free squares (king-step blocks) the
//             side's pieces can occupy, each piece entering ONE of the blocks
//             touching it (the square it leaves becomes a hole)
//   1 Q1      free squares within 1 queen move (sliding through free squares)
//   2 Q2      ... within 2 queen moves
//   3 Q3      ... within 3 queen moves
//   4 QFIRST  free squares reached in strictly fewer queen moves than the other side
//   5 QW      Σ over the reached free squares of 2^−(distance − 1), rounded down
//   6 LATE9   free squares of the zones (3 × 3 grid, files and ranks 3/2/3) the
//             other side reaches first (fewer queen moves to the zone's nearest square)
//   7 MOB     number of legal moves
// All are invariant under the a <-> h mirror of the white view.
namespace space {

constexpr int NB = 16;
constexpr int PER_SIDE = 8;
enum Id { CORR, Q1, Q2, Q3, QFIRST, QW, LATE9, MOB };
constexpr uint32_t ALL = 0xFFFF;
constexpr uint32_t NO_CORR = ALL & ~((1u << CORR) | (1u << (PER_SIDE + CORR)));
inline const char* name(int id) {
    static const char* names[PER_SIDE] = {"corr", "q1", "q2", "q3", "qfirst", "qw", "late9", "mob"};
    return names[id % PER_SIDE];
}

// Every free square one queen move away from a square of gen, in direction D
// (occluded fill, Kogge-Stone: 3 doubling steps, the whole set at once).
template <Direction D> inline uint64_t slide(uint64_t gen, uint64_t free) {
    uint64_t pro = free;
    gen |= pro & shift<D>(gen);
    pro &= shift<D>(pro);
    gen |= pro & shift<D>(shift<D>(gen));
    pro &= shift<D>(shift<D>(pro));
    gen |= pro & shift<D>(shift<D>(shift<D>(shift<D>(gen))));
    return shift<D>(gen) & free;
}

inline uint64_t queen_reach(uint64_t from, uint64_t free) {
    return slide<NORTH>(from, free) | slide<SOUTH>(from, free) | slide<EAST>(from, free) |
           slide<WEST>(from, free) | slide<NORTH_EAST>(from, free) | slide<NORTH_WEST>(from, free) |
           slide<SOUTH_EAST>(from, free) | slide<SOUTH_WEST>(from, free);
}

// levels[k] = the free squares at queen distance k + 1; returns the number of levels.
inline int queen_levels(uint64_t pieces, uint64_t free, uint64_t levels[64]) {
    int n = 0;
    uint64_t seen = 0, frontier = pieces;
    for (;;) {
        const uint64_t next = queen_reach(frontier, free) & ~seen;
        if (!next) return n;
        levels[n++] = next;
        seen |= next;
        frontier = next;
    }
}

// Corrected flood fill of `pieces`: the largest total size of the blocks they
// can occupy, each piece choosing one block touching it (or none).
inline int corrected_flood_fill(uint64_t pieces, uint64_t free, const uint64_t* comps, const int* size, int nc) {
    uint64_t options[8];
    int np = 0;
    for (uint64_t b = pieces; b && np < 8; b &= b - 1) {
        const uint64_t around = shift_all_directions(b & -b);
        uint64_t o = 0;
        for (int i = 0; i < nc; i++) if (comps[i] & around) o |= uint64_t(1) << i;
        options[np++] = o;
    }
    // depth-first over the pieces' choices (at most 4 pieces, a few blocks each)
    int best = 0;
    auto rec = [&](auto&& self, int k, uint64_t covered) -> void {
        if (k == np) {
            int s = 0;
            for (uint64_t c = covered; c; c &= c - 1) s += size[std::countr_zero(c)];
            best = std::max(best, s);
            return;
        }
        self(self, k + 1, covered);
        for (uint64_t o = options[k] & ~covered; o; o &= o - 1) self(self, k + 1, covered | (o & -o));
    };
    rec(rec, 0, 0);
    return best;
}

inline const uint64_t* zones9() {
    static const struct Z {
        uint64_t z[9]{};
        Z() {
            const int cut[4] = {0, 3, 5, 8};
            for (int zr = 0; zr < 3; zr++) for (int zc = 0; zc < 3; zc++)
                for (int r = cut[zr]; r < cut[zr + 1]; r++) for (int c = cut[zc]; c < cut[zc + 1]; c++)
                    z[zr * 3 + zc] |= uint64_t(1) << (r * 8 + c);
        }
    } zs;
    return zs.z;
}

// The 16 features of the position seen from `us` (the side to move): out[0..7]
// for us, out[8..15] for them. `mask`: the features needed (bit = id); the
// others are left at 0 (the corrected flood fill, the costliest, is skipped
// when neither CORR bit is set).
inline void compute(uint64_t us, uint64_t them, uint64_t free, uint8_t out[NB], uint32_t mask = ALL) {
    for (int i = 0; i < NB; i++) out[i] = 0;
    const uint64_t pcs[2] = { us, them };
    auto put = [&](int side, int id, int v) { out[side * PER_SIDE + id] = uint8_t(std::clamp(v, 0, 64)); };
    if (mask & ((1u << CORR) | (1u << (PER_SIDE + CORR)))) {
        uint64_t comps[64];
        int size[64], nc = 0;
        for (uint64_t rest = free; rest;) {
            uint64_t c = rest & -rest, prev = 0;
            while (c != prev) { prev = c; c |= shift_all_directions(c) & free; }
            comps[nc] = c;
            size[nc++] = std::popcount(c);
            rest &= ~c;
        }
        for (int sd = 0; sd < 2; sd++) put(sd, CORR, corrected_flood_fill(pcs[sd], free, comps, size, nc));
    }
    uint64_t lv[2][64];
    const int nl[2] = { queen_levels(us, free, lv[0]), queen_levels(them, free, lv[1]) };
    for (int sd = 0; sd < 2; sd++) {
        uint64_t cover = 0;
        int w = 0;
        int q[3] = {0, 0, 0};
        for (int k = 0; k < nl[sd]; k++) {
            cover |= lv[sd][k];
            if (k < 3) q[k] = std::popcount(cover);
            if (k < 6) w += std::popcount(lv[sd][k]) << (5 - k);
        }
        for (int k = std::max(nl[sd], 1); k < 3; k++) q[k] = q[k - 1];
        put(sd, Q1, q[0]); put(sd, Q2, q[1]); put(sd, Q3, q[2]);
        put(sd, QW, w >> 5);
        int mob = 0;
        for (uint64_t b = pcs[sd]; b; b &= b - 1)
            mob += std::popcount(attacks_bb(Square(std::countr_zero(b)), ~free) & free);
        put(sd, MOB, mob);
    }
    // reached first
    {
        uint64_t seen0 = 0, seen1 = 0;
        int first0 = 0, first1 = 0;
        for (int k = 0; k < std::max(nl[0], nl[1]); k++) {
            const uint64_t a = k < nl[0] ? lv[0][k] : 0, b = k < nl[1] ? lv[1][k] : 0;
            first0 += std::popcount(a & ~b & ~seen1);
            first1 += std::popcount(b & ~a & ~seen0);
            seen0 |= a; seen1 |= b;
        }
        put(0, QFIRST, first0); put(1, QFIRST, first1);
    }
    // zones reached late
    {
        const uint64_t* z9 = zones9();
        auto first_level = [&](int sd, uint64_t zone) {
            for (int k = 0; k < nl[sd]; k++) if (lv[sd][k] & zone) return k;
            return 99;
        };
        int late[2] = {0, 0};
        for (int i = 0; i < 9; i++) {
            const uint64_t z = z9[i] & free;
            if (!z) continue;
            const int f0 = first_level(0, z), f1 = first_level(1, z);
            if (f1 < f0) late[0] += std::popcount(z);
            if (f0 < f1) late[1] += std::popcount(z);
        }
        put(0, LATE9, late[0]); put(1, LATE9, late[1]);
    }
}

} // namespace space

#endif
