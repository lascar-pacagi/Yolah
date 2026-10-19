#ifndef SPACE_FEATURES_H
#define SPACE_FEATURES_H
#include <algorithm>
#include <array>
#include <tuple>
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
    if (!(mask & ~((1u << CORR) | (1u << (PER_SIDE + CORR))))) return;    // no feature of the queen levels
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

// ── Every candidate feature, for the ablation (doc/nnue_roadmap.qmd) ─────────
// NB_ALL columns, integers in [0, 64] like the 16 above, which are columns
// 0–15 unchanged. Each GROUP is one feature of the ablation (removed whole):
//   G_CORR      CORR                     0, 8
//   G_QDIST     Q1, Q2, Q3               1–3, 9–11
//   G_QFIRST    QFIRST                   4, 12
//   G_QW        QW                       5, 13
//   G_LATE9     LATE9                    6, 14
//   G_MOB       MOB                      7, 15
//   G_INFL      influence (heuristic::influence): free squares reached first
//               by king steps, both sides spreading at once; squares reached
//               together are neutral and spread too                         16, 17
//   G_CONNECT   per piece, each side's pieces most isolated first (us 18–37,
//               them 38–57), 5 numbers each:
//                 block   free squares it reaches by king steps
//                 contact 16 × the other side's pieces in those blocks
//                 king    64 / king steps to land next to one of them (0: never)
//                 go      64 / queen moves to land next to one of them
//                 come    64 / queen moves for one of them to land next to it
//   G_ENCIRCLE  per side (us 58–74, them 75–91), the encirclement of ITS pieces
//               by the other side's: 9 sums over its 4 pieces
//                 gene    queen moves the other side's pieces take from them
//                 contrib 4 × other-side pieces stopping one of their lines first
//                 opp     4 × neighbours that are other-side pieces
//                 att     2 × liberties the other side reaches in one queen move
//                 lib     2 × liberties (free neighbours)
//                 short   2 × lines with only 1 or 2 free squares
//                 tight   16 × pieces with at most 1 liberty
//                 shut    16 × pieces with no liberty
//                 close   16 × pieces with 1 liberty the other side reaches
//               then, its pieces most encircled first, 8 × att and 64 × close
// The 1 / distance and the multipliers keep every column in [0, 64]; the
// orders break ties on all the numbers, so the columns stay invariant under
// the a <-> h mirror.
enum Group { G_CORR, G_QDIST, G_QFIRST, G_QW, G_LATE9, G_MOB, G_INFL, G_CONNECT, G_ENCIRCLE, NB_GROUPS };
constexpr int INFL0 = 16, CONNECT0 = 18, ENCIRCLE0 = 58, NB_ALL = 92;
constexpr int CONNECT_PER_PIECE = 5, ENCIRCLE_PER_SIDE = 17;

inline const char* group_name(int g) {
    static const char* names[NB_GROUPS] = {"corr", "qdist", "qfirst", "qw", "late9", "mob", "infl", "connect", "encircle"};
    return names[g];
}

inline int group_of(int col) {
    if (col >= ENCIRCLE0) return G_ENCIRCLE;
    if (col >= CONNECT0) return G_CONNECT;
    if (col >= INFL0) return G_INFL;
    switch (col % PER_SIDE) {
        case CORR: return G_CORR;
        case Q1: case Q2: case Q3: return G_QDIST;
        case QFIRST: return G_QFIRST;
        case QW: return G_QW;
        case LATE9: return G_LATE9;
        default: return G_MOB;
    }
}

// Influence, as the player computes it (heuristic::influence): both sides
// spread one king step at a time through the free squares; a square reached
// by both at once is neutral, and the neutral squares spread too, blocking
// both. Returns the free squares each side ends up owning.
inline void influence(uint64_t us, uint64_t them, uint64_t free, int out[2]) {
    auto step = [&](uint64_t b) { return shift_all_directions(b) & free; };
    uint64_t own[2] = {us, them}, front[2] = {us, them}, neutral = 0, prev[2] = {0, 0};
    while (prev[0] != own[0] || prev[1] != own[1]) {
        front[0] = step(front[0]) & ~own[1];
        front[1] = step(front[1]) & ~own[0];
        neutral |= step(neutral) | (front[0] & front[1]);
        front[0] &= ~neutral; front[1] &= ~neutral;
        prev[0] = own[0]; prev[1] = own[1];
        own[0] |= front[0]; own[1] |= front[1];
    }
    out[0] = std::popcount(own[0] & free);
    out[1] = std::popcount(own[1] & free);
}

// The 8 lines of a piece: SLIDES[d](b, free) = the free squares it reaches in direction d.
inline uint64_t (*const SLIDES[8])(uint64_t, uint64_t) = {
    slide<NORTH>, slide<SOUTH>, slide<EAST>, slide<WEST>,
    slide<NORTH_EAST>, slide<NORTH_WEST>, slide<SOUTH_EAST>, slide<SOUTH_WEST>};

inline int inverse64(int d) { return d ? (64 + d / 2) / d : 0; }     // 64 / d rounded, 0 = never

// Connectivity of one piece with the other side (`opp`): the 5 numbers above.
inline std::array<int, CONNECT_PER_PIECE> connect_piece(uint64_t piece, uint64_t opp, uint64_t free,
                                                         uint64_t opp_reach_levels[], int opp_nl) {
    const uint64_t near_opp = shift_all_directions(opp);
    // king steps through the free squares
    uint64_t seen = 0, front = piece;
    int king = 0;
    for (int d = 1; (front = shift_all_directions(front) & free & ~seen); d++) {
        if (!king && (front & near_opp)) king = d;
        seen |= front;
    }
    int contact = 0;
    for (uint64_t b = opp; b; b &= b - 1) contact += (shift_all_directions(b & -b) & seen) != 0;
    // queen moves, going (from the piece) and coming (from the other side's pieces)
    uint64_t lv[64];
    const int nl = queen_levels(piece, free, lv);
    int go = 0, come = 0;
    for (int k = 0; k < nl && !go; k++) if (lv[k] & near_opp) go = k + 1;
    const uint64_t near_me = shift_all_directions(piece);
    for (int k = 0; k < opp_nl && !come; k++) if (opp_reach_levels[k] & near_me) come = k + 1;
    return {std::popcount(seen), 16 * contact, inverse64(king), inverse64(go), inverse64(come)};
}

// The 8 rays of each square on an empty board: RAYS[sq][d] = SLIDES[d] with every square free.
inline const auto& rays() {
    static const struct R {
        uint64_t r[64][8];
        R() { for (int sq = 0; sq < 64; sq++) for (int d = 0; d < 8; d++) r[sq][d] = SLIDES[d](uint64_t(1) << sq, ~uint64_t(0)); }
    } rs;
    return rs.r;
}

// The free squares the queens on `pieces` reach in one move (= queen_reach, by magic lookups).
inline uint64_t queen_moves(uint64_t pieces, uint64_t free) {
    uint64_t r = 0;
    for (uint64_t b = pieces; b; b &= b - 1) r |= attacks_bb(Square(std::countr_zero(b)), ~free);
    return r & free;
}

// Encirclement of one piece by the other side (`opp`): lib, opp, short, gene, contrib, att.
struct Encircle { int lib, opp, shrt, gene, contrib, att; };
inline Encircle encircle_piece(uint64_t piece, uint64_t opp, uint64_t free, uint64_t opp_reach1) {
    const uint64_t around = shift_all_directions(piece);
    Encircle e{std::popcount(around & free), std::popcount(around & opp), 0, 0, 0,
               std::popcount(around & free & opp_reach1)};
    // a queen on the piece's square: its moves, then the same with his pieces as free squares,
    // each cut into its 8 lines by the rays
    const Square sq = Square(std::countr_zero(piece));
    const uint64_t all_real = queen_moves(piece, free), all_freed = queen_moves(piece, free | opp);
    for (const uint64_t ray : rays()[sq]) {
        const uint64_t real = all_real & ray, freed = all_freed & ray;
        const int len = std::popcount(real);
        e.shrt += len == 1 || len == 2;
        e.gene += std::popcount(freed) - len;       // the squares they take from it, theirs included
        e.contrib += (freed & opp) != 0;            // the first obstacle of this line is one of them
    }
    return e;
}

constexpr uint32_t ALL_GROUPS = (1u << NB_GROUPS) - 1;

// The ids of the 16 first columns (compute's mask) that belong to `groups`.
inline uint32_t ids_of_groups(uint32_t groups) {
    uint32_t m = 0;
    for (int i = 0; i < NB; i++) if (groups >> group_of(i) & 1) m |= 1u << i;
    return m;
}

// The NB_ALL columns of the position seen from `us` (the side to move), only
// those of `groups` (bit = Group) computed, the others 0: what a network
// reading some groups needs (NNUE_Persp::value), at the cost of those only.
inline void compute_groups(uint64_t us, uint64_t them, uint64_t free, uint8_t out[NB_ALL], uint32_t groups) {
    compute(us, them, free, out, ids_of_groups(groups));
    for (int i = NB; i < NB_ALL; i++) out[i] = 0;
    auto put = [&](int col, int v) { out[col] = uint8_t(std::clamp(v, 0, 64)); };
    const uint64_t pcs[2] = {us, them};
    if (groups >> G_INFL & 1) {
        int infl[2];
        influence(us, them, free, infl);
        put(INFL0, infl[0]); put(INFL0 + 1, infl[1]);
    }
    const bool connect = groups >> G_CONNECT & 1, encircle = groups >> G_ENCIRCLE & 1;
    if (!connect && !encircle) return;
    uint64_t lv[2][64];
    int nl[2] = {0, 0};
    if (connect) { nl[0] = queen_levels(us, free, lv[0]); nl[1] = queen_levels(them, free, lv[1]); }
    for (int sd = 0; sd < 2; sd++) {
        const uint64_t mine = pcs[sd], opp = pcs[1 - sd];
        int n = 0;
        if (connect) {
            // connectivity, most isolated first (fewest contacts, smallest block, then the rest)
            std::array<std::array<int, CONNECT_PER_PIECE>, 4> c{};
            for (uint64_t b = mine; b && n < 4; b &= b - 1) c[n++] = connect_piece(b & -b, opp, free, lv[1 - sd], nl[1 - sd]);
            std::sort(c.begin(), c.begin() + n, [](const auto& x, const auto& y) {
                return std::tie(x[1], x[0], x[2], x[3], x[4]) < std::tie(y[1], y[0], y[2], y[3], y[4]); });
            for (int i = 0; i < n; i++)
                for (int j = 0; j < CONNECT_PER_PIECE; j++) put(CONNECT0 + (sd * 4 + i) * CONNECT_PER_PIECE + j, c[i][j]);
        }
        if (encircle) {
            // encirclement of my pieces by his (queen_reach = his first queen level)
            std::array<Encircle, 4> e{};
            const uint64_t reach1 = queen_moves(opp, free);
            n = 0;
            for (uint64_t b = mine; b && n < 4; b &= b - 1) e[n++] = encircle_piece(b & -b, opp, free, reach1);
            int sum[9] = {};
            for (int i = 0; i < n; i++) {
                const Encircle& x = e[i];
                const int v[9] = {x.gene, x.contrib, x.opp, x.att, x.lib, x.shrt, x.lib <= 1, x.lib == 0, x.lib == 1 && x.att};
                for (int j = 0; j < 9; j++) sum[j] += v[j];
            }
            const int mult[9] = {1, 4, 4, 2, 2, 2, 16, 16, 16};
            const int base = ENCIRCLE0 + sd * ENCIRCLE_PER_SIDE;
            for (int j = 0; j < 9; j++) put(base + j, mult[j] * sum[j]);
            // most encircled first (fewest liberties, most gene, most attacked liberties)
            std::sort(e.begin(), e.begin() + n, [](const Encircle& x, const Encircle& y) {
                return std::tie(x.lib, y.gene, y.att) < std::tie(y.lib, x.gene, x.att); });
            for (int i = 0; i < n; i++) {
                put(base + 9 + 2 * i, 8 * e[i].att);
                put(base + 10 + 2 * i, 64 * (e[i].lib == 1 && e[i].att > 0));
            }
        }
    }
}

// All NB_ALL columns of the position seen from `us` (the side to move).
inline void compute_all(uint64_t us, uint64_t them, uint64_t free, uint8_t out[NB_ALL]) {
    compute_groups(us, them, free, out, ALL_GROUPS);
}

} // namespace space

#endif
