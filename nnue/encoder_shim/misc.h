// Slim shim for misc.h — sourced ahead of ../misc/misc.h via the encoder
// build's -I order.
//
// The real misc.h pulls in <boost/asio> + <boost/beast> for its websocket /
// socket server classes. The encoder build does not use any of that — only
// PRNG (magic.cpp + zobrist.cpp), the tiny header-only helpers (`mul_hi64`,
// `reduce`, `debug`), and the AROUND[] table + `around()` function used by
// game.h. Keeping the shim header-only avoids needing libboost-dev
// (~hundreds of MB) inside the SIF.
//
// All symbols below are copied VERBATIM from misc.h. If misc.h ever
// changes meaning for any of these, update this shim to match — the
// encoder must produce bit-identical features to the C++ player runtime.
#ifndef MISC_H
#define MISC_H
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <numeric>
#include <algorithm>

// xorshift64star Pseudo-Random Number Generator
// This class is based on original code written and dedicated
// to the public domain by Sebastiano Vigna (2014).
class PRNG {
    uint64_t s;

    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

   public:
    PRNG(uint64_t seed) : s(seed) {}

    uint64_t seed() { return s; }

    template<typename T>
    T rand() {
        return T(rand64());
    }

    // Special generator used to fast init magic numbers.
    // Output values only have 1/8th of their bits set on average.
    template<typename T>
    T sparse_rand() {
        return T(rand64() & rand64() & rand64());
    }
};

inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
    uint64_t aL = uint32_t(a), aH = a >> 32;
    uint64_t bL = uint32_t(b), bH = b >> 32;
    uint64_t c1 = (aL * bL) >> 32;
    uint64_t c2 = aH * bL + c1;
    uint64_t c3 = aL * bH + uint32_t(c2);
    return aH * bH + (c2 >> 32) + (c3 >> 32);
}

constexpr uint32_t reduce(uint32_t x, uint32_t N) {
    return ((uint64_t) x * (uint64_t) N) >> 32;
}

static constexpr bool DEBUG = false;

void debug(auto&& print) {
    if constexpr (DEBUG) {
        print();
    }
}

// Lookup table indexed by square (0..63) → bitboard of the 8 neighbouring
// squares. Last (65th) entry is a sentinel reused for the LSB-set case.
constexpr uint64_t AROUND[65] = {
    0x302,
    0x705,
    0xe0a,
    0x1c14,
    0x3828,
    0x7050,
    0xe0a0,
    0xc040,
    0x30203,
    0x70507,
    0xe0a0e,
    0x1c141c,
    0x382838,
    0x705070,
    0xe0a0e0,
    0xc040c0,
    0x3020300,
    0x7050700,
    0xe0a0e00,
    0x1c141c00,
    0x38283800,
    0x70507000,
    0xe0a0e000,
    0xc040c000,
    0x302030000,
    0x705070000,
    0xe0a0e0000,
    0x1c141c0000,
    0x3828380000,
    0x7050700000,
    0xe0a0e00000,
    0xc040c00000,
    0x30203000000,
    0x70507000000,
    0xe0a0e000000,
    0x1c141c000000,
    0x382838000000,
    0x705070000000,
    0xe0a0e0000000,
    0xc040c0000000,
    0x3020300000000,
    0x7050700000000,
    0xe0a0e00000000,
    0x1c141c00000000,
    0x38283800000000,
    0x70507000000000,
    0xe0a0e000000000,
    0xc040c000000000,
    0x302030000000000,
    0x705070000000000,
    0xe0a0e0000000000,
    0x1c141c0000000000,
    0x3828380000000000,
    0x7050700000000000,
    0xe0a0e00000000000,
    0xc040c00000000000,
    0x203000000000000,
    0x507000000000000,
    0xa0e000000000000,
    0x141c000000000000,
    0x2838000000000000,
    0x5070000000000000,
    0xa0e0000000000000,
    0x40c0000000000000,
    0x302,
};

constexpr uint64_t around(uint64_t stone) {
    return AROUND[std::countr_zero(stone)];
}

#endif
