#ifndef SPARSE_MATVEC_H
#define SPARSE_MATVEC_H
#include <cstdint>
#include <cstring>
#include <bit>
#include <immintrin.h>

// The sparse product of the first hidden layer (doc/nnue_roadmap.qmd, "Sauter
// les zéros de l'accumulateur"). h1 = clamp(accumulator, 0, 64) is mostly zero
// (87 % of the values, 59 % of the groups of 4 with the 193 network), and a
// zero input adds nothing to the next layer: only the groups of 4 inputs with
// at least one non-zero value are multiplied.
//
// Groups of 4 because one dpbusd (or maddubs + madd) multiplies and adds 4
// bytes per 32-bit lane. The weights are laid out by group (sparse_layout):
// block g = the 4 columns 4g..4g+3 for the m outputs, m × 4 contiguous bytes,
// lane l of register o holding the 4 weights of output 8o + l. The 4 values of
// the group are broadcast to the 8 lanes; each output accumulates in its own
// lane, so no horizontal add is needed at the end.
//
// The result is EXACTLY that of the dense matvec (integer sums, the order
// does not matter, the zero groups add 0).

namespace sparse_nnue {

// out (by group) ← w (m × n int8, row-major, one row per output).
inline void layout(const int8_t* w, int m, int n, int8_t* out) {
    for (int g = 0; g < n / 4; g++)
        for (int o = 0; o < m; o++)
            for (int t = 0; t < 4; t++)
                out[(size_t(g) * m + o) * 4 + t] = w[size_t(o) * n + 4 * g + t];
}

// For each byte mask, the positions of its set bits (the rest: don't care).
// The indices of the non-zero groups are written 8 at a time without a
// branch: a loop over the bits of each mask mispredicts about once per 32
// bytes of h1, which costs more than the multiplications saved.
struct Lut {
    alignas(16) uint16_t idx[256][8];
    constexpr Lut() : idx{} {
        for (int m = 0; m < 256; m++) {
            int k = 0;
            for (int b = 0; b < 8; b++) if (m >> b & 1) idx[m][k++] = uint16_t(b);
        }
    }
};
inline constexpr Lut lut{};

static inline void dot4(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i p = _mm256_maddubs_epi16(a, b);
    p = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
    acc = _mm256_add_epi32(acc, p);
#endif
}

// c (m int8 in [0, 64]) = clamp((W · b + bias) >> 6, 0, 64), W given by
// layout(); b = n int8 in [0, 64], 32-byte aligned. Same result as the dense
// matvec<m, n> of nnue_quantized.cpp / nnue_persp.cpp.
template<int m, int n>
static inline void matvec(const int8_t* __restrict__ w, const int8_t* __restrict__ b, int8_t* __restrict__ c,
                          const int16_t* __restrict__ bias) {
    static_assert(m % 8 == 0 && m / 8 <= 8 && n % 32 == 0);
    constexpr int R = m / 8;                  // registers of outputs
    constexpr int G = n / 4;                  // groups of 4 inputs
    // The non-zero groups: 8 groups per 32 bytes of b, one bit each.
    alignas(16) uint16_t nz[G + 8];
    int count = 0;
    const __m256i zero = _mm256_setzero_si256();
    __m128i base = _mm_setzero_si128();
    const __m128i eight = _mm_set1_epi16(8);
    for (int j = 0; j < n / 32; j++) {
        const __m256i in = _mm256_load_si256((const __m256i*)&b[j * 32]);
        const __m256i is_zero = _mm256_cmpeq_epi32(in, zero);
        const uint32_t mask = ~uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(is_zero))) & 0xFF;
        const __m128i idx = _mm_load_si128((const __m128i*)lut.idx[mask]);
        _mm_storeu_si128((__m128i*)&nz[count], _mm_add_epi16(idx, base));
        count += std::popcount(mask);
        base = _mm_add_epi16(base, eight);
    }
    __m256i sum[R];
    for (int r = 0; r < R; r++) sum[r] = _mm256_setzero_si256();
    auto broadcast = [&](int g) {
        int32_t four;
        std::memcpy(&four, &b[4 * g], 4);
        return _mm256_set1_epi32(four);
    };
    int i = 0;
#if !defined (USE_VNNI)
    // Two groups at a time: their maddubs products are added in 16 bits before
    // the single madd (no saturation: |64 · 127 · 2| · 2 = 32512 < 32767).
    const __m256i one = _mm256_set1_epi16(1);
    for (; i + 1 < count; i += 2) {
        const __m256i in0 = broadcast(nz[i]), in1 = broadcast(nz[i + 1]);
        const int8_t* block0 = w + size_t(nz[i]) * m * 4;
        const int8_t* block1 = w + size_t(nz[i + 1]) * m * 4;
        for (int r = 0; r < R; r++) {
            const __m256i p = _mm256_add_epi16(
                _mm256_maddubs_epi16(in0, _mm256_load_si256((const __m256i*)&block0[r * 32])),
                _mm256_maddubs_epi16(in1, _mm256_load_si256((const __m256i*)&block1[r * 32])));
            sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(p, one));
        }
    }
#endif
    for (; i < count; i++) {
        const __m256i in = broadcast(nz[i]);
        const int8_t* block = w + size_t(nz[i]) * m * 4;
        for (int r = 0; r < R; r++)
            dot4(sum[r], in, _mm256_load_si256((const __m256i*)&block[r * 32]));
    }
    const __m256i factor = _mm256_set1_epi32(64);
    for (int r = 0; r < R; r++) {
        __m256i v = _mm256_add_epi32(sum[r], _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)&bias[r * 8])));
        v = _mm256_srai_epi32(v, 6);
        v = _mm256_min_epi32(_mm256_max_epi32(v, zero), factor);
        const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storel_epi64((__m128i*)&c[r * 8], _mm_packs_epi16(p16, _mm_setzero_si128()));
    }
}

} // namespace sparse_nnue

#endif
