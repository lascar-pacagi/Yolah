#ifndef FFNN_H
#define FFNN_H

#include <immintrin.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <cmath>
#include <algorithm>

namespace ffnn_detail {

// Pack 8 x int32 (in a 256-bit register) down to 8 x int8 (in low 64 bits of a 128-bit register).
// [a0 a1 a2 a3 | a4 a5 a6 a7] (int32)
//   -> [a0 a1 a2 a3 a4 a5 a6 a7] (int16, via saturating pack)
//   -> [a0 a1 a2 a3 a4 a5 a6 a7 0 0 0 0 0 0 0 0] (int8, via saturating pack with zeros)
inline __m128i convert_8x32_to_8x8(__m256i input32) {
    __m128i lo = _mm256_castsi256_si128(input32);        // low 128 bits: [a0 a1 a2 a3] as int32
    __m128i hi = _mm256_extracti128_si256(input32, 1);   // high 128 bits: [a4 a5 a6 a7] as int32
    __m128i packed16 = _mm_packs_epi32(lo, hi);          // saturating int32 -> int16: [a0..a7]
    return _mm_packs_epi16(packed16, _mm_setzero_si128()); // saturating int16 -> int8: [a0..a7, 0..0]
}

// Horizontal reduction of 4 x __m256i accumulators into 4 x int32 results + bias.
// Each sum_k holds 8 partial int32 sums for one output neuron.
// After 3 rounds of pairwise horizontal add and a cross-lane combine,
// we get one int32 per accumulator = the full dot product for that neuron.
inline __m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    // hadd_epi32: pairwise horizontal add of adjacent int32 lanes
    // [a0 a1 a2 a3 | a4 a5 a6 a7] hadd [b0 b1 b2 b3 | b4 b5 b6 b7]
    //   = [a01 a23 b01 b23 | a45 a67 b45 b67]
    sum0 = _mm256_hadd_epi32(sum0, sum1);   // interleave partial sums of neuron 0 and 1
    sum2 = _mm256_hadd_epi32(sum2, sum3);   // interleave partial sums of neuron 2 and 3
    sum0 = _mm256_hadd_epi32(sum0, sum2);   // interleave all 4 neurons
    // Now sum0 = [s0_lo s1_lo s2_lo s3_lo | s0_hi s1_hi s2_hi s3_hi]
    // where s_k = s_k_lo + s_k_hi is the full dot product for neuron k
    __m128i lo = _mm256_castsi256_si128(sum0);           // low 128: [s0_lo s1_lo s2_lo s3_lo]
    __m128i hi = _mm256_extracti128_si256(sum0, 1);      // high 128: [s0_hi s1_hi s2_hi s3_hi]
    return _mm_add_epi32(_mm_add_epi32(lo, hi), bias);   // [s0 s1 s2 s3] + bias
}

// Same as haddx4 but for 8 accumulators -> 8 int32 results in a __m256i.
// Neurons 0-3 are reduced in sum0/sum2, neurons 4-7 in sum4/sum6,
// then the two 128-bit halves are combined with their respective bias halves.
inline __m256i m256_haddx8(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3,
                           __m256i sum4, __m256i sum5, __m256i sum6, __m256i sum7, __m256i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);   // pairwise horizontal add (neurons 0,1)
    sum2 = _mm256_hadd_epi32(sum2, sum3);   // (neurons 2,3)
    sum4 = _mm256_hadd_epi32(sum4, sum5);   // (neurons 4,5)
    sum6 = _mm256_hadd_epi32(sum6, sum7);   // (neurons 6,7)
    sum0 = _mm256_hadd_epi32(sum0, sum2);   // second round: neurons 0-3
    sum4 = _mm256_hadd_epi32(sum4, sum6);   // second round: neurons 4-7
    // Each 256-bit register now has [lo_half | hi_half] that need to be added across lanes
    __m128i lo0 = _mm256_castsi256_si128(sum0);          // neurons 0-3 low partial sums
    __m128i hi0 = _mm256_extracti128_si256(sum0, 1);     // neurons 0-3 high partial sums
    __m128i lo1 = _mm256_castsi256_si128(sum4);          // neurons 4-7 low partial sums
    __m128i hi1 = _mm256_extracti128_si256(sum4, 1);     // neurons 4-7 high partial sums
    __m128i res0 = _mm_add_epi32(_mm_add_epi32(lo0, hi0), _mm256_castsi256_si128(bias));     // [s0 s1 s2 s3] + bias[0..3]
    __m128i res1 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), _mm256_extracti128_si256(bias, 1)); // [s4 s5 s6 s7] + bias[4..7]
    return _mm256_set_m128i(res1, res0);  // combine into [res0 | res1] = 8 x int32
}

// Dot-product accumulate: acc += dot(a_uint8, b_int8) for 32 byte pairs.
// 'a' is treated as unsigned bytes (uint8), 'b' as signed bytes (int8).
//
// With VNNI (AVX-512 extension): single instruction does it all.
// Without VNNI (AVX2 fallback):
//   maddubs: multiply 32 (uint8 * int8) pairs -> 16 int16 (with pairwise horizontal add)
//   madd:    multiply 16 int16 by 1 and pairwise add -> 8 int32
//   add:     accumulate into acc
inline void mm256_dpwssds_avx_epi32(__m256i& acc, __m256i a, __m256i b) {
#if defined(USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);               // single-instruction uint8*int8 dot + accumulate
#else
    __m256i product = _mm256_maddubs_epi16(a, b);       // 32x (u8*i8) -> 16x i16 (adjacent pairs summed)
    product = _mm256_madd_epi16(product, _mm256_set1_epi16(1)); // 16x i16 -> 8x i32 (adjacent pairs summed)
    acc = _mm256_add_epi32(acc, product);                // accumulate into 8x int32
#endif
}

// Hidden-layer matvec: m outputs (multiple of 8) from n inputs (multiple of 32).
// first_layer=true  : inputs are uint8 features in [0,255], bias at scale 64.
//                     Exact /255 via float round-trip after integer dot product.
// first_layer=false : activations in [0,64], bias at scale 64^2. Pure integer >>6.
// Output is int8 clamped to [0, 64].
template<int m, int n, bool first_layer = false>
inline void matvec(const int8_t* __restrict__ w, const int8_t* __restrict__ x,
                   int8_t* __restrict__ y, const int16_t* __restrict__ bias) {
    static_assert(m % 8 == 0, "m must be a multiple of 8");
    static_assert(n % 32 == 0, "n must be a multiple of 32");

    constexpr int register_width = 32;
    constexpr int num_in_chunks = n / register_width;
    constexpr int num_out_chunks = m / 8;

    const __m256i zero_i    = _mm256_setzero_si256();
    const __m256i clamp_max = _mm256_set1_epi32(64);

    for (int i = 0; i < num_out_chunks; ++i) {
        const int base = i * 8;

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        __m256i sum4 = _mm256_setzero_si256();
        __m256i sum5 = _mm256_setzero_si256();
        __m256i sum6 = _mm256_setzero_si256();
        __m256i sum7 = _mm256_setzero_si256();

        // Inner loop: accumulate dot products for 8 output neurons simultaneously.
        // Each iteration processes 32 input bytes (one AVX2 register width).
        // load_si256: load 32 aligned bytes into a 256-bit register.
        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((__m256i*)&x[j * register_width]);  // 32 input bytes
            mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&w[(base + 0) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&w[(base + 1) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&w[(base + 2) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&w[(base + 3) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum4, in, _mm256_load_si256((__m256i*)&w[(base + 4) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum5, in, _mm256_load_si256((__m256i*)&w[(base + 5) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum6, in, _mm256_load_si256((__m256i*)&w[(base + 6) * n + j * register_width]));
            mm256_dpwssds_avx_epi32(sum7, in, _mm256_load_si256((__m256i*)&w[(base + 7) * n + j * register_width]));
        }

        if constexpr (first_layer) {
            // First layer: dot products are sum(W_q * x_uint8) at scale 64 * 1 = 64.
            // We need: 64 * (W·(x/255) + b) = dot/255 + bias_q.
            // Horizontal add with zero bias to get raw int32 dot products.
            __m256i hsum = m256_haddx8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, zero_i);
            // Exact /255 via float round-trip (5 instructions, no precision loss for our range).
            __m256 dot_f = _mm256_cvtepi32_ps(hsum);                    // 8x int32 -> 8x float
            dot_f = _mm256_mul_ps(dot_f, _mm256_set1_ps(1.0f / 255.0f)); // exact divide by 255
            // Load bias (int16 at scale 64) -> widen to int32 -> convert to float, then add.
            __m256 bias_f = _mm256_cvtepi32_ps(                         // 8x int32 -> 8x float
                _mm256_cvtepi16_epi32(                                  // 8x int16 -> 8x int32
                    _mm_load_si128((__m128i*)&bias[base])));            // load 8x int16
            dot_f = _mm256_add_ps(dot_f, bias_f);                       // dot/255 + bias
            // Clamp to [0, 64] in float, then convert back to int32 and pack to int8.
            dot_f = _mm256_min_ps(_mm256_max_ps(dot_f, _mm256_setzero_ps()), _mm256_set1_ps(64.0f));
            __m256i outval = _mm256_cvtps_epi32(dot_f);                 // 8x float -> 8x int32 (round to nearest)
            _mm_storel_epi64((__m128i*)&y[base], convert_8x32_to_8x8(outval)); // pack int32 -> int8, store 8 bytes
        } else {
            // Deeper layers: dot products are sum(W_q * h_q) at scale 64*64.
            // Bias is also at scale 64^2, so we add it directly in the horizontal reduction.
            // cvtepi16_epi32: sign-extend 8x int16 -> 8x int32.
            __m256i bias_vec = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i*)&bias[base]));
            __m256i outval = m256_haddx8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, bias_vec);
            // srai_epi32: arithmetic right shift each int32 by 6 (= divide by 64, keeping sign).
            outval = _mm256_srai_epi32(outval, 6);
            // Clamp to [0, 64]: max with 0, then min with 64.
            outval = _mm256_min_epi32(_mm256_max_epi32(outval, zero_i), clamp_max);
            _mm_storel_epi64((__m128i*)&y[base], convert_8x32_to_8x8(outval)); // pack int32 -> int8, store 8 bytes
        }
    }
}

// Output-layer matvec: m outputs (<=4) from n inputs (multiple of 32).
// Uses if constexpr to emit only the needed dot-product accumulators.
// No clamp — results are stored as int32 for float softmax.
template<int m, int n>
inline void matvec_output(const int8_t* __restrict__ w, const int8_t* __restrict__ x,
                          int32_t* __restrict__ y, const int16_t* __restrict__ bias) {
    static_assert(m <= 4, "m must be <= 4");
    static_assert(n % 32 == 0, "n must be a multiple of 32");

    constexpr int register_width = 32;
    constexpr int num_in_chunks = n / register_width;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    // Accumulate dot products. if constexpr eliminates unused rows at compile time
    // (e.g. for m=3, sum3 stays zero — no wasted instructions).
    for (int j = 0; j < num_in_chunks; ++j) {
        const __m256i in = _mm256_load_si256((__m256i*)&x[j * register_width]);  // 32 input bytes
        if constexpr (m >= 1) mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&w[0 * n + j * register_width]));
        if constexpr (m >= 2) mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&w[1 * n + j * register_width]));
        if constexpr (m >= 3) mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&w[2 * n + j * register_width]));
        if constexpr (m >= 4) mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&w[3 * n + j * register_width]));
    }

    // loadl_epi64: load 8 bytes (4x int16) into low half of 128-bit register.
    // cvtepi16_epi32: sign-extend 4x int16 -> 4x int32.
    const __m128i bias_vec = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)bias));
    // Horizontal reduction of 4 accumulators -> 4x int32 dot products + bias.
    __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias_vec);
    // srai_epi32: arithmetic right shift by 6 (divide by 64).
    // Result is at scale 64 (one factor of 64 removed from the 64^2 accumulation).
    // Will be divided by 64.0f in the caller to recover float logits for softmax.
    outval = _mm_srai_epi32(outval, 6);
    // store_si128: store 4x int32 (128 bits) to aligned memory.
    _mm_store_si128((__m128i*)y, outval);
}

template<int M, int N, int N_STRIDE>
void read_matrix(std::ifstream& ifs, int8_t* weights) {
    std::string type;
    int m, n, v;
    ifs >> type;
    if (type != "W")
        throw std::runtime_error("expected 'W', got '" + type + "'");
    if (!(ifs >> m >> n))
        throw std::runtime_error("expected matrix dimensions");
    if (m != M || n != N)
        throw std::runtime_error(
            "bad matrix dimension: expected " + std::to_string(M) + "x" + std::to_string(N) +
            ", got " + std::to_string(m) + "x" + std::to_string(n));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            ifs >> v;
            weights[i * N_STRIDE + j] = static_cast<int8_t>(v);
        }
    }
}

template<int N>
void read_bias(std::ifstream& ifs, int16_t* bias) {
    std::string type;
    int n, v;
    ifs >> type;
    if (type != "B")
        throw std::runtime_error("expected 'B', got '" + type + "'");
    if (!(ifs >> n))
        throw std::runtime_error("expected bias dimension");
    if (n != N)
        throw std::runtime_error(
            "bad bias dimension: expected " + std::to_string(N) + ", got " + std::to_string(n));
    for (int i = 0; i < N; i++) {
        ifs >> v;
        bias[i] = static_cast<int16_t>(v);
    }
}

} // namespace ffnn_detail

// Quantized 3-layer feedforward neural network: I -> H1 -> H2 -> O.
// Weights are int8 (scale 64), biases are int16 (scale 64 for fc1, scale 64^2 for fc2/fc3).
// Activation: clamp(x, 0, 1) quantized as clamp(x_q, 0, 64).
// Output: softmax over O logits -> (black_win, draw, white_win) probabilities.
//
// Usage:
//   FFNN<YolahFeatures::NB_FEATURES, 128, 64, 3> net("features_quantized.txt");
//   alignas(64) uint8_t features[decltype(net)::I_PADDED]{};  // zero-init once
//   YolahFeatures::set_features(features, yolah);              // writes [0, NB_FEATURES)
//   auto [black, draw, white] = net(features);                 // no copy, direct SIMD
template <int I, int H1, int H2, int O>
struct FFNN {
    static_assert(H1 % 32 == 0, "H1 must be a multiple of 32");
    static_assert(H2 % 32 == 0, "H2 must be a multiple of 32");
    static_assert(O <= 4, "O must be <= 4");

    // Round up input dimension to next multiple of 32 (AVX2 register width in bytes).
    // Padding columns are zero-initialized -> contribute nothing to dot products.
    static constexpr int I_PADDED = ((I + 31) / 32) * 32;

    // All arrays are 64-byte aligned for AVX2 aligned loads (_mm256_load_si256).
    // Zero-initialized with {} so padding bytes don't contain garbage.
    alignas(64) int8_t  fc1_weight[H1 * I_PADDED]{};   // [H1 x I_PADDED] row-major, scale 64
    alignas(64) int16_t fc1_bias[H1]{};                 // [H1], scale 64
    alignas(64) int8_t  fc2_weight[H2 * H1]{};          // [H2 x H1] row-major, scale 64
    alignas(64) int16_t fc2_bias[H2]{};                  // [H2], scale 64^2
    alignas(64) int8_t  fc3_weight[O * H2]{};            // [O x H2] row-major, scale 64
    alignas(64) int16_t fc3_bias[8]{};                   // [O] padded to 8 for SIMD, scale 64^2

    FFNN(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs)
            throw std::runtime_error("cannot open " + filename);
        ffnn_detail::read_matrix<H1, I, I_PADDED>(ifs, fc1_weight);
        ffnn_detail::read_bias<H1>(ifs, fc1_bias);
        ffnn_detail::read_matrix<H2, H1, H1>(ifs, fc2_weight);
        ffnn_detail::read_bias<H2>(ifs, fc2_bias);
        ffnn_detail::read_matrix<O, H2, H2>(ifs, fc3_weight);
        ffnn_detail::read_bias<O>(ifs, fc3_bias);
        if (ifs.fail())
            throw std::runtime_error("error reading weights from " + filename);
    }

    // Fast path: caller provides a 64-byte aligned, I_PADDED-sized buffer.
    // Bytes [0, I) are the features, bytes [I, I_PADDED) must be zero.
    // No copy — the pointer is passed straight to the SIMD dot products.
    //
    // Usage:
    //   alignas(64) uint8_t features[FFNN::I_PADDED]{};   // zero-init once
    //   YolahFeatures::set_features(features, yolah);      // fills [0, I)
    //   auto [b, d, w] = net(features);
    std::tuple<float, float, float> operator()(const uint8_t* __restrict__ input) const {
        // Layer 1: uint8 features -> int8 activations in [0, 64].
        // first_layer=true: uses float round-trip for exact /255 normalization.
        // Reinterpret cast: matvec's signature says int8_t*, but for the first layer
        // the SIMD path treats these bytes as unsigned (vpmaddubsw operand 'a').
        // The bytes themselves are unchanged; only the C++ pointer type is bridged.
        alignas(64) int8_t h1[H1];
        ffnn_detail::matvec<H1, I_PADDED, true>(fc1_weight, reinterpret_cast<const int8_t*>(input), h1, fc1_bias);

        // Layer 2: int8 [0,64] -> int8 [0,64]. Pure integer path (>>6).
        alignas(64) int8_t h2[H2]{};
        ffnn_detail::matvec<H2, H1>(fc2_weight, h1, h2, fc2_bias);

        // Output layer: int8 [0,64] -> int32 (raw logits at scale 64, no clamp).
        alignas(64) int32_t out[4]{};
        ffnn_detail::matvec_output<O, H2>(fc3_weight, h2, out, fc3_bias);

        float out0 = out[0] / 64.0f;
        float out1 = out[1] / 64.0f;
        float out2 = out[2] / 64.0f;
        
        float m = std::max({out0, out1, out2});
        
        // Softmax: divide by 64 to recover float logits, then exp + normalize.
        const float e0 = std::exp(out0 - m);
        const float e1 = std::exp(out1 - m);
        const float e2 = std::exp(out2 - m);
        const float sum = e0 + e1 + e2;
        return {e0 / sum, e1 / sum, e2 / sum};
    }

    // Convenience overload: accepts a plain std::array<uint8_t, I>.
    // Copies into an aligned, zero-padded buffer before evaluation.
    std::tuple<float, float, float> operator()(const std::array<uint8_t, I>& features) const {
        alignas(64) uint8_t input[I_PADDED]{};
        std::memcpy(input, features.data(), I);
        return (*this)(input);
    }
};

#endif
