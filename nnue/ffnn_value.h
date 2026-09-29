#ifndef FFNN_VALUE_H
#define FFNN_VALUE_H
// ffnn_value.h — the features networks with a VALUE head
// (features_119x256x64x1*.pt, trained by features_net119x256x64x1*.py).
//
// Same trunk as FFNN / FFNNFloat (ffnn.h, ffnn_float.h): the features, as
// bytes, divided by 255, then two layers with clamp(x, 0, 1) activations.
// What differs is the head: ONE output, the value of the position FOR THE
// SIDE TO MOVE, tanh(fc3 · h2 + b3) ∈ (−1, 1) — instead of the three logits
// (black wins, draw, white wins) of the original networks.
//
// The value head is not clamped during training (weights up to ~80), so it
// stays in float in both classes (64 multiplications).
//
// Why the quantized version is int16 and not int8 like FFNN: with int8 at
// scale 64, the error on the value is ~0.10–0.14 on average (3–4 % of the
// positions change sign) — the value head, whose weights sum to ~500 in
// absolute value, amplifies the 1/64 steps of the weights and activations
// (measured with nets_check / a PyTorch simulation). At scale 1024 in int16
// the steps are 16 times finer; the arithmetic stays integer (madd_epi16:
// int16 × int16 summed in int32) and int32 sums cannot overflow:
//   fc1: 255 × 2030 × 128 < 2^31,   fc2: 1024 × 2030 × 256 < 2^31.
//
// Weights files (written by nnue/export_value_nets.py), layout of ffnn.h
// ("W m n" + values, "B n" + values):
//   FFNNValue      (quantized): fc1 W int16 ×1024, fc1 B ×1024, fc2 W int16
//                               ×1024, fc2 B ×1024², then fc3 W and B in float;
//   FFNNValueFloat (float)    : every layer in float.
#include "ffnn_float.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <stdexcept>
#include <string>

namespace ffnn_value_detail {

constexpr int SCALE = 1024;

template<typename T>
inline void read_matrix(std::ifstream& ifs, T* w, int M, int N, int stride) {
    std::string type;
    int m, n;
    ifs >> type >> m >> n;
    if (type != "W" || m != M || n != N)
        throw std::runtime_error("bad matrix: expected W " + std::to_string(M) + "x" + std::to_string(N) +
                                 ", got " + type + " " + std::to_string(m) + "x" + std::to_string(n));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double v;
            ifs >> v;
            w[i * stride + j] = static_cast<T>(v);
        }
}

template<typename T>
inline void read_bias(std::ifstream& ifs, T* b, int N) {
    std::string type;
    int n;
    ifs >> type >> n;
    if (type != "B" || n != N)
        throw std::runtime_error("bad bias: expected B " + std::to_string(N));
    for (int i = 0; i < N; i++) {
        double v;
        ifs >> v;
        b[i] = static_cast<T>(v);
    }
}

// Σ over n int16 pairs, in int32 (n multiple of 16).
template<int n>
inline int32_t dot_i16(const int16_t* __restrict__ w, const __m256i* __restrict__ x) {
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < n / 16; j++)
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x[j], _mm256_load_si256((const __m256i*)&w[16 * j])));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

} // namespace ffnn_value_detail

// Quantized (int16, scale 1024) trunk, float value head.
template <int I, int H1, int H2>
struct FFNNValue {
    static_assert(H1 % 16 == 0 && H2 % 8 == 0, "H1 must be a multiple of 16, H2 of 8");
    static constexpr int I_PADDED = ((I + 31) / 32) * 32;   // features buffer size (as FFNN)
    static constexpr int SCALE = ffnn_value_detail::SCALE;

    alignas(64) int16_t fc1_weight[H1 * I_PADDED]{};   // ×1024, rows padded with zeros
    alignas(64) int32_t fc1_bias[H1]{};                 // ×1024
    alignas(64) int16_t fc2_weight[H2 * H1]{};          // ×1024
    alignas(64) int32_t fc2_bias[H2]{};                 // ×1024²
    alignas(64) float   fc3_weight[H2]{};               // value head, float
    float               fc3_bias = 0;

    explicit FFNNValue(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("cannot open " + filename);
        using namespace ffnn_value_detail;
        read_matrix(ifs, fc1_weight, H1, I, I_PADDED);
        read_bias(ifs, fc1_bias, H1);
        read_matrix(ifs, fc2_weight, H2, H1, H1);
        read_bias(ifs, fc2_bias, H2);
        read_matrix(ifs, fc3_weight, 1, H2, H2);
        read_bias(ifs, &fc3_bias, 1);
        if (ifs.fail()) throw std::runtime_error("error reading weights from " + filename);
    }

    // input: 64-byte aligned, I_PADDED bytes, the features in [0, I) and zeros after.
    float value(const uint8_t* __restrict__ input) const {
        using namespace ffnn_value_detail;
        // Layer 1: raw bytes (x·255) × W (×1024) → /255 + b → clamp to [0, 1024].
        __m256i x[I_PADDED / 16];
        for (int j = 0; j < I_PADDED / 16; j++)
            x[j] = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)&input[16 * j]));
        alignas(64) int16_t h1[H1];
        for (int i = 0; i < H1; i++) {
            const float v = dot_i16<I_PADDED>(&fc1_weight[i * I_PADDED], x) / 255.0f + fc1_bias[i];
            h1[i] = static_cast<int16_t>(std::nearbyint(std::clamp(v, 0.0f, float(SCALE))));
        }
        // Layer 2: h1 (×1024) × W (×1024) + b (×1024²) → clamp(·/1024², 0, 1) in float.
        __m256i h[H1 / 16];
        for (int j = 0; j < H1 / 16; j++) h[j] = _mm256_load_si256((const __m256i*)&h1[16 * j]);
        float s = fc3_bias;
        for (int i = 0; i < H2; i++) {
            const float v = (dot_i16<H1>(&fc2_weight[i * H1], h) + (int64_t)fc2_bias[i]) / float(SCALE * SCALE);
            s += fc3_weight[i] * std::clamp(v, 0.0f, 1.0f);
        }
        return std::tanh(s);
    }
};

// Float everywhere (same arithmetic as FFNNFloat), value head.
template <int I, int H1, int H2>
struct FFNNValueFloat {
    static_assert(H1 % 8 == 0 && H2 % 8 == 0, "H1 and H2 must be multiples of 8");
    static constexpr int I_PADDED = ((I + 7) / 8) * 8;

    alignas(32) float fc1_weight[H1 * I_PADDED]{};
    alignas(32) float fc1_bias[H1]{};
    alignas(32) float fc2_weight[H2 * H1]{};
    alignas(32) float fc2_bias[H2]{};
    alignas(32) float fc3_weight[H2]{};
    float             fc3_bias = 0;

    explicit FFNNValueFloat(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("cannot open " + filename);
        ffnn_float_detail::read_matrix<H1, I, I_PADDED>(ifs, fc1_weight);
        ffnn_float_detail::read_bias<H1>(ifs, fc1_bias);
        ffnn_float_detail::read_matrix<H2, H1, H1>(ifs, fc2_weight);
        ffnn_float_detail::read_bias<H2>(ifs, fc2_bias);
        ffnn_value_detail::read_matrix(ifs, fc3_weight, 1, H2, H2);
        ffnn_value_detail::read_bias(ifs, &fc3_bias, 1);
        if (ifs.fail()) throw std::runtime_error("error reading weights from " + filename);
    }

    // input: the feature bytes (I of them); divided by 255 as in training.
    float value(const uint8_t* __restrict__ input) const {
        alignas(32) float x[I_PADDED]{};
        for (int i = 0; i < I; i++) x[i] = input[i] / 255.0f;
        alignas(32) float h1[H1];
        ffnn_float_detail::matvec<H1, I_PADDED>(fc1_weight, x, h1, fc1_bias);
        alignas(32) float h2[H2]{};
        ffnn_float_detail::matvec<H2, H1>(fc2_weight, h1, h2, fc2_bias);
        float s = fc3_bias;
        for (int i = 0; i < H2; i++) s += fc3_weight[i] * h2[i];
        return std::tanh(s);
    }
};

#endif
