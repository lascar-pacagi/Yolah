#ifndef FFNN_FLOAT_H
#define FFNN_FLOAT_H

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

namespace ffnn_float_detail {

// Hidden-layer matvec: m outputs from n inputs, all in float.
// Activation: clamp(x, 0, 1).
// n is padded to a multiple of 8 (AVX2 register width for float).
template<int m, int n>
inline void matvec(const float* __restrict__ w, const float* __restrict__ x,
                   float* __restrict__ y, const float* __restrict__ bias) {
    static_assert(n % 8 == 0, "n must be a multiple of 8");

    for (int i = 0; i < m; ++i) {
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < n; j += 8) {
            __m256 xv = _mm256_load_ps(&x[j]);
            __m256 wv = _mm256_load_ps(&w[i * n + j]);
            sum = _mm256_fmadd_ps(wv, xv, sum);
        }
        // Horizontal reduction: sum all 8 floats
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);            // 4 floats
        __m128 shuf = _mm_movehdup_ps(lo);  // [1,1,3,3]
        lo = _mm_add_ps(lo, shuf);          // [0+1, _, 2+3, _]
        shuf = _mm_movehl_ps(shuf, lo);     // [2+3, ...]
        lo = _mm_add_ss(lo, shuf);          // scalar sum
        float val = _mm_cvtss_f32(lo) + bias[i];
        // clamp(val, 0, 1)
        y[i] = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
    }
}

// Output-layer matvec: m outputs from n inputs, no clamp.
// Raw logits are stored for softmax.
template<int m, int n>
inline void matvec_output(const float* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, const float* __restrict__ bias) {
    static_assert(n % 8 == 0, "n must be a multiple of 8");

    for (int i = 0; i < m; ++i) {
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < n; j += 8) {
            __m256 xv = _mm256_load_ps(&x[j]);
            __m256 wv = _mm256_load_ps(&w[i * n + j]);
            sum = _mm256_fmadd_ps(wv, xv, sum);
        }
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, lo);
        lo = _mm_add_ss(lo, shuf);
        y[i] = _mm_cvtss_f32(lo) + bias[i];
    }
}

template<int M, int N, int N_STRIDE>
void read_matrix(std::ifstream& ifs, float* weights) {
    std::string type;
    int m, n;
    float v;
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
            weights[i * N_STRIDE + j] = v;
        }
    }
}

template<int N>
void read_bias(std::ifstream& ifs, float* bias) {
    std::string type;
    int n;
    float v;
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
        bias[i] = v;
    }
}

} // namespace ffnn_float_detail

// Floating-point 3-layer feedforward neural network: I -> H1 -> H2 -> O.
// Weights and biases are stored as float32.
// Activation: clamp(x, 0, 1).
// Output: softmax over O logits -> (black_win, draw, white_win) probabilities.
//
// Usage:
//   FFNNFloat<YolahFeatures::NB_FEATURES, 128, 64, 3> net("features_float.txt");
//   alignas(32) float features[decltype(net)::I_PADDED]{};
//   for (int j = 0; j < NB_FEATURES; j++) features[j] = raw_features[j] / 255.0f;
//   auto [black, draw, white] = net(features);
template <int I, int H1, int H2, int O>
struct FFNNFloat {
    static_assert(H1 % 8 == 0, "H1 must be a multiple of 8");
    static_assert(H2 % 8 == 0, "H2 must be a multiple of 8");

    // Round up input dimension to next multiple of 8 (AVX2 register width in floats).
    static constexpr int I_PADDED = ((I + 7) / 8) * 8;

    alignas(32) float fc1_weight[H1 * I_PADDED]{};
    alignas(32) float fc1_bias[H1]{};
    alignas(32) float fc2_weight[H2 * H1]{};
    alignas(32) float fc2_bias[H2]{};
    alignas(32) float fc3_weight[O * H2]{};
    alignas(32) float fc3_bias[O]{};

    FFNNFloat(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs)
            throw std::runtime_error("cannot open " + filename);
        ffnn_float_detail::read_matrix<H1, I, I_PADDED>(ifs, fc1_weight);
        ffnn_float_detail::read_bias<H1>(ifs, fc1_bias);
        ffnn_float_detail::read_matrix<H2, H1, H1>(ifs, fc2_weight);
        ffnn_float_detail::read_bias<H2>(ifs, fc2_bias);
        ffnn_float_detail::read_matrix<O, H2, H2>(ifs, fc3_weight);
        ffnn_float_detail::read_bias<O>(ifs, fc3_bias);
        if (ifs.fail())
            throw std::runtime_error("error reading weights from " + filename);
    }

    // Fast path: caller provides a 32-byte aligned, I_PADDED-sized float buffer.
    // Values [0, I) are the features (already divided by 255), [I, I_PADDED) must be 0.
    std::tuple<double, double, double> operator()(const float* __restrict__ input) const {
        alignas(32) float h1[H1];
        ffnn_float_detail::matvec<H1, I_PADDED>(fc1_weight, input, h1, fc1_bias);

        alignas(32) float h2[H2]{};
        ffnn_float_detail::matvec<H2, H1>(fc2_weight, h1, h2, fc2_bias);

        alignas(32) float out[O]{};
        ffnn_float_detail::matvec_output<O, H2>(fc3_weight, h2, out, fc3_bias);

        const float m = std::max({out[0], out[1], out[2]});
        const float e0 = std::exp(out[0] - m);
        const float e1 = std::exp(out[1] - m);
        const float e2 = std::exp(out[2] - m);
        const float sum = e0 + e1 + e2;
        return {static_cast<double>(e0 / sum), static_cast<double>(e1 / sum), static_cast<double>(e2 / sum)};
    }

    // Convenience overload with uint8_t features (like the quantized version).
    // Converts to float internally.
    std::tuple<double, double, double> operator()(const uint8_t* __restrict__ input) const {
        alignas(32) float finput[I_PADDED]{};
        for (int i = 0; i < I; ++i)
            finput[i] = input[i] / 255.0f;
        return (*this)(finput);
    }

    std::tuple<double, double, double> operator()(const std::array<uint8_t, I>& features) const {
        return (*this)(features.data());
    }
};

#endif
