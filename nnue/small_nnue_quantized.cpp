#include "small_nnue_quantized.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>

void Small_NNUE_Quantized::load(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line, token;
    int m, n;

    // Helper to read a matrix
    auto read_matrix = [&](int8_t* weights, int rows, int cols) {
        std::getline(ifs, line);
        std::istringstream header(line);
        header >> token >> m >> n;
        if (token != "W" || m != rows || n != cols) {
            throw std::runtime_error("Invalid weight matrix header");
        }
        for (int i = 0; i < rows; i++) {
            std::getline(ifs, line);
            std::istringstream row(line);
            for (int j = 0; j < cols; j++) {
                int v;
                row >> v;
                weights[i * cols + j] = static_cast<int8_t>(v);
            }
        }
    };

    // Helper to read a bias
    auto read_bias = [&](int16_t* bias, int size) {
        std::getline(ifs, line);
        std::istringstream header(line);
        header >> token >> n;
        if (token != "B" || n != size) {
            throw std::runtime_error("Invalid bias header");
        }
        std::getline(ifs, line);
        std::istringstream values(line);
        for (int i = 0; i < size; i++) {
            int v;
            values >> v;
            bias[i] = static_cast<int16_t>(v);
        }
    };

    // Helper to read scale
    auto read_scale = [&]() -> float {
        std::getline(ifs, line);
        std::istringstream ss(line);
        float scale;
        ss >> token >> scale;
        if (token != "S") {
            throw std::runtime_error("Invalid scale header");
        }
        return scale;
    };

    // Check for normalization parameters
    std::getline(ifs, line);
    std::istringstream first_line(line);
    first_line >> token;

    if (token == "NORM") {
        has_normalization = true;
        first_line >> n;
        if (n != INPUT_SIZE) {
            throw std::runtime_error("Invalid normalization size");
        }

        // Read mean
        std::getline(ifs, line);
        std::istringstream mean_line(line);
        for (int i = 0; i < INPUT_SIZE; i++) {
            mean_line >> norm_mean[i];
        }

        // Read std
        std::getline(ifs, line);
        std::istringstream std_line(line);
        for (int i = 0; i < INPUT_SIZE; i++) {
            std_line >> norm_std[i];
        }

        // Read first layer header
        std::getline(ifs, line);
    }

    // Put back the first layer header line
    // (We already read it, so we parse it directly)
    {
        std::istringstream header(line);
        header >> token >> m >> n;
        if (token != "W" || m != H1_SIZE || n != INPUT_SIZE) {
            throw std::runtime_error("Invalid fc1 weight header");
        }
        for (int i = 0; i < H1_SIZE; i++) {
            std::getline(ifs, line);
            std::istringstream row(line);
            for (int j = 0; j < INPUT_SIZE; j++) {
                int v;
                row >> v;
                fc1_weight[i * INPUT_SIZE + j] = static_cast<int8_t>(v);
            }
        }
    }
    read_bias(fc1_bias, H1_SIZE);
    fc1_scale = read_scale();

    // Layer 2
    read_matrix(fc2_weight, H2_SIZE, H1_SIZE);
    read_bias(fc2_bias, H2_SIZE);
    fc2_scale = read_scale();

    // Layer 3
    read_matrix(fc3_weight, OUTPUT_SIZE, H2_SIZE);
    read_bias(fc3_bias, OUTPUT_SIZE);
    fc3_scale = read_scale();

    loaded = true;
}

void Small_NNUE_Quantized::normalize_input(const Features& input, float* normalized) const {
    if (has_normalization) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            normalized[i] = (input[i] - norm_mean[i]) / norm_std[i];
        }
    } else {
        for (int i = 0; i < INPUT_SIZE; i++) {
            normalized[i] = input[i];
        }
    }
}

void Small_NNUE_Quantized::quantize_input(const float* normalized, int8_t* quantized) const {
    // Scale normalized floats to int8 range for first layer
    // The scaling should match what was used during training export
    for (int i = 0; i < INPUT_SIZE; i++) {
        float scaled = normalized[i] * fc1_scale;
        scaled = std::clamp(scaled, -127.0f, 127.0f);
        quantized[i] = static_cast<int8_t>(std::round(scaled));
    }
}

std::array<float, Small_NNUE_Quantized::OUTPUT_SIZE>
Small_NNUE_Quantized::raw_output(const Features& features) const {
    alignas(32) float normalized[INPUT_SIZE];
    normalize_input(features, normalized);

    // Layer 1: float input -> int32 accumulator -> clipped int8 output
    alignas(32) int32_t h1_acc[H1_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        float acc = fc1_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            acc += normalized[j] * fc1_weight[i * INPUT_SIZE + j];
        }
        // Clipped ReLU: clamp to [0, FACTOR]
        int32_t v = static_cast<int32_t>(acc);
        h1_acc[i] = std::clamp(v >> SHIFT, 0, static_cast<int>(FACTOR));
    }

    // Convert to int8 for next layer
    alignas(32) int8_t h1[H1_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        h1[i] = static_cast<int8_t>(h1_acc[i]);
    }

    // Layer 2: H1_SIZE -> H2_SIZE
    alignas(32) int8_t h2[H2_SIZE];
    for (int i = 0; i < H2_SIZE; i++) {
        int32_t acc = fc2_bias[i];
        for (int j = 0; j < H1_SIZE; j++) {
            acc += static_cast<int32_t>(h1[j]) * static_cast<int32_t>(fc2_weight[i * H1_SIZE + j]);
        }
        // Clipped ReLU
        int32_t v = acc >> SHIFT;
        h2[i] = static_cast<int8_t>(std::clamp(v, 0, static_cast<int>(FACTOR)));
    }

    // Layer 3: H2_SIZE -> OUTPUT_SIZE (no activation, raw logits)
    std::array<float, OUTPUT_SIZE> output;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int32_t acc = fc3_bias[i];
        for (int j = 0; j < H2_SIZE; j++) {
            acc += static_cast<int32_t>(h2[j]) * static_cast<int32_t>(fc3_weight[i * H2_SIZE + j]);
        }
        output[i] = static_cast<float>(acc) / static_cast<float>(FACTOR);
    }

    return output;
}

std::tuple<float, float, float>
Small_NNUE_Quantized::predict(const Features& features) const {
    auto logits = raw_output(features);

    // Softmax
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& v : logits) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (auto& v : logits) {
        v /= sum;
    }

    return {logits[0], logits[1], logits[2]};
}

// Optional: SIMD-optimized version for Layer 2 (32 inputs)
#ifdef USE_AVX2
// Layer 2 can benefit from AVX2 since H1_SIZE=32 fits exactly in one __m256i
static inline int32_t dot_product_32(const int8_t* a, const int8_t* b) {
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    __m256i vb = _mm256_loadu_si256((__m256i*)b);

    // Multiply and add adjacent pairs (int8 * int8 -> int16)
    __m256i prod = _mm256_maddubs_epi16(va, vb);

    // Horizontal sum: int16 pairs -> int32
    __m256i one = _mm256_set1_epi16(1);
    __m256i sum32 = _mm256_madd_epi16(prod, one);

    // Reduce to single int32
    __m128i lo = _mm256_castsi256_si128(sum32);
    __m128i hi = _mm256_extracti128_si256(sum32, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return _mm_cvtsi128_si32(sum128);
}
#endif
