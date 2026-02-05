#ifndef SMALL_NNUE_QUANTIZED_H
#define SMALL_NNUE_QUANTIZED_H

#include <array>
#include <cstdint>
#include <string>
#include <tuple>

/**
 * Small NNUE with heuristic features as input.
 *
 * Architecture: 24 -> 32 -> 16 -> 3
 * Total weights: ~700 (vs ~200k for full NNUE)
 *
 * Uses int8 quantization for weights and SIMD for inference.
 * Much faster than full NNUE due to smaller size and no accumulator updates.
 */
class Small_NNUE_Quantized {
public:
    // Network dimensions
    static constexpr int INPUT_SIZE = 24;    // heuristic features
    static constexpr int H1_SIZE = 32;
    static constexpr int H2_SIZE = 16;
    static constexpr int OUTPUT_SIZE = 3;    // black_win, draw, white_win

    // Quantization factor
    static constexpr int16_t FACTOR = 64;
    static constexpr int SHIFT = 6;

    // Feature input type (from heuristic_features.h)
    using Features = std::array<float, INPUT_SIZE>;

    /**
     * Load quantized weights from file.
     * Format: W/B markers followed by dimensions and values.
     */
    void load(const std::string& filename);

    /**
     * Run inference on heuristic features.
     * Returns (P(black_win), P(draw), P(white_win)).
     */
    std::tuple<float, float, float> predict(const Features& features) const;

    /**
     * Get raw logits (before softmax) for debugging.
     */
    std::array<float, OUTPUT_SIZE> raw_output(const Features& features) const;

    /**
     * Check if model is loaded.
     */
    bool is_loaded() const { return loaded; }

private:
    bool loaded = false;

    // Normalization parameters (if using pre-normalized features)
    bool has_normalization = false;
    alignas(32) float norm_mean[INPUT_SIZE];
    alignas(32) float norm_std[INPUT_SIZE];

    // Layer 1: INPUT_SIZE -> H1_SIZE
    alignas(32) int8_t fc1_weight[H1_SIZE * INPUT_SIZE];
    alignas(32) int16_t fc1_bias[H1_SIZE];
    float fc1_scale;

    // Layer 2: H1_SIZE -> H2_SIZE
    alignas(32) int8_t fc2_weight[H2_SIZE * H1_SIZE];
    alignas(32) int16_t fc2_bias[H2_SIZE];
    float fc2_scale;

    // Layer 3: H2_SIZE -> OUTPUT_SIZE
    alignas(32) int8_t fc3_weight[OUTPUT_SIZE * H2_SIZE];
    alignas(32) int16_t fc3_bias[OUTPUT_SIZE];
    float fc3_scale;

    // Internal inference functions
    void normalize_input(const Features& input, float* normalized) const;
    void quantize_input(const float* normalized, int8_t* quantized) const;
};

#endif
