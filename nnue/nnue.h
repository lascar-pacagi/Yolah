#ifndef NNUE_H
#define NNUE_H
#include <cstddef>
#include <vector>
#include <tuple>
#include "game.h"
#include <fstream>
#include <string>
#include <iomanip>
#include "Eigen/Dense"

struct NNUE {
    // black positions + white positions + empty positions + turn
    static constexpr int INPUT_SIZE = 64 + 64 + 64 + 1;
    static constexpr int OUTPUT_SIZE = 3;

    static constexpr int H1_SIZE = 1024;
    static constexpr int H2_SIZE = 64;
    static constexpr int H3_SIZE = 32;
    static constexpr int H1_BIAS = 0;    
    static constexpr int INPUT_TO_H1 = H1_SIZE;
    static constexpr int H2_BIAS = H1_SIZE + INPUT_SIZE * H1_SIZE;
    static constexpr int H1_TO_H2 = H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE;
    static constexpr int H3_BIAS = H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE;
    static constexpr int H2_TO_H3 = H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE;
    static constexpr int OUTPUT_BIAS = H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE;
    static constexpr int H3_TO_OUTPUT = H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE;
    struct Accumulator {
        float* acc;
        Accumulator() {
            acc = (float*)aligned_alloc(64, 4 * H1_SIZE);
            memset(acc, 0, 4 * H1_SIZE);
        }
        ~Accumulator() {
            free(acc);
        }
    };    
    float* weights_and_biases;
    NNUE();
    void load(const std::string& filename);
    Accumulator make_accumulator() const;
    void init(const Yolah& yolah, Accumulator& a);
    void play(uint8_t player, const Move& m, Accumulator& a);
    void undo(uint8_t player, const Move& m, Accumulator& a);
    std::tuple<float, float, float> output(Accumulator& a);
    ~NNUE();    
    void save_quantized(const std::string& filename, float scale = 64);
};

#endif