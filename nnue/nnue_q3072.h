#ifndef NNUE_Q3072_H
#define NNUE_Q3072_H
#include <cstddef>
#include <vector>
#include <tuple>
#include "game.h"
#include <fstream>
#include <string>
#include <iomanip>

struct NNUE_Q3072 {  
    // black positions + white positions + empty positions + turn 
    static constexpr int INPUT_SIZE = 64 + 64 + 64 + 1;
    static constexpr int OUTPUT_SIZE = 3;

    static constexpr int H1_SIZE = 3072;
    static constexpr int H2_SIZE = 16;
    static constexpr int H3_SIZE = 32;
    alignas(64) int16_t h1_bias[H1_SIZE];
    alignas(64) int8_t input_to_h1[INPUT_SIZE * H1_SIZE];
    alignas(64) int16_t h2_bias[H2_SIZE];
    alignas(64) int8_t h1_to_h2[H1_SIZE * H2_SIZE];
    alignas(64) int16_t h3_bias[H3_SIZE];
    alignas(64) int8_t h2_to_h3[H1_SIZE * H2_SIZE];
    alignas(64) int16_t output_bias[OUTPUT_SIZE + 1]{};
    alignas(64) int8_t h3_to_output[H3_SIZE * OUTPUT_SIZE];
    struct Accumulator {
        int16_t* acc;
        Accumulator() {
            acc = (int16_t*)aligned_alloc(64, 2 * H1_SIZE);
            memset(acc, 0, H1_SIZE);
        }
        ~Accumulator() {
            free(acc);
        }
    };

    void load(const std::string& filename);
    Accumulator make_accumulator() const;
    void init(const Yolah& yolah, Accumulator& a);
    void play(uint8_t player, const Move& m, Accumulator& a);
    void undo(uint8_t player, const Move& m, Accumulator& a);
    std::tuple<float, float, float> output(Accumulator& a);
};

#endif
