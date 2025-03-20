#ifndef NNUE_Q_H
#define NNUE_Q_H
#include <cstddef>
#include <vector>
#include <tuple>
#include "game.h"
#include <fstream>
#include <string>
#include <iomanip>

// black positions + white positions + empty positions + occupied positions + free positions + turn 
constexpr int INPUT_SIZE = 64 + 64 + 64 + 64 + 64 + 64;
constexpr int OUTPUT_SIZE = 3;

struct NNUE_Q {  
    static constexpr int H1_SIZE = 4096;
    static constexpr int H2_SIZE = 64;
    static constexpr int H3_SIZE = 64;
    static constexpr int TURN_WHITE = 0;
    static constexpr int H1_BIAS = 2 * H1_SIZE;    
    static constexpr int INPUT_TO_H1 = 3 * H1_SIZE;
    static constexpr int H2_BIAS = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE;
    static constexpr int H1_TO_H2 = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE;
    static constexpr int H3_BIAS = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE;
    static constexpr int H2_TO_H3 = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE;
    static constexpr int OUTPUT_BIAS = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE;
    static constexpr int H3_TO_OUTPUT = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE;
    struct Accumulator {
        int32_t* acc;
        Accumulator() {
            acc = (int32_t*)aligned_alloc(32, 4 * H1_SIZE);
            memset(acc, H1_SIZE, 0);
        }
        ~Accumulator() {
            free(acc);
        }
    };
    int8_t* weights_and_biases;
    NNUE_Q();
    void load(const std::string& filename);
    Accumulator make_accumulator() const;
    void init(const Yolah& yolah, Accumulator& a);
    void play(uint8_t player, const Move& m, Accumulator& a);
    void undo(uint8_t player, const Move& m, Accumulator& a);
    std::tuple<float, float, float> output(Accumulator& a);
    ~NNUE_Q();
};

#endif
