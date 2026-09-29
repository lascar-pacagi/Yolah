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
    // Number of outputs of the loaded network, read from the weights file:
    //   3 — the original networks: logits of (black wins, draw, white wins),
    //       read with output();
    //   1 — the value networks (nnue_193x1024x64x32x1*.pt): one logit, the
    //       value is tanh(logit) FOR THE SIDE TO MOVE, read with value().
    int nb_outputs = OUTPUT_SIZE;
    NNUE();
    void load(const std::string& filename);
    Accumulator make_accumulator() const;
    void init(const Yolah& yolah, Accumulator& a);
    void play(uint8_t player, const Move& m, Accumulator& a);
    void undo(uint8_t player, const Move& m, Accumulator& a);
    std::tuple<float, float, float> output(Accumulator& a);
    // Value in [-1, 1] for the side to move, whatever the kind of network:
    // P(side to move wins) − P(it loses) for a 3-output network, tanh of the
    // value head for a 1-output one.
    float value(Accumulator& a, uint8_t side_to_move);
    ~NNUE();    
    void save_quantized(const std::string& filename, float scale = 64);
};

#endif