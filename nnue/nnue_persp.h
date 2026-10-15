#ifndef NNUE_PERSP_H
#define NNUE_PERSP_H
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include "game.h"
#include "move.h"

// The NNUE seen from the side to move, with one head per game phase (step 1 of
// doc/nnue_roadmap.qmd; trained by nnue/nnue_persp_distill.py, exported by
// nnue/export_value_nets.py). Quantized like NNUE_Quantized: int8 weights at
// scale 64, int16 accumulator, value heads in float.
//
// Inputs, 192 per point of view: my pieces (rows 0–63), his pieces (64–127),
// holes (128–191). Two points of view, one per colour, both kept up to date:
//     view BLACK : mine = black, his = white, square sq → row 63 − sq
//     view WHITE : mine = white, his = black, square sq → row (63 − sq) ^ 7
// (the white view is mirrored a <-> h, so that both colours start from the
// same squares; 63 − sq is the MSB-first order of the training cache). A move
// by `player` goes into rows 0–63 of its own view and 64–127 of the other.
//
// value() reads the view of the side to move ("us"); with `them`, the next
// layer also reads the other view, always in the order [us, them]. The head
// is chosen by the number of holes: bucket = holes · K / 57.
//
// Space features (step 2, weights file tag "PERSPS" + mask): the next layer
// also reads up to 16 numbers computed from the position by value()
// (nnue/space_features.h), stored after the h1 values, padded to 32 columns.
//
// Accumulator: 2 × H1_SIZE int16, view v at acc + v · H1_SIZE (v = the colour).
struct NNUE_Persp {
    static constexpr int INPUT_SIZE = 192;
    static constexpr int H1_SIZE = 1024;
    static constexpr int H2_SIZE = 64;
    static constexpr int H3_SIZE = 32;
    static constexpr int ACC_SIZE = 2 * H1_SIZE;      // int16 per accumulator (both views)
    static constexpr int MAX_HOLES = 56;

    int k = 1;                // number of phase heads
    bool them = false;        // fc2 reads [us, them] (2 · H1_SIZE inputs) instead of us
    int h2_inputs = H1_SIZE;  // h1 values read by fc2 (H1_SIZE or 2 · H1_SIZE)
    uint32_t space_mask = 0;  // space features read by fc2 (bit = id in space_features.h)
    int n_space = 0;
    int space_ids[16]{};
    int h2_stride = H1_SIZE;  // columns of fc2 in memory: h2_inputs (+ 32 with space features)
    bool sparse = false;      // value(): sparse product of fc2 (same results, zero groups of h1 skipped)

    alignas(64) int16_t h1_bias[H1_SIZE];
    alignas(64) int8_t input_to_h1[INPUT_SIZE * H1_SIZE];   // one row of H1_SIZE per input
    int8_t* h1_to_h2 = nullptr;     // k × (H2_SIZE × h2_stride)
    int8_t* h1_to_h2_sparse = nullptr;  // the same by groups of 4 inputs (nnue/sparse_matvec.h)
    int16_t* h2_bias = nullptr;     // k × H2_SIZE
    int8_t* h2_to_h3 = nullptr;     // k × (H3_SIZE × H2_SIZE)
    int16_t* h3_bias = nullptr;     // k × H3_SIZE
    float* value_weight = nullptr;  // k × H3_SIZE
    float* value_bias = nullptr;    // k

    struct Accumulator {
        int16_t* acc;
        Accumulator() {
            acc = (int16_t*)aligned_alloc(64, ACC_SIZE * sizeof(int16_t));
            memset(acc, 0, ACC_SIZE * sizeof(int16_t));
        }
        ~Accumulator() { free(acc); }
        Accumulator(const Accumulator&) = delete;
        Accumulator& operator=(const Accumulator&) = delete;
    };

    NNUE_Persp() = default;
    explicit NNUE_Persp(const std::string& filename) { load(filename); }
    ~NNUE_Persp();
    NNUE_Persp(const NNUE_Persp&) = delete;
    NNUE_Persp& operator=(const NNUE_Persp&) = delete;

    // True if `filename` is a NNUE_Persp weights file (it starts with "PERSP").
    static bool is_persp_file(const std::string& filename);
    void load(const std::string& filename);

    static int row(int view, int sq) { return view == Yolah::BLACK ? 63 - sq : (63 - sq) ^ 7; }
    int bucket(const Yolah& yolah) const {
        const int holes = std::popcount(yolah.empty_bitboard());
        return std::min(k - 1, holes * k / (MAX_HOLES + 1));
    }

    void init(const Yolah& yolah, int16_t* acc) const;
    // out = in + the move m by `player`, both views, without modifying `in`.
    void update(const int16_t* __restrict in, int16_t* __restrict out, uint8_t player, Move m) const;
    void play(uint8_t player, Move m, int16_t* acc) const;
    void undo(uint8_t player, Move m, int16_t* acc) const;
    // Value in [-1, 1] for the side to move of `yolah` (the position of `acc`).
    float value(const int16_t* acc, const Yolah& yolah) const;

    // The same with Accumulator objects (nets_check, the non-lazy search).
    void init(const Yolah& yolah, Accumulator& a) const { init(yolah, a.acc); }
    void play(uint8_t player, Move m, Accumulator& a) const { play(player, m, a.acc); }
    void undo(uint8_t player, Move m, Accumulator& a) const { undo(player, m, a.acc); }
    float value(const Accumulator& a, const Yolah& yolah) const { return value(a.acc, yolah); }
};

#endif
