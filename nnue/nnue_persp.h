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
// Space features (step 2): the next layer also reads numbers computed from
// the position by value() (nnue/space_features.h), stored after the h1
// values, padded to a multiple of 32 columns. Weights file tag "PERSPS" + a
// mask of the 16 first columns, or "PERSPC" + the number of columns and the
// columns themselves (any of the NB_ALL of the ablation: influence,
// encirclement, ...); value() computes only their groups.
//
// Direct path (step 3): optionally, a "DIRECT" section after the value heads,
// k × n_space floats: value() adds Σ_j direct[b][j] · feature_j / 64 to the
// head's sum before the tanh (nnue_persp.py, direct=True).
//
// First layer: h1 = 1024 or 512 values per view (the l1 of the weights file);
// the kernels are compiled for each size.
//
// First layer per phase (nnue_persp.py fc1_buckets = N, the factorised sets
// merged by the exporter): fc1 has N · l1 rows in the file, set b = rows b · l1 …,
// chosen by the holes like the heads, fc1 bucket = holes · N / 57 (N = 4: holes
// 0–14, 15–28, 29–42, 43–56). The accumulator holds the sums of its position's
// set only; a move into another set (the holes only grow: 3 times per game for
// N = 4) recomputes it from the position, which the accumulator also keeps
// (3 bitboards after the int16 values) since update() / play() / undo() do not
// get the position. N = 1: the file of before, nothing changes.
// Refresh cache (RefreshCache, one per search thread, optional; Stockfish's
// "finny tables"): per set, the last accumulator recomputed and its position.
// A refresh starts from it and applies only the squares that differ — few
// within one search — instead of every piece and hole (≈ 60–120 rows).
//
// Accumulator: 2 × H1_SIZE int16 (the largest size), view v at acc + v · h1_size
// (v = the colour), then the position: black, white, holes (uint64_t, at
// acc + ACC_SIZE; kept up to date only with N > 1).
struct NNUE_Persp {
    static constexpr int INPUT_SIZE = 192;
    static constexpr int H1_SIZE = 1024;              // the largest first layer
    static constexpr int H1_SMALL = 512;              // the other one
    static constexpr int H2_SIZE = 64;
    static constexpr int H3_SIZE = 32;
    static constexpr int ACC_SIZE = 2 * H1_SIZE;      // int16 per accumulator (both views)
    static constexpr int MAX_HOLES = 56;
    static constexpr int MAX_FC1_BUCKETS = 8;

    int k = 1;                // number of phase heads
    int fc1_buckets = 1;      // sets of fc1 weights by phase (rows of fc1 in the file / l1)
    int h1_size = H1_SIZE;    // values of the first layer per view (H1_SIZE or H1_SMALL)
    bool them = false;        // fc2 reads [us, them] (2 · H1_SIZE inputs) instead of us
    int h2_inputs = H1_SIZE;  // h1 values read by fc2 (h1_size or 2 · h1_size)
    static constexpr int MAX_SPACE = 96;              // space columns, padded (space::NB_ALL = 92)
    int n_space = 0;          // space columns read by fc2
    int space_cols[MAX_SPACE]{};  // their columns in space::compute_groups (0..NB_ALL − 1)
    uint32_t space_groups = 0;    // the groups to compute (bit = space::Group)
    int h2_stride = H1_SIZE;  // columns of fc2 in memory: h2_inputs + n_space padded to 32
    bool sparse = false;      // value(): sparse product of fc2 (same results, zero groups of h1 skipped)

    int16_t* h1_bias = nullptr;     // fc1_buckets × h1_size
    int8_t* input_to_h1 = nullptr;  // fc1_buckets × (INPUT_SIZE × h1_size): one row of h1_size per input
    int8_t* h1_to_h2 = nullptr;     // k × (H2_SIZE × h2_stride)
    int8_t* h1_to_h2_sparse = nullptr;  // the same by groups of 4 inputs (nnue/sparse_matvec.h)
    int16_t* h2_bias = nullptr;     // k × H2_SIZE
    int8_t* h2_to_h3 = nullptr;     // k × (H3_SIZE × H2_SIZE)
    int16_t* h3_bias = nullptr;     // k × H3_SIZE
    float* value_weight = nullptr;  // k × H3_SIZE
    float* value_bias = nullptr;    // k
    float* direct = nullptr;        // k × n_space, or nullptr: the direct path of the space features

    struct Accumulator {
        int16_t* acc;
        static constexpr size_t BYTES = ACC_SIZE * sizeof(int16_t) + 64;   // + the position (3 uint64_t)
        Accumulator() {
            acc = (int16_t*)aligned_alloc(64, BYTES);
            memset(acc, 0, BYTES);
        }
        ~Accumulator() { free(acc); }
        Accumulator(const Accumulator&) = delete;
        Accumulator& operator=(const Accumulator&) = delete;
    };

    // Per search thread, for one network: entry b = the accumulator of set b for
    // its position (valid[b]). Correct for any position; only speed depends on it.
    struct RefreshCache {
        Accumulator entries[MAX_FC1_BUCKETS];
        bool valid[MAX_FC1_BUCKETS]{};
        void clear() { for (bool& v : valid) v = false; }
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
    // The position kept after the values of an accumulator: black, white, holes.
    static uint64_t* position_of(int16_t* acc) { return reinterpret_cast<uint64_t*>(acc + ACC_SIZE); }
    static const uint64_t* position_of(const int16_t* acc) { return reinterpret_cast<const uint64_t*>(acc + ACC_SIZE); }
    int fc1_bucket(uint64_t holes) const {
        return std::min(fc1_buckets - 1, std::popcount(holes) * fc1_buckets / (MAX_HOLES + 1));
    }
    int bucket(const Yolah& yolah) const {
        const int holes = std::popcount(yolah.empty_bitboard());
        return std::min(k - 1, holes * k / (MAX_HOLES + 1));
    }

    void init(const Yolah& yolah, int16_t* acc) const;
    // out = in + the move m by `player`, both views, without modifying `in`.
    // cache: refreshes start from it (nullptr: from scratch); the result is the same.
    void update(const int16_t* __restrict in, int16_t* __restrict out, uint8_t player, Move m,
                RefreshCache* cache = nullptr) const;
    void play(uint8_t player, Move m, int16_t* acc, RefreshCache* cache = nullptr) const;
    void undo(uint8_t player, Move m, int16_t* acc, RefreshCache* cache = nullptr) const;
    // Value in [-1, 1] for the side to move of `yolah` (the position of `acc`).
    float value(const int16_t* acc, const Yolah& yolah) const;

    // The same with Accumulator objects (nets_check, the non-lazy search).
    void init(const Yolah& yolah, Accumulator& a) const { init(yolah, a.acc); }
    void play(uint8_t player, Move m, Accumulator& a, RefreshCache* cache = nullptr) const { play(player, m, a.acc, cache); }
    void undo(uint8_t player, Move m, Accumulator& a, RefreshCache* cache = nullptr) const { undo(player, m, a.acc, cache); }
    float value(const Accumulator& a, const Yolah& yolah) const { return value(a.acc, yolah); }

private:
    template<int H> void init_n(const uint64_t* pos, int16_t* acc) const;   // pos: black, white, holes
    // out = the accumulator of pos in its set (a set change), from the cache if any.
    template<int H> void refresh_n(const uint64_t* pos, int16_t* out, RefreshCache* cache) const;
    template<int H> void update_n(const int16_t* __restrict in, int16_t* __restrict out, uint8_t player, Move m,
                                  RefreshCache* cache) const;
    template<int H, int SIGN> void play_n(uint8_t player, Move m, int16_t* acc, RefreshCache* cache) const;
    template<int H> float value_n(const int16_t* acc, const Yolah& yolah) const;
};

#endif
