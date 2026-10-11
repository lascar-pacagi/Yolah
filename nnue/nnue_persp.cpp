#include "nnue_persp.h"
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <stdexcept>

// The int8 kernels are those of nnue_quantized.cpp: activations in [0, 64]
// (scale 64), int8 weights, int32 sums shifted back by 6 and clamped.
static constexpr int16_t FACTOR = 64;
static constexpr int SHIFT = 6;

static inline __m128i convert_8x32_to_8x8(__m256i input32) {
    __m128i lo = _mm256_castsi256_si128(input32);
    __m128i hi = _mm256_extracti128_si256(input32, 1);
    __m128i packed16 = _mm_packs_epi32(lo, hi);
    return _mm_packs_epi16(packed16, _mm_setzero_si128());
}

static inline __m256i m256_haddx8(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3,
                                  __m256i sum4, __m256i sum5, __m256i sum6, __m256i sum7, __m256i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    __m128i lo0 = _mm256_castsi256_si128(sum0);
    __m128i hi0 = _mm256_extracti128_si256(sum0, 1);
    __m128i lo1 = _mm256_castsi256_si128(sum4);
    __m128i hi1 = _mm256_extracti128_si256(sum4, 1);
    __m128i res0 = _mm_add_epi32(_mm_add_epi32(lo0, hi0), _mm256_castsi256_si128(bias));
    __m128i res1 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), _mm256_extracti128_si256(bias, 1));
    return _mm256_set_m128i(res1, res0);
}

static inline void dot_accumulate(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc = _mm256_add_epi32(acc, product0);
#endif
}

// c (m int8 in [0, 64]) = clamp((a (m × n int8, row-major) · b (n int8 in [0, 64]) + bias) >> 6)
template<int m, int n>
static inline void matvec(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c,
                          const int16_t* __restrict__ bias) {
    constexpr int register_width = 256 / 8;
    constexpr int num_in_chunks = n / register_width;
    constexpr int num_out_chunks = m / 8;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i factor = _mm256_set1_epi32(FACTOR);
    for (int i = 0; i < num_out_chunks; ++i) {
        __m256i sum[8];
        for (int r = 0; r < 8; r++) sum[r] = _mm256_setzero_si256();
        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((const __m256i*)&b[j * register_width]);
            for (int r = 0; r < 8; r++)
                dot_accumulate(sum[r], in, _mm256_load_si256((const __m256i*)&a[(i * 8 + r) * n + j * register_width]));
        }
        const __m256i _bias = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)&bias[i * 8]));
        __m256i outval = m256_haddx8(sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7], _bias);
        outval = _mm256_srai_epi32(outval, SHIFT);
        outval = _mm256_min_epi32(_mm256_max_epi32(outval, zero), factor);
        _mm_storel_epi64((__m128i*)&c[i * 8], convert_8x32_to_8x8(outval));
    }
}

template<typename T>
static T* aligned_array(size_t n) {
    const size_t bytes = ((n * sizeof(T) + 63) / 64) * 64;
    T* p = (T*)aligned_alloc(64, bytes);
    if (!p) throw std::bad_alloc();
    memset(p, 0, bytes);
    return p;
}

NNUE_Persp::~NNUE_Persp() {
    free(h1_to_h2); free(h2_bias); free(h2_to_h3); free(h3_bias); free(value_weight); free(value_bias);
}

bool NNUE_Persp::is_persp_file(const std::string& filename) {
    std::ifstream ifs(filename);
    std::string tag;
    return (ifs >> tag) && tag == "PERSP";
}

// "W\n<rows>\n<cols>\n" + the values row by row, as written by export_value_nets.py.
template<typename T>
static void read_matrix(std::ifstream& ifs, int rows, int cols, T* out, bool transpose, const std::string& what) {
    std::string type;
    int m, n;
    if (!(ifs >> type >> m >> n) || type != "W" || m != rows || n != cols)
        throw std::runtime_error("NNUE_Persp: bad matrix " + what);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            double v;
            ifs >> v;
            if (transpose) out[j * rows + i] = T(v);
            else           out[i * cols + j] = T(v);
        }
}

template<typename T>
static void read_bias(std::ifstream& ifs, int size, T* out, const std::string& what) {
    std::string type;
    int n;
    if (!(ifs >> type >> n) || type != "B" || n != size) throw std::runtime_error("NNUE_Persp: bad bias " + what);
    for (int i = 0; i < size; i++) {
        double v;
        ifs >> v;
        out[i] = T(v);
    }
}

void NNUE_Persp::load(const std::string& filename) {
    std::ifstream ifs(filename);
    std::string tag;
    int l1, l2, l3, t;
    if (!(ifs >> tag >> k >> t >> l1 >> l2 >> l3) || tag != "PERSP")
        throw std::runtime_error("NNUE_Persp: " + filename + " is not a PERSP weights file");
    if (l1 != H1_SIZE || l2 != H2_SIZE || l3 != H3_SIZE || k < 1)
        throw std::runtime_error("NNUE_Persp: unsupported sizes in " + filename);
    them = t != 0;
    h2_inputs = them ? 2 * H1_SIZE : H1_SIZE;
    // fc1 is stored (out, in) by PyTorch: transposed into one row per input.
    read_matrix(ifs, H1_SIZE, INPUT_SIZE, input_to_h1, true, "fc1");
    read_bias(ifs, H1_SIZE, h1_bias, "fc1");
    h1_to_h2 = aligned_array<int8_t>(size_t(k) * H2_SIZE * h2_inputs);
    h2_bias = aligned_array<int16_t>(size_t(k) * H2_SIZE);
    h2_to_h3 = aligned_array<int8_t>(size_t(k) * H3_SIZE * H2_SIZE);
    h3_bias = aligned_array<int16_t>(size_t(k) * H3_SIZE);
    value_weight = aligned_array<float>(size_t(k) * H3_SIZE);
    value_bias = aligned_array<float>(k);
    read_matrix(ifs, k * H2_SIZE, h2_inputs, h1_to_h2, false, "fc2");
    read_bias(ifs, k * H2_SIZE, h2_bias, "fc2");
    for (int b = 0; b < k; b++) {
        read_matrix(ifs, H3_SIZE, H2_SIZE, h2_to_h3 + b * H3_SIZE * H2_SIZE, false, "w3");
        read_bias(ifs, H3_SIZE, h3_bias + b * H3_SIZE, "b3");
    }
    read_matrix(ifs, k, H3_SIZE, value_weight, false, "w4");
    read_bias(ifs, k, value_bias, "b4");
    if (!ifs) throw std::runtime_error("NNUE_Persp: cannot read " + filename);
}

void NNUE_Persp::init(const Yolah& yolah, int16_t* acc) const {
    const uint64_t pieces[2] = { yolah.bitboard(Yolah::BLACK), yolah.bitboard(Yolah::WHITE) };
    const uint64_t holes = yolah.empty_bitboard();
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        int16_t* a = acc + v * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) a[j] = h1_bias[j];
        auto add = [&](uint64_t bb, int plane) {
            for (; bb; bb &= bb - 1) {
                const int8_t* w = input_to_h1 + (plane + row(v, std::countr_zero(bb))) * H1_SIZE;
                for (int j = 0; j < H1_SIZE; j++) a[j] += w[j];
            }
        };
        add(pieces[v], 0);          // mine
        add(pieces[1 - v], 64);     // his
        add(holes, 128);
    }
}

void NNUE_Persp::update(const int16_t* __restrict in, int16_t* __restrict out, uint8_t player, Move m) const {
    if (m == Move::none()) {        // a pass: no feature changes, only the view that value() reads
        memcpy(out, in, ACC_SIZE * sizeof(int16_t));
        return;
    }
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        const int plane = v == player ? 0 : 64;
        const int from = row(v, m.from_sq()), to = row(v, m.to_sq());
        const int8_t* __restrict w_from = input_to_h1 + (plane + from) * H1_SIZE;
        const int8_t* __restrict w_to   = input_to_h1 + (plane + to) * H1_SIZE;
        const int8_t* __restrict w_hole = input_to_h1 + (128 + from) * H1_SIZE;
        const int16_t* __restrict a = in + v * H1_SIZE;
        int16_t* __restrict b = out + v * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) b[j] = int16_t(a[j] - w_from[j] + w_to[j] + w_hole[j]);
    }
}

void NNUE_Persp::play(uint8_t player, Move m, int16_t* acc) const {
    if (m == Move::none()) return;
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        const int plane = v == player ? 0 : 64;
        const int from = row(v, m.from_sq()), to = row(v, m.to_sq());
        const int8_t* w_from = input_to_h1 + (plane + from) * H1_SIZE;
        const int8_t* w_to   = input_to_h1 + (plane + to) * H1_SIZE;
        const int8_t* w_hole = input_to_h1 + (128 + from) * H1_SIZE;
        int16_t* a = acc + v * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) a[j] += -w_from[j] + w_to[j] + w_hole[j];
    }
}

void NNUE_Persp::undo(uint8_t player, Move m, int16_t* acc) const {
    if (m == Move::none()) return;
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        const int plane = v == player ? 0 : 64;
        const int from = row(v, m.from_sq()), to = row(v, m.to_sq());
        const int8_t* w_from = input_to_h1 + (plane + from) * H1_SIZE;
        const int8_t* w_to   = input_to_h1 + (plane + to) * H1_SIZE;
        const int8_t* w_hole = input_to_h1 + (128 + from) * H1_SIZE;
        int16_t* a = acc + v * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) a[j] += w_from[j] - w_to[j] - w_hole[j];
    }
}

float NNUE_Persp::value(const int16_t* acc, const Yolah& yolah) const {
    const int stm = yolah.current_player();
    const int b = bucket(yolah);
    alignas(64) int8_t h1[2 * H1_SIZE];
    alignas(64) int8_t h2[H2_SIZE];
    alignas(64) int8_t h3[H3_SIZE];
    const int16_t* us = acc + stm * H1_SIZE;
    for (int i = 0; i < H1_SIZE; i++) { const int16_t v = us[i]; h1[i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v); }
    if (them) {
        const int16_t* other = acc + (1 - stm) * H1_SIZE;
        for (int i = 0; i < H1_SIZE; i++) { const int16_t v = other[i]; h1[H1_SIZE + i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v); }
        matvec<H2_SIZE, 2 * H1_SIZE>(h1_to_h2 + size_t(b) * H2_SIZE * 2 * H1_SIZE, h1, h2, h2_bias + b * H2_SIZE);
    } else {
        matvec<H2_SIZE, H1_SIZE>(h1_to_h2 + size_t(b) * H2_SIZE * H1_SIZE, h1, h2, h2_bias + b * H2_SIZE);
    }
    matvec<H3_SIZE, H2_SIZE>(h2_to_h3 + b * H3_SIZE * H2_SIZE, h2, h3, h3_bias + b * H3_SIZE);
    const float* w = value_weight + b * H3_SIZE;
    float s = 0;
    for (int i = 0; i < H3_SIZE; i++) s += w[i] * h3[i];      // h3 at scale 64
    return std::tanh(s / FACTOR + value_bias[b]);
}
