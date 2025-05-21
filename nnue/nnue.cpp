#include "nnue.h"
#include <fstream>
#include <string>
#include <regex>
#include "game.h"
#include "move.h"
#include "types.h"
#include "magic.h"
#include "zobrist.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <stdlib.h>
#include <bit>
#include <algorithm>
#include <immintrin.h>

NNUE::NNUE() {
    constexpr int n = 1 + H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE + H3_SIZE * OUTPUT_SIZE;
    weights_and_biases = (float*)aligned_alloc(64, 4 * 32 * (n + 31) / 32);    
}

NNUE::~NNUE() {
    free(weights_and_biases);
}

NNUE::Accumulator NNUE::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = weights_and_biases[H1_BIAS + i];
    }
    return a;
}

static inline __m128 m256_haddx4(__m256 sum0, __m256 sum1, __m256 sum2, __m256 sum3, __m128 bias) {
    sum0 = _mm256_hadd_ps(sum0, sum1);
    sum2 = _mm256_hadd_ps(sum2, sum3);

    sum0 = _mm256_hadd_ps(sum0, sum2);

    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);

    return _mm_add_ps(_mm_add_ps(lo, hi), bias);
};

static inline __m256 m256_haddx8(__m256 sum0, __m256 sum1, __m256 sum2, __m256 sum3, 
                                    __m256 sum4, __m256 sum5, __m256 sum6, __m256 sum7, __m256 bias) {
    sum0 = _mm256_hadd_ps(sum0, sum1);
    sum2 = _mm256_hadd_ps(sum2, sum3);

    sum4 = _mm256_hadd_ps(sum4, sum5);
    sum6 = _mm256_hadd_ps(sum6, sum7);

    sum0 = _mm256_hadd_ps(sum0, sum2);
    sum4 = _mm256_hadd_ps(sum4, sum6);

    __m128 lo0 = _mm256_castps256_ps128(sum0);
    __m128 hi0 = _mm256_extractf128_ps(sum0, 1);

    __m128 lo1 = _mm256_castps256_ps128(sum4);
    __m128 hi1 = _mm256_extractf128_ps(sum4, 1);
    
    __m128 res0 = _mm_add_ps(_mm_add_ps(lo0, hi0), _mm256_castps256_ps128(bias));
    __m128 res1 = _mm_add_ps(_mm_add_ps(lo1, hi1), _mm256_extractf128_ps(bias, 1));

    return _mm256_set_m128(res1, res0);
};

template<int m, int n>
static inline void matvec(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, 
                            const float* __restrict__ bias) {
    constexpr int register_width = 256 / 32;    
    constexpr int num_in_chunks = n / register_width;
    constexpr int num_out_chunks = m / 8;

    const __m256 zero    = _mm256_setzero_ps();
    const __m256 one     = _mm256_set1_ps(1); 

    for (int i = 0; i < num_out_chunks; ++i) {
        const int offset0 = (i * 8 + 0) * n;
        const int offset1 = (i * 8 + 1) * n;
        const int offset2 = (i * 8 + 2) * n;
        const int offset3 = (i * 8 + 3) * n;
        const int offset4 = (i * 8 + 4) * n;
        const int offset5 = (i * 8 + 5) * n;
        const int offset6 = (i * 8 + 6) * n;
        const int offset7 = (i * 8 + 7) * n;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 sum5 = _mm256_setzero_ps();
        __m256 sum6 = _mm256_setzero_ps();
        __m256 sum7 = _mm256_setzero_ps();

        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256 in = _mm256_load_ps(&b[j * register_width]);
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(in, _mm256_load_ps(&a[offset0 + j * register_width])));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(in, _mm256_load_ps(&a[offset1 + j * register_width])));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(in, _mm256_load_ps(&a[offset2 + j * register_width])));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(in, _mm256_load_ps(&a[offset3 + j * register_width])));
            sum4 = _mm256_add_ps(sum4, _mm256_mul_ps(in, _mm256_load_ps(&a[offset4 + j * register_width])));
            sum5 = _mm256_add_ps(sum5, _mm256_mul_ps(in, _mm256_load_ps(&a[offset5 + j * register_width])));
            sum6 = _mm256_add_ps(sum6, _mm256_mul_ps(in, _mm256_load_ps(&a[offset6 + j * register_width])));
            sum7 = _mm256_add_ps(sum7, _mm256_mul_ps(in, _mm256_load_ps(&a[offset7 + j * register_width])));
        }
        const __m256 _bias = _mm256_load_ps(&bias[i * 8]);
        __m256 outval = m256_haddx8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, _bias);
        outval = _mm256_min_ps(_mm256_max_ps(outval, zero), one);        
        _mm256_store_ps(&c[i * 8], outval);
    }
}

static inline void matvec_3x32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                                const float* __restrict__ bias) {
    constexpr int register_width = 256 / 32;    
    constexpr int n = 32;
    constexpr int num_in_chunks = n / register_width;
    
    const int offset0 = 0 * n;
    const int offset1 = 1 * n;
    const int offset2 = 2 * n;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (int j = 0; j < num_in_chunks; ++j) {
        const __m256 in = _mm256_load_ps(&b[j * register_width]);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(in, _mm256_load_ps(&a[offset0 + j * register_width])));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(in, _mm256_load_ps(&a[offset1 + j * register_width])));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(in, _mm256_load_ps(&a[offset2 + j * register_width])));
    }
    const __m128 _bias = _mm_load_ps(bias);
    __m128 outval = m256_haddx4(sum0, sum1, sum2, sum3, _bias);
    _mm_store_ps(c, outval);
}

std::tuple<float, float, float> NNUE::output(Accumulator& a) {        
    alignas(64) float output[OUTPUT_SIZE + 1]{};
    alignas(64) float h1[H1_SIZE];
    alignas(64) float h2[H2_SIZE];
    alignas(64) float h3[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        const float v = a.acc[i];        
        h1[i] = v <= 0 ? 0 : (v >= 1 ? 1 : v);
    }
    matvec<64, 1024>(weights_and_biases + H1_TO_H2, h1, h2, weights_and_biases + H2_BIAS);
    matvec<32, 64>(weights_and_biases + H2_TO_H3, h2, h3, weights_and_biases + H3_BIAS);
    matvec_3x32(weights_and_biases + H3_TO_OUTPUT, h3, output, weights_and_biases + OUTPUT_BIAS);
    float e1 = std::exp(output[0]);
    float e2 = std::exp(output[1]);
    float e3 = std::exp(output[2]);
    float sum = e1 + e2 + e3;
    return { e1 / sum, e2 / sum, e3 / sum };
}

static constexpr bool TRANSPOSE = true; 

template<int M, int N, bool transpose = false>
static void read_matrix(std::ifstream& ifs, float* weights) {
    int m, n;
    float v;
    std::string type;
    ifs >> type;
    if (type != "W") {
        throw "W expected";
    }
    if (!(ifs >> m >> n)) {
        throw "matrix size expected";
    }
    if (m != M || n != N) {
        throw "bad matrix dimension";
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            ifs >> v;
            if constexpr (transpose) weights[j * M + i] =  v; 
            else weights[i * N + j] =  v;
        }
    }
}

template<int N>
void read_bias(std::ifstream& ifs, float* weights) {
    int n;
    float v;
    std::string type;
    ifs >> type;
    if (type != "B") {
        throw "B expected";
    }
    if (!(ifs >> n)) {
        throw "bias size expected";
    }
    if (n != N) {
        throw "bad bias dimension";
    }
    for (int i = 0; i < N; i++) {
        ifs >> v;
        weights[i] = v;
    }
}

void NNUE::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::in);
    read_matrix<H1_SIZE, INPUT_SIZE, TRANSPOSE>(ifs, weights_and_biases + INPUT_TO_H1);
    read_bias<H1_SIZE>(ifs, weights_and_biases + H1_BIAS);
    read_matrix<H2_SIZE, H1_SIZE>(ifs, weights_and_biases + H1_TO_H2);
    read_bias<H2_SIZE>(ifs, weights_and_biases + H2_BIAS);
    read_matrix<H3_SIZE, H2_SIZE>(ifs, weights_and_biases + H2_TO_H3);
    read_bias<H3_SIZE>(ifs, weights_and_biases + H3_BIAS);
    read_matrix<OUTPUT_SIZE, H3_SIZE>(ifs, weights_and_biases + H3_TO_OUTPUT);
    read_bias<OUTPUT_SIZE>(ifs, weights_and_biases + OUTPUT_BIAS);
}

template<int M, int N, bool transpose = false>
static void save_matrix_quantized(std::ofstream& ofs, float* weights, float scale = 64) {
    ofs << "W\n" << M << '\n' << N << '\n';
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int w;                                    
            if constexpr (transpose) w = static_cast<int32_t>(std::round(scale * weights[j * M + i])); 
            else w = static_cast<int32_t>(std::round(scale * weights[i * N + j]));
            ofs << w << '\n';            
        }
    }
}

template<int N>
static void save_bias_quantized(std::ofstream& ofs, float* weights, float scale = 64) {
    ofs << "B\n" << N << '\n';
    for (int i = 0; i < N; i++) {
        int w = static_cast<int32_t>(std::round(scale * weights[i]));    
        ofs << w << '\n';
    }
}

void NNUE::save_quantized(const std::string& filename, float scale) {
    std::ofstream ofs(filename, std::ofstream::out);
    save_matrix_quantized<H1_SIZE, INPUT_SIZE, TRANSPOSE>(ofs, weights_and_biases + INPUT_TO_H1, scale);
    save_bias_quantized<H1_SIZE>(ofs, weights_and_biases + H1_BIAS, scale);
    save_matrix_quantized<H2_SIZE, H1_SIZE>(ofs, weights_and_biases + H1_TO_H2, scale);    
    save_bias_quantized<H2_SIZE>(ofs, weights_and_biases + H2_BIAS, scale * scale);
    save_matrix_quantized<H3_SIZE, H2_SIZE>(ofs, weights_and_biases + H2_TO_H3, scale);
    save_bias_quantized<H3_SIZE>(ofs, weights_and_biases + H3_BIAS, scale * scale);
    save_matrix_quantized<OUTPUT_SIZE, H3_SIZE>(ofs, weights_and_biases + H3_TO_OUTPUT, scale);
    save_bias_quantized<OUTPUT_SIZE>(ofs, weights_and_biases + OUTPUT_BIAS, scale * scale);
}

static inline std::tuple<uint64_t, uint64_t, uint64_t> encode_yolah(const Yolah& yolah) {
    // black positions + white positions + empty positions 
    const uint64_t black = yolah.bitboard(Yolah::BLACK);
    const uint64_t white = yolah.bitboard(Yolah::WHITE);
    const uint64_t empty = yolah.empty_bitboard();
    return { black, white, empty };
}

void NNUE::init(const Yolah& yolah, Accumulator& a) {
    const float* h1_bias = weights_and_biases + H1_BIAS;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }    
    const auto [black, white, empty] = encode_yolah(yolah);     
    const float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    int delta = 0;
    for (uint64_t bitboard : { black, white, empty }) {
        while (bitboard) {
            uint64_t pos = std::countr_zero(bitboard & -bitboard);
            int row = (delta + 63 - pos) * H1_SIZE;
            for (int j = 0; j < H1_SIZE; j++) {
                a.acc[j] += input_to_h1[row + j];
            }
            bitboard &= bitboard - 1;
        }
        delta += 64;
    }
    if (yolah.current_player() == Yolah::WHITE) {
        const float* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE; 
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn[i];
        }
    }
}

void NNUE::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int j = 0; j < H1_SIZE; j++) {
        float v1 = -input_to_h1[from_offset + j];
        float v2 = input_to_h1[to_offset + j];
        float v3 = input_to_h1[empty_offset + j];
        a.acc[j] += v1 + v2 + v3;
    }
    const float* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
    if (player == Yolah::BLACK) {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn[i];
        }
    } else {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] -= turn[i];
        }
    }
}

void NNUE::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int j = 0; j < H1_SIZE; j++) {
        float v1 = input_to_h1[from_offset + j];
        float v2 = -input_to_h1[to_offset + j];
        float v3 = -input_to_h1[empty_offset + j];
        a.acc[j] += v1 + v2 + v3;
    }
    float* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
    if (player == Yolah::BLACK) {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] -= turn[i];
        }
    } else {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn[i];
        }
    }
}

// g++ -std=c++2a -O3 -march=native -mavx2 -ffast-math -funroll-loops -I../game -I../misc -I../eigen ../game/zobrist.cpp ../game/magic.cpp ../game/game.cpp nnue.cpp
// int main(int argc, char* argv[]) {
//     using namespace std;
//     NNUE nnue;
//     nnue.load("nnue.txt");
//     nnue.save_quantized("nnue_quantized.txt", 64);
//     auto acc = nnue.make_accumulator();
//     ifstream ifs(argv[1], std::ifstream::in);
//     regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
//     size_t i = 0;
//     while (ifs) {
//         Yolah yolah;
//         nnue.init(yolah, acc);
//         string line; 
//         getline(ifs, line);
//         if (line == "") continue;
//         for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
//             const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
//             cout << setprecision(17) << black_proba << '\n';
//             cout << draw_proba << '\n';
//             cout << white_proba << '\n';
//             smatch match = *it;
//             string match_str = match.str();
//             Square sq1 = make_square(match[2].str());
//             Square sq2 = make_square(match[3].str());
//             Move m(sq1, sq2);
//             nnue.play(yolah.current_player(), m, acc);
//             yolah.play(m);        
//         }
//     }
// }
