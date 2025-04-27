#include "nnue_q512.h"
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

static constexpr int16_t FACTOR = 4096;
static constexpr int SHIFT = 12;

static inline __m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i sum128lo = _mm256_castsi256_si128(sum0);
    __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
};

static inline void mm256_dpwssds_avx_epi32(__m256i& acc, __m256i a, __m256i b) {
    #if defined (USE_VNNI)
        acc = _mm256_dpwssds_avx_epi32(acc, a, b);    
    #else    
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));    
    #endif
}

static inline void matvec_64x512(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int16_t* __restrict__ c, 
                                    const int32_t* __restrict__ bias) {
    constexpr int register_width = 256 / 16;    
    constexpr int m = 64;
    constexpr int n = 512;
    const int num_in_chunks = n / register_width;
    const int num_out_chunks = m / 4;

    const __m128i zero    = _mm_setzero_si128();
    const __m128i factor  = _mm_set1_epi32(FACTOR); 

    for (int i = 0; i < num_out_chunks; ++i) {
        const int offset0 = (i * 4 + 0) * n;
        const int offset1 = (i * 4 + 1) * n;
        const int offset2 = (i * 4 + 2) * n;
        const int offset3 = (i * 4 + 3) * n;

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((__m256i*)&b[j * register_width]);
            mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&a[offset0 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&a[offset1 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&a[offset2 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&a[offset3 + j * register_width]));
        }
        const __m128i _bias = _mm_load_si128((__m128i*)&bias[i * 4]);
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, _bias);
        outval = _mm_srai_epi32(outval, SHIFT);
        outval = _mm_min_epi32(_mm_max_epi32(outval, zero), factor);
        outval = _mm_packs_epi32(outval, zero);
        _mm_storeu_si64(&c[i * 4], outval);
    }
}

static inline void matvec_32x64(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int16_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {    
    constexpr int register_width = 256 / 16;    
    constexpr int m = 32;
    constexpr int n = 64;
    const int num_in_chunks = n / register_width;
    const int num_out_chunks = m / 4;

    const __m128i zero    = _mm_setzero_si128();
    const __m128i factor  = _mm_set1_epi32(FACTOR); 

    for (int i = 0; i < num_out_chunks; ++i) {
        const int offset0 = (i * 4 + 0) * n;
        const int offset1 = (i * 4 + 1) * n;
        const int offset2 = (i * 4 + 2) * n;
        const int offset3 = (i * 4 + 3) * n;

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((__m256i*)&b[j * register_width]);
            mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&a[offset0 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&a[offset1 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&a[offset2 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&a[offset3 + j * register_width]));
        }
        const __m128i _bias = _mm_load_si128((__m128i*)&bias[i * 4]);
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, _bias);
        outval = _mm_srai_epi32(outval, SHIFT);
        outval = _mm_min_epi32(_mm_max_epi32(outval, zero), factor);
        outval = _mm_packs_epi32(outval, zero);
        _mm_storeu_si64(&c[i * 4], outval);
    }
}

static inline void matvec_3x32(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {
    constexpr int register_width = 256 / 16;    
    constexpr int n = 32;
    const int num_in_chunks = n / register_width;
    
    const int offset0 = 0 * n;
    const int offset1 = 1 * n;
    const int offset2 = 2 * n;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    for (int j = 0; j < num_in_chunks; ++j) {
        const __m256i in = _mm256_load_si256((__m256i*)&b[j * register_width]);
        mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&a[offset0 + j * register_width]));
        mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&a[offset1 + j * register_width]));
        mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&a[offset2 + j * register_width]));
    }
    const __m128i _bias = _mm_load_si128((__m128i*)bias);
    __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, _bias);
    outval = _mm_srai_epi32(outval, SHIFT);

    _mm_store_si128((__m128i*)c, outval);
}

static inline void clamp(int n, int32_t* input, int16_t* output) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] <= 0 ? 0 : (input[i] >= FACTOR ? FACTOR : input[i]);
    }
}

NNUE_Q512::Accumulator NNUE_Q512::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Q512::output(Accumulator& a) {        
    alignas(32) int32_t output[OUTPUT_SIZE + 1]{};
    alignas(32) int16_t h1_16[H1_SIZE];
    alignas(32) int16_t h2_16[H2_SIZE];
    alignas(32) int16_t h3_16[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        const int32_t v = a.acc[i];        
        h1_16[i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v);
    }
    matvec_64x512(h1_to_h2, h1_16, h2_16, h2_bias);    
    matvec_32x64(h2_to_h3, h2_16, h3_16, h3_bias);
    matvec_3x32(h3_to_output, h3_16, output, output_bias);    
    float e1 = std::exp(output[0] / ((float)FACTOR));
    float e2 = std::exp(output[1] / ((float)FACTOR));
    float e3 = std::exp(output[2] / ((float)FACTOR));
    float sum = e1 + e2 + e3;
    return { e1 / sum, e2 / sum, e3 / sum };
}

static constexpr bool TRANSPOSE = true; 

template<typename T, int M, int N, bool transpose = false>
static void read_matrix(std::ifstream& ifs, T* weights) {
    int m, n;
    int v;
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

template<typename T, int N>
void read_bias(std::ifstream& ifs, T* weights) {
    int n;
    int v;
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

void NNUE_Q512::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::in);
    read_matrix<int16_t, H1_SIZE, INPUT_SIZE, TRANSPOSE>(ifs, input_to_h1);
    read_bias<int32_t, H1_SIZE>(ifs, h1_bias);
    read_matrix<int16_t, H2_SIZE, H1_SIZE>(ifs, h1_to_h2);
    read_bias<int32_t, H2_SIZE>(ifs, h2_bias);
    read_matrix<int16_t, H3_SIZE, H2_SIZE>(ifs, h2_to_h3);
    read_bias<int32_t, H3_SIZE>(ifs, h3_bias);
    read_matrix<int16_t, OUTPUT_SIZE, H3_SIZE>(ifs, h3_to_output);
    read_bias<int32_t, OUTPUT_SIZE>(ifs, output_bias);
}

static inline std::tuple<uint64_t, uint64_t, uint64_t> encode_yolah(const Yolah& yolah) {
    // black positions + white positions + empty positions 
    const uint64_t black = yolah.bitboard(Yolah::BLACK);
    const uint64_t white = yolah.bitboard(Yolah::WHITE);
    const uint64_t empty = yolah.empty_bitboard();
    return { black, white, empty };
}

void NNUE_Q512::init(const Yolah& yolah, Accumulator& a) {
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];       
    }
    const auto [black, white, empty] = encode_yolah(yolah);
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
        const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn[i];
        }
    }
}

void NNUE_Q512::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int16_t v1 = -input_to_h1[from_offset + j];
        int16_t v2 = input_to_h1[to_offset + j];
        int16_t v3 = input_to_h1[empty_offset + j];
        a.acc[j] += v1 + v2 + v3;
    }
    const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
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

void NNUE_Q512::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int16_t v1 = input_to_h1[from_offset + j];
        int16_t v2 = -input_to_h1[to_offset + j];
        int16_t v3 = -input_to_h1[empty_offset + j];
        a.acc[j] += v1 + v2 + v3;
    }
    const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
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

// int main(int argc, char* argv[]) {
//     using namespace std;
//     NNUE_Q512 nnue;
//     try {
//         nnue.load("nnue_q_512x64x32x3.txt");
//     } catch (const char* e) {
//         cout << e << endl;
//         exit(EXIT_FAILURE);
//     }    
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
//             //nnue.init(yolah, acc);
//             const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
//             cout << setprecision(17) << black_proba << '\n';
//             cout << draw_proba << '\n';
//             cout << white_proba << '\n';
//             //return 0;
//             smatch match = *it;
//             string match_str = match.str();
//             //cout << match_str << '\n';
//             Square sq1 = make_square(match[2].str());
//             Square sq2 = make_square(match[3].str());
//             //cout << sq1 << ':' << sq2 << '\n';
//             Move m(sq1, sq2);
//             nnue.play(yolah.current_player(), m, acc);                        
//             yolah.play(m);
//             // yolah.undo(m);
//             // nnue.undo(yolah.current_player(), m, acc);
//             // nnue.play(yolah.current_player(), m, acc);                        
//             // yolah.play(m);
//             //cout << yolah << '\n';            
//         }
//     }
// }
