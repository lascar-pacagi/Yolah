#include "nnue_q768x160x64x3.h"
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

static constexpr int16_t FACTOR = 64;
static constexpr int SHIFT = 6;

static inline __m128i convert_8x32_to_8x8(__m256i input32) {
    __m128i lo = _mm256_castsi256_si128(input32);
    __m128i hi = _mm256_extracti128_si256(input32, 1);
    __m128i packed16 = _mm_packs_epi32(lo, hi);
    return _mm_packs_epi16(packed16, _mm_setzero_si128());
}

static inline __m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i lo = _mm256_castsi256_si128(sum0);
    __m128i hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(_mm_add_epi32(lo, hi), bias);
};

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
};

static inline void mm256_dpwssds_avx_epi32(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    __m256i one = _mm256_set1_epi16(1);
    product0 = _mm256_madd_epi16(product0, one);
    acc = _mm256_add_epi32(acc, product0);
#endif
}

static inline void matvec_160x768(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c, 
                                    const int16_t* __restrict__ bias) {
    constexpr int register_width = 256 / 8;    
    constexpr int m = 160;
    constexpr int n = 768;
    constexpr int num_in_chunks = n / register_width;
    constexpr int num_out_chunks = m / 8;

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i factor  = _mm256_set1_epi32(FACTOR); 

    for (int i = 0; i < num_out_chunks; ++i) {
        const int offset0 = (i * 8 + 0) * n;
        const int offset1 = (i * 8 + 1) * n;
        const int offset2 = (i * 8 + 2) * n;
        const int offset3 = (i * 8 + 3) * n;
        const int offset4 = (i * 8 + 4) * n;
        const int offset5 = (i * 8 + 5) * n;
        const int offset6 = (i * 8 + 6) * n;
        const int offset7 = (i * 8 + 7) * n;

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        __m256i sum4 = _mm256_setzero_si256();
        __m256i sum5 = _mm256_setzero_si256();
        __m256i sum6 = _mm256_setzero_si256();
        __m256i sum7 = _mm256_setzero_si256();

        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((__m256i*)&b[j * register_width]);
            mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&a[offset0 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&a[offset1 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&a[offset2 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&a[offset3 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum4, in, _mm256_load_si256((__m256i*)&a[offset4 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum5, in, _mm256_load_si256((__m256i*)&a[offset5 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum6, in, _mm256_load_si256((__m256i*)&a[offset6 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum7, in, _mm256_load_si256((__m256i*)&a[offset7 + j * register_width]));
        }
        const __m256i _bias = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i*)&bias[i * 8]));
        __m256i outval = m256_haddx8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, _bias);
        outval = _mm256_srai_epi32(outval, SHIFT);
        outval = _mm256_min_epi32(_mm256_max_epi32(outval, zero), factor);        
        _mm_storel_epi64((__m128i*)&c[i * 8], convert_8x32_to_8x8(outval));
    }
}

static inline void matvec_64x160(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c,
                                    const int16_t* __restrict__ bias) {    
    constexpr int register_width = 256 / 8;    
    constexpr int m = 64;
    constexpr int n = 160;
    constexpr int num_in_chunks = n / register_width;
    constexpr int num_out_chunks = m / 8;

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i factor  = _mm256_set1_epi32(FACTOR); 

    for (int i = 0; i < num_out_chunks; ++i) {
        const int offset0 = (i * 8 + 0) * n;
        const int offset1 = (i * 8 + 1) * n;
        const int offset2 = (i * 8 + 2) * n;
        const int offset3 = (i * 8 + 3) * n;
        const int offset4 = (i * 8 + 4) * n;
        const int offset5 = (i * 8 + 5) * n;
        const int offset6 = (i * 8 + 6) * n;
        const int offset7 = (i * 8 + 7) * n;

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        __m256i sum4 = _mm256_setzero_si256();
        __m256i sum5 = _mm256_setzero_si256();
        __m256i sum6 = _mm256_setzero_si256();
        __m256i sum7 = _mm256_setzero_si256();

        for (int j = 0; j < num_in_chunks; ++j) {
            const __m256i in = _mm256_load_si256((__m256i*)&b[j * register_width]);
            mm256_dpwssds_avx_epi32(sum0, in, _mm256_load_si256((__m256i*)&a[offset0 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum1, in, _mm256_load_si256((__m256i*)&a[offset1 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum2, in, _mm256_load_si256((__m256i*)&a[offset2 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum3, in, _mm256_load_si256((__m256i*)&a[offset3 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum4, in, _mm256_load_si256((__m256i*)&a[offset4 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum5, in, _mm256_load_si256((__m256i*)&a[offset5 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum6, in, _mm256_load_si256((__m256i*)&a[offset6 + j * register_width]));
            mm256_dpwssds_avx_epi32(sum7, in, _mm256_load_si256((__m256i*)&a[offset7 + j * register_width]));
        }
        const __m256i _bias = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i*)&bias[i * 8]));
        __m256i outval = m256_haddx8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, _bias);
        outval = _mm256_srai_epi32(outval, SHIFT);
        outval = _mm256_min_epi32(_mm256_max_epi32(outval, zero), factor);
        _mm_storel_epi64((__m128i*)&c[i * 8], convert_8x32_to_8x8(outval));
    }
}

static inline void matvec_3x64(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c,
                                const int16_t* __restrict__ bias) {
    constexpr int register_width = 256 / 8;    
    constexpr int n = 64;
    constexpr int num_in_chunks = n / register_width;
    
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
    const __m128i _bias = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i*)bias));
    __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, _bias);
    outval = _mm_srai_epi32(outval, SHIFT);
    _mm_store_si128((__m128i*)c, outval);
}

NNUE_Q768x160x64x3::Accumulator NNUE_Q768x160x64x3::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Q768x160x64x3::output(Accumulator& a) {        
    alignas(64) int32_t output[OUTPUT_SIZE + 1]{};
    alignas(64) int8_t h1[H1_SIZE];
    alignas(64) int8_t h2[H2_SIZE];
    alignas(64) int8_t h3[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        const int16_t v = a.acc[i];        
        h1[i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v);
        //std::cout << (int)h1[i] << ' ';
    }
    //std::cout << std::endl;
    matvec_160x768(h1_to_h2, h1, h2, h2_bias);
    // for (int i = 0; i < H2_SIZE; i++) {
    //     std::cout << (int)h2[i] << ' ';
    // }
    //std::cout << std::endl;
    matvec_64x160(h2_to_h3, h2, h3, h3_bias);
    // for (int i = 0; i < H3_SIZE; i++) {
    //     std::cout << (int)h3[i] << ' ';
    // }
    // std::cout << std::endl;
    matvec_3x64(h3_to_output, h3, output, output_bias);
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     std::cout << (int)output[i] << ' ';
    // }
    // std::cout << std::endl;
    float e1 = std::exp(output[0] / ((float)FACTOR));
    float e2 = std::exp(output[1] / ((float)FACTOR));
    float e3 = std::exp(output[2] / ((float)FACTOR));
    //std::cout << e1 << ' ' << e2 << ' ' << e3 << std::endl;
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

void NNUE_Q768x160x64x3::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::in);
    read_matrix<int8_t, H1_SIZE, INPUT_SIZE, TRANSPOSE>(ifs, input_to_h1);
    read_bias<int16_t, H1_SIZE>(ifs, h1_bias);
    read_matrix<int8_t, H2_SIZE, H1_SIZE>(ifs, h1_to_h2);
    read_bias<int16_t, H2_SIZE>(ifs, h2_bias);
    read_matrix<int8_t, H3_SIZE, H2_SIZE>(ifs, h2_to_h3);
    read_bias<int16_t, H3_SIZE>(ifs, h3_bias);
    read_matrix<int8_t, OUTPUT_SIZE, H3_SIZE>(ifs, h3_to_output);
    read_bias<int16_t, OUTPUT_SIZE>(ifs, output_bias);
}

static inline std::tuple<uint64_t, uint64_t, uint64_t> encode_yolah(const Yolah& yolah) {
    // black positions + white positions + empty positions 
    const uint64_t black = yolah.bitboard(Yolah::BLACK);
    const uint64_t white = yolah.bitboard(Yolah::WHITE);
    const uint64_t empty = yolah.empty_bitboard();
    return { black, white, empty };
}

void NNUE_Q768x160x64x3::init(const Yolah& yolah, Accumulator& a) {
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
        const int8_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn[i];
        }
    }
}

void NNUE_Q768x160x64x3::play(uint8_t player, const Move& m, Accumulator& a) {
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
    const int8_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
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

void NNUE_Q768x160x64x3::undo(uint8_t player, const Move& m, Accumulator& a) {
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
    const int8_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
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
//     NNUE_Q768x160x64x3 nnue;
//     try {
//         nnue.load("nnue_q_768x160x64x3.5.txt");
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
