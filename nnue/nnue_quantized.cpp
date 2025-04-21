#include "nnue_quantized.h"
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
static constexpr int A = 1;

// static inline void matvec_128(int n, const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c) {    
//     for (int i = 0; i < 128; i++) {
//         int32_t sum = 0;
//         for (int j = 0; j < n; j++) {
//             sum += (int32_t)a[i * n + j] * b[j];
//         }
//         c[i] = sum;
//     }
// }

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

static inline void matvec_128(int n, const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c, 
                                const int32_t* __restrict__ bias) {
    constexpr int register_width = 256 / 16;    
    constexpr int m = 128;
    const int num_in_chunks = n / register_width;
    const int num_out_chunks = m / 4;

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
        _mm_store_si128((__m128i*)&c[i * 4], outval);
    }
}

// typedef int32_t vec8 __attribute__ (( vector_size(8 * 4) ));

// static void matvec_128(int n, const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ c) {    
//     vec8 sum[8]{};
//     for (int i = 0; i < n; i++) {
//         vec8 bb = vec8{} + b[i];
//         const vec8* aa = (vec8*)&a[i * 128];
//         for (int k = 0; k < 8; k++) {
//             sum[k] += aa[k] * bb;
//         }
//     }
//     for (int k = 0; k < 8; k++) {
//         *((vec8*)&c[k * 8]) = sum[k];
//     }
//     memset(sum, 0, sizeof sum);
//     for (int i = 0; i < n; i++) {
//         vec8 bb = vec8{} + b[i];
//         const vec8* aa = (vec8*)&a[i * 128 + 64];
//         for (int k = 0; k < 8; k++) {
//             sum[k] += aa[k] * bb;
//         }
//     }
//     for (int k = 0; k < 8; k++) {
//         *((vec8*)&c[64 + k * 8]) = sum[k];
//     }
// }

static inline void matvec_64(int n, const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {    
    constexpr int register_width = 256 / 16;    
    constexpr int m = 64;
    const int num_in_chunks = n / register_width;
    const int num_out_chunks = m / 4;

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
        _mm_store_si128((__m128i*)&c[i * 4], outval);
    }
}

// static inline void matvec_64(int n, const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ c) {    
//     vec8 sum[8]{};
//     for (int i = 0; i < n; i++) {
//         vec8 bb = vec8{} + b[i];
//         const vec8* aa = (vec8*)&a[i * 64];
//         for (int k = 0; k < 8; k++) {
//             sum[k] += aa[k] * bb;
//         }
//     }
//     for (int k = 0; k < 8; k++) {
//         *((vec8*)&c[k * 8]) = sum[k];
//     }
// }

static inline void matvec3x64(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {
    constexpr int register_width = 256 / 16;    
    constexpr int n = 64;
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

// static inline void relu(int n, int32_t* input, int32_t* output) {
//     for (int i = 0; i < n; i++) {
//         output[i] = std::min(std::max(0, input[i]), FACTOR);
//     }
// }

static inline void clamp(int n, int32_t* input, int16_t* output) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] <= 0 ? 0 : (input[i] >= FACTOR ? FACTOR : input[i]);
    }
}

//_mm256_set1_epi32

// static inline void clamp(int n, const int32_t* input, int16_t* output) {
//     constexpr int in_register_width = 256 / 32;
//     constexpr int out_register_width = 256 / 16;
//     const int num_in_chunks = n / in_register_width;

//     const __m256i zero    = _mm256_setzero_si256();
//     const __m256i factor  = _mm256_set1_epi16(FACTOR); 
//     const __m256i control = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

//     for (int i = 0; i < num_in_chunks; i += 2) {
//         const __m256i in =
//             _mm256_packs_epi32(
//                 _mm256_load_si256((__m256i*)&input[i * in_register_width]),
//                 _mm256_load_si256((__m256i*)&input[(i + 1) * in_register_width])
//             );
//         const __m256i res =
//             _mm256_permutevar8x32_epi32(
//                 _mm256_min_epi16(_mm256_max_epi16(in, zero), factor),
//                 control
//             );
//         _mm256_store_si256((__m256i*)&output[(i / 2) * out_register_width], res);
//     }
// }

// static inline void addvec(int n, const int32_t* __restrict__ src, int32_t* __restrict__ dst) {
//     for (int i = 0; i < n; i++) {
//         dst[i] += src[i];
//     }
// }

NNUE_Quantized::Accumulator NNUE_Quantized::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Quantized::output(Accumulator& a) {        
    alignas(32) int32_t h2[H2_SIZE];
    alignas(32) int32_t h3[H3_SIZE];
    alignas(32) int32_t output[OUTPUT_SIZE + 1]{};
    alignas(32) int16_t h1[H1_SIZE];
    alignas(32) int16_t h2_16[H2_SIZE];
    alignas(32) int16_t h3_16[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        int32_t v = a.acc[i];
        //std::cout << v << ' ';
        //std::cout << (float)(v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v)) / FACTOR << " ";
        //h1[i] = std::min(std::max(0, v), 127);
        h1[i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v);
        //std::cout << (float)h1[i] / FACTOR << " ";
    }
    //std::cout << '\n' << "#################\n";
    matvec_128(H1_SIZE, h1_to_h2, h1, h2, h2_bias);    
    // for (int i = 0; i < H2_SIZE; i++) {
    //     std::cout << (float)h2[i] / (FACTOR * FACTOR) << ' ';
    // }
    // std::cout << '\n' << "#################\n";
    //addvec(H2_SIZE, h2_bias, h2);
    // for (int i = 0; i < H2_SIZE; i++) {
    //     std::cout << (float)h2[i] / (FACTOR * FACTOR) << ' ';
    // }
    // for (int i = 0; i < H2_SIZE; i++) {
    //     h2[i] >>= SHIFT;
    //     //std::cout << h2[i] << ' ';
    // }
    //std::cout << '\n' << "#################\n";
    clamp(H2_SIZE, h2, h2_16);
    // for (int i = 0; i < H2_SIZE; i++) {
    //     std::cout << (int)h2_8[i] / FACTOR << ' ';
    // }
    // std::cout << '\n' << "#################\n";
    matvec_64(H2_SIZE, h2_to_h3, h2_16, h3, h3_bias);
    // addvec(H3_SIZE, h3_bias, h3);
    // for (int i = 0; i < H3_SIZE; i++) {
    //     h3[i] >>= SHIFT;
    // }
    clamp(H3_SIZE, h3, h3_16);
    // for (int i = 0; i < H3_SIZE; i++) {
    //     std::cout << (int)h3_8[i] << ' ';
    // }
    // std::cout << '\n' << "#################\n";    
    matvec3x64(h3_to_output, h3_16, output, output_bias);
    // for (int i = 0; i < 4; i++) {
    //     std::cout << output[i] << ' ';
    // }
    // std::cout << std::endl;
    //addvec(OUTPUT_SIZE, output_bias, output);
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     output[i] >>= SHIFT;
    //     //std::cout << h2[i] << ' ';
    // }
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     std::cout << output[i] << ' ';
    // }

    // std::cout << '\n' << "#################\n";

    float e1 = std::exp(output[0] / ((float)FACTOR * A));
    float e2 = std::exp(output[1] / ((float)FACTOR * A));
    float e3 = std::exp(output[2] / ((float)FACTOR * A));
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
        weights[i] = v * A;
    }
}

void NNUE_Quantized::load(const std::string& filename) {
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

void NNUE_Quantized::init(const Yolah& yolah, Accumulator& a) {
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];       
    }
    const auto [black, white, empty] = encode_yolah(yolah);
    //std::cout << "popcount: " << std::popcount(empty) << std::endl;
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
    //for (int i = 0; i < H1_SIZE; i++) {
    //     std::cout << a.acc[i] << ' ';
    // }
    // std::cout << "\n---------------\n";
}

// void NNUE_Quantized::init(const Yolah& yolah, Accumulator& a) {
//     constexpr int register_width = 256 / 32;
//     constexpr int num_chunks = H1_SIZE / register_width;
//     __m256i regs[num_chunks];
//     for (int i = 0; i < num_chunks; i++) {
//         regs[i] = _mm256_load_si256((__m256i*)&h1_bias[i * register_width]);
//     }
//     const auto [black, white, empty] = encode_yolah(yolah);
//     int delta = 0;
//     for (uint64_t bitboard : { black, white, empty }) {
//         while (bitboard) {
//             uint64_t pos = std::countr_zero(bitboard & -bitboard);
//             int row = (delta + 63 - pos) * H1_SIZE;
//             for (int j = 0; j < num_chunks; j += 2) {
//                 __m256i v = _mm256_load_si256((__m256i*)&input_to_h1[row + j * register_width]);
//                 __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//                 __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//                 regs[j] = _mm256_add_epi32(regs[j], lo);
//                 regs[j + 1] = _mm256_add_epi32(regs[j + 1], hi);
//             }
//             bitboard &= bitboard - 1;
//         }
//         delta += 64;
//     }
//     if (yolah.current_player() == Yolah::WHITE) {
//         const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
//         for (int i = 0; i < num_chunks; i += 2) {
//             __m256i v = _mm256_load_si256((__m256i*)&turn[i * register_width]);
//             __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//             __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//             regs[i] = _mm256_add_epi32(regs[i], lo);
//             regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi);
//         }
//     }
//     for (int i = 0; i < num_chunks; i++) {
//         _mm256_store_si256((__m256i*)&a.acc[i * register_width], regs[i]);
//     }
// }

void NNUE_Quantized::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int16_t v1 = -input_to_h1[from_offset + j] * A;
        int16_t v2 = input_to_h1[to_offset + j] * A;
        int16_t v3 = input_to_h1[empty_offset + j] * A;
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

// void NNUE_Quantized::play(uint8_t player, const Move& m, Accumulator& a) {
//     constexpr int register_width = 256 / 32;
//     constexpr int num_chunks = H1_SIZE / register_width;
//     __m256i regs[num_chunks];
//     for (int i = 0; i < num_chunks; i++) {
//         regs[i] = _mm256_load_si256((__m256i*)&a.acc[i * register_width]);
//     }
//     int from = 63 - m.from_sq();
//     int to = 63 - m.to_sq();
//     // black positions + white positions + empty positions
//     int pos = (player == Yolah::BLACK) ? 0 : 64;
//     int from_offset = (pos + from) * H1_SIZE;
//     int to_offset = (pos + to) * H1_SIZE;
//     int empty_offset = (128 + from) * H1_SIZE;    
//     for (int i = 0; i < num_chunks; i += 2) {
//         __m256i v1 = _mm256_load_si256((__m256i*)&input_to_h1[from_offset + i * register_width]);
//         __m256i lo1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1));
//         __m256i hi1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1, 1));
//         regs[i] = _mm256_sub_epi32(regs[i], lo1);
//         regs[i + 1] = _mm256_sub_epi32(regs[i + 1], hi1);
//         __m256i v2 = _mm256_load_si256((__m256i*)&input_to_h1[to_offset + i * register_width]);
//         __m256i lo2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2));
//         __m256i hi2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2, 1));
//         regs[i] = _mm256_add_epi32(regs[i], lo2);
//         regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi2);
//         __m256i v3 = _mm256_load_si256((__m256i*)&input_to_h1[empty_offset + i * register_width]);
//         __m256i lo3 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v3));
//         __m256i hi3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v3, 1));
//         regs[i] = _mm256_add_epi32(regs[i], lo3);
//         regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi3);
//     }
//     const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
//     if (player == Yolah::BLACK) {
//         for (int i = 0; i < num_chunks; i += 2) {
//             __m256i v = _mm256_load_si256((__m256i*)&turn[i * register_width]);
//             __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//             __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//             regs[i] = _mm256_add_epi32(regs[i], lo);
//             regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi);
//         }
//     } else {
//         for (int i = 0; i < num_chunks; i += 2) {
//             __m256i v = _mm256_load_si256((__m256i*)&turn[i * register_width]);
//             __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//             __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//             regs[i] = _mm256_sub_epi32(regs[i], lo);
//             regs[i + 1] = _mm256_sub_epi32(regs[i + 1], hi);
//         }
//     }
//     for (int i = 0; i < num_chunks; i++) {
//         _mm256_store_si256((__m256i*)&a.acc[i * register_width], regs[i]);
//     }
// }

void NNUE_Quantized::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int16_t v1 = input_to_h1[from_offset + j] * A;
        int16_t v2 = -input_to_h1[to_offset + j] * A;
        int16_t v3 = -input_to_h1[empty_offset + j] * A;
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

// void NNUE_Quantized::undo(uint8_t player, const Move& m, Accumulator& a) {
//     constexpr int register_width = 256 / 32;
//     constexpr int num_chunks = H1_SIZE / register_width;
//     __m256i regs[num_chunks];
//     for (int i = 0; i < num_chunks; i++) {
//         regs[i] = _mm256_load_si256((__m256i*)&a.acc[i * register_width]);
//     }
//     int from = 63 - m.from_sq();
//     int to = 63 - m.to_sq();
//     // black positions + white positions + empty positions
//     int pos = (player == Yolah::BLACK) ? 0 : 64;
//     int from_offset = (pos + from) * H1_SIZE;
//     int to_offset = (pos + to) * H1_SIZE;
//     int empty_offset = (128 + from) * H1_SIZE;    
//     for (int i = 0; i < num_chunks; i += 2) {
//         __m256i v1 = _mm256_load_si256((__m256i*)&input_to_h1[from_offset + i * register_width]);
//         __m256i lo1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1));
//         __m256i hi1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1, 1));
//         regs[i] = _mm256_add_epi32(regs[i], lo1);
//         regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi1);
//         __m256i v2 = _mm256_load_si256((__m256i*)&input_to_h1[to_offset + i * register_width]);
//         __m256i lo2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2));
//         __m256i hi2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2, 1));
//         regs[i] = _mm256_sub_epi32(regs[i], lo2);
//         regs[i + 1] = _mm256_sub_epi32(regs[i + 1], hi2);
//         __m256i v3 = _mm256_load_si256((__m256i*)&input_to_h1[empty_offset + i * register_width]);
//         __m256i lo3 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v3));
//         __m256i hi3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v3, 1));
//         regs[i] = _mm256_sub_epi32(regs[i], lo3);
//         regs[i + 1] = _mm256_sub_epi32(regs[i + 1], hi3);
//     }
//     const int16_t* turn = input_to_h1 + (INPUT_SIZE - 1) * H1_SIZE;
//     if (player == Yolah::BLACK) {
//         for (int i = 0; i < num_chunks; i += 2) {
//             __m256i v = _mm256_load_si256((__m256i*)&turn[i * register_width]);
//             __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//             __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//             regs[i] = _mm256_sub_epi32(regs[i], lo);
//             regs[i + 1] = _mm256_sub_epi32(regs[i + 1], hi);
//         }
//     } else {
//         for (int i = 0; i < num_chunks; i += 2) {
//             __m256i v = _mm256_load_si256((__m256i*)&turn[i * register_width]);
//             __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
//             __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
//             regs[i] = _mm256_add_epi32(regs[i], lo);
//             regs[i + 1] = _mm256_add_epi32(regs[i + 1], hi);
//         }
//     }
//     for (int i = 0; i < num_chunks; i++) {
//         _mm256_store_si256((__m256i*)&a.acc[i * register_width], regs[i]);
//     }
// }

// int main(int argc, char* argv[]) {
//     using namespace std;
//     NNUE_Quantized nnue;
//     try {
//         nnue.load("nnue_q_1024x128x64x3.20.txt");
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
