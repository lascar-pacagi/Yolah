#include "nnue_q3.h"
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

static inline void matvec_128(int n, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {    
    for (int i = 0; i < 128; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)a[i * n + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void matvec_64(int n, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {    
    for (int i = 0; i < 64; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)a[i * n + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void matvec3x64(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {
    for (int i = 0; i < 3; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 64; j++) {
            sum += (int32_t)a[i * 64 + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void relu(int n, int32_t* input, int8_t* output) {
    for (int i = 0; i < n; i++) {
        output[i] = std::min(std::max(0, input[i]), 64);
    }
}

static inline void clamp(int n, int32_t* input, int8_t* output) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] <= 0 ? 0 : (input[i] >= 64 ? 64 : input[i]);
    }
}

static inline void addvec(int n, const int16_t* __restrict__ src, int32_t* __restrict__ dst) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i] * 64;
    }
}

static constexpr float FACTOR = 64.0;
static constexpr int SHIFT = 6;
static constexpr int A = 1;

NNUE_Q3::Accumulator NNUE_Q3::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Q3::output(Accumulator& a) {        
    int32_t h2[H2_SIZE];
    int32_t h3[H3_SIZE];
    int32_t output[OUTPUT_SIZE];
    int8_t h1[H1_SIZE];
    int8_t h2_8[H2_SIZE];
    int8_t h3_8[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        int32_t v = a.acc[i];// >> SHIFT;
        //h1[i] = std::min(std::max(0, v), 127);
        h1[i] = v <= 0 ? 0 : (v >= 64 ? 64 : v);
        std::cout << (float)h1[i] / 64 << " ";
    }
    std::cout << '\n' << "#################\n";
    matvec_128(H1_SIZE, h1_to_h2, h1, h2);    
    addvec(H2_SIZE, h2_bias, h2);
    for (int i = 0; i < H2_SIZE; i++) {
        h2[i] >>= SHIFT;
        //std::cout << h2[i] << ' ';
    }
    //std::cout << '\n' << "#################\n";
    clamp(H2_SIZE, h2, h2_8);
    for (int i = 0; i < H2_SIZE; i++) {
        std::cout << (int)h2_8[i] / 64.0 << ' ';
    }
    std::cout << '\n' << "#################\n";
    matvec_64(H2_SIZE, h2_to_h3, h2_8, h3);
    addvec(H3_SIZE, h3_bias, h3);
    for (int i = 0; i < H3_SIZE; i++) {
        h3[i] >>= SHIFT;
    }
    clamp(H3_SIZE, h3, h3_8);
    // for (int i = 0; i < H3_SIZE; i++) {
    //     std::cout << (int)h3_8[i] << ' ';
    // }
    // std::cout << '\n' << "#################\n";
    matvec3x64(h3_to_output, h3_8, output);
    addvec(OUTPUT_SIZE, output_bias, output);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] >>= SHIFT;
        //std::cout << h2[i] << ' ';
    }
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     std::cout << output[i] << ' ';
    // }

    // std::cout << '\n' << "#################\n";

    float e1 = std::exp(output[0] / (FACTOR * A));
    float e2 = std::exp(output[1] / (FACTOR * A));
    float e3 = std::exp(output[2] / (FACTOR * A));
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

void NNUE_Q3::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::in);
    read_matrix<int8_t, H1_SIZE, INPUT_SIZE, TRANSPOSE>(ifs, input_to_h1);
    read_bias<int16_t, H1_SIZE>(ifs, h1_bias);
    read_matrix<int8_t, H2_SIZE, H1_SIZE, TRANSPOSE>(ifs, h1_to_h2);
    read_bias<int16_t, H2_SIZE>(ifs, h2_bias);
    read_matrix<int8_t, H3_SIZE, H2_SIZE, TRANSPOSE>(ifs, h2_to_h3);
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

void NNUE_Q3::init(const Yolah& yolah, Accumulator& a) {
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
                a.acc[j] += input_to_h1[row + j] * A;
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
    // for (int i = 0; i < H1_SIZE; i++) {
    //     std::cout << a.acc[i] << ' ';
    // }
    // std::cout << "\n---------------\n";
}

void NNUE_Q3::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int32_t v1 = -input_to_h1[from_offset + j] * A;
        int32_t v2 = input_to_h1[to_offset + j] * A;
        int32_t v3 = input_to_h1[empty_offset + j] * A;
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

void NNUE_Q3::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    for (int j = 0; j < H1_SIZE; j++) {
        int32_t v1 = input_to_h1[from_offset + j] * A;
        int32_t v2 = -input_to_h1[to_offset + j] * A;
        int32_t v3 = -input_to_h1[empty_offset + j] * A;
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

int main(int argc, char* argv[]) {
    using namespace std;
    NNUE_Q3 nnue;
    try {
        nnue.load("nnue_q_1024x128x64x3.20.txt");
    } catch (const char* e) {
        cout << e << endl;
        exit(EXIT_FAILURE);
    }    
    auto acc = nnue.make_accumulator();
    ifstream ifs(argv[1], std::ifstream::in);
    regex re_moves(R"(((\w\d):(\w\d))+)", regex_constants::ECMAScript);
    size_t i = 0;
    while (ifs) {
        Yolah yolah;
        nnue.init(yolah, acc);
        string line;
        getline(ifs, line);
        if (line == "") continue;
        for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            //nnue.init(yolah, acc);
            const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
            cout << setprecision(17) << black_proba << '\n';
            cout << draw_proba << '\n';
            cout << white_proba << '\n';
            return 0;
            smatch match = *it;
            string match_str = match.str();
            //cout << match_str << '\n';
            Square sq1 = make_square(match[2].str());
            Square sq2 = make_square(match[3].str());
            //cout << sq1 << ':' << sq2 << '\n';
            Move m(sq1, sq2);
            nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            // yolah.undo(m);
            // nnue.undo(yolah.current_player(), m, acc);
            // nnue.play(yolah.current_player(), m, acc);                        
            // yolah.play(m);
            //cout << yolah << '\n';            
        }
    }
}
