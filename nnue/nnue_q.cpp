#include "nnue_q.h"
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

static inline void matvec(int n, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {    
    for (int i = 0; i < 64; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int16_t)a[i * n + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void matvec3x64(const int8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {
    for (int i = 0; i < 3; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 64; j++) {
            sum += (int16_t)a[i * 64 + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void relu(int n, int32_t* input, int8_t* output) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] >= 0 ? input[i] & 0x7F : 0;
    }
}

static inline void addvec(int n, const int8_t* __restrict__ src, int32_t* __restrict__ dst) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

static constexpr float FACTOR = 128.0;
static constexpr int SHIFT = 7;

NNUE_Q::NNUE_Q() {
    constexpr int n = 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE + H3_SIZE * OUTPUT_SIZE;
    weights_and_biases = (int8_t*)aligned_alloc(32, 32 * (n + 31) / 32);    
}

NNUE_Q::~NNUE_Q() {
    free(weights_and_biases);
}

NNUE_Q::Accumulator NNUE_Q::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = weights_and_biases[H1_BIAS + i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Q::output(Accumulator& a) {        
    int32_t h2[H2_SIZE];
    int32_t h3[H3_SIZE];
    int32_t output[OUTPUT_SIZE];
    int8_t h1[H1_SIZE];
    int8_t h2_8[H2_SIZE];
    int8_t h3_8[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        int32_t v = a.acc[i] >> SHIFT;
        h1[i] = v >= 0 ? v & 0x7F : 0;
        std::cout << (int)h1[i] << " ";
    }
    std::cout << '\n' << "#################\n";
    matvec(H1_SIZE, weights_and_biases + H1_TO_H2, h1, h2);    
    addvec(H2_SIZE, weights_and_biases + H2_BIAS, h2);
    for (int i = 0; i < H2_SIZE; i++) {
        h2[i] >>= SHIFT;
    }
    relu(H2_SIZE, h2, h2_8);
    for (int i = 0; i < H2_SIZE; i++) {
        std::cout << (int)h2_8[i] << ' ';
    }
    std::cout << '\n' << "#################\n";
    matvec(H2_SIZE, weights_and_biases + H2_TO_H3, h2_8, h3);
    addvec(H3_SIZE, weights_and_biases + H3_BIAS, h3);
    for (int i = 0; i < H3_SIZE; i++) {
        h3[i] >>= SHIFT;
    }
    relu(H3_SIZE, h3, h3_8);
    for (int i = 0; i < H3_SIZE; i++) {
        std::cout << (int)h3_8[i] << ' ';
    }
    std::cout << '\n' << "#################\n";
    matvec3x64(weights_and_biases + H3_TO_OUTPUT, h3_8, output);
    addvec(OUTPUT_SIZE, weights_and_biases + OUTPUT_BIAS, output);
    
    float e1 = std::exp(output[0] / FACTOR);
    float e2 = std::exp(output[1] / FACTOR);
    float e3 = std::exp(output[2] / FACTOR);
    float sum = e1 + e2 + e3;
    return { e1 / sum, e2 / sum, e3 / sum };
}

static constexpr bool TRANSPOSE = true; 

template<int M, int N, bool transpose = false>
static void read_matrix(std::ifstream& ifs, int8_t* weights) {
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

template<int N>
void read_bias(std::ifstream& ifs, int8_t* weights) {
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

void NNUE_Q::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::in);
    read_matrix<H1_SIZE, INPUT_SIZE, TRANSPOSE>(ifs, weights_and_biases + INPUT_TO_H1);
    read_bias<H1_SIZE>(ifs, weights_and_biases + H1_BIAS);
    read_matrix<H2_SIZE, H1_SIZE, TRANSPOSE>(ifs, weights_and_biases + H1_TO_H2);
    read_bias<H2_SIZE>(ifs, weights_and_biases + H2_BIAS);
    read_matrix<H3_SIZE, H2_SIZE, TRANSPOSE>(ifs, weights_and_biases + H2_TO_H3);
    read_bias<H3_SIZE>(ifs, weights_and_biases + H3_BIAS);
    read_matrix<OUTPUT_SIZE, H3_SIZE>(ifs, weights_and_biases + H3_TO_OUTPUT);
    read_bias<OUTPUT_SIZE>(ifs, weights_and_biases + OUTPUT_BIAS);
    constexpr int pos = 64 * 5;
    int16_t* turn_white = (int16_t*)(weights_and_biases + TURN_WHITE);
    int8_t* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int i = 0; i < 64; i++) {
        int row = (pos + i) * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) {
            turn_white[j] += input_to_h1[row + j];
        }
    }
    // for (int i = 0; i < 3 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE + H3_SIZE * OUTPUT_SIZE; i++) {
    //     std::cout << (int)weights_and_biases[i] << std::endl;
    // }
}

static inline std::tuple<uint64_t, uint64_t, uint64_t , uint64_t, uint64_t> encode_yolah(const Yolah& yolah) {
    // black positions + white positions + empty positions + occupied positions + free positions 
    const uint64_t black = yolah.bitboard(Yolah::BLACK);
    const uint64_t white = yolah.bitboard(Yolah::WHITE);
    const uint64_t empty = yolah.empty_bitboard();
    const uint64_t occupied = yolah.occupied_squares();
    const uint64_t free = yolah.free_squares();
    return { black, white, empty, occupied, free };
}

void NNUE_Q::init(const Yolah& yolah, Accumulator& a) {
    int8_t* h1_bias = weights_and_biases + H1_BIAS;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];       
    }
    const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
    int16_t* turn_white = (int16_t*)(weights_and_biases + TURN_WHITE);
    int8_t* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    int delta = 0;
    for (uint64_t bitboard : {black, white, empty, occupied, free}) {
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
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn_white[i];
        }
    }
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] *= 16;
        std::cout << a.acc[i] << ' ';
    }
    std::cout << "\n---------------\n";
}

void NNUE_Q::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions + occupied positions + free positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    int occupied_offset = (192 + to) * H1_SIZE;
    int free_offset = (256 + to) * H1_SIZE;
    int8_t* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int j = 0; j < H1_SIZE; j++) {
        int32_t v1 = -input_to_h1[from_offset + j];
        int32_t v2 = input_to_h1[to_offset + j];
        int32_t v3 = input_to_h1[empty_offset + j];
        int32_t v4 = input_to_h1[occupied_offset + j];
        int32_t v5 = -input_to_h1[free_offset + j];
        a.acc[j] += v1 + v2 + v3 + v4 + v5;
    }
    int16_t* turn_white = (int16_t*)(weights_and_biases + TURN_WHITE);
    if (player == Yolah::BLACK) {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn_white[i];
        }
    } else {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] -= turn_white[i];
        }
    }
}

void NNUE_Q::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions + occupied positions + free positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    int occupied_offset = (192 + to) * H1_SIZE;
    int free_offset = (256 + to) * H1_SIZE;
    int8_t* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int j = 0; j < H1_SIZE; j++) {
        int32_t v1 = input_to_h1[from_offset + j];
        int32_t v2 = -input_to_h1[to_offset + j];
        int32_t v3 = -input_to_h1[empty_offset + j];
        int32_t v4 = -input_to_h1[occupied_offset + j];
        int32_t v5 = input_to_h1[free_offset + j];
        a.acc[j] += v1 + v2 + v3 + v4 + v5;
    }
    int16_t* turn_white = (int16_t*)(weights_and_biases + TURN_WHITE);
    if (player == Yolah::BLACK) {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] -= turn_white[i];
        }
    } else {
        for (int i = 0; i < H1_SIZE; i++) {
            a.acc[i] += turn_white[i];
        }
    }
}

int main(int argc, char* argv[]) {
    using namespace std;
    NNUE_Q nnue;
    try {
        nnue.load("nnue_q_parameters.txt");
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
        //nnue.init(yolah, acc);
        string line;
        getline(ifs, line);
        if (line == "") continue;
        for (auto it = sregex_iterator(begin(line), end(line), re_moves); it != sregex_iterator(); ++it) {
            nnue.init(yolah, acc);
            const auto [black_proba, draw_proba, white_proba] = nnue.output(acc);
            cout << setprecision(17) << black_proba << '\n';
            cout << draw_proba << '\n';
            cout << white_proba << '\n';
            smatch match = *it;
            string match_str = match.str();
            //cout << match_str << '\n';
            Square sq1 = make_square(match[2].str());
            Square sq2 = make_square(match[3].str());
            //cout << sq1 << ':' << sq2 << '\n';
            Move m(sq1, sq2);
            //nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            return 0;
            // yolah.undo(m);
            // nnue.undo(yolah.current_player(), m, acc);
            // nnue.play(yolah.current_player(), m, acc);                        
            // yolah.play(m);
            //cout << yolah << '\n';            
        }
    }
}