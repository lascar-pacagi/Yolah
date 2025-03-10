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

typedef float vec8 __attribute__ (( vector_size(8 * 4) ));

static inline void matvec(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {    
    vec8 sum[8]{};
    for (int i = 0; i < n; i++) {
        vec8 bb = vec8{} + b[i];
        const vec8* aa = (vec8*)&a[i * 64];
        for (int k = 0; k < 8; k++) {
            sum[k] += aa[k] * bb;
        }
    }
    for (int k = 0; k < 8; k++) {
        *((vec8*)&c[k * 8]) = sum[k];
    }
}

static inline void matvec3x64(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < 64; j++) {
            sum += a[i * 64 + j] * b[j];
        }
        c[i] = sum;
    }
}

static inline void relu(int n, float* output) {
    for (int i = 0; i < n; i++) {
        output[i] = output[i] >= 0 ? output[i] : 0;
    }
}

static inline void addvec(int n, const float* __restrict__ src, float* __restrict__ dst) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

NNUE::NNUE() {
    constexpr int n = 2 * H1_SIZE + INPUT_SIZE * H1_SIZE + H2_SIZE + H1_SIZE * H2_SIZE + H3_SIZE + H2_SIZE * H3_SIZE + OUTPUT_SIZE + H3_SIZE * OUTPUT_SIZE;
    weights_and_biases = (float*)aligned_alloc(32, 4 * 32 * (n + 31) / 32);    
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

std::tuple<float, float, float> NNUE::output(Accumulator& a) {    
    float h1[H1_SIZE];
    float h2[H2_SIZE];
    float h3[H3_SIZE];
    float output[OUTPUT_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        h1[i] = a.acc[i] >= 0 ? a.acc[i] : 0;
    }
    matvec(H1_SIZE, weights_and_biases + H1_TO_H2, h1, h2);    
    addvec(H2_SIZE, weights_and_biases + H2_BIAS, h2);
    relu(H2_SIZE, h2);
    
    matvec(H2_SIZE, weights_and_biases + H2_TO_H3, h2, h3);
    addvec(H3_SIZE, weights_and_biases + H3_BIAS, h3);
    relu(H3_SIZE, h3);
    
    matvec3x64(weights_and_biases + H3_TO_OUTPUT, h3, output);
    addvec(OUTPUT_SIZE, weights_and_biases + OUTPUT_BIAS, output);
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
    read_matrix<H2_SIZE, H1_SIZE, TRANSPOSE>(ifs, weights_and_biases + H1_TO_H2);
    read_bias<H2_SIZE>(ifs, weights_and_biases + H2_BIAS);
    read_matrix<H3_SIZE, H2_SIZE, TRANSPOSE>(ifs, weights_and_biases + H2_TO_H3);
    read_bias<H3_SIZE>(ifs, weights_and_biases + H3_BIAS);
    read_matrix<OUTPUT_SIZE, H3_SIZE>(ifs, weights_and_biases + H3_TO_OUTPUT);
    read_bias<OUTPUT_SIZE>(ifs, weights_and_biases + OUTPUT_BIAS);
    constexpr int pos = 64 * 5;
    float* turn_white = weights_and_biases + TURN_WHITE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    for (int i = 0; i < 64; i++) {
        int row = (pos + i) * H1_SIZE;
        for (int j = 0; j < H1_SIZE; j++) {
            turn_white[j] += input_to_h1[row + j];
        }
    }
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

void NNUE::init(const Yolah& yolah, Accumulator& a) {
    float* h1_bias = weights_and_biases + H1_BIAS;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
    float* turn_white = weights_and_biases + TURN_WHITE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
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
}

void NNUE::play(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions + occupied positions + free positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    int occupied_offset = (192 + to) * H1_SIZE;
    int free_offset = (256 + to) * H1_SIZE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    float* turn_white = weights_and_biases + TURN_WHITE;
    for (int j = 0; j < H1_SIZE; j++) {
        float v1 = -input_to_h1[from_offset + j];
        float v2 = input_to_h1[to_offset + j];
        float v3 = input_to_h1[empty_offset + j];
        float v4 = input_to_h1[occupied_offset + j];
        float v5 = -input_to_h1[free_offset + j];
        a.acc[j] += v1 + v2 + v3 + v4 + v5;
    }
    float* turn = weights_and_biases + TURN_WHITE;
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

void NNUE::undo(uint8_t player, const Move& m, Accumulator& a) {
    int from = 63 - m.from_sq();
    int to = 63 - m.to_sq();
    // black positions + white positions + empty positions + occupied positions + free positions
    int pos = (player == Yolah::BLACK) ? 0 : 64;    
    int from_offset = (pos + from) * H1_SIZE;
    int to_offset = (pos + to) * H1_SIZE;
    int empty_offset = (128 + from) * H1_SIZE;
    int occupied_offset = (192 + to) * H1_SIZE;
    int free_offset = (256 + to) * H1_SIZE;
    float* input_to_h1 = weights_and_biases + INPUT_TO_H1;
    float* turn_white = weights_and_biases + TURN_WHITE;
    for (int j = 0; j < H1_SIZE; j++) {
        float v1 = input_to_h1[from_offset + j];
        float v2 = -input_to_h1[to_offset + j];
        float v3 = -input_to_h1[empty_offset + j];
        float v4 = -input_to_h1[occupied_offset + j];
        float v5 = input_to_h1[free_offset + j];
        a.acc[j] += v1 + v2 + v3 + v4 + v5;
    }
    float* turn = weights_and_biases + TURN_WHITE;
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

// g++ -std=c++2a -O3 -march=native -mavx2 -ffast-math -funroll-loops -I../game -I../misc -I../eigen ../game/zobrist.cpp ../game/magic.cpp ../game/game.cpp nnue.cpp
/*
int main(int argc, char* argv[]) {
    using namespace std;
    NNUE nnue;
    nnue.load("nnue_parameters.txt");
    auto acc = nnue.make_accumulator();
    // Yolah yolah;
    // cout << yolah << '\n';
    // yolah.play(Move(make_square("a1"), make_square("a7")));
    // cout << yolah << '\n';
    // nnue.output_linear(yolah);
    //nnue.write(cout);
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
            smatch match = *it;
            string match_str = match.str();
            //cout << match_str << '\n';
            Square sq1 = make_square(match[2].str());
            Square sq2 = make_square(match[3].str());
            //cout << sq1 << ':' << sq2 << '\n';
            Move m(sq1, sq2);
            nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            yolah.undo(m);
            nnue.undo(yolah.current_player(), m, acc);
            nnue.play(yolah.current_player(), m, acc);                        
            yolah.play(m);
            //cout << yolah << '\n';            
        }
    }
}
*/
/*
int main() {
    magic::init();
    zobrist::init();
    NNUE<4096, 64, 64> nnue;
    auto acc = nnue.make_accumulator();
    float res = 0;
    PRNG prng(42);
    for (size_t i = 0; i < 10000; i++) {
        Yolah yolah;
        nnue.init(yolah, acc);
        Yolah::MoveList moves;
        while (!yolah.game_over()) {
            yolah.moves(moves);        
            Move m = moves[prng.rand<size_t>() % moves.size()];
            const auto [black_proba, ignore, white_proba] = nnue.output_softmax(acc);
            res += black_proba - white_proba;
            nnue.play(yolah.current_player(), m, acc);
            yolah.play(m);
        }
    }
    std::cout << res << std::endl;
}
*/
/*
using type = float;

void matmul(const type *a, const type *_b, type * __restrict__ c, size_t N) {
    alignas(32) type *b = new type[N * N];
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            b[i * N + j] = _b[j * N + i];
        }            
    }        
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[j * N + k];
            }                
        }            
    }        
}

void rinit(type* a, size_t N) {
    std::default_random_engine g(42);
    std::uniform_real_distribution<type> d;
    for (size_t i = 0; i < N; i++) {
        a[i] = d(g); 
    }
}

int main() {
    constexpr size_t N = 2048;
    alignas(32) type* a = new type[N * N];
    alignas(32) type* b = new type[N * N];
    alignas(32) type* c = new type[N * N];
    rinit(a, N * N);
    rinit(b, N * N);
    matmul(a, b, c, N);
    std::cout << c[0] << std::endl;
}
*/