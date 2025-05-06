#include "nnue_q3072_basic.h"
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

static constexpr int FACTOR = 4096;
static constexpr int SHIFT = 12;

// static constexpr int FACTOR2 = 512;
// static constexpr int SHIFT2 = 9;


static inline void matvec_16x3072(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int16_t* __restrict__ c, 
                                    const int32_t* __restrict__ bias) {
    constexpr int m = 16;
    constexpr int n = 3072;
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)a[i * n + j] * int32_t(b[i * n + j]); 
            // std::cout << (int32_t)a[i * n + j] * int32_t(b[i * n + j]) << ' ' << sum << '\n';
            // std::string _;
            // std::getline(std::cin, _);
        }
        sum += bias[i];
        // std::cout << sum / (8192.0 * 8192.0) << '\n';
        sum >>= SHIFT;
        c[i] = std::min(std::max(0, sum), FACTOR);
    }
    std::cout << '\n';
}

static inline void matvec_16x32(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int16_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {    
    constexpr int m = 16;
    constexpr int n = 32;
    for (int i = 0; i < m; i++) {
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)a[i * n + j] * (int32_t)b[i * n + j]; 
        }
        sum += bias[i];
        sum >>= SHIFT;
        c[i] = std::min(std::max(0, sum), FACTOR);
    }
}

static inline void matvec_3x32(const int16_t* __restrict__ a, const int16_t* __restrict__ b, int32_t* __restrict__ c,
                                const int32_t* __restrict__ bias) {
    constexpr int m = 3;
    constexpr int n = 32;
    for (int i = 0; i < m; i++) {
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)a[i * n + j] * (int32_t)b[i * n + j];
        }
        sum += bias[i];
        c[i] = sum;
    }
}

NNUE_Q3072_BASIC::Accumulator NNUE_Q3072_BASIC::make_accumulator() const {
    Accumulator a;
    for (int i = 0; i < H1_SIZE; i++) {
        a.acc[i] = h1_bias[i];
    }
    return a;
}

std::tuple<float, float, float> NNUE_Q3072_BASIC::output(Accumulator& a) {        
    alignas(64) int32_t output[OUTPUT_SIZE + 1]{};
    alignas(64) int16_t h1[H1_SIZE];
    alignas(64) int16_t h2[H2_SIZE];
    alignas(64) int16_t h3[H3_SIZE];
    for (int i = 0; i < H1_SIZE; i++) {
        int32_t v = a.acc[i];
        h1[i] = v <= 0 ? 0 : (v >= FACTOR ? FACTOR : v);
        std::cout << h1[i] / (float)FACTOR << ' ';
    }
    std::cout << '\n' << "####################\n";
    matvec_16x3072(h1_to_h2, h1, h2, h2_bias);
    for (int i = 0; i < H2_SIZE; i++) {
        std::cout << h2[i] / (float)FACTOR << ' ';
    }
    std::cout << '\n' << "####################\n";
    matvec_16x32(h2_to_h3, h2, h3, h3_bias);
    for (int i = 0; i < H3_SIZE; i++) {
        std::cout << h3[i] / (float)FACTOR << ' ';
    }
    std::cout << '\n' << "####################\n";
    matvec_3x32(h3_to_output, h3, output, output_bias);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << h3[i] / (float)FACTOR << ' ';
    }
    std::cout << '\n' << "####################\n";
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

void NNUE_Q3072_BASIC::load(const std::string& filename) {
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

void NNUE_Q3072_BASIC::init(const Yolah& yolah, Accumulator& a) {
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

void NNUE_Q3072_BASIC::play(uint8_t player, const Move& m, Accumulator& a) {
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

void NNUE_Q3072_BASIC::undo(uint8_t player, const Move& m, Accumulator& a) {
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

int main(int argc, char* argv[]) {
    using namespace std;
    NNUE_Q3072_BASIC nnue;
    try {
        nnue.load("nnue_q_3072x16x32x3.txt");
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
