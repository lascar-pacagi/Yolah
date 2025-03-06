#ifndef NNUE_H
#define NNUE_H
#include <cstddef>
#include <vector>
#include <tuple>
#include "game.h"
#include <fstream>
#include <string>
#include <iomanip>
#include "Eigen/Dense"

// black positions + white positions + empty positions + occupied positions + free positions + turn 
constexpr size_t INPUT_SIZE = 64 + 64 + 64 + 64 + 64 + 64;
constexpr size_t OUTPUT_SIZE = 3;

struct NNUE {  
    static constexpr int H1 = 4096;
    static constexpr int H2 = 64;
    static constexpr int H3 = 64;
    static constexpr int OUTPUT = 3;  
    float* acc;
    float* h1_to_h2;
    float* h2_to_h3;
    float* h3_to_output;
    nnue();
    float output();
    ~nnue();
};

/*
constexpr int VECTOR_LANES = 8;
constexpr int VECTOR_SIZE = 4 * VECTOR_LANES;
typedef float vec __attribute__ (( vector_size(VECTOR_SIZE) ));

template<int H1_SIZE, int H2_SIZE, int H3_SIZE>
class NNUE {
    int round_to_vector_size(int n) {
        return (n + VECTOR_SIZE - 1) / VECTOR_SIZE * VECTOR_SIZE; 
    }
    vec* alloc(int n) {
        vec* ptr = (vec*) std::aligned_alloc(VECTOR_SIZE, 4 * n);
        memset(ptr, 0, 4 * n);
        return ptr;
    }
    void kernel(float *a, vec *b, vec *c, int x, int y, int l, int r, int n) {
        vec t[6][2]{}; // will be zero-filled and stored in ymm registers

        for (int k = l; k < r; k++) {
            for (int i = 0; i < 6; i++) {
                // broadcast a[x + i][k] into a register
                vec alpha = vec{} + a[(x + i) * n + k]; // converts to a broadcast
                // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
                for (int j = 0; j < 2; j++) {
                    t[i][j] += alpha * b[(k * n + y) / VECTOR_SIZE + j]; // converts to an fma
                }                 
            }
        }
        // write the results back to C
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 2; j++) {
                c[((x + i) * n + y) / VECTOR_SIZE + j] += t[i][j];
            }
        }
    }
    void matmul(const float *_a, const float *_b, float *_c, int n) {
        // to simplify the implementation, we pad the height and width
        // so that they are divisible by 6 and 16 respectively
        int nx = round_to_vector_size((n + 5) / 6 * 6);
        int ny = round_to_vector_size((n + 15) / 16 * 16);
        
        for (int i = 0; i < n; i++) {
            memcpy(&a[i * ny], &_a[i * n], 4 * n);
            memcpy(&b[i * ny], &_b[i * n], 4 * n); // we don't need to transpose b this time
        }

        const int s3 = 64;  // how many columns of B to select
        const int s2 = 120; // how many rows of A to select 
        const int s1 = 240; // how many rows of B to select

        for (int i3 = 0; i3 < ny; i3 += s3) {
            // now we are working with b[:][i3:i3+s3]
            for (int i2 = 0; i2 < nx; i2 += s2) {
                // now we are working with a[i2:i2+s2][:]
                for (int i1 = 0; i1 < ny; i1 += s1) {
                    // now we are working with b[i1:i1+s1][i3:i3+s3]
                    // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
                    for (int x = i2; x < std::min(i2 + s2, nx); x += 6) {
                        for (int y = i3; y < std::min(i3 + s3, ny); y += 16) {
                            kernel(a, (vec*) b, (vec*) c, x, y, i1, std::min(i1 + s1, n), ny);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < n; i++) {
            memcpy(&_c[i * n], &c[i * ny], 4 * n);
        }
    }
    struct Accumulator {
        float* acc;
        float* output;
        Accumulator() {
            acc = (float*)alloc(H1_SIZE);
            output = (float*)alloc(std::max({H1_SIZE, H2_SIZE, H3_SIZE, OUTPUT_SIZE});
        }
        ~Accumulator() {
            delete[] acc;
            delete[] output;
        }
    };
private:
    float* turn_black;
    float* turn_white;
    float* input_to_h1;
    float* h1_bias;    
    float* h1_to_h2;
    float* h2_bias;
    float* h2_to_h3;
    float* h3_bias;
    float* h3_to_output;
    float* output_bias;
public:
    NNUE() {
        turn_black   = (float*)alloc(H1_SIZE);
        turn_white   = (float*)alloc(H1_SIZE);
        input_to_h1  = (float*)alloc(INPUT_SIZE * H1_SIZE); 
        h1_bias      = (float*)alloc(H1_SIZE);
        h1_to_h2     = (float*)alloc(H1_SIZE * H2_SIZE);
        h2_bias      = (float*)alloc(H2_SIZE);
        h2_to_h3     = (float*)alloc(H2_SIZE * H3_SIZE);
        h3_bias      = (float*)alloc(H3_SIZE);
        h3_to_output = (float*)alloc(H3_SIZE * OUTPUT_SIZE);
        output_bias  = (float*)alloc(OUTPUT_SIZE);
    }
    ~NNUE() {
        delete[] turn_black;
        delete[] turn_white;
        delete[] input_to_h1;
        delete[] h1_bias;
        delete[] h1_to_h2;
        delete[] h2_bias;
        delete[] h2_to_h3;
        delete[] h3_bias;
        delete[] h3_to_output;
        delete[] output_bias;
    }

    Accumulator make_accumulator() {
        return {};                
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ifstream::in);
        size_t n, m;
        float v;
        std::string type;
        bool first = true;
        for (auto& [N, M, weights, bias]: {
                std::make_tuple(H1_SIZE, INPUT_SIZE, input_to_h1, h1_bias), 
                std::make_tuple(H2_SIZE, H1_SIZE, h1_to_h2, h2_bias), 
                std::make_tuple(H3_SIZE, H2_SIZE, h2_to_h3, h3_bias),
                std::make_tuple(OUTPUT_SIZE, H3_SIZE, h3_to_output, output_bias)}) {
            ifs >> type;
            if (type != "W") {
                throw "W expected";
            }
            if (!(ifs >> n >> m)) {
                throw "matrix size expected";
            }
            if (n != N || m != M) {
                throw "bad matrix dimension";
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < M; j++) {
                    ifs >> v;
                    if (first) weights[j * N + i] =  v; 
                    else weights[i * M + j] =  v;                    
                }
            }
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
            for (size_t i = 0; i < N; i++) {
                ifs >> v;
                bias[i] = v;
            }
            first = false;
        }
        constexpr size_t pos = 64 * 5;
        for (size_t i = 0; i < 64; i++) {
            for (size_t j = 0; j < H1_SIZE; j++) {
                turn_white[j] += input_to_h1[(pos + i) * H1_SIZE + j];
            }     
        }    
        for (size_t i = 0; i < H1_SIZE; i++) {
            turn_black[i] = -turn_white[i];
        }
    }
    std::tuple<uint64_t, uint64_t, uint64_t , uint64_t, uint64_t> encode_yolah(const Yolah& yolah) const {
        // black positions + white positions + empty positions + occupied positions + free positions 
        const uint64_t black = yolah.bitboard(Yolah::BLACK);
        const uint64_t white = yolah.bitboard(Yolah::WHITE);
        const uint64_t empty = yolah.empty_bitboard();
        const uint64_t occupied = yolah.occupied_squares();
        const uint64_t free = yolah.free_squares();
        return {black, white, empty, occupied, free};
    }
    void init(const Yolah& yolah, Accumulator& a) {
        a.acc = h1_bias;
        const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
        size_t delta = 0;
        for (uint64_t bitboard : {black, white, empty, occupied, free}) {
            while (bitboard) {
                uint64_t pos = std::countr_zero(bitboard & -bitboard);
                size_t offset = delta + 63 - pos;
                //a.acc += input_to_h1.col(offset);
                a.add(input_to_h1, offset);
                bitboard &= bitboard - 1;
            }
            delta += 64;
        }
        if (yolah.current_player() == Yolah::WHITE) {
            a.add += turn_white;  
        }
    }
    std::tuple<float, float, float> output_linear(const Accumulator& a) {
        VectorXf h1_output = a.acc.array().max(0);        
        VectorXf h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        VectorXf h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        VectorXf output    = h3_to_output * h3_output + output_bias;
        return {output(0), output(1), output(2)};
    }
    std::tuple<float, float, float> output_softmax(const Accumulator& a) {
        VectorXf h1_output = a.acc.array().max(0);
        VectorXf h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        VectorXf h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        VectorXf output    = (h3_to_output * h3_output + output_bias).array().exp();
        auto sum  = output.sum();
        output    /= sum;
        return {output(0), output(1), output(2)};        
    }
    void play(uint8_t player, const Move& m, Accumulator& a) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;        
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        a.acc += input_to_h1.col(to_offset) 
                - input_to_h1.col(from_offset)
                + input_to_h1.col(empty_offset)
                + input_to_h1.col(occupied_offset)
                - input_to_h1.col(free_offset)
                + turn;
    }
    void undo(uint8_t player, const Move& m, Accumulator& a) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        a.acc += input_to_h1.col(from_offset)
                - input_to_h1.col(to_offset)
                - input_to_h1.col(empty_offset)
                - input_to_h1.col(occupied_offset)
                + input_to_h1.col(free_offset)
                - turn;
    }
};
*/
/*
using MatrixXf = Eigen::MatrixXf;
using RowVectorXf = Eigen::RowVectorXf;
using VectorXf = Eigen::VectorXf;
//using MatrixX = Eigen::MatrixX<int>;
//using VectorX = Eigen::VectorX<int>;

template<size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE>
class NNUE {
public:
    struct Accumulator {
        VectorXf acc;
        Accumulator() : acc(H1_SIZE) {}
    };
private:
    VectorXf turn_black;
    VectorXf turn_white;
    MatrixXf input_to_h1;
    VectorXf h1_bias;
    MatrixXf h1_to_h2;
    VectorXf h2_bias;
    MatrixXf h2_to_h3;
    VectorXf h3_bias;
    MatrixXf h3_to_output;
    VectorXf output_bias;
public:
    NNUE() :
        turn_black(H1_SIZE),
        turn_white(H1_SIZE),
        input_to_h1(H1_SIZE, INPUT_SIZE),
        h1_bias(H1_SIZE),
        h1_to_h2(H2_SIZE, H1_SIZE),
        h2_bias(H2_SIZE),
        h2_to_h3(H3_SIZE, H2_SIZE),
        h3_bias(H3_SIZE),
        h3_to_output(OUTPUT_SIZE, H3_SIZE),
        output_bias(OUTPUT_SIZE) {}

    Accumulator make_accumulator() {
        return {};        
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ifstream::in);
        size_t n, m;
        float v;
        std::string type;
        for (auto& [N, M, weights, bias]: {
                std::make_tuple(H1_SIZE, INPUT_SIZE, input_to_h1.data(), h1_bias.data()), 
                std::make_tuple(H2_SIZE, H1_SIZE, h1_to_h2.data(), h2_bias.data()), 
                std::make_tuple(H3_SIZE, H2_SIZE, h2_to_h3.data(), h3_bias.data()),
                std::make_tuple(OUTPUT_SIZE, H3_SIZE, h3_to_output.data(), output_bias.data())}) {
            ifs >> type;
            if (type != "W") {
                throw "W expected";
            }
            if (!(ifs >> n >> m)) {
                throw "matrix size expected";
            }
            if (n != N || m != M) {
                throw "bad matrix dimension";
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < M; j++) {
                    ifs >> v;
                    weights[j * N + i] =  v;                    
                }
            }
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
            for (size_t i = 0; i < N; i++) {
                ifs >> v;
                bias[i] = v;
            }
        }
        turn_white.fill(0);
        constexpr size_t pos = 64 * 5;
        for (size_t i = 0; i < 64; i++) {
            turn_white += input_to_h1.col(pos + i);                        
        }    
        turn_black = -turn_white;
    }
    std::tuple<uint64_t, uint64_t, uint64_t , uint64_t, uint64_t> encode_yolah(const Yolah& yolah) const {
        // black positions + white positions + empty positions + occupied positions + free positions 
        const uint64_t black = yolah.bitboard(Yolah::BLACK);
        const uint64_t white = yolah.bitboard(Yolah::WHITE);
        const uint64_t empty = yolah.empty_bitboard();
        const uint64_t occupied = yolah.occupied_squares();
        const uint64_t free = yolah.free_squares();
        return {black, white, empty, occupied, free};
    }
    void init(const Yolah& yolah, Accumulator& a) {
        a.acc = h1_bias;
        const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
        size_t delta = 0;
        for (uint64_t bitboard : {black, white, empty, occupied, free}) {
            while (bitboard) {
                uint64_t pos = std::countr_zero(bitboard & -bitboard);
                size_t offset = delta + 63 - pos;
                a.acc += input_to_h1.col(offset);
                bitboard &= bitboard - 1;
            }
            delta += 64;
        }
        if (yolah.current_player() == Yolah::WHITE) {
            a.acc += turn_white;  
        }
    }
    std::tuple<float, float, float> output_linear(const Accumulator& a) {
        VectorXf h1_output = a.acc.array().max(0);        
        VectorXf h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        VectorXf h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        VectorXf output    = h3_to_output * h3_output + output_bias;
        return {output(0), output(1), output(2)};
    }
    std::tuple<float, float, float> output_softmax(const Accumulator& a) {
        VectorXf h1_output = a.acc.array().max(0);
        VectorXf h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        VectorXf h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        VectorXf output    = (h3_to_output * h3_output + output_bias).array().exp();
        auto sum  = output.sum();
        output    /= sum;
        return {output(0), output(1), output(2)};        
    }
    void play(uint8_t player, const Move& m, Accumulator& a) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;        
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        a.acc += input_to_h1.col(to_offset) 
                - input_to_h1.col(from_offset)
                + input_to_h1.col(empty_offset)
                + input_to_h1.col(occupied_offset)
                - input_to_h1.col(free_offset)
                + turn;
    }
    void undo(uint8_t player, const Move& m, Accumulator& a) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        a.acc += input_to_h1.col(from_offset)
                - input_to_h1.col(to_offset)
                - input_to_h1.col(empty_offset)
                - input_to_h1.col(occupied_offset)
                + input_to_h1.col(free_offset)
                - turn;
    }
};
*/
/*
// black positions + white positions + empty positions + occupied positions + free positions + turn 
constexpr size_t INPUT_SIZE = 64 + 64 + 64 + 64 + 64 + 64;
constexpr size_t OUTPUT_SIZE = 3;

using MatrixXf = Eigen::MatrixXf;
using RowVectorXf = Eigen::RowVectorXf;
using VectorXf = Eigen::VectorXf;

template<size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE>
class NNUE {
    VectorXf accumulator;
    VectorXf turn_black;
    VectorXf turn_white;
    MatrixXf input_to_h1;
    VectorXf h1_bias;
    VectorXf h1_output;
    MatrixXf h1_to_h2;
    VectorXf h2_bias;
    VectorXf h2_output;
    MatrixXf h2_to_h3;
    VectorXf h3_bias;
    VectorXf h3_output;
    MatrixXf h3_to_output;
    VectorXf output_bias;
    VectorXf output;
public:
    NNUE() :
        accumulator(H1_SIZE),
        turn_black(H1_SIZE),
        turn_white(H1_SIZE),
        input_to_h1(H1_SIZE, INPUT_SIZE),
        h1_bias(H1_SIZE),
        h1_output(H1_SIZE),
        h1_to_h2(H2_SIZE, H1_SIZE),
        h2_bias(H2_SIZE),
        h2_output(H2_SIZE),
        h2_to_h3(H3_SIZE, H2_SIZE),
        h3_bias(H3_SIZE),
        h3_output(H3_SIZE),
        h3_to_output(OUTPUT_SIZE, H3_SIZE),
        output_bias(OUTPUT_SIZE),
        output(OUTPUT_SIZE) {}

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ifstream::in);
        size_t n, m;
        float v;
        std::string type;
        for (auto& [N, M, weights, bias]: {
                std::make_tuple(H1_SIZE, INPUT_SIZE, input_to_h1.data(), h1_bias.data()), 
                std::make_tuple(H2_SIZE, H1_SIZE, h1_to_h2.data(), h2_bias.data()), 
                std::make_tuple(H3_SIZE, H2_SIZE, h2_to_h3.data(), h3_bias.data()),
                std::make_tuple(OUTPUT_SIZE, H3_SIZE, h3_to_output.data(), output_bias.data())}) {
            ifs >> type;
            if (type != "W") {
                throw "W expected";
            }
            if (!(ifs >> n >> m)) {
                throw "matrix size expected";
            }
            if (n != N || m != M) {
                throw "bad matrix dimension";
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < M; j++) {
                    ifs >> v;
                    weights[j * N + i] =  v;                    
                }
            }
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
            for (size_t i = 0; i < N; i++) {
                ifs >> v;
                bias[i] = v;
            }
        }
        accumulator.fill(0);
        turn_white.fill(0);
        constexpr size_t pos = 64 * 5;
        for (size_t i = 0; i < 64; i++) {
            turn_white += input_to_h1.col(pos + i);                        
        }    
        turn_black = -turn_white;
    }
    std::tuple<uint64_t, uint64_t, uint64_t , uint64_t, uint64_t> encode_yolah(const Yolah& yolah) const {
        // black positions + white positions + empty positions + occupied positions + free positions 
        const uint64_t black = yolah.bitboard(Yolah::BLACK);
        const uint64_t white = yolah.bitboard(Yolah::WHITE);
        const uint64_t empty = yolah.empty_bitboard();
        const uint64_t occupied = yolah.occupied_squares();
        const uint64_t free = yolah.free_squares();
        return {black, white, empty, occupied, free};
    }
    void init(const Yolah& yolah) {
        accumulator.fill(0);
        const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
        size_t delta = 0;
        for (uint64_t bitboard : {black, white, empty, occupied, free}) {
            while (bitboard) {
                uint64_t pos = std::countr_zero(bitboard & -bitboard);
                size_t offset = delta + 63 - pos;
                accumulator += input_to_h1.col(offset);
                bitboard &= bitboard - 1;
            }
            delta += 64;
        }
        if (yolah.current_player() == Yolah::WHITE) {
            accumulator += turn_white;  
        }
    }
    std::tuple<float, float, float> output_linear() {
        h1_output = (accumulator + h1_bias).array().max(0);        
        h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        output    = h3_to_output * h3_output + output_bias;
        return {output(0), output(1), output(2)};
    }
    std::tuple<float, float, float> output_softmax() {
        h1_output = (accumulator + h1_bias).array().max(0);
        h2_output = (h1_to_h2 * h1_output + h2_bias).array().max(0);
        h3_output = (h2_to_h3 * h2_output + h3_bias).array().max(0);
        output    = (h3_to_output * h3_output + output_bias).array().exp();
        auto sum  = output.sum();
        output    /= sum;
        return {output(0), output(1), output(2)};
    }
    void play(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;        
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        accumulator += input_to_h1.col(to_offset) 
                        - input_to_h1.col(from_offset)
                        + input_to_h1.col(empty_offset)
                        + input_to_h1.col(occupied_offset)
                        - input_to_h1.col(free_offset)
                        + turn;
    }
    void undo(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
        size_t from_offset = pos + from;
        size_t to_offset = pos + to;
        size_t empty_offset = 128 + from;
        size_t occupied_offset = 192 + to;
        size_t free_offset = 256 + to;
        accumulator += input_to_h1.col(from_offset)
                        - input_to_h1.col(to_offset)
                        - input_to_h1.col(empty_offset)
                        - input_to_h1.col(occupied_offset)
                        + input_to_h1.col(free_offset)
                        - turn;
    }
};
*/
/*
constexpr bool NNUE_BASIC = true;
constexpr bool NNUE_SIMD = !NNUE_BASIC;

template<size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE, bool IMPL = NNUE_BASIC>
class NNUE {
    alignas(32) std::vector<float> accumulator;
    alignas(32) std::vector<float> turn_black;
    alignas(32) std::vector<float> turn_white;
    alignas(32) std::vector<float> input_to_h1;
    alignas(32) std::vector<float> h1_bias;
    alignas(32) std::vector<float> h1_output;
    alignas(32) std::vector<float> h1_to_h2;
    alignas(32) std::vector<float> h2_bias;
    alignas(32) std::vector<float> h2_output;
    alignas(32) std::vector<float> h2_to_h3;
    alignas(32) std::vector<float> h3_bias;
    alignas(32) std::vector<float> h3_output;
    alignas(32) std::vector<float> h3_to_output;
    alignas(32) std::vector<float> output_bias;
    alignas(32) std::vector<float> output;
public:
    NNUE() :
        accumulator(H1_SIZE),
        turn_black(H1_SIZE),
        turn_white(H1_SIZE),
        input_to_h1(INPUT_SIZE * H1_SIZE),
        h1_bias(H1_SIZE),
        h1_output(H1_SIZE),
        h1_to_h2(H2_SIZE * H1_SIZE),
        h2_bias(H2_SIZE),
        h2_output(H2_SIZE),
        h2_to_h3(H3_SIZE * H2_SIZE),
        h3_bias(H3_SIZE),
        h3_output(H3_SIZE),
        h3_to_output(OUTPUT_SIZE * H3_SIZE),
        output_bias(OUTPUT_SIZE),
        output(OUTPUT_SIZE) {}

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ifstream::in);
        size_t n, m;
        float v;
        std::string type;
        bool first = true;
        for (auto& [N, M, weights, bias]: {
                std::make_tuple(H1_SIZE, INPUT_SIZE, input_to_h1.data(), h1_bias.data()), 
                std::make_tuple(H2_SIZE, H1_SIZE, h1_to_h2.data(), h2_bias.data()), 
                std::make_tuple(H3_SIZE, H2_SIZE, h2_to_h3.data(), h3_bias.data()),
                std::make_tuple(OUTPUT_SIZE, H3_SIZE, h3_to_output.data(), output_bias.data())}) {
            ifs >> type;
            if (type != "W") {
                throw "W expected";
            }
            if (!(ifs >> n >> m)) {
                throw "matrix size expected";
            }
            if (n != N || m != M) {
                throw "bad matrix dimension";
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < M; j++) {
                    ifs >> v;
                    if (first) {
                        weights[j * N + i] =  v;
                    } else {
                        weights[i * M + j] =  v;
                        //std::cout << v << ' ';
                    }
                }
            }
            first = false;
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
            for (size_t i = 0; i < N; i++) {
                ifs >> v;
                bias[i] = v;
            }
        }
        std::fill(begin(accumulator), end(accumulator), 0.0);
        std::fill(begin(turn_white), end(turn_white), 0.0);
        constexpr size_t pos = 64 * 5;
        for (size_t i = 0; i < 64; i++) {
            size_t offset = (pos + i) * H1_SIZE;
            for (size_t j = 0; j < H1_SIZE; j++) {
                turn_white[j] += input_to_h1[offset + j];
            }            
        }    
        for (size_t j = 0; j < H1_SIZE; j++) {
            turn_black[j] = -turn_white[j];
        }
    }
    void write(std::ostream& os) {
        os << "accumulator\n" << H1_SIZE << '\n';
        for (auto v : accumulator) {
            os << std::setprecision(17) << v << '\n';
        }
        os << "turn black\n" << H1_SIZE << '\n';
        for (auto v : turn_black) {
            os << v << '\n';
        }
        os << "turn white\n" << H1_SIZE << '\n';
        for (auto v : turn_white) {
            os << v << '\n';
        }
        bool first = true;
        for (auto& [N, M, weights, bias]: {
                std::make_tuple(H1_SIZE, INPUT_SIZE, input_to_h1.data(), h1_bias.data()), 
                std::make_tuple(H2_SIZE, H1_SIZE, h1_to_h2.data(), h2_bias.data()), 
                std::make_tuple(H3_SIZE, H2_SIZE, h2_to_h3.data(), h3_bias.data()),
                std::make_tuple(OUTPUT_SIZE, H3_SIZE, h3_to_output.data(), output_bias.data())}) {
            os << "W\n";
            os << N << '\n';
            os << M << '\n';
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < M; j++) {
                    os << (first ? weights[j * N + i] : weights[i * M + j]) << '\n';
                }
            }
            first = false;
            os << "B\n";
            os << N << '\n';
            for (size_t i = 0; i < N; i++) {
                os << bias[i] << '\n';
            }
        }
    }
    std::tuple<uint64_t, uint64_t, uint64_t , uint64_t, uint64_t> encode_yolah(const Yolah& yolah) const {
        // black positions + white positions + empty positions + occupied positions + free positions 
        const uint64_t black = yolah.bitboard(Yolah::BLACK);
        const uint64_t white = yolah.bitboard(Yolah::WHITE);
        const uint64_t empty = yolah.empty_bitboard();
        const uint64_t occupied = yolah.occupied_squares();
        const uint64_t free = yolah.free_squares();
        return {black, white, empty, occupied, free};
    }
    void init(const Yolah& yolah) {
        std::fill(begin(accumulator), end(accumulator), 0.0);
        const auto [black, white, empty, occupied, free] = encode_yolah(yolah);
        // std::cout << yolah << '\n';
        // std::cout << std::hex << black << '\n';
        // std::cout << std::hex << white << '\n';
        // std::cout << std::hex << empty << '\n';
        // std::cout << std::hex << occupied << '\n';
        // std::cout << std::hex << free << '\n';
        // std::string _;
        // std::cin >> _;        
        size_t delta = 0;
        for (uint64_t bitboard : {black, white, empty, occupied, free}) {
            while (bitboard) {
                uint64_t pos = std::countr_zero(bitboard & -bitboard);
                size_t offset = (delta + 63 - pos) * H1_SIZE;
                //std::cout << std::dec << pos << ' ' << delta << '\n';
                if constexpr (IMPL == NNUE_BASIC) {
                    for (size_t i = 0; i < H1_SIZE; i++) {
                        accumulator[i] += input_to_h1[offset + i];
                    }
                } else {

                }
                bitboard &= bitboard - 1;
            }
            delta += 64;
        }
        if constexpr (IMPL == NNUE_BASIC) {
            if (yolah.current_player() == Yolah::WHITE) {
                for (size_t i = 0; i < H1_SIZE; i++) {
                    accumulator[i] += turn_white[i];
                }
            }            
        } else {

        }
    }
    float relu(float x) const {
        return x < 0.0 ? 0.0 : x;
    }
    std::tuple<float, float, float> output_linear() {
        if constexpr (IMPL == NNUE_BASIC) {            
            for (size_t i = 0; i < H1_SIZE; i++) {
                h1_output[i] = relu(accumulator[i] + h1_bias[i]);
            }
            for (size_t i = 0; i < H2_SIZE; i++) {
                h2_output[i] = 0;
                for (size_t j = 0; j < H1_SIZE; j++) {
                    h2_output[i] += h1_output[j] * h1_to_h2[i * H1_SIZE + j];
                }
                h2_output[i] = relu(h2_output[i] + h2_bias[i]);
            }
            for (size_t i = 0; i < H3_SIZE; i++) {
                h3_output[i] = 0;
                for (size_t j = 0; j < H2_SIZE; j++) {
                    h3_output[i] += h2_output[j] * h2_to_h3[i * H2_SIZE + j];
                }
                h3_output[i] = relu(h3_output[i] + h3_bias[i]);
            }
            for (size_t i = 0; i < OUTPUT_SIZE; i++) {
                output[i] = 0;
                for (size_t j = 0; j < H3_SIZE; j++) {
                    output[i] += h3_output[j] * h3_to_output[i * H3_SIZE + j];
                }
                output[i] += output_bias[i];
            }
            return {output[0], output[1], output[2]};
        } else {
            return {};
        }
    }
    // std::tuple<double, double, double> output_linear(const Yolah& yolah) {
    //     if constexpr (BASIC) {
    //         const auto [black, white, empty, occupied, free, turn] = encode_yolah(yolah);
    //         for (size_t i = 0; i < H1_SIZE; i++) {
    //             h1_output[i] = 0;
    //             size_t delta = 0;
    //             for (uint64_t bitboard : {black, white, empty, occupied, free, turn}) {
    //                 while (bitboard) {
    //                     uint64_t pos = std::countr_zero(bitboard & -bitboard);                   
    //                     h1_output[i] += input_to_h1[i * INPUT_SIZE + delta + (63 - pos)];
    //                     bitboard &= bitboard - 1;
    //                 }
    //                 delta += 64;
    //             }
    //             h1_output[i] = relu(h1_output[i] + h1_bias[i]);                
    //             //std::cout << h1_output[i] << ' ';
    //         }            
    //         for (size_t i = 0; i < H2_SIZE; i++) {
    //             h2_output[i] = 0;
    //             for (size_t j = 0; j < H1_SIZE; j++) {
    //                 h2_output[i] += h1_output[j] * h1_to_h2[i * H1_SIZE + j];
    //             }
    //             h2_output[i] = relu(h2_output[i] + h2_bias[i]);
    //         }
    //         for (size_t i = 0; i < H3_SIZE; i++) {
    //             h3_output[i] = 0;
    //             for (size_t j = 0; j < H2_SIZE; j++) {
    //                 h3_output[i] += h2_output[j] * h2_to_h3[i * H2_SIZE + j];
    //             }
    //             h3_output[i] = relu(h3_output[i] + h3_bias[i]);
    //         }
    //         for (size_t i = 0; i < OUTPUT_SIZE; i++) {
    //             output[i] = 0;
    //             for (size_t j = 0; j < H3_SIZE; j++) {
    //                 output[i] += h3_output[j] * h3_to_output[i * H3_SIZE + j];
    //             }
    //             output[i] += output_bias[i];
    //             //std::cout << output[i] << '\n';
    //         }
    //         return {output[0], output[1], output[2]};
    //     } else {
    //         return {};
    //     }
    // }    
    std::tuple<float, float, float> output_softmax() {
        (void)output_linear();
        // float max_output = std::numeric_limits<float>::lowest();
        // for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        //     float v = output[i];
        //     max_output = (v > max_output) ? v : max_output;
        // }
        // float sum = 0.0;
        // for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        //     sum += output[i];// - max_output;
        // }
        float e0 = std::exp(output[0]);
        float e1 = std::exp(output[1]);
        float e2 = std::exp(output[2]);
        float sum = (e0 + e1) + e2;
        return {
            e0 / sum,
            e1 / sum,
            e2 / sum
        };
        // return {
        //     std::exp(output[0] - max_output) / sum,
        //     std::exp(output[1] - max_output) / sum,
        //     std::exp(output[2] - max_output) / sum
        // };
    }
    void play(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
        if constexpr (IMPL == NNUE_BASIC) {
            float v1, v2, v3, v4, v5;
            size_t from_offset = (pos + from) * H1_SIZE;
            size_t to_offset = (pos + to) * H1_SIZE;
            size_t empty_offset = (128 + from) * H1_SIZE;
            size_t occupied_offset = (192 + to) * H1_SIZE;
            size_t free_offset = (256 + to) * H1_SIZE;
            for (size_t i = 0; i < H1_SIZE; i++) {
                v1 = -input_to_h1[from_offset + i];
                v2 = input_to_h1[to_offset + i];
                v3 = input_to_h1[empty_offset + i];
                v4 = input_to_h1[occupied_offset + i];
                v5 = -input_to_h1[free_offset + i];
                accumulator[i] += (v1 + v2) + (v3 + v4) + (v5 + turn[i]);
            }
        } else {

        }
    }
    void undo(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
        if constexpr (IMPL == NNUE_BASIC) {
            float v1, v2, v3, v4, v5;
            size_t from_offset = (pos + from) * H1_SIZE;
            size_t to_offset = (pos + to) * H1_SIZE;
            size_t empty_offset = (128 + from) * H1_SIZE;
            size_t occupied_offset = (192 + to) * H1_SIZE;
            size_t free_offset = (256 + to) * H1_SIZE;
            for (size_t i = 0; i < H1_SIZE; i++) {
                v1 = input_to_h1[from_offset + i];
                v2 = -input_to_h1[to_offset + i];
                v3 = -input_to_h1[empty_offset + i];
                v4 = -input_to_h1[occupied_offset + i];
                v5 = input_to_h1[free_offset + i];
                accumulator[i] += (v1 + v2) + (v3 + v4) + (v5 - turn[i]);
            }
        } else {

        }
    }
};
*/
#endif