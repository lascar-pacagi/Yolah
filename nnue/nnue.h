#ifndef NNUE_H
#define NNUE_H
#include <cstddef>
#include <vector>
#include <tuple>
#include "game.h"
#include <fstream>
#include <string>
#include <iomanip>
#include "vectorclass.h"
#include "Eigen/Dense"

// black positions + white positions + empty positions + occupied positions + free positions + turn 
constexpr size_t INPUT_SIZE = 64 + 64 + 64 + 64 + 64 + 64;
constexpr size_t OUTPUT_SIZE = 3;

using MatrixXf = Eigen::MatrixXf;
using RowVectorXf = Eigen::RowVectorXf;

template<size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE>
class NNUE {
    RowVectorXf accumulator;
    RowVectorXf turn_black;
    RowVectorXf turn_white;
    MatrixXf input_to_h1;
    RowVectorXf h1_bias;
    RowVectorXf h1_output;
    MatrixXf h1_to_h2;
    RowVectorXf h2_bias;
    RowVectorXf h2_output;
    MatrixXf h2_to_h3;
    RowVectorXf h3_bias;
    RowVectorXf h3_output;
    MatrixXf h3_to_output;
    RowVectorXf output_bias;
    RowVectorXf output;
public:
    NNUE() :
        accumulator(H1_SIZE),
        turn_black(H1_SIZE),
        turn_white(H1_SIZE),
        input_to_h1(INPUT_SIZE, H1_SIZE),
        h1_bias(H1_SIZE),
        h1_output(H1_SIZE),
        h1_to_h2(H1_SIZE, H2_SIZE),
        h2_bias(H2_SIZE),
        h2_output(H2_SIZE),
        h2_to_h3(H2_SIZE, H3_SIZE),
        h3_bias(H3_SIZE),
        h3_output(H3_SIZE),
        h3_to_output(H3_SIZE, OUTPUT_SIZE),
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
        accumulator.fill(0.0f);
        turn_white.fill(0.0f);
        constexpr size_t pos = 64 * 5;
        for (size_t i = 0; i < 64; i++) {
            turn_white += input_to_h1[(pos + i) * H1_SIZE];                        
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
        std::fill(begin(accumulator), end(accumulator), 0.0);
        const auto [black, white, empty, occupied, free] = encode_yolah(yolah);        
        size_t delta = 0;
        for (uint64_t bitboard : {black, white, empty, occupied, free}) {
            while (bitboard) {
                uint64_t pos = std::countr_zero(bitboard & -bitboard);
                size_t offset = (delta + 63 - pos) * H1_SIZE;
                
                    for (size_t i = 0; i < H1_SIZE; i++) {
                        accumulator[i] += input_to_h1[offset + i];
                    }
                
                bitboard &= bitboard - 1;
            }
            delta += 64;
        }
        if (yolah.current_player() == Yolah::WHITE) {
            for (size_t i = 0; i < H1_SIZE; i++) {
                accumulator[i] += turn_white[i];
            }
        }        
    }
    float relu(float x) const {
        return x < 0.0 ? 0.0 : x;
    }
    std::tuple<float, float, float> output_linear() {
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
    }
    std::tuple<float, float, float> output_softmax() {
        (void)output_linear();
        float e0 = std::exp(output[0]);
        float e1 = std::exp(output[1]);
        float e2 = std::exp(output[2]);
        float sum = (e0 + e1) + e2;
        return {
            e0 / sum,
            e1 / sum,
            e2 / sum
        };
    }
    void play(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;        
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
    }
    void undo(uint8_t player, const Move& m) {
        size_t from = 63 - m.from_sq();
        size_t to = 63 - m.to_sq();
        // black positions + white positions + empty positions + occupied positions + free positions
        size_t pos = (player == Yolah::BLACK) ? 0 : 64;
        const auto& turn = (player == Yolah::BLACK) ? turn_white : turn_black;
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
    }
};

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