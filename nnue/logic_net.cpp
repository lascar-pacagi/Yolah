#include "logic_net.h"
#include <chrono>
#include <bit>
#include <immintrin.h>

LogicNet::Layer::Layer(std::mt19937_64& mt) {
    std::uniform_int_distribution<uint16_t> in(0, 511);
    std::uniform_int_distribution<uint8_t> g(0, 15);
    for (int i = 0; i < 256; i++) {
        inputs1[i] = in(mt);
        inputs2[i] = in(mt);
        gates[i]   = g(mt);
    }
}

LogicNet::Layer::Layer(uint8_t gate) {
    for (int i = 0; i < 256; i++) {
        inputs1[i] = i;
        inputs2[i] = i;
        gates[i]   = gate;
    }
}

LogicNet::Layer::Layer() {
    inputs1.fill(0);
    inputs2.fill(0);
    gates.fill(0);
}

// static void print__m256i(__m256i r) {
//     alignas(64) uint8_t v[32];
//     _mm256_store_si256((__m256i*)v, r);
//     std::cout << "print__m256i\n";
//     for (int i = 0; i < 32; i++) {
//         std::cout << "[#" << i << ' ' << (int)v[i] << ']';
//     } 
//     std::cout << '\n';
// }

void LogicNet::Layer::forward(const uint8_t* input_prev, uint8_t* __restrict__ output) const {
    alignas(64) uint8_t out_idx[32];
    for (int i = 0; i < 256; i += 32) {
        __m256i g  = _mm256_load_si256((__m256i*)&gates[i]);
        __m256i v1 = _mm256_set_epi8(
            input_prev[inputs1[i + 31]], input_prev[inputs1[i + 30]],
            input_prev[inputs1[i + 29]], input_prev[inputs1[i + 28]],
            input_prev[inputs1[i + 27]], input_prev[inputs1[i + 26]],
            input_prev[inputs1[i + 25]], input_prev[inputs1[i + 24]],
            input_prev[inputs1[i + 23]], input_prev[inputs1[i + 22]],
            input_prev[inputs1[i + 21]], input_prev[inputs1[i + 20]],
            input_prev[inputs1[i + 19]], input_prev[inputs1[i + 18]],
            input_prev[inputs1[i + 17]], input_prev[inputs1[i + 16]],
            input_prev[inputs1[i + 15]], input_prev[inputs1[i + 14]],
            input_prev[inputs1[i + 13]], input_prev[inputs1[i + 12]],
            input_prev[inputs1[i + 11]], input_prev[inputs1[i + 10]],
            input_prev[inputs1[i + 9]], input_prev[inputs1[i + 8]],
            input_prev[inputs1[i + 7]], input_prev[inputs1[i + 6]],
            input_prev[inputs1[i + 5]], input_prev[inputs1[i + 4]],
            input_prev[inputs1[i + 3]], input_prev[inputs1[i + 2]],
            input_prev[inputs1[i + 1]], input_prev[inputs1[i + 0]]
        );
        //print__m256i(v1);
        __m256i v2 = _mm256_set_epi8(
            input_prev[inputs2[i + 31]], input_prev[inputs2[i + 30]],
            input_prev[inputs2[i + 29]], input_prev[inputs2[i + 28]],
            input_prev[inputs2[i + 27]], input_prev[inputs2[i + 26]],
            input_prev[inputs2[i + 25]], input_prev[inputs2[i + 24]],
            input_prev[inputs2[i + 23]], input_prev[inputs2[i + 22]],
            input_prev[inputs2[i + 21]], input_prev[inputs2[i + 20]],
            input_prev[inputs2[i + 19]], input_prev[inputs2[i + 18]],
            input_prev[inputs2[i + 17]], input_prev[inputs2[i + 16]],
            input_prev[inputs2[i + 15]], input_prev[inputs2[i + 14]],
            input_prev[inputs2[i + 13]], input_prev[inputs2[i + 12]],
            input_prev[inputs2[i + 11]], input_prev[inputs2[i + 10]],
            input_prev[inputs2[i + 9]], input_prev[inputs2[i + 8]],
            input_prev[inputs2[i + 7]], input_prev[inputs2[i + 6]],
            input_prev[inputs2[i + 5]], input_prev[inputs2[i + 4]],
            input_prev[inputs2[i + 3]], input_prev[inputs2[i + 2]],
            input_prev[inputs2[i + 1]], input_prev[inputs2[i + 0]]
        );        
        //print__m256i(v2);
        _mm256_store_si256((__m256i*)out_idx, _mm256_adds_epu8(_mm256_slli_epi16(g, 2), _mm256_adds_epu8(_mm256_slli_epi16(v1, 1), v2)));
        for (int j = 0; j < 32; j++) {
            output[i + j] = gates_output[out_idx[j]];
        }
    }
}

std::array<int, 16> LogicNet::Layer::gates_count() const {
    std::array<int, 16> res{};
    for (int i = 0; i < 256; i++) {
        res[gates[i]]++;    
    }
    return res;
}

std::ostream& operator<<(std::ostream& os, const LogicNet::Layer& l) {    
    static const char* const prev_repr[] = {
        "I", ">"
    };
    for (int i = 0; i < 256; i++) {
        os << "[#" << i << ' ';
        os << prev_repr[l.inputs1[i] > 255] << l.inputs1[i] << ' '; 
        os << prev_repr[l.inputs2[i] > 255] << l.inputs2[i] << ' ';
        os << LogicNet::gates_repr[l.gates[i]];
        os << ']';
    }
    return os;
}   

static void to_json(json& j, const LogicNet::Layer& l) {
    j = json{
        {"inputs1", l.inputs1},
        {"inputs2", l.inputs2},
        {"gates",   l.gates}
    };
}

static void from_json(const json& j, LogicNet::Layer& l) {
    j.at("inputs1").get_to(l.inputs1);
    j.at("inputs2").get_to(l.inputs2);
    j.at("gates").get_to(l.gates);
}

static void to_json(json& j, const LogicNet& net) {
    j = json{{"layers", net.layers}};
}

static void from_json(const json& j, LogicNet& net) {
    j.at("layers").get_to(net.layers);
}

std::string LogicNet::Layer::to_json() const {
    json j = *this;
    return j.dump(4);
}

LogicNet::Layer LogicNet::Layer::from_json(std::istream& is) {
    json j = json::parse(is);
    return j.get<Layer>();
}

LogicNet::LogicNet(int nb_layers) {
    std::mt19937_64 mt(std::chrono::system_clock::now().time_since_epoch().count());
    for (int i = 0; i < nb_layers; i++) {
        layers.emplace_back(mt);
    }
}

std::array<int, 16> LogicNet::gates_count() const {
    std::array<int, 16> res{};
    std::array<int, 16> count;
    for (const auto& layer : layers) {
        count = layer.gates_count();
        for (int i = 0; i < 16; i++) {
            res[i] += count[i];
        }
    }
    return res;
}

std::tuple<float, float, float> LogicNet::forward(const Yolah& yolah) const {
    alignas(64) uint8_t input_prev[2][512];
    uint64_t black = yolah.bitboard(Yolah::BLACK);
    uint64_t white = yolah.bitboard(Yolah::WHITE);
    uint64_t empty = yolah.empty_bitboard();
    uint8_t turn   = yolah.nb_plies() & 1;
    int k = 0;
    for (int i = 0; i < 64; i++) {
        input_prev[0][k + 256]       = input_prev[0][k]       = input_prev[1][k] = black >> i & 1ULL;
        input_prev[0][k + 256 + 64]  = input_prev[0][k + 64]  = input_prev[1][k + 64] = white >> i & 1ULL;
        input_prev[0][k + 256 + 128] = input_prev[0][k + 128] = input_prev[1][k + 128] = empty >> i & 1ULL;
        input_prev[0][k + 256 + 192] = input_prev[0][k + 192] = input_prev[1][k + 192] = turn;
        k++;
    }
    int idx = 0;
    for (const auto& l : layers) {
        l.forward(input_prev[idx], input_prev[1 - idx] + 256);
        idx = 1 - idx;
    }
    int sum_black = 0;
    int sum_draw  = 0;
    int sum_white = 0;
    for (int i = 0; i < 80; i++) {
        sum_black += input_prev[idx][256 + i];
        sum_draw  += input_prev[idx][256 + 64 + i];
        sum_white += input_prev[idx][256 + 128 + i];
    }
    constexpr float DIV = 30;
    float b = std::exp(sum_black / DIV);
    float d = std::exp(sum_draw / DIV);
    float w = std::exp(sum_white / DIV);
    float s = b + d + w;
    return { b / s, d / s, w / s };
}

std::ostream& operator<<(std::ostream& os, const LogicNet& net) {
    const auto count = net.gates_count();
    float sum = 0;
    sum = std::accumulate(begin(count), end(count), 0);
    for (int i = 0; i < 16; i++) {
        os << '[' << LogicNet::gates_repr[i] << ' ' << count[i] / sum <<  ']';
    }
    os << '\n';
    for (int i = 0; i < (int)net.layers.size(); i++) {
        os << "\n****LAYER " << i << ":\n" << net.layers[i] << '\n';
    }
    return os;
}

std::string LogicNet::to_json() const {
    json j = *this;
    return j.dump(4);
}

/*
json j;
    j["ply"]   = to_string(ply);
    j["black"] = to_string(black);
    j["white"] = to_string(white);
    j["empty"] = to_string(empty);
    j["black score"] = to_string(black_score);
    j["white score"] = to_string(white_score);
    return j.dump();
*/

LogicNet LogicNet::from_json(std::istream& is) {
    json j = json::parse(is);
    return j.get<LogicNet>();
}

void test1() {
    LogicNet::Layer l;
    alignas(64) uint8_t input_prev[512];
    alignas(64) uint8_t output[256];
    input_prev[0] = 0;
    input_prev[1] = 0;
    input_prev[2] = 0;
    input_prev[3] = 1;
    input_prev[4] = 1;
    input_prev[5] = 0;
    input_prev[6] = 1;
    input_prev[7] = 1;
    l.inputs1[0] = 0;
    l.inputs2[0] = 1;
    l.inputs1[1] = 2;
    l.inputs2[1] = 3;
    l.inputs1[2] = 4;
    l.inputs2[2] = 5;
    l.inputs1[3] = 6;
    l.inputs2[3] = 7;
    for (int i = 0; i < 16; i++) {
        l.gates[0] = i;
        l.gates[1] = i;
        l.gates[2] = i;
        l.gates[3] = i;
        l.forward(input_prev, output);
        for (int j = 0; j < 4; j++) {
            int expected = LogicNet::gates_output[i * 4 + j];
            if (output[j] != expected) {
                std::cout << "test1 for gate " << i << " and input " << j << " expected " << expected << " got " << output[j] << '\n';
            }
        }
    }        
}

void test2() {
    LogicNet::Layer l;
    alignas(64) uint8_t input_prev[512];
    alignas(64) uint8_t output[256];
    input_prev[455] = 0;
    input_prev[456] = 0;
    input_prev[457] = 0;
    input_prev[458] = 1;
    input_prev[459] = 1;
    input_prev[460] = 0;
    input_prev[461] = 1;
    input_prev[462] = 1;
    l.inputs1[200] = 455;
    l.inputs2[200] = 456;
    l.inputs1[201] = 457;
    l.inputs2[201] = 458;
    l.inputs1[202] = 459;
    l.inputs2[202] = 460;
    l.inputs1[203] = 461;
    l.inputs2[203] = 462;
    for (int i = 0; i < 16; i++) {
        l.gates[200] = i;
        l.gates[201] = i;
        l.gates[202] = i;
        l.gates[203] = i;
        l.forward(input_prev, output);
        for (int j = 0; j < 4; j++) {
            int expected = LogicNet::gates_output[i * 4 + j];
            if (output[200 + j] != expected) {
                std::cout << "test2 for gate " << i << " and input " << j << " expected " << expected << " got " << output[j] << '\n';
            }
        }
    }        
}

void test3() {
    LogicNet::Layer l;
    alignas(64) uint8_t input_prev[512];
    alignas(64) uint8_t output[256];
    input_prev[0] = 0;
    input_prev[256] = 0;
    input_prev[1] = 0;
    input_prev[257] = 1;
    input_prev[2] = 1;
    input_prev[258] = 0;
    input_prev[3] = 1;
    input_prev[259] = 1;
    l.inputs1[0] = 0;
    l.inputs2[0] = 256;
    l.inputs1[1] = 1;
    l.inputs2[1] = 257;
    l.inputs1[2] = 2;
    l.inputs2[2] = 258;
    l.inputs1[3] = 3;
    l.inputs2[3] = 259;
    for (int i = 0; i < 16; i++) {
        l.gates[0] = i;
        l.gates[1] = i;
        l.gates[2] = i;
        l.gates[3] = i;
        l.forward(input_prev, output);
        for (int j = 0; j < 4; j++) {
            int expected = LogicNet::gates_output[i * 4 + j];
            if (output[j] != expected) {
                std::cout << "test3 for gate " << i << " and input " << j << " expected " << expected << " got " << output[j] << '\n';
            }
        }
    }        
}

// int main() {
//     test1();
//     test2();
//     test3();
//     // std::vector<int> ones{ 4, 9, 12, 13, 18, 20, 22, 25, 26, 28, 29, 30, 35, 36, 39, 41, 43, 44, 45, 47, 50, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63 };
//     // std::vector<int> zeroes{ 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 19, 21, 23, 24, 27, 31, 32, 33, 34, 37, 38, 40, 42, 46, 48, 49, 53, 56 };
//     // for (int i = 1; i < 5000; i++) {
//     //     for (int x : ones) {
            
//     //     }
//     //     for (int x : zeroes) {
            
//     //     }
//     //     std::cout << "found: " << i << '\n';
//     //     bad:;
//     // }
//     // LogicNet::Layer l(8);
//     // std::cout << l << '\n';
//     // alignas(64) uint8_t input_prev[512];
//     // std::memset(input_prev, 1, sizeof(input_prev));
//     // alignas(64) uint8_t output[256];    
//     // for (int i = 0; i < 10000000; i++) {
//     //     l.forward(input_prev, output);
//     // }    
//     // for (int i = 0; i < 256; i++) {
//     //     std::cout << "[#" << i << ' ' << (int)output[i] << ']'; 
//     // }
//     // std::cout << '\n';    
//     // LogicNet net(2);
//     // std::cout << net << '\n';
//     // std::cout << net.to_json() << '\n';
//     // Yolah yolah;
//     // float black = 0, draw = 0, white = 0;
//     // for (int i = 0; i < 10000000; i++) {
//     //     const auto [b, d, w] = net.forward(yolah);
//     //     black += b;
//     //     draw += d;
//     //     white += w;
//     // }
//     // std::cout << black << ' ' << draw << ' ' << white << '\n';    
// }
