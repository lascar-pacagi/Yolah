#include "logic_net.h"
#include <chrono>
#include <bit>
#include <immintrin.h>
#include <format>
#include <sstream>
#include <set>
#include <fstream>

LogicNet::Layer::Layer(std::mt19937_64& mt) {
    std::uniform_int_distribution<uint16_t> in(0, 2 * SIZE - 1);
    std::uniform_int_distribution<uint8_t> g(0, 15);
    for (int i = 0; i < SIZE; i++) {
        inputs1[i] = in(mt);
        inputs2[i] = in(mt);
        gates[i]   = g(mt);
    }
}

LogicNet::Layer::Layer(uint8_t gate) {
    for (int i = 0; i < SIZE; i++) {
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

void LogicNet::Layer::forward(const uint8_t* __restrict__ input, const uint8_t* __restrict__ prev, uint8_t* __restrict__ output) const {
    for (int i = 0; i < SIZE; i++) {
        int i1 = inputs1[i];
        int i2 = inputs2[i];
        int a = i1 < SIZE ? input[i1] : prev[i1 - SIZE];
        int b = i2 < SIZE ? input[i2] : prev[i2 - SIZE];
        int gate = gates[i];
        output[i] = gates_output[gate * 4 + a * 2 + b];
    }
}

std::array<int, 16> LogicNet::Layer::gates_count() const {
    std::array<int, 16> res{};
    for (int i = 0; i < SIZE; i++) {
        res[gates[i]]++;
    }
    return res;
}

std::ostream& operator<<(std::ostream& os, const LogicNet::Layer& l) {    
    static const char* const prev_repr[] = {
        "I", ">"
    };
    for (int i = 0; i < LogicNet::Layer::SIZE; i++) {
        os << "[#" << i << ' ';
        os << prev_repr[l.inputs1[i] >= LogicNet::Layer::SIZE] << l.inputs1[i] << ' '; 
        os << prev_repr[l.inputs2[i] >= LogicNet::Layer::SIZE] << l.inputs2[i] << ' ';
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

LogicNet::LogicNet(int nb_layers, int gate) {
    if (gate == -1) {
        std::mt19937_64 mt(std::chrono::system_clock::now().time_since_epoch().count());
        for (int i = 0; i < nb_layers; i++) {
            layers.emplace_back(mt);
        }
    } else {
        for (int i = 0; i < nb_layers; i++) {
            layers.emplace_back(gate);
        }
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
    alignas(64) uint8_t input[Layer::SIZE], prev[Layer::SIZE], output[Layer::SIZE];
    uint64_t black = yolah.bitboard(Yolah::BLACK);
    uint64_t white = yolah.bitboard(Yolah::WHITE);
    uint64_t empty = yolah.empty_bitboard();
    uint8_t turn   = yolah.nb_plies() & 1;
    int n = Layer::SIZE / (64 * 3 + 1);
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 64; j++) {
            input[k] = prev[k] = black >> i & 1ULL;
            k++;
        }
        for (int j = 0; j < 64; j++) {
            input[k] = prev[k] = white >> i & 1ULL;
            k++;
        }
        for (int j = 0; j < 64; j++) {
            input[k] = prev[k] = empty >> i & 1ULL;
            k++;
        }
        input[k] = prev[k] = turn;
        k++;        
    }
    for (; k < Layer::SIZE; k++) {
        input[k] = prev[k] = turn;
    }
    //initial_layers(input, prev);
    for (const auto& l : layers) {
        l.forward(input, prev, output);
        for (int i = 0; i < Layer::SIZE; i++) {
            prev[i] = output[i];
        }
    }
    int sum_black = 0;
    int sum_draw  = 0;
    int sum_white = 0;
    for (int i = 0; i < Layer::SIZE / 3; i++) {
        sum_black += output[i];
        sum_draw  += output[Layer::SIZE / 3 + i];
        sum_white += output[2 * (Layer::SIZE / 3) + i];
    }
    constexpr float DIV = 3.0 / Layer::SIZE;
    float b = std::exp(sum_black * DIV);
    float d = std::exp(sum_draw * DIV);
    float w = std::exp(sum_white * DIV);
    float s = b + d + w;
    return { b / s, d / s, w / s };
}

std::string LogicNet::gate_to_c_expression(std::string_view i1, const std::string_view i2, int gate) const {
    using namespace std;
    switch (gate) {
        case 0: // FALSE
            return "0";
        case 1: // A NOR B
            return format("!({} || {})", i1, i2);
        case 2: // ~(B=>A)
            return format("(!{} && {})", i1, i2);
        case 3: // ~A
            return format("!{}", i1);
        case 4: // ~(A=>B)
            return format("({} && !{})", i1, i2);
        case 5: // ~B
            return format("!{}", i2);
        case 6: // A XOR B
            return format("({} ^ {})", i1, i2);
        case 7: // A NAND B
            return format("!({} && {})", i1, i2);
        case 8: // A AND B
            return format("({} && {})", i1, i2);
        case 9: // ~(A XOR B)
            return format("!({} ^ {})", i1, i2);
        case 10: // B
            return format("{}", i2);
        case 11: // A=>B
            return format("(!{} || {})", i1, i2);
        case 12: // A
            return format("{}", i1);
        case 13: // B=>A
            return format("({} || !{})", i1, i2);
        case 14: // A OR B
            return format("({} || {})", i1, i2);
        case 15: // TRUE
            return "1";
        default:
            std::unreachable();
    }
}

std::string LogicNet::c_expression_from_layer(int l) const {
    using namespace std;
    ostringstream ss;
    vector<set<int>> useful(l);
    for (int i = l; i > 0; i--) {
        for (int j = 0; j < Layer::SIZE; j++) {
            int i1 = layers[i].inputs1[j];
            int i2 = layers[i].inputs2[j];
            int gate = layers[i].gates[j];
            if (i1 >= Layer::SIZE && (i == l || useful[i].contains(j))) {
                if (gate != 0 && gate != 5 && gate != 10 && gate != 15) {
                    useful[i - 1].insert(i1 - Layer::SIZE);
                }
            }
            if (i2 >= Layer::SIZE && (i == l || useful[i].contains(j))) {
                if (gate != 0 && gate != 3 && gate != 12 && gate != 15) {
                    useful[i - 1].insert(i2 - Layer::SIZE);
                }
            }
        }
    }
    for (int j = 0; j < Layer::SIZE; j++) {
        if (!useful[0].contains(j)) continue;
        int i1 = layers[0].inputs1[j];
        if (i1 >= Layer::SIZE) i1 -= Layer::SIZE;
        int i2 = layers[0].inputs2[j];
        if (i2 >= Layer::SIZE) i2 -= Layer::SIZE;
        int gate = layers[0].gates[j];
        string res = gate_to_c_expression(format("input[{}]", i1), format("input[{}]", i2), gate);
        ss << format("uint8_t x_0_{} = {};\n", j, res);
    }
    for (int i = 1; i <= l; i++) {
        for (int j = 0; j < Layer::SIZE; j++) {
            if (i != l && !useful[i].contains(j)) continue;
            int i1 = layers[i].inputs1[j];
            int i2 = layers[i].inputs2[j];
            int gate = layers[i].gates[j];
            string in1 = i1 < Layer::SIZE ? format("input[{}]", i1) : format("x_{}_{}", i - 1, i1 - Layer::SIZE); 
            string in2 = i2 < Layer::SIZE ? format("input[{}]", i2) : format("x_{}_{}", i - 1, i2 - Layer::SIZE); 
            string res = gate_to_c_expression(in1, in2, gate);
            if (i != l) {
                ss << format("int8_t x_{}_{} = {};\n", i, j, res);
            } else {
                ss << format("output[{}] = {};\n", j, res);
            }
        }
    }
    return ss.str();
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

LogicNet LogicNet::from_json(std::istream& is) {
    json j = json::parse(is);
    return j.get<LogicNet>();
}

// int main() {
//     // std::ifstream ifs("model.txt");
//     // LogicNet net = LogicNet::from_json(ifs);
//     // std::cout << net << std::endl;
//     // std::cout << net.c_expression_from_layer(3);
//     LogicNet net(5);
//     Yolah yolah;
//     float black = 0, draw = 0, white = 0;
//     for (int i = 0; i < 1000000; i++) {
//         const auto [b, d, w] = net.forward(yolah);
//         black += b;
//         draw += d;
//         white += w;
//     }
//     std::cout << black << ' ' << draw << ' ' << white << '\n';    
// }

void LogicNet::initial_layers(const uint8_t* __restrict__ input, uint8_t* __restrict__ output) {
}
