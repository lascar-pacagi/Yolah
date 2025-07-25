#include "logic_net.h"
#include <chrono>
#include <bit>
#include <immintrin.h>
#include <format>
#include <sstream>
#include <set>
#include <fstream>

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

void LogicNet::Layer::forward(const uint8_t* input, const uint8_t* prev, uint8_t* __restrict__ output) const {
    for (int i = 0; i < 256; i++) {
        int i1 = inputs1[i];
        int i2 = inputs2[i];
        int a = i1 < 256 ? input[i1] : prev[i1 - 256];
        int b = i2 < 256 ? input[i2] : prev[i2 - 256];
        int gate = gates[i];
        output[i] = gates_output[gate * 4 + a * 2 + b];
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
    alignas(64) uint8_t input[256], prev[256], output[256];
    uint64_t black = yolah.bitboard(Yolah::BLACK);
    uint64_t white = yolah.bitboard(Yolah::WHITE);
    uint64_t empty = yolah.empty_bitboard();
    uint8_t turn   = yolah.nb_plies() & 1;
    for (int i = 0; i < 64; i++) {
        input[i] = prev[i] = black >> i & 1ULL;
        input[i + 64] = prev[i + 64] = white >> i & 1ULL;
        input[i + 128] = prev[i + 128] = empty >> i & 1ULL;
        input[i + 192] = prev[i + 192] = turn;
    }
    //initial_layers(input, prev);
    for (const auto& l : layers) {
        l.forward(input, prev, output);
        for (int i = 0; i < 256; i++) {
            prev[i] = output[i];
        }
    }
    int sum_black = 0;
    int sum_draw  = 0;
    int sum_white = 0;
    for (int i = 0; i < 85; i++) {
        sum_black += output[i];
        sum_draw  += output[85 + i];
        sum_white += output[170 + i];
    }
    constexpr float DIV = 30;
    float b = std::exp(sum_black / DIV);
    float d = std::exp(sum_draw / DIV);
    float w = std::exp(sum_white / DIV);
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

void LogicNet::initial_layers(const uint8_t* input, uint8_t* __restrict__ output) {
    uint8_t x_0_0 = !input[97];
    uint8_t x_0_2 = 0;
    uint8_t x_0_15 = (input[108] ^ input[4]);
    uint8_t x_0_16 = (input[219] && input[174]);
    uint8_t x_0_18 = !(input[52] ^ input[105]);
    uint8_t x_0_21 = !(input[141] && input[113]);
    uint8_t x_0_27 = !(input[101] ^ input[3]);
    uint8_t x_0_28 = (input[241] ^ input[122]);
    uint8_t x_0_35 = !(input[106] || input[41]);
    uint8_t x_0_41 = input[240];
    uint8_t x_0_47 = (input[45] || !input[254]);
    uint8_t x_0_51 = (input[247] && input[151]);
    uint8_t x_0_57 = !(input[199] || input[116]);
    uint8_t x_0_58 = (!input[204] || input[227]);
    uint8_t x_0_59 = input[78];
    uint8_t x_0_65 = !(input[175] ^ input[166]);
    uint8_t x_0_66 = !(input[109] && input[155]);
    uint8_t x_0_70 = (input[85] || !input[243]);
    uint8_t x_0_71 = (!input[187] || input[198]);
    uint8_t x_0_87 = !input[82];
    uint8_t x_0_92 = 1;
    uint8_t x_0_99 = !(input[111] && input[254]);
    uint8_t x_0_104 = !(input[81] ^ input[97]);
    uint8_t x_0_110 = 0;
    uint8_t x_0_111 = !input[202];
    uint8_t x_0_114 = input[16];
    uint8_t x_0_118 = !(input[77] ^ input[242]);
    uint8_t x_0_125 = !input[203];
    uint8_t x_0_131 = input[100];
    uint8_t x_0_146 = !(input[151] ^ input[100]);
    uint8_t x_0_153 = !(input[40] && input[137]);
    uint8_t x_0_157 = (!input[24] || input[87]);
    uint8_t x_0_161 = (input[212] || input[230]);
    uint8_t x_0_171 = (input[195] || input[150]);
    uint8_t x_0_181 = 1;
    uint8_t x_0_187 = 0;
    uint8_t x_0_190 = (input[246] || input[223]);
    uint8_t x_0_193 = !(input[167] && input[73]);
    uint8_t x_0_196 = !input[224];
    uint8_t x_0_200 = (!input[239] || input[213]);
    uint8_t x_0_201 = !(input[136] && input[78]);
    uint8_t x_0_212 = (input[104] && !input[207]);
    uint8_t x_0_213 = !input[18];
    uint8_t x_0_227 = !(input[165] && input[223]);
    uint8_t x_0_228 = 0;
    uint8_t x_0_231 = !input[48];
    uint8_t x_0_233 = (input[77] || !input[149]);
    uint8_t x_0_244 = (!input[202] || input[3]);
    uint8_t x_0_246 = (input[54] || input[75]);
    uint8_t x_0_248 = !input[89];
    uint8_t x_0_249 = 0;
    int8_t x_1_0 = !input[91];
    int8_t x_1_4 = (!x_0_227 || x_0_201);
    int8_t x_1_11 = !(input[171] && input[91]);
    int8_t x_1_16 = (x_0_110 && input[23]);
    int8_t x_1_17 = 0;
    int8_t x_1_23 = 1;
    int8_t x_1_25 = 1;
    int8_t x_1_32 = !input[157];
    int8_t x_1_36 = (input[8] || x_0_16);
    int8_t x_1_38 = !input[216];
    int8_t x_1_39 = input[184];
    int8_t x_1_41 = 1;
    int8_t x_1_45 = (!x_0_181 || x_0_15);
    int8_t x_1_46 = (!input[2] || x_0_187);
    int8_t x_1_47 = (input[241] || input[6]);
    int8_t x_1_56 = x_0_104;
    int8_t x_1_61 = 1;
    int8_t x_1_66 = !(x_0_153 ^ input[180]);
    int8_t x_1_69 = !x_0_228;
    int8_t x_1_73 = x_0_231;
    int8_t x_1_74 = !(input[174] || x_0_71);
    int8_t x_1_76 = !(x_0_57 && input[187]);
    int8_t x_1_77 = !x_0_28;
    int8_t x_1_80 = (input[120] || x_0_47);
    int8_t x_1_82 = input[78];
    int8_t x_1_84 = input[184];
    int8_t x_1_87 = (input[40] || !x_0_131);
    int8_t x_1_89 = input[184];
    int8_t x_1_92 = !x_0_248;
    int8_t x_1_93 = !x_0_58;
    int8_t x_1_94 = 0;
    int8_t x_1_97 = (x_0_212 && !x_0_41);
    int8_t x_1_104 = !input[236];
    int8_t x_1_106 = 0;
    int8_t x_1_107 = 0;
    int8_t x_1_108 = !input[38];
    int8_t x_1_109 = (!x_0_16 || x_0_118);
    int8_t x_1_115 = 0;
    int8_t x_1_116 = !input[104];
    int8_t x_1_118 = (x_0_200 ^ input[2]);
    int8_t x_1_119 = (x_0_157 ^ x_0_18);
    int8_t x_1_121 = !(input[221] || input[65]);
    int8_t x_1_124 = !input[221];
    int8_t x_1_125 = !(input[16] || input[41]);
    int8_t x_1_126 = !input[13];
    int8_t x_1_127 = (!input[7] && input[175]);
    int8_t x_1_128 = 0;
    int8_t x_1_130 = !(input[25] && x_0_41);
    int8_t x_1_139 = x_0_35;
    int8_t x_1_141 = (input[93] || x_0_233);
    int8_t x_1_143 = (input[88] ^ x_0_51);
    int8_t x_1_144 = (!x_0_249 || x_0_70);
    int8_t x_1_147 = (input[52] && x_0_104);
    int8_t x_1_148 = !input[94];
    int8_t x_1_149 = input[229];
    int8_t x_1_153 = !(input[186] || x_0_70);
    int8_t x_1_154 = !(input[37] ^ x_0_190);
    int8_t x_1_156 = x_0_196;
    int8_t x_1_159 = (x_0_66 ^ input[16]);
    int8_t x_1_160 = !(input[237] ^ input[196]);
    int8_t x_1_161 = (!x_0_146 && input[107]);
    int8_t x_1_163 = !(input[161] || input[56]);
    int8_t x_1_165 = 1;
    int8_t x_1_166 = 1;
    int8_t x_1_169 = (input[150] || x_0_161);
    int8_t x_1_172 = (!input[6] && input[39]);
    int8_t x_1_173 = x_0_244;
    int8_t x_1_174 = (x_0_118 || input[185]);
    int8_t x_1_175 = input[112];
    int8_t x_1_176 = (x_0_244 || !x_0_65);
    int8_t x_1_177 = (x_0_111 && x_0_193);
    int8_t x_1_178 = !(x_0_92 ^ input[205]);
    int8_t x_1_185 = 1;
    int8_t x_1_186 = x_0_246;
    int8_t x_1_198 = 1;
    int8_t x_1_201 = (!x_0_114 && input[93]);
    int8_t x_1_202 = !input[42];
    int8_t x_1_209 = (!x_0_59 && x_0_99);
    int8_t x_1_210 = (input[83] && input[199]);
    int8_t x_1_214 = !x_0_0;
    int8_t x_1_218 = (input[13] ^ input[92]);
    int8_t x_1_219 = !(input[22] || x_0_27);
    int8_t x_1_220 = (input[167] && input[154]);
    int8_t x_1_225 = 1;
    int8_t x_1_231 = x_0_87;
    int8_t x_1_234 = (input[30] || input[178]);
    int8_t x_1_243 = (input[32] || !x_0_213);
    int8_t x_1_244 = 1;
    int8_t x_1_245 = 0;
    int8_t x_1_246 = !(input[244] || input[253]);
    int8_t x_1_247 = !(x_0_171 && x_0_125);
    int8_t x_1_248 = (x_0_21 || !x_0_2);
    int8_t x_1_251 = 1;
    int8_t x_1_252 = !(input[162] ^ input[75]);
    int8_t x_2_0 = input[156];
    int8_t x_2_1 = (x_1_147 ^ x_1_0);
    int8_t x_2_2 = !(x_1_234 ^ input[47]);
    int8_t x_2_3 = !(x_1_154 && x_1_245);
    int8_t x_2_7 = !input[147];
    int8_t x_2_8 = x_1_201;
    int8_t x_2_9 = !(x_1_87 || x_1_36);
    int8_t x_2_13 = (input[50] || input[38]);
    int8_t x_2_18 = !input[9];
    int8_t x_2_19 = !(x_1_69 ^ input[106]);
    int8_t x_2_20 = !input[240];
    int8_t x_2_23 = (!x_1_107 || input[132]);
    int8_t x_2_24 = !x_1_84;
    int8_t x_2_25 = !input[38];
    int8_t x_2_26 = !(input[48] && input[105]);
    int8_t x_2_27 = (input[155] ^ x_1_45);
    int8_t x_2_28 = (input[146] && input[17]);
    int8_t x_2_33 = (input[251] || !input[117]);
    int8_t x_2_36 = (!x_1_202 || x_1_178);
    int8_t x_2_37 = 1;
    int8_t x_2_39 = (x_1_244 && !x_1_251);
    int8_t x_2_40 = (x_1_245 && input[27]);
    int8_t x_2_42 = (input[250] && !input[237]);
    int8_t x_2_43 = (input[1] && !input[197]);
    int8_t x_2_44 = x_1_56;
    int8_t x_2_45 = !(x_1_139 && input[212]);
    int8_t x_2_46 = !(x_1_69 || input[235]);
    int8_t x_2_47 = (x_1_141 || x_1_165);
    int8_t x_2_48 = !x_1_177;
    int8_t x_2_49 = !(input[164] && input[135]);
    int8_t x_2_50 = (input[153] && x_1_176);
    int8_t x_2_53 = (x_1_109 && x_1_38);
    int8_t x_2_54 = (!input[139] || x_1_186);
    int8_t x_2_56 = 1;
    int8_t x_2_61 = 0;
    int8_t x_2_64 = !(x_1_143 && input[100]);
    int8_t x_2_65 = !input[217];
    int8_t x_2_66 = 1;
    int8_t x_2_68 = (!x_1_94 || x_1_219);
    int8_t x_2_69 = (input[102] || !x_1_23);
    int8_t x_2_72 = (x_1_185 ^ x_1_248);
    int8_t x_2_73 = (!input[199] && x_1_148);
    int8_t x_2_74 = x_1_77;
    int8_t x_2_77 = !(x_1_74 || x_1_144);
    int8_t x_2_79 = (input[214] && input[96]);
    int8_t x_2_80 = 1;
    int8_t x_2_81 = !x_1_76;
    int8_t x_2_82 = (!x_1_163 && input[188]);
    int8_t x_2_83 = (input[200] && x_1_82);
    int8_t x_2_86 = (input[245] || !x_1_173);
    int8_t x_2_87 = input[254];
    int8_t x_2_88 = 1;
    int8_t x_2_91 = 1;
    int8_t x_2_94 = input[126];
    int8_t x_2_96 = (!x_1_244 && input[58]);
    int8_t x_2_99 = (x_1_36 || !x_1_25);
    int8_t x_2_100 = 0;
    int8_t x_2_103 = (input[120] && !x_1_66);
    int8_t x_2_104 = 0;
    int8_t x_2_106 = (input[42] && !x_1_130);
    int8_t x_2_107 = !x_1_119;
    int8_t x_2_109 = !(x_1_97 && x_1_169);
    int8_t x_2_111 = 0;
    int8_t x_2_114 = 1;
    int8_t x_2_115 = (input[254] ^ x_1_147);
    int8_t x_2_117 = (!input[188] || input[170]);
    int8_t x_2_119 = (x_1_32 ^ x_1_220);
    int8_t x_2_122 = !(x_1_17 ^ input[239]);
    int8_t x_2_125 = !(input[33] || x_1_47);
    int8_t x_2_126 = x_1_161;
    int8_t x_2_128 = 0;
    int8_t x_2_132 = (!input[245] || x_1_153);
    int8_t x_2_133 = (input[229] || !x_1_76);
    int8_t x_2_136 = !(x_1_231 && input[89]);
    int8_t x_2_137 = !(x_1_209 && x_1_115);
    int8_t x_2_138 = 1;
    int8_t x_2_139 = (!x_1_178 || x_1_4);
    int8_t x_2_140 = !(x_1_247 || x_1_116);
    int8_t x_2_144 = !x_1_92;
    int8_t x_2_150 = (!input[215] && x_1_210);
    int8_t x_2_155 = (!x_1_118 || input[150]);
    int8_t x_2_158 = (x_1_156 && !x_1_154);
    int8_t x_2_159 = (!x_1_130 && x_1_104);
    int8_t x_2_161 = (!input[196] && x_1_243);
    int8_t x_2_163 = x_1_210;
    int8_t x_2_164 = 1;
    int8_t x_2_165 = x_1_175;
    int8_t x_2_166 = (x_1_125 && !input[9]);
    int8_t x_2_167 = 0;
    int8_t x_2_169 = (input[6] || input[231]);
    int8_t x_2_171 = !(x_1_214 || x_1_159);
    int8_t x_2_173 = 0;
    int8_t x_2_174 = !input[220];
    int8_t x_2_176 = !input[35];
    int8_t x_2_178 = (x_1_149 ^ x_1_246);
    int8_t x_2_179 = x_1_156;
    int8_t x_2_182 = 0;
    int8_t x_2_183 = 1;
    int8_t x_2_184 = (input[73] || x_1_252);
    int8_t x_2_185 = (input[6] && !x_1_202);
    int8_t x_2_190 = input[224];
    int8_t x_2_195 = 1;
    int8_t x_2_198 = !(x_1_17 ^ x_1_169);
    int8_t x_2_199 = !(input[227] && x_1_128);
    int8_t x_2_202 = (input[12] ^ input[211]);
    int8_t x_2_203 = (input[176] && !x_1_11);
    int8_t x_2_204 = !x_1_61;
    int8_t x_2_207 = !(input[157] ^ input[115]);
    int8_t x_2_208 = !x_1_87;
    int8_t x_2_210 = (!input[223] && input[37]);
    int8_t x_2_212 = (!x_1_41 || input[139]);
    int8_t x_2_215 = (input[85] && !input[113]);
    int8_t x_2_216 = (x_1_89 ^ x_1_174);
    int8_t x_2_218 = (x_1_36 ^ x_1_106);
    int8_t x_2_219 = (input[148] && x_1_80);
    int8_t x_2_220 = (input[60] || x_1_124);
    int8_t x_2_221 = (!x_1_127 || x_1_38);
    int8_t x_2_222 = (!x_1_166 && x_1_172);
    int8_t x_2_225 = x_1_198;
    int8_t x_2_228 = x_1_218;
    int8_t x_2_231 = (input[30] && !x_1_92);
    int8_t x_2_233 = (input[134] && !input[58]);
    int8_t x_2_234 = !input[193];
    int8_t x_2_236 = 1;
    int8_t x_2_237 = !(x_1_160 || input[26]);
    int8_t x_2_238 = !input[48];
    int8_t x_2_239 = !x_1_225;
    int8_t x_2_241 = (x_1_93 && !x_1_126);
    int8_t x_2_242 = (x_1_121 && x_1_202);
    int8_t x_2_243 = !(input[125] ^ input[171]);
    int8_t x_2_247 = (input[112] || x_1_39);
    int8_t x_2_248 = (x_1_108 || !x_1_147);
    int8_t x_2_249 = x_1_73;
    int8_t x_2_250 = !x_1_46;
    int8_t x_2_251 = !(input[53] && x_1_16);
    int8_t x_2_252 = (input[79] || input[152]);
    int8_t x_2_254 = (input[74] || !input[230]);
    output[0] = !(input[253] ^ x_2_69);
    output[1] = x_2_204;
    output[2] = input[95];
    output[3] = (!x_2_190 || x_2_37);
    output[4] = x_2_43;
    output[5] = (x_2_236 && x_2_144);
    output[6] = (x_2_19 || x_2_82);
    output[7] = (!input[45] || x_2_136);
    output[8] = (!x_2_46 || x_2_103);
    output[9] = (input[41] && input[93]);
    output[10] = !x_2_47;
    output[11] = x_2_221;
    output[12] = 0;
    output[13] = !input[204];
    output[14] = (!x_2_25 && input[162]);
    output[15] = 0;
    output[16] = (input[65] && !input[113]);
    output[17] = (input[80] || input[238]);
    output[18] = (!input[85] && input[48]);
    output[19] = !(x_2_106 && x_2_107);
    output[20] = input[0];
    output[21] = x_2_184;
    output[22] = !x_2_68;
    output[23] = (x_2_190 || !x_2_251);
    output[24] = !(input[167] && input[242]);
    output[25] = !(x_2_208 ^ input[142]);
    output[26] = x_2_39;
    output[27] = 0;
    output[28] = (input[250] || !x_2_126);
    output[29] = !(input[185] ^ input[105]);
    output[30] = (input[70] && !x_2_64);
    output[31] = 1;
    output[32] = !(input[232] && input[183]);
    output[33] = !x_2_65;
    output[34] = !(input[230] ^ x_2_19);
    output[35] = (input[203] && x_2_0);
    output[36] = (x_2_237 && !x_2_254);
    output[37] = 0;
    output[38] = 1;
    output[39] = (!input[237] && x_2_195);
    output[40] = !input[147];
    output[41] = 0;
    output[42] = !(x_2_61 ^ x_2_236);
    output[43] = 1;
    output[44] = !(x_2_248 ^ x_2_133);
    output[45] = !input[233];
    output[46] = !(x_2_44 || x_2_221);
    output[47] = x_2_247;
    output[48] = 1;
    output[49] = (!input[119] && x_2_126);
    output[50] = !(x_2_111 && input[120]);
    output[51] = !input[226];
    output[52] = (input[190] && !x_2_185);
    output[53] = !x_2_23;
    output[54] = (!input[15] && x_2_155);
    output[55] = (x_2_99 ^ x_2_79);
    output[56] = !(x_2_111 && input[114]);
    output[57] = (x_2_103 || input[225]);
    output[58] = !x_2_150;
    output[59] = input[55];
    output[60] = (x_2_13 && input[201]);
    output[61] = (input[74] ^ x_2_88);
    output[62] = (x_2_3 || x_2_182);
    output[63] = (input[92] || input[6]);
    output[64] = 1;
    output[65] = 0;
    output[66] = 1;
    output[67] = (input[56] && input[137]);
    output[68] = 1;
    output[69] = (x_2_249 || !input[84]);
    output[70] = !(input[215] && x_2_242);
    output[71] = input[23];
    output[72] = !(x_2_86 ^ input[116]);
    output[73] = !(x_2_86 ^ input[28]);
    output[74] = !x_2_42;
    output[75] = (x_2_202 && x_2_8);
    output[76] = !(input[212] || x_2_114);
    output[77] = 0;
    output[78] = (input[100] && !input[105]);
    output[79] = (input[115] && !input[166]);
    output[80] = !(input[228] && x_2_163);
    output[81] = (x_2_54 && input[140]);
    output[82] = input[212];
    output[83] = 0;
    output[84] = !(input[63] && x_2_183);
    output[85] = !input[191];
    output[86] = (x_2_100 || !x_2_164);
    output[87] = (!input[50] || x_2_236);
    output[88] = !input[102];
    output[89] = !(input[156] || x_2_27);
    output[90] = (input[94] || input[67]);
    output[91] = !input[38];
    output[92] = !(x_2_136 ^ input[219]);
    output[93] = 0;
    output[94] = 1;
    output[95] = (!x_2_237 || input[210]);
    output[96] = x_2_107;
    output[97] = 1;
    output[98] = (!input[190] && x_2_47);
    output[99] = (!x_2_37 && x_2_109);
    output[100] = (input[232] && !x_2_169);
    output[101] = !input[189];
    output[102] = (x_2_173 ^ input[111]);
    output[103] = (!input[43] && input[175]);
    output[104] = !(x_2_49 ^ x_2_37);
    output[105] = !(input[234] || input[213]);
    output[106] = (!x_2_26 && x_2_133);
    output[107] = (input[167] && !input[110]);
    output[108] = x_2_73;
    output[109] = !(input[166] ^ x_2_119);
    output[110] = (x_2_241 || x_2_122);
    output[111] = x_2_53;
    output[112] = (!x_2_215 || input[217]);
    output[113] = !(x_2_207 || input[223]);
    output[114] = (input[98] && x_2_77);
    output[115] = (x_2_231 || !x_2_220);
    output[116] = (!x_2_81 && x_2_20);
    output[117] = (x_2_23 || input[200]);
    output[118] = (input[246] && !input[37]);
    output[119] = !x_2_164;
    output[120] = 1;
    output[121] = (x_2_117 && x_2_18);
    output[122] = 0;
    output[123] = !x_2_36;
    output[124] = (input[175] || !x_2_125);
    output[125] = (!input[5] && x_2_46);
    output[126] = (!input[242] && x_2_174);
    output[127] = input[6];
    output[128] = (x_2_115 && !input[212]);
    output[129] = !(x_2_222 && x_2_161);
    output[130] = 1;
    output[131] = 1;
    output[132] = (x_2_9 || !x_2_74);
    output[133] = !(x_2_234 || input[121]);
    output[134] = (x_2_83 ^ input[182]);
    output[135] = (input[216] && x_2_64);
    output[136] = (!input[243] || x_2_225);
    output[137] = !(x_2_37 || x_2_173);
    output[138] = !(input[122] && x_2_140);
    output[139] = 1;
    output[140] = 0;
    output[141] = (input[100] && input[151]);
    output[142] = (input[99] || x_2_66);
    output[143] = (!input[208] && input[188]);
    output[144] = input[92];
    output[145] = !x_2_25;
    output[146] = (!x_2_243 && x_2_218);
    output[147] = (input[206] || !input[115]);
    output[148] = (input[27] && !x_2_94);
    output[149] = (input[31] || !x_2_178);
    output[150] = !x_2_239;
    output[151] = 1;
    output[152] = !(input[230] || x_2_210);
    output[153] = x_2_220;
    output[154] = (x_2_132 || !input[101]);
    output[155] = 0;
    output[156] = (x_2_190 && input[163]);
    output[157] = !(input[111] || x_2_174);
    output[158] = !x_2_83;
    output[159] = (!x_2_176 && input[123]);
    output[160] = 1;
    output[161] = !input[85];
    output[162] = x_2_65;
    output[163] = (input[156] && x_2_94);
    output[164] = (input[218] || input[95]);
    output[165] = !x_2_167;
    output[166] = !(input[60] || x_2_155);
    output[167] = (!x_2_165 || input[65]);
    output[168] = !x_2_24;
    output[169] = (!input[31] || input[65]);
    output[170] = input[247];
    output[171] = (!input[32] || x_2_238);
    output[172] = !(x_2_248 ^ x_2_50);
    output[173] = 1;
    output[174] = (!x_2_72 || input[222]);
    output[175] = (input[97] && x_2_216);
    output[176] = !x_2_179;
    output[177] = !(x_2_219 && x_2_176);
    output[178] = x_2_33;
    output[179] = !x_2_91;
    output[180] = !(x_2_7 ^ x_2_251);
    output[181] = (x_2_56 || x_2_40);
    output[182] = (x_2_164 && x_2_87);
    output[183] = 0;
    output[184] = (x_2_36 && !input[208]);
    output[185] = (!x_2_3 && input[25]);
    output[186] = (x_2_212 && x_2_171);
    output[187] = !input[87];
    output[188] = (!x_2_28 && x_2_80);
    output[189] = input[202];
    output[190] = !(input[210] || x_2_234);
    output[191] = !input[45];
    output[192] = !input[177];
    output[193] = input[65];
    output[194] = (input[240] && input[218]);
    output[195] = (!x_2_207 && input[113]);
    output[196] = (x_2_33 ^ x_2_128);
    output[197] = !(x_2_96 && input[85]);
    output[198] = input[111];
    output[199] = (!input[39] && input[61]);
    output[200] = !x_2_199;
    output[201] = (input[238] || !input[110]);
    output[202] = x_2_74;
    output[203] = (x_2_138 && !input[252]);
    output[204] = (x_2_43 || input[14]);
    output[205] = !input[189];
    output[206] = (input[19] ^ x_2_233);
    output[207] = !(input[137] || input[246]);
    output[208] = (x_2_198 && input[228]);
    output[209] = (x_2_77 ^ input[64]);
    output[210] = (x_2_158 || !x_2_19);
    output[211] = (!x_2_203 || input[34]);
    output[212] = 1;
    output[213] = (input[24] || !input[253]);
    output[214] = (input[66] || !x_2_23);
    output[215] = x_2_73;
    output[216] = !input[221];
    output[217] = (input[58] ^ x_2_53);
    output[218] = !(input[182] ^ x_2_104);
    output[219] = (!input[213] && input[106]);
    output[220] = !(input[63] && input[182]);
    output[221] = (input[100] ^ input[89]);
    output[222] = (input[25] || !input[189]);
    output[223] = !(x_2_159 ^ x_2_164);
    output[224] = (input[109] && input[245]);
    output[225] = !(input[158] && input[203]);
    output[226] = (x_2_241 && !x_2_111);
    output[227] = (x_2_48 || x_2_167);
    output[228] = (!input[201] || input[134]);
    output[229] = (x_2_1 ^ input[56]);
    output[230] = (!input[201] || x_2_155);
    output[231] = (x_2_166 || x_2_45);
    output[232] = x_2_139;
    output[233] = (input[109] ^ x_2_2);
    output[234] = !(x_2_138 ^ x_2_137);
    output[235] = (!input[83] && x_2_252);
    output[236] = 0;
    output[237] = 1;
    output[238] = (input[139] ^ input[16]);
    output[239] = 1;
    output[240] = (input[165] && !input[42]);
    output[241] = (x_2_239 || !input[248]);
    output[242] = !(x_2_250 || input[34]);
    output[243] = 0;
    output[244] = (x_2_210 && input[227]);
    output[245] = (input[172] && x_2_204);
    output[246] = (input[94] && !input[125]);
    output[247] = input[212];
    output[248] = !(input[34] ^ input[23]);
    output[249] = !(x_2_158 && input[23]);
    output[250] = !input[185];
    output[251] = (input[139] && input[39]);
    output[252] = (!input[51] && x_2_7);
    output[253] = 1;
    output[254] = x_2_208;
    output[255] = (!input[145] && x_2_228);
}

std::tuple<float, float, float> LogicNet::forward2(const Yolah& yolah) {
    uint8_t input[256];
    uint8_t output[256];
    uint64_t black = yolah.bitboard(Yolah::BLACK);
    uint64_t white = yolah.bitboard(Yolah::WHITE);
    uint64_t empty = yolah.empty_bitboard();
    uint8_t turn   = yolah.nb_plies() & 1;
    for (int i = 0; i < 64; i++) {
        input[i] = black >> i & 1ULL;
        input[i + 64] = white >> i & 1ULL;
        input[i + 128] = empty >> i & 1ULL;
        input[i + 192] = turn;
    } 
    initial_layers(input, output);
    int sum_black = 0;
    int sum_draw  = 0;
    int sum_white = 0;    
    for (int i = 0; i < 85; i++) {
        sum_black += output[i];
        sum_draw  += output[85 + i];
        sum_white += output[170 + i];
    }
    constexpr float DIV = 30;
    float b = std::exp(sum_black / DIV);
    float d = std::exp(sum_draw / DIV);
    float w = std::exp(sum_white / DIV);
    float s = b + d + w;
    return { b / s, d / s, w / s };
}