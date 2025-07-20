#ifndef LOGIC_NET_H
#define LOGIC_NET_H

#include <cstdint>
#include <array>
#include <vector>
#include "game.h"
#include <tuple>
#include <iostream>
#include <random>
#include <map>
#include <utility>
#include <string>
#include <string_view>

struct LogicNet {
    /*
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+        
        | A | B | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13| 14| 15|
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
        | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
        | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
        | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
        | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    0: FALSE           8: A AND B
    1: A NOR B         9: ~(A XOR B)
    2: ~(B=>A)        10: B 
    3: ~A             11: A=>B
    4: ~(A=>B)        12: A
    5: ~B             13: B=>A
    6: A XOR B        14: A OR B
    7: A NAND B       15: TRUE
    */
   static constexpr const char* const gates_repr[] = {
        "FALSE", "A NOR B", "~(B => A)", "~A", "~(A => B)", "~B", "A XOR B", "A NAND B",
        "A AND B", "~(A XOR B)", "B", "(A => B)", "A", "B => A", "A OR B", "TRUE"
    };
   static constexpr uint8_t gates_output[16 * 4] = {
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        1, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 1, 0,
        0, 1, 1, 0,
        1, 1, 1, 0,
        0, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 0, 1,
        1, 1, 0, 1,
        0, 0, 1, 1,
        1, 0, 1, 1,
        0, 1, 1, 1,
        1, 1, 1, 1,
   };
   struct Layer {
    static constexpr int SIZE = 256;
    alignas(64) std::array<uint16_t, SIZE> inputs1;
    alignas(64) std::array<uint16_t, SIZE> inputs2;
    alignas(64) std::array<uint8_t, SIZE> gates;
    Layer(std::mt19937_64&);
    Layer(uint8_t gate);
    Layer();
    void forward(const uint8_t* input, const uint8_t* prev, uint8_t* __restrict__ output) const;
    std::array<int, 16> gates_count() const;
    friend std::ostream& operator<<(std::ostream&, const Layer&);
    std::string to_json() const;
    static Layer from_json(std::istream&);
   };
   std::vector<Layer> layers;
   LogicNet(int nb_layers = 4, int gate = -1);
   std::tuple<float, float, float> forward(const Yolah& yolah) const;
   std::string c_expression_from_layer(int layer) const;
   std::array<int, 16> gates_count() const;
   friend std::ostream& operator<<(std::ostream&, const LogicNet&);
   std::string to_json() const;
   static LogicNet from_json(std::istream&);
   static std::tuple<float, float, float> forward2(const Yolah& yolay);
private:
    static void initial_layers(const uint8_t* input, uint8_t* __restrict__ output);
    std::string gate_to_c_expression(std::string_view i1, std::string_view i2, int gate) const;
};

#endif
