#include <bits/stdc++.h>

using namespace std;

constexpr uint64_t BLACK_INITIAL_POSITION =
0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
constexpr uint64_t WHITE_INITIAL_POSITION =
0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;

class Yolah {    
    uint64_t black = BLACK_INITIAL_POSITION;
    uint16_t black_score = 0;    
    uint64_t white = WHITE_INITIAL_POSITION;
    uint16_t white_score = 0;
    uint64_t holes = 0;
    uint16_t ply = 0;    
public:
};

int main() {
    cout << sizeof(Yolah) << '\n';
}
