#include "types.h"
#include <iostream>

using namespace std;

uint64_t around(uint64_t bb) {
    uint64_t north      = shift<NORTH>(bb);
    uint64_t south      = shift<SOUTH>(bb);
    uint64_t east       = shift<EAST>(bb);
    uint64_t west       = shift<WEST>(bb);
    uint64_t north_east = shift<NORTH_EAST>(bb);
    uint64_t south_east = shift<SOUTH_EAST>(bb);
    uint64_t north_west = shift<NORTH_WEST>(bb);
    uint64_t south_west = shift<SOUTH_WEST>(bb);
    return north | south | east | west | north_east | south_east | north_west | south_west;
}

int main() {
    cout << "constexpr uint64_t AROUND[64] = {\n";
    for (int i = 0; i < 64; i++) {
        //cout << Bitboard::pretty(around(uint64_t(1) << i)) << '\n';
        cout << hex << "    0x" << around(uint64_t(1) << i) << ",\n";
        // string _;
        // getline(cin, _);
    }
    cout << "};\n";
}
