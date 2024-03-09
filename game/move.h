#ifndef MOVE_H 
#define MOVE_H 

#include "types.h"
#include <iostream>

class Move {
    uint16_t data;
public:
    Move() = default;
    constexpr explicit Move(uint16_t d) : data(d) {}
    constexpr explicit Move(Square from, Square to) : data((to << 6) + from) {}
    constexpr Square to_sq() const {
        return Square((data >> 6) & 0x3F);
    }
    constexpr Square from_sq() const {
        return Square(data & 0x3F);
    }
    static constexpr Move none() { return Move(0); }
    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }
    constexpr explicit operator bool() const { return data != 0; }
    constexpr uint16_t raw() const { return data; }
};

std::ostream& operator<<(std::ostream& os, const Move& m);
std::istream& operator>>(std::istream& is, Move& m);

#endif