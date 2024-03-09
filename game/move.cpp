#include "move.h"
#include <string>
#include <cassert>

using std::string;

static const std::string square2string[SQUARE_NB] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
};

std::ostream& operator<<(std::ostream& os, const Move& m) {
    os << square2string[m.from_sq()] << ':' << square2string[m.to_sq()];
    return os;
}

std::istream& operator>>(std::istream& is, Move& m) {
    string from, to;
    is >> from >> to;
    auto get_square = [&](const string& sq) {
        int idx = std::find(begin(square2string), end(square2string), sq) - begin(square2string);
        assert(0 <= idx && idx < SQUARE_NB);
        return Square(idx);
    };
    m = Move(get_square(from), get_square(to));
    return is;
}
