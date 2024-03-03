#ifndef TYPES_H
#define TYPES_H 

#include <cstdint>
#include <cmath>
#include <string>
#include <bit>

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,

    SQUARE_ZERO = 0,
    SQUARE_NB   = 64
};

enum Direction : int {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,

    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

enum File : int {
    FILE_A,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H,
    FILE_NB
};

enum Rank : int {
    RANK_1,
    RANK_2,
    RANK_3,
    RANK_4,
    RANK_5,
    RANK_6,
    RANK_7,
    RANK_8,
    RANK_NB
};

constexpr uint64_t FileABB = uint64_t(0x0101010101010101);
constexpr uint64_t FileBBB = FileABB << 1;
constexpr uint64_t FileCBB = FileABB << 2;
constexpr uint64_t FileDBB = FileABB << 3;
constexpr uint64_t FileEBB = FileABB << 4;
constexpr uint64_t FileFBB = FileABB << 5;
constexpr uint64_t FileGBB = FileABB << 6;
constexpr uint64_t FileHBB = FileABB << 7;

constexpr uint64_t Rank1BB = uint64_t(0xFF);
constexpr uint64_t Rank2BB = Rank1BB << (8 * 1);
constexpr uint64_t Rank3BB = Rank1BB << (8 * 2);
constexpr uint64_t Rank4BB = Rank1BB << (8 * 3);
constexpr uint64_t Rank5BB = Rank1BB << (8 * 4);
constexpr uint64_t Rank6BB = Rank1BB << (8 * 5);
constexpr uint64_t Rank7BB = Rank1BB << (8 * 6);
constexpr uint64_t Rank8BB = Rank1BB << (8 * 7);

constexpr uint64_t BLACK_INITIAL_POSITION = uint64_t(0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001);
constexpr uint64_t WHITE_INITIAL_POSITION = uint64_t(0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000); 
constexpr uint64_t FULL = uint64_t(0xFFFFFFFFFFFFFFFF);
    
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

constexpr File file_of(Square s) { return File(s & 7); }

constexpr Rank rank_of(Square s) { return Rank(s >> 3); }

constexpr uint64_t square_bb(Square s) {
    return uint64_t(1) << s;
}

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }

constexpr uint64_t rank_bb(Rank r) { return Rank1BB << (8 * r); }

constexpr uint64_t rank_bb(Square s) { return rank_bb(rank_of(s)); }

constexpr uint64_t file_bb(File f) { return FileABB << f; }

constexpr uint64_t file_bb(Square s) { return file_bb(file_of(s)); }

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline Square&   operator+=(Square& s, Direction d) { return s = s + d; }
inline Square&   operator-=(Square& s, Direction d) { return s = s - d; }
inline Square&   operator++(Square& d) { return d = Square(int(d) + 1); }
inline Square&   operator--(Square& d) { return d = Square(int(d) - 1); }

// Returns the least significant bit in a non-zero bitboard.
inline Square lsb(uint64_t b) {
    return Square(std::countr_zero(b));
}

// Returns the most significant bit in a non-zero bitboard.
inline Square msb(uint64_t b) {
    return Square(63 ^ std::countl_zero(b));
}

// Returns the bitboard of the least significant
// square of a non-zero bitboard. It is equivalent to square_bb(lsb(bb)).
inline uint64_t least_significant_square_bb(uint64_t b) {
    return b & -b;
}

// Finds and clears the least significant bit in a non-zero bitboard.
inline Square pop_lsb(uint64_t& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline uint64_t  operator&(uint64_t b, Square s) { return b & square_bb(s); }
inline uint64_t  operator|(uint64_t b, Square s) { return b | square_bb(s); }
inline uint64_t  operator^(uint64_t b, Square s) { return b ^ square_bb(s); }
inline uint64_t& operator|=(uint64_t& b, Square s) { return b |= square_bb(s); }
inline uint64_t& operator^=(uint64_t& b, Square s) { return b ^= square_bb(s); }

inline uint64_t operator&(Square s, uint64_t b) { return b & s; }
inline uint64_t operator|(Square s, uint64_t b) { return b | s; }
inline uint64_t operator^(Square s, uint64_t b) { return b ^ s; }

inline uint64_t operator|(Square s1, Square s2) { return square_bb(s1) | s2; }

constexpr bool more_than_one(uint64_t b) { return b & (b - 1); }

// Moves a bitboard one or two steps as specified by the direction D
template<Direction D>
constexpr uint64_t shift(uint64_t b) {
    return D == NORTH         ? b << 8
         : D == SOUTH         ? b >> 8
         : D == EAST          ? (b & ~FileHBB) << 1
         : D == WEST          ? (b & ~FileABB) >> 1
         : D == NORTH_EAST    ? (b & ~FileHBB) << 9
         : D == NORTH_WEST    ? (b & ~FileABB) << 7
         : D == SOUTH_EAST    ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST    ? (b & ~FileABB) >> 9
                              : 0;
}

#define ENABLE_INCR_OPERATORS_ON(T) \
        inline T& operator++(T& d) { return d = T(int(d) + 1); } \
        inline T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)

#undef ENABLE_INCR_OPERATORS_ON

namespace Bitboard {
    inline std::string pretty(uint64_t b) {

        std::string s = "+---+---+---+---+---+---+---+---+\n";

        for (Rank r = RANK_8; r >= RANK_1; --r) {
            for (File f = FILE_A; f <= FILE_H; ++f)
                s += b & make_square(f, r) ? "| X " : "|   ";

            s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";
        }
        s += "  a   b   c   d   e   f   g   h\n";

        return s;
    }
}

#endif