#ifndef TYPES_H
#define TYPES_H 
#include <cstdint>
#include <bit>
enum Square : int8_t {
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
enum Direction : int8_t {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,
    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};
enum File : uint8_t {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
    FILE_NB
};
enum Rank : uint8_t {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
    RANK_NB
};
constexpr uint64_t FileABB = 0x0101010101010101;
constexpr uint64_t FileBBB = FileABB << 1;
constexpr uint64_t FileCBB = FileABB << 2;
constexpr uint64_t FileDBB = FileABB << 3;
constexpr uint64_t FileEBB = FileABB << 4;
constexpr uint64_t FileFBB = FileABB << 5;
constexpr uint64_t FileGBB = FileABB << 6;
constexpr uint64_t FileHBB = FileABB << 7;
constexpr uint64_t Rank1BB = 0xFF;
constexpr uint64_t Rank2BB = Rank1BB << (8 * 1);
constexpr uint64_t Rank3BB = Rank1BB << (8 * 2);
constexpr uint64_t Rank4BB = Rank1BB << (8 * 3);
constexpr uint64_t Rank5BB = Rank1BB << (8 * 4);
constexpr uint64_t Rank6BB = Rank1BB << (8 * 5);
constexpr uint64_t Rank7BB = Rank1BB << (8 * 6);
constexpr uint64_t Rank8BB = Rank1BB << (8 * 7);
constexpr uint64_t BLACK_INITIAL_POSITION =
0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
constexpr uint64_t WHITE_INITIAL_POSITION =
0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;
constexpr uint64_t FULL = 0xFFFFFFFFFFFFFFFF;
constexpr Square make_square(File f, Rank r) { 
    return Square((r << 3) + f); 
}
inline Square lsb(uint64_t b) {
    return Square(std::countr_zero(b));
}
inline Square pop_lsb(uint64_t& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}
constexpr bool more_than_one(uint64_t b) { 
    return b & (b - 1); 
}
template<Direction D>
constexpr uint64_t shift(uint64_t b) {
    if constexpr (D == NORTH)           return b << 8;
    else if constexpr (D == SOUTH)      return b >> 8;
    else if constexpr (D == EAST)       return (b & ~FileHBB) << 1;
    else if constexpr (D == WEST)       return (b & ~FileABB) >> 1;
    else if constexpr (D == NORTH_EAST) return (b & ~FileHBB) << 9;
    else if constexpr (D == NORTH_WEST) return (b & ~FileABB) << 7;
    else if constexpr (D == SOUTH_EAST) return (b & ~FileHBB) >> 7;
    else if constexpr (D == SOUTH_WEST) return (b & ~FileABB) >> 9;
    else return 0;
}
constexpr uint64_t shift_all_directions(uint64_t b) {
    uint64_t b1 = b & ~FileHBB;
    uint64_t b2 = b & ~FileABB;
    return b << 8 | b >> 8 | b1 << 1 | b1 << 9 
            | b1 >> 7 | b2 >> 1 | b2 << 7 | b2 >> 9;
}
#endif