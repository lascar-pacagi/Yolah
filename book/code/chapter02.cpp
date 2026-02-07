#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <immintrin.h>
using namespace std;

// 8  56   57   58   59   60   61   62   63
// 7  48   49   50   51   52   53   54   55
// 6  40   41   42   43   44   45   46   47
// 5  32   33   34   35   36   37   38   39
// 4  24   25   26   27   28   29   30   31
// 3  16   17   18   19   20   21   22   23
// 2   8    9   10   11   12   13   14   15
// 1   0    1    2    3    4    5    6    7
//     a    b    c    d    e    f    g    h
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

// 8  56     57     58     59     60     61     62     63
// 7  48     49     50     51     52     53     54     55
// 6  40     41     42     43     44     45     46     47
// 5  32     33     34     35     36     37     38     39
// 4  24     25     26     27     28     29     30     31
// 3  16     17     18     19     20     21     22     23
// 2   8      9     10     11     12     13     14     15
// 1   0      1      2      3      4      5      6      7
//  FILE_A FILE_B FILE_C FILE_D FILE_E FILE_F FILE_G FILE_H
enum File : uint8_t {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, 
    FILE_F, FILE_G, FILE_H, FILE_NB
};

//  RANK_8 | 56  57  58  59  60  61  62  63
//  RANK_7 | 48  49  50  51  52  53  54  55
//  RANK_6 | 40  41  42  43  44  45  46  47
//  RANK_5 | 32  33  34  35  36  37  38  39
//  RANK_4 | 24  25  26  27  28  29  30  31
//  RANK_3 | 16  17  18  19  20  21  22  23
//  RANK_2 |  8   9  10  11  12  13  14  15
//  RANK_1 |  0   1   2   3   4   5   6   7
//            a   b   c   d   e   f   g   h
enum Rank : uint8_t {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, 
    RANK_6, RANK_7, RANK_8, RANK_NB
};

// Directions from sq = 27 (d4):
//
//   +------+------+------+------+------+------+------+------+
// 8 |  56  |  57  |  58  |  59  |  60  |  61  |  62  |  63  |
//   +------+------+------+------+------+------+------+------+
// 7 |  48  |  49  |  50  |  51  |  52  |  53  |  54  |  55  |
//   +------+------+------+------+------+------+------+------+
// 6 |  40  |  41  |  42  |  43  |  44  |  45  |  46  |  47  |
//   +------+------+------+------+------+------+------+------+
// 5 |  32  |  33  |NW 34 |N  35 |NE 36 |  37  |  38  |  39  |
//   +------+------+------+------+------+------+------+------+
// 4 |  24  |  25  |W  26 |>>27<<|E  28 |  29  |  30  |  31  |
//   +------+------+------+------+------+------+------+------+
// 3 |  16  |  17  |SW 18 |S  19 |SE 20 |  21  |  22  |  23  |
//   +------+------+------+------+------+------+------+------+
// 2 |   8  |   9  |  10  |  11  |  12  |  13  |  14  |  15  |
//   +------+------+------+------+------+------+------+------+
// 1 |   0  |   1  |   2  |   3  |   4  |   5  |   6  |   7  |
//   +------+------+------+------+------+------+------+------+
//      a      b      c      d      e      f      g      h
enum Direction : int8_t {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH, // -8
    WEST  = -EAST,  // -1
    NORTH_EAST = NORTH + EAST, // 9
    SOUTH_EAST = SOUTH + EAST, // -7
    SOUTH_WEST = SOUTH + WEST, // -9
    NORTH_WEST = NORTH + WEST  // 7
};

// ORTHOGONAL moves from d4:         DIAGONAL moves from d4:
//
// 8  .  .  .  *  .  .  .  .         .  .  .  .  .  .  .  *
// 7  .  .  .  *  .  .  .  .         *  .  .  .  .  .  *  .
// 6  .  .  .  *  .  .  .  .         .  *  .  .  .  *  .  .
// 5  .  .  .  *  .  .  .  .         .  .  *  .  *  .  .  .
// 4  *  *  * sq  *  *  *  *         .  .  . sq  .  .  .  .
// 3  .  .  .  *  .  .  .  .         .  .  *  .  *  .  .  .
// 2  .  .  .  *  .  .  .  .         .  *  .  .  .  *  .  .
// 1  .  .  .  *  .  .  .  .         *  .  .  .  .  .  *  .
//    a  b  c  d  e  f  g  h         a  b  c  d  e  f  g  h
enum MoveType {
    ORTHOGONAL,
    DIAGONAL
};

// Black player
constexpr uint8_t BLACK = 0;
// White Player
constexpr uint8_t WHITE = 1;
// Square is destroyed
constexpr uint8_t HOLE  = 2;
// Square is free
constexpr uint8_t FREE  = 3;

// FileABB: bitboard for file A, bits on file A are set to 1,
// all other bits are set to 0. Other file bitboards are derived
// by shifting FileABB to the left.
//
//    FileABB           FileBBB           FileCBB
// 8  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 7  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 6  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 5  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 4  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 3  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 2  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
// 1  1 . . . . . . .   . 1 . . . . . .   . . 1 . . . . .
//    a b c d e f g h   a b c d e f g h   a b c d e f g h
//
//    FileDBB           FileEBB           FileFBB
// 8  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 7  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 6  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 5  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 4  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 3  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 2  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
// 1  . . . 1 . . . .   . . . . 1 . . .   . . . . . 1 . .
//    a b c d e f g h   a b c d e f g h   a b c d e f g h
//
//    FileGBB           FileHBB
// 8  . . . . . . 1 .   . . . . . . . 1
// 7  . . . . . . 1 .   . . . . . . . 1
// 6  . . . . . . 1 .   . . . . . . . 1
// 5  . . . . . . 1 .   . . . . . . . 1
// 4  . . . . . . 1 .   . . . . . . . 1
// 3  . . . . . . 1 .   . . . . . . . 1
// 2  . . . . . . 1 .   . . . . . . . 1
// 1  . . . . . . 1 .   . . . . . . . 1
//    a b c d e f g h   a b c d e f g h
constexpr uint64_t FileABB = 0x0101010101010101;
constexpr uint64_t FileBBB = FileABB << 1;
constexpr uint64_t FileCBB = FileABB << 2;
constexpr uint64_t FileDBB = FileABB << 3;
constexpr uint64_t FileEBB = FileABB << 4;
constexpr uint64_t FileFBB = FileABB << 5;
constexpr uint64_t FileGBB = FileABB << 6;
constexpr uint64_t FileHBB = FileABB << 7;

// Rank1BB: bitboard for rank 1, bits on rank 1 are set to 1,
// all other bits are set to 0. Other rank bitboards are derived
// by shifting Rank1BB to the left by multiples of 8.
//
//    Rank1BB           Rank2BB           Rank3BB
// 8  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 7  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 6  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 5  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 4  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 3  . . . . . . . .   . . . . . . . .   1 1 1 1 1 1 1 1
// 2  . . . . . . . .   1 1 1 1 1 1 1 1   . . . . . . . .
// 1  1 1 1 1 1 1 1 1   . . . . . . . .   . . . . . . . .
//    a b c d e f g h   a b c d e f g h   a b c d e f g h
//
//    Rank4BB           Rank5BB           Rank6BB
// 8  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 7  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 6  . . . . . . . .   . . . . . . . .   1 1 1 1 1 1 1 1
// 5  . . . . . . . .   1 1 1 1 1 1 1 1   . . . . . . . .
// 4  1 1 1 1 1 1 1 1   . . . . . . . .   . . . . . . . .
// 3  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 2  . . . . . . . .   . . . . . . . .   . . . . . . . .
// 1  . . . . . . . .   . . . . . . . .   . . . . . . . .
//    a b c d e f g h   a b c d e f g h   a b c d e f g h
//
//    Rank7BB           Rank8BB
// 8  . . . . . . . .   1 1 1 1 1 1 1 1
// 7  1 1 1 1 1 1 1 1   . . . . . . . .
// 6  . . . . . . . .   . . . . . . . .
// 5  . . . . . . . .   . . . . . . . .
// 4  . . . . . . . .   . . . . . . . .
// 3  . . . . . . . .   . . . . . . . .
// 2  . . . . . . . .   . . . . . . . .
// 1  . . . . . . . .   . . . . . . . .
//    a b c d e f g h   a b c d e f g h
constexpr uint64_t Rank1BB = 0xFF;
constexpr uint64_t Rank2BB = Rank1BB << (8 * 1);
constexpr uint64_t Rank3BB = Rank1BB << (8 * 2);
constexpr uint64_t Rank4BB = Rank1BB << (8 * 3);
constexpr uint64_t Rank5BB = Rank1BB << (8 * 4);
constexpr uint64_t Rank6BB = Rank1BB << (8 * 5);
constexpr uint64_t Rank7BB = Rank1BB << (8 * 6);
constexpr uint64_t Rank8BB = Rank1BB << (8 * 7);

// Initial positions: Black starts at a1, d4, e5, h8.
// White starts at a8, d5, e4, h1.
//
// BLACK_INITIAL_POSITION       WHITE_INITIAL_POSITION
//
// 8  .  .  .  .  .  .  .  B    W  .  .  .  .  .  .  .
// 7  .  .  .  .  .  .  .  .    .  .  .  .  .  .  .  .
// 6  .  .  .  .  .  .  .  .    .  .  .  .  .  .  .  .
// 5  .  .  .  .  B  .  .  .    .  .  .  W  .  .  .  .
// 4  .  .  .  B  .  .  .  .    .  .  .  .  W  .  .  .
// 3  .  .  .  .  .  .  .  .    .  .  .  .  .  .  .  .
// 2  .  .  .  .  .  .  .  .    .  .  .  .  .  .  .  .
// 1  B  .  .  .  .  .  .  .    .  .  .  .  .  .  .  W
//    a  b  c  d  e  f  g  h    a  b  c  d  e  f  g  h
constexpr uint64_t BLACK_INITIAL_POSITION =
0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
constexpr uint64_t WHITE_INITIAL_POSITION =
0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;

// Returns the square at rank i, file j
constexpr Square square_of(int i, int j) {
    return Square(i * 8 + j);
}

// Checks whether a square is within the valid range [A1, H8]
constexpr bool is_ok(Square s) {
    return s >= SQ_A1 && s <= SQ_H8;
}

// Extracts the file (column) from a square: s % 8
constexpr File file_of(Square s) {
    return File(s & 7);
}

// Extracts the rank (row) from a square: s / 8
constexpr Rank rank_of(Square s) {
    return Rank(s >> 3);
}

// Returns a bitboard with all bits set on rank r
constexpr uint64_t rank_bb(Rank r) {
    return Rank1BB << (8 * r);
}

// Returns a bitboard with all bits set on the rank of square s
constexpr uint64_t rank_bb(Square s) {
    return rank_bb(rank_of(s));
}

// Returns a bitboard with all bits set on file f
constexpr uint64_t file_bb(File f) {
    return FileABB << f;
}

// Returns a bitboard with all bits set on the file of square s
constexpr uint64_t file_bb(Square s) {
    return file_bb(file_of(s));
}

// Returns a bitboard with only the bit corresponding to square s set
constexpr uint64_t square_bb(Square s) {
    return uint64_t(1) << s;
}

// Moves a square in a given direction
constexpr Square operator+(Square s, Direction d) {
    return Square(int(s) + int(d));
}

// Pre-increment operator to iterate over squares
Square& operator++(Square& d) {
    return d = Square(int(d) + 1);
}

// Returns the least significant bit as a Square (index 
// of first set bit)
constexpr Square lsb(uint64_t b) {
    return Square(std::countr_zero(b));
}

// Extracts and clears the least significant bit, returns it as 
// a Square
Square pop_lsb(uint64_t& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

// Computes the Manhattan distance (|rank1 - rank2| + |file1 - file2|)
int manhattan_distance(Square sq1, Square sq2) {
    int d_rank = std::abs(rank_of(sq1) - rank_of(sq2));
    int d_file = std::abs(file_of(sq1) - file_of(sq2));
    return d_rank + d_file;
}

// Shifts all bits of a bitboard in direction D. For horizontal and
// diagonal shifts, bits on the edge file are masked out to prevent 
// wrapping around the board (e.g., a piece on file H shifted east 
// would appear on file A)
template<Direction D>
constexpr uint64_t shift(uint64_t b) {
    if constexpr (D == NORTH)
        return b << NORTH;
    else if constexpr (D == SOUTH)
        return b >> -SOUTH;
    else if constexpr (D == EAST)
        return (b & ~FileHBB) << EAST;
    else if constexpr (D == WEST)
        return (b & ~FileABB) >> -WEST;
    else if constexpr (D == NORTH_EAST)
        return (b & ~FileHBB) << NORTH_EAST;
    else if constexpr (D == NORTH_WEST)
        return (b & ~FileABB) << NORTH_WEST;
    else if constexpr (D == SOUTH_EAST)
        return (b & ~FileHBB) >> -SOUTH_EAST;
    else if constexpr (D == SOUTH_WEST)
        return (b & ~FileABB) >> -SOUTH_WEST;
    else return 0;
}

// Computes all squares reachable from sq by sliding in the given move 
// type (orthogonal or diagonal), stopping at occupied squares or
// board edges. Used during magic bitboard initialization to build the 
// move tables
uint64_t reachable_squares(MoveType mt, Square sq, 
                           uint64_t occupied) {
    uint64_t  moves    = 0;
    Direction o_dir[4]={NORTH, SOUTH, EAST, WEST};
    Direction d_dir[4]={NORTH_EAST,SOUTH_EAST,SOUTH_WEST,NORTH_WEST};
    for (Direction d : (mt == ORTHOGONAL ? o_dir : d_dir)) {
        Square s = sq;
        while (true) {
            Square to = s + d;
            if (!is_ok(to) || manhattan_distance(s, to) > 2) break;
            uint64_t bb = square_bb(to);
            if ((bb & occupied) != 0) break;
            moves |= bb;
            s = to;
        }
    }
    return moves;
}

// Magic bitboard entry for one square. The mask isolates the relevant
// occupancy bits, the magic number maps them to a unique index, and 
// the moves pointer references the precomputed move bitboards in the 
// table
struct Magic {
    uint64_t  mask;   // relevant occupancy mask (edges excluded)
    uint64_t  magic;  // magic multiplier for this square
    uint64_t* moves;  // pointer into the move lookup table
    uint32_t  shift;  // right shift amount = 64 - popcount(mask)

    // Computes the table index for a given occupancy configuration
    uint32_t index(uint64_t occupied) const {
        return uint32_t( ((occupied & mask) * magic) >> shift );
    }
};

// Precomputed move lookup tables for all squares
uint64_t orthogonalTable[102400];
uint64_t diagonalTable[5248];

// Magic bitboard entries for each square
Magic orthogonalMagics[SQUARE_NB];
Magic diagonalMagics[SQUARE_NB];

// Combines orthogonal and diagonal magic table lookups to return a 
// bitboard of all squares that a piece on sq can reach, given the 
// board occupancy
uint64_t moves_bb(Square sq, uint64_t occupied) {
    uint32_t idx_omoves = orthogonalMagics[sq].index(occupied);
    uint32_t idx_dmoves = diagonalMagics[sq].index(occupied);
    return orthogonalMagics[sq].moves[idx_omoves] | diagonalMagics[sq].moves[idx_dmoves];
}

// Initializes magic bitboard tables for a given move type (orthogonal 
// or diagonal). For each square, enumerates all possible occupancy 
// patterns using the Carry-Rippler trick, computes the corresponding 
// reachable squares, and stores them in the lookup table indexed by 
// the magic hash
void init_magics(MoveType mt, uint64_t table[], Magic magics[]) {
    static constexpr uint64_t O_MAGIC[64] = {
        0x80011040002082,   0x40022002100040,   0x1880200081181000,
        0x2080240800100080, 0x8080024400800800, 0x4100080400024100,
        0xc080028001000a00, 0x80146043000080,   0x8120802080034004,
        0x8401000200240,    0x202001282002044,  0x81010021000b1000,
        0x808044000800,     0x300080800c000200, 0x8c000268411004,
        0x810080058020c100, 0xc248608010400080, 0x30024040002000,
        0x9001010042102000, 0x210009001002,     0xa0061d0018001100,
        0x2410808004000600, 0x6400240008025001, 0xc10600010340a4,
        0x628080044011,     0x4810014040002000, 0x380200080801000,
        0x10018580080010,   0x101040080180180,  0x9208020080040080,
        0x10400a21008,      0x6800104200010484, 0x21400280800020,
        0x9400402008401001, 0x8430006800200400, 0x8104411202000820,
        0x8010171000408,    0x1202000402001008, 0x881100904002208,
        0x15a0800a49802100, 0x224001808004,     0x4420201002424000,
        0xc04500020008080,  0x2503009004210008, 0x42801010010,
        0x2000400090100,    0x8080011810040002, 0x44401c008046000d,
        0x4000800521104100, 0x82000b080400080,  0x10821022420200,
        0x9488a82104100100, 0x1004800041100,    0x81600a0034008080,
        0xa00056210280400,  0x5124088200,       0x4210410010228202,
        0x1802230840001081, 0x1002102000400901, 0x1100c46010000901,
        0x281000408001003,  0xc001001c00028809, 0x10020008008c4102,
        0x280005008c014222,
    };
    static constexpr uint64_t D_MAGIC[64] = {
        0x811100100408200,  0x412100401044020,  0x404044c00408002,
        0xa0c070200010102,  0x104042001400008,  0x8802013008080000,
        0x1001008860080080, 0x20220044202800,   0x2002610802080160,
        0x4080800808610,    0x91c2800a10a0132,  0x400242401822000,
        0x8530040420040001, 0x142010c210048,    0x8841820801241004,
        0x804212084108801,  0x2032402094100484, 0x40202110010210a2,
        0x8010000800202020, 0x800240421a800,    0x62200401a00444,
        0x224082200820845,  0x106021492012000,  0x8481020082849000,
        0x40a110c59602800,  0x10020108020400,   0x208c020844080010,
        0x2000480004012020, 0x8001004004044000, 0xa044104128080200,
        0x1108008015cc1400, 0x8284004801844400, 0x8180a020c2004,
        0x9101004080100,    0x8840264108800c0,  0xc004200900200900,
        0x8040008020020020, 0x20010802e1920200, 0x80204000480a0,
        0xc0a80a100008400,  0x4018808114000,    0x90092200b9000,
        0x80020c0048000400, 0x6018005500,       0x80a0204110a00,
        0x4018808407201,    0x6050040806500280, 0x108208400c40180,
        0x803081210840480,  0x201210402200200,  0x200010400920042,
        0x902000a884110010, 0x851002021004,     0x43c08020120,
        0x6140500501010044, 0x200a04440400c028, 0x14a002084046000,
        0x10002409041040,   0x100022020500880b, 0x1000000000460802,
        0x21084104410,      0x8000001053300104, 0x4000182008c20048,
        0x112088105020200,
    };
    int size = 0;
    vector<uint64_t> occupancies;
    vector<uint64_t> possible_moves;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        occupancies.clear();
        possible_moves.clear();
        Magic& m = magics[sq];
        uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) |
                            ((FileABB | FileHBB) & ~file_bb(sq));
        uint64_t moves_bb = reachable_squares(mt, sq, 0) & ~edges;
        m.mask = moves_bb;
        m.shift = 64 - popcount(m.mask);
        m.magic = (mt == ORTHOGONAL ? O_MAGIC : D_MAGIC)[sq];
        m.moves = table + size;
        uint64_t b = 0;
        do {
            occupancies.push_back(b);
            possible_moves.push_back(reachable_squares(mt, sq, b));
            b = (b - moves_bb) & moves_bb;
            size++;
        } while (b);
        for (size_t j = 0; j < occupancies.size(); j++) {
            int32_t index = m.index(occupancies[j]);
            m.moves[index] = possible_moves[j];
        }
    }
}

// Initializes both orthogonal and diagonal magic bitboard tables
void init_all_magics() {
    init_magics(ORTHOGONAL, orthogonalTable, orthogonalMagics);
    init_magics(DIAGONAL, diagonalTable, diagonalMagics);
}

// Compact move representation: packs source and destination squares 
// into a 16-bit integer (6 bits each). Move::none() encodes a 
// pass (data = 0, i.e., from = a1, to = a1), used when a player has
// no legal move
class Move {
    uint16_t data;
public:
    constexpr Move() noexcept = default;
    constexpr explicit Move(uint16_t d) noexcept : data(d) {}
    constexpr explicit Move(Square from, Square to) noexcept
        : data((to << 6) + from) {}
    constexpr Square from_sq() const noexcept {
        return Square(data & 0x3F);
    }
    constexpr Square to_sq() const noexcept {
        return Square((data >> 6) & 0x3F);
    }
    constexpr uint16_t raw() const noexcept { return data; }
    static constexpr Move none() noexcept { return Move(0); }
    constexpr bool operator==(const Move& m) const noexcept {
        return data == m.data;
    }
    constexpr bool operator!=(const Move& m) const noexcept {
        return data != m.data;
    }
    constexpr bool operator<(const Move& m) const noexcept {
        return make_pair(from_sq(), to_sq()) < 
               make_pair(m.from_sq(), m.to_sq());
    }
    constexpr explicit operator bool() const noexcept { 
        return data != 0; 
    }
};

// Prints a move in "from:to" notation (e.g., "d4:d7")
ostream& operator<<(ostream& os, const Move& m) {
    static constexpr std::string_view square2string[SQUARE_NB] = {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    };
    os << square2string[m.from_sq()] << ':' 
       << square2string[m.to_sq()];
    return os;
}

// Stack-allocated move list with a fixed capacity. Avoids heap 
// allocation during move generation. 
// Over millions of random games, the observed maximum number of 
// moves never exceeded 73; 75 is a safe upper bound
static constexpr uint16_t MAX_NB_MOVES = 75;
class MoveList {
    Move move_list[MAX_NB_MOVES], *last;
public:
    constexpr MoveList() noexcept : last(move_list) {}
    constexpr const Move* begin() const noexcept { return move_list; }
    constexpr const Move* end() const noexcept { return last; }
    constexpr Move* begin() noexcept { return move_list; }
    constexpr Move* end() noexcept { return last; }
    constexpr size_t size() const noexcept { 
        return last - move_list; 
    }
    constexpr const Move& operator[](size_t i) const { 
        return move_list[i]; 
    }
    constexpr Move& operator[](size_t i) { return move_list[i]; }
    constexpr Move* data() noexcept { return move_list; }
    friend class Yolah;
};

// Forward declaration of test function
namespace test {
    void random_games(size_t nb_games, 
                      optional<uint64_t> seed = nullopt);
}
struct YolahWithMoves;

// Board game. Uses three bitboards (black, white, holes) to 
// represent the board. Each player starts with 4 pieces. On each 
// turn, a piece slides orthogonally or diagonally to a free square; 
// the departure square becomes a hole. The game ends when no piece 
// can move. The player with the most moves wins
class Yolah {
    uint64_t black = BLACK_INITIAL_POSITION;
    uint64_t white = WHITE_INITIAL_POSITION;
    uint64_t holes = 0;        // destroyed squares
    uint8_t black_score = 0;   // number of moves played by black
    uint8_t white_score = 0;   // number of moves played by white
    uint8_t ply = 0;           // current ply (0 = black's first turn)
public:
    // The game is over when no piece of either player has an adjacent 
    // free square
    constexpr bool game_over() const noexcept {
        uint64_t possible = ~holes & ~black & ~white;
        uint64_t players  = black | white;
        uint64_t around_players = shift<NORTH>(players) |
            shift<SOUTH>(players) | shift<EAST>(players) |
            shift<WEST>(players) | shift<NORTH_EAST>(players) |
            shift<NORTH_WEST>(players) | shift<SOUTH_EAST>(players) |
            shift<SOUTH_WEST>(players);
        return (around_players & possible) == 0;
    }

    // Returns BLACK (0) or WHITE (1) based on parity of ply
    constexpr uint8_t current_player() const noexcept {
        return ply & 1;
    }

    constexpr uint8_t nb_plies() const noexcept {
        return ply;
    }

    constexpr pair<uint8_t, uint8_t> score() const noexcept {
        return {black_score, white_score};
    }

    // Returns the content of a square: BLACK, WHITE, HOLE, or FREE
    constexpr uint8_t get(Square sq) const noexcept {
        uint64_t bb = square_bb(sq);
        if (holes & bb) return HOLE;
        if (black & bb) return BLACK;
        if (white & bb) return WHITE;
        return FREE;
    }

    constexpr uint8_t get(int i, int j) const noexcept {        
        return get(square_of(i, j));
    }

    // Generates all legal moves for the given player into the 
    // provided MoveList. Since each player always has exactly 
    // 4 pieces, the first loop is fully unrolled (version 2) 
    // to avoid branch mispredictions. If no move is available,
    // a single Move::none() is added
    void moves(uint8_t player, MoveList& moves) const noexcept {
        Move* move_list = moves.move_list;
        uint64_t occupied = black | white | holes;
        uint64_t bb = player == BLACK ? black : white;

        // Version 1: generic loop (commented out, kept for reference)
        // while (bb) {
        //     Square from = pop_lsb(bb);
        //     uint64_t b = moves_bb(from, occupied) & ~occupied;
        //     while (b) {
        //         *move_list++ = Move(from, pop_lsb(b));
        //     }
        // }

        // Version 2: unrolled loop for the 4 pieces
        Square from0 = pop_lsb(bb);
        Square from1 = pop_lsb(bb);
        Square from2 = pop_lsb(bb);
        Square from3 = pop_lsb(bb);

        uint64_t b0 = moves_bb(from0, occupied) & ~occupied;
        uint64_t b1 = moves_bb(from1, occupied) & ~occupied;
        uint64_t b2 = moves_bb(from2, occupied) & ~occupied;
        uint64_t b3 = moves_bb(from3, occupied) & ~occupied;

        while (b0) {
            *move_list++ = Move(from0, pop_lsb(b0));
        }
        while (b1) {
            *move_list++ = Move(from1, pop_lsb(b1));
        }
        while (b2) {
            *move_list++ = Move(from2, pop_lsb(b2));
        }
        while (b3) {
            *move_list++ = Move(from3, pop_lsb(b3));
        }
        if (move_list == moves.move_list) [[unlikely]] {
            *move_list++ = Move::none();
        }

        moves.last = move_list;
    }

    // Generates moves for the current player
    void moves(MoveList& moves) const noexcept {
        this->moves(current_player(), moves);
    }

    // Selects a uniformly random legal move without allocating a
    // move list while minimizing branches
    Move random_move(mt19937& mt) const noexcept {
        uint64_t occupied = black | white | holes;
        uint64_t player_bb = 
            current_player() == BLACK ? black : white;
        Square from0 = pop_lsb(player_bb);
        Square from1 = pop_lsb(player_bb);
        Square from2 = pop_lsb(player_bb);
        Square from3 = pop_lsb(player_bb);
        uint64_t b0 = moves_bb(from0, occupied) & ~occupied;
        uint64_t b1 = moves_bb(from1, occupied) & ~occupied;
        uint64_t b2 = moves_bb(from2, occupied) & ~occupied;
        uint64_t b3 = moves_bb(from3, occupied) & ~occupied;
        int n0 = popcount(b0);
        int n1 = popcount(b1);
        int n2 = popcount(b2);
        int n3 = popcount(b3);
        int n = n0 + n1 + n2 + n3;
        if (n == 0) [[unlikely]] {
            return Move::none();
        }
        uniform_int_distribution<int> d(0, n - 1);
        int bit = d(mt);
        int bb_index = (bit >= n0) + (bit >= n0 + n1) + 
                       (bit >= n0 + n1 + n2);
        bit -= (bb_index > 0) * n0 + (bb_index > 1) * n1 + 
               (bb_index > 2) * n2;
        Square from = array{from0, from1, from2, from3}[bb_index];
        uint64_t bb = array{ b0, b1, b2, b3 }[bb_index];        
        // _pdep_u64 extracts the bit-th destination from bb without
        // a loop. Example: bb = 0b01010100 (3 set bits), 
        // bit = 2 (3rd move):
        //   1ULL << 2           = 0b00000100
        //   _pdep_u64(0b100,bb) = 0b01000000  (the 3rd set bit of bb)
        //   countr_zero(...)    = 6           (square index)
        //
        // Without _pdep_u64, the equivalent code would be:
        // for (int i = 0; i < bit; i++) bb &= bb - 1; 
                                            // bb &= bb - 1 
                                            // clears the least
                                            // significant bit of bb 
        // Square to = countr_zero(bb);                          
        Square to = Square(countr_zero(_pdep_u64(1ULL << bit, bb)));
        return Move(from, to);
    }

    // Applies a move: moves the piece from source to destination, 
    // turns the source square into a hole, and increments the 
    // player's score.
    // For Move::none() (pass), only the ply counter is incremented
    void play(Move m) noexcept {
        if (m != Move::none()) [[likely]] {
            uint64_t pos1 = square_bb(m.from_sq());
            uint64_t pos2 = square_bb(m.to_sq());
            if (ply & 1) {
                white ^= pos1 | pos2;
                white_score++;
            } else {
                black ^= pos1 | pos2;
                black_score++;
            }
            holes |= pos1;
        }
        ply++;
    }

    // Reverses a move: restores the piece, removes the hole, 
    // decrements score
    void undo(Move m) noexcept {
        ply--;
        if (m != Move::none()) [[likely]] {
            uint64_t pos1 = square_bb(m.from_sq());
            uint64_t pos2 = square_bb(m.to_sq());
            if (ply & 1) {
                white ^= pos1 | pos2;
                white_score--;
            } else {
                black ^= pos1 | pos2;
                black_score--;
            }
            holes ^= pos1;
        }
    }

    // Equality: compares all state fields
    constexpr bool operator==(const Yolah& other) const noexcept {
        return black == other.black
            && white == other.white
            && holes == other.holes
            && black_score == other.black_score
            && white_score == other.white_score
            && ply == other.ply;
    }

    constexpr bool operator!=(const Yolah& other) const noexcept {
        return !(*this == other);
    }
};

// Pairs a Yolah board with its legal moves for display purposes
struct YolahWithMoves {
    const Yolah& yolah;
    const MoveList& moves;
    YolahWithMoves(const Yolah& y, 
                   const MoveList& m) : yolah(y), moves(m) {}
};

// Displays the board with Unicode box-drawing, marking 
// reachable squares
ostream& operator<<(ostream& os, const YolahWithMoves& ym) {
    char grid[8][8];
    const Yolah& yolah = ym.yolah;
    const MoveList& moves = ym.moves;
    static constexpr string_view players[] = {
        "Black player",
        "White player"
    };
    static constexpr char content[] = {'B', 'W', 'H', '.'};
    uint8_t player = yolah.current_player();
    os << players[player] << '\n';
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Square dst = square_of(i, j);
            if (any_of(begin(moves), end(moves), [&](const Move& m) {
                return m.to_sq() == dst;
            })) {
                grid[i][j] = player == BLACK ? 'X' : 'x';
            }
            else grid[i][j] = content[yolah.get(dst)];
        }
    }
    os << "\n  ┌───┬───┬───┬───┬───┬───┬───┬───┐\n";
    for (int i = 7; i >= 0; i--) {
        os << i + 1 << " │";
        for (int j = 0; j < 8; j++) {
            char c = grid[i][j];
            if (c == 'B') os << " ○ ";
            else if (c == 'W') os << " ● ";
            else if (c == 'H') os << "   ";
            else if (c == '.') os << " · ";
            else os << ' ' << c << ' ';
            os << "│";
        }
        os << '\n';
        if (i > 0) os << "  ├───┼───┼───┼───┼───┼───┼───┼───┤\n";
    }
    os << "  └───┴───┴───┴───┴───┴───┴───┴───┘\n";
    os << "    a   b   c   d   e   f   g   h\n";
    const auto [black_score, white_score] = yolah.score();
    os << "score: " << int(black_score) << '/' 
       << int(white_score) << '\n';
    return os;
}

// Displays the board without move annotations
ostream& operator<<(ostream& os, const Yolah& yolah) {
    return os << YolahWithMoves(yolah, MoveList());    
}

// Plays random games using standard move generation. When 
// STEP_BY_STEP is true, prints board and moves at each turn 
// and waits for user input
template<bool STEP_BY_STEP = true>
void play_random_games(size_t nb_games, 
                       optional<uint64_t> seed = nullopt) {
    MoveList moves;
    random_device rd;
    mt19937 mt(seed.value_or(rd()));
    size_t black_wins = 0;
    size_t white_wins = 0;
    size_t draws = 0;

    for (size_t i = 0; i < nb_games; i++) {
        Yolah yolah;    
        while (!yolah.game_over()) {                 
            if constexpr (STEP_BY_STEP) cout << yolah << '\n';        
            yolah.moves(moves);
            if constexpr (STEP_BY_STEP) {
                sort(begin(moves), end(moves));
                cout << format("# moves: {}\n", moves.size());
                for (const auto& m : moves) {
                    cout << m << ' ';
                }
                cout << "\n\n";
                cout << YolahWithMoves(yolah, moves) << '\n';
            }
            uniform_int_distribution<int> d(0, moves.size() - 1);
            Move m = moves[d(mt)];
            if constexpr (STEP_BY_STEP) {
                cout << m << '\n';
                std::string _;
                std::getline(std::cin, _);
            }
            yolah.play(m);
        }
        if constexpr (STEP_BY_STEP) cout << yolah << '\n';
        
        auto [black_score, white_score] = yolah.score();
        if (black_score > white_score) {
            black_wins++;
        } else if (white_score > black_score) {
            white_wins++;
        } else {
            draws++;
        }
    }

    if constexpr (!STEP_BY_STEP) {
        cout << format("\n=== Game Statistics ===\n");
        cout << format("Total games: {}\n", nb_games);
        cout << format("Black wins:  {} ({:.1f}%)\n", black_wins, 
                       100.0 * black_wins / nb_games);
        cout << format("White wins:  {} ({:.1f}%)\n", white_wins, 
                       100.0 * white_wins / nb_games);
        cout << format("Draws:       {} ({:.1f}%)\n", draws, 
                       100.0 * draws / nb_games);
    }
}

// Plays random games using random_move (no move list allocation)
void play_random_games_fast(size_t nb_games, 
                            optional<uint64_t> seed = nullopt) {
    random_device rd;
    mt19937 mt(seed.value_or(rd()));
    size_t black_wins = 0;
    size_t white_wins = 0;
    size_t draws = 0;
    for (size_t i = 0; i < nb_games; i++) {
        Yolah yolah;
        while (!yolah.game_over()) {
            Move m = yolah.random_move(mt);
            yolah.play(m);
        }
        auto [black_score, white_score] = yolah.score();
        if (black_score > white_score) {
            black_wins++;
        } else if (white_score > black_score) {
            white_wins++;
        } else {
            draws++;
        }
    }
    cout << format("\n=== Game Statistics ===\n");
    cout << format("Total games: {}\n", nb_games);
    cout << format("Black wins:  {} ({:.1f}%)\n", black_wins, 
                   100.0 * black_wins / nb_games);
    cout << format("White wins:  {} ({:.1f}%)\n", white_wins, 
                   100.0 * white_wins / nb_games);
    cout << format("Draws:       {} ({:.1f}%)\n", draws, 
                   100.0 * draws / nb_games);
}

// Differential and property-based tests: differential testing 
// validates move generation by comparing against a naive reference 
// implementation; property-based testing checks invariants like undo 
// reversibility and game-over detection
namespace test {
    namespace {
        using std::format;

        constexpr string_view RED    = "\033[1;31m";
        constexpr string_view GREEN  = "\033[1;32m";
        constexpr string_view YELLOW = "\033[1;33m";
        constexpr string_view RESET  = "\033[0m";
        constexpr string_view BOLD   = "\033[1m";

        struct TestResult {
            bool passed;
            string message;
            operator bool() const { return passed; }
        };

        TestResult pass() { return {true, ""}; }
        TestResult fail(string msg) { 
            return {false, std::move(msg)}; 
        }

        // Reference move generator: brute-force loop over all 
        // squares/directions
        vector<Move> slow_moves_generation(const Yolah& yolah) {
            vector<Move> res;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    Square from = square_of(i, j);
                    if (yolah.get(from) != yolah.current_player()) {
                        continue;
                    }
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            if (di == 0 && dj == 0) continue;
                            int ii = i + di;
                            int jj = j + dj;
                            for(;;) {
                                if (ii < 0 || ii >= 8 || 
                                    jj < 0 || jj >= 8) break;
                                Square to = square_of(ii,jj);                                
                                if (yolah.get(to) != FREE) break;                                
                                res.emplace_back(from, to);
                                ii += di;
                                jj += dj;
                            }
                        }
                    }
                }
            }
            if (res.empty()) res.push_back(Move::none());
            return res;
        }

        // Verifies that magic and naive generators produce the same 
        // move count
        TestResult check_move_count(const MoveList& fast, 
                                    const vector<Move>& expected) {
            if (fast.size() != expected.size()) {
                return fail(format("# of moves: expected {} got {}", 
                                   expected.size(), fast.size()));
            }
            return pass();
        }

        // Sorts magic and naive lists and checks element-wise 
        // equality, reports differences
        TestResult check_move_lists_equal(MoveList& fast, 
                                          vector<Move>& expected, 
                                          const Yolah& yolah) {
            sort(begin(fast), end(fast));
            sort(begin(expected), end(expected));
            if (equal(begin(fast), end(fast), 
                      begin(expected), end(expected))) {
                return pass();
            }
            ostringstream oss;
            oss << "move lists differ\n" << yolah << '\n';
            vector<Move> only_in_fast, only_in_expected;
            set_difference(begin(fast), end(fast), begin(expected), 
                           end(expected), 
                           back_inserter(only_in_fast));
            set_difference(begin(expected), end(expected), begin(fast), 
                           end(fast), 
                           back_inserter(only_in_expected));
            if (!only_in_expected.empty()) {
                oss << "  Only in expected: ";
                for (const auto& m : only_in_expected) {
                    oss << m << ' ';
                }
                oss << '\n';
            }
            if (!only_in_fast.empty()) {
                oss << "  Only in fast: ";
                for (const auto& m : only_in_fast) oss << m << ' ';
                oss << '\n';
            }
            return fail(oss.str());
        }

        // Verifies that undo restores the board to its previous state
        TestResult check_undo(const Yolah& before, 
                              const Yolah& after) {
            if (before == after) return pass();
            ostringstream oss;
            oss << "undo failed\n  Previous state:\n" << before
                << "\n  State after undo:\n" << after << '\n';
            return fail(oss.str());
        }

        // When the game is over, the only legal move must be 
        // Move::none()
        TestResult check_game_over_moves(const Yolah& yolah, 
                                         const MoveList& moves) {
            if (moves.size() == 1 && moves[0] == Move::none()) {
                return pass();
            }
            ostringstream oss;
            oss << "only Move::none() should be available when"
                << " game is over\n"
                << YolahWithMoves(yolah, moves) << '\n';
            return fail(oss.str());
        }

        // Move::none() must not alter the board, only increment the 
        // ply counter
        TestResult check_none_move_execution(const Yolah& before, 
                                             const Yolah& after) {
            bool ok = true;
            for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
                if (before.get(sq) != after.get(sq)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                return fail("Move::none() must not change the board content");
            }
            if (before.nb_plies() + 1 != after.nb_plies()) {
                return fail("Move::none() must increment the number of plies");
            }
            return pass();
        }

        // A regular move must: place a piece at 'to', create a hole 
        // at 'from', increment ply, and leave all other squares 
        // unchanged
        TestResult check_regular_move_execution(const Yolah& before, 
                                                const Yolah& after, 
                                                Move m) {
            Square from = m.from_sq();
            Square to = m.to_sq();
            bool ok = true;
            for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
                if (sq == from || sq == to) continue;
                if (before.get(sq) != after.get(sq)) {
                    ok = false;
                    break;
                }
            }
            if (ok && before.nb_plies() + 1 == after.nb_plies() &&
                after.get(from) == HOLE && 
                after.get(to) == before.current_player()) {
                    return pass();
            }
            ostringstream oss;
            oss << "Move execution incorrect\n" << after 
                << "\n  Move: " << m << '\n';
            if (!ok) {
                oss << " Squares not concerned by the Move must not changed";
            }
            if (after.get(from) != HOLE) {
                oss << "  From square should be hole\n";
            }
            if (after.get(to) != before.current_player()) {
                oss << " To square should contain a piece from " 
                    << (before.current_player() == BLACK ? "black" : 
                                                           "white") 
                    << " player\n";
            }
            if (before.nb_plies() + 1 != after.nb_plies()) {
                oss << " The number of plies must be incremented\n";
            }
            return fail(oss.str());
        }
    }

    // Runs nb_games random games, testing move generation, play/undo, 
    // and game_over at each step
    void random_games(size_t nb_games, optional<uint64_t> seed) {
        MoveList fast_moves;
        random_device rd;
        mt19937 mt(seed.value_or(rd()));
        size_t total_tests = 0;
        size_t passed_tests = 0;

        auto run_test = [&](TestResult result) -> bool {
            total_tests++;
            if (result) {
                passed_tests++;
                return true;
            }
            cout << format("{}FAIL:{} {}\n", RED, RESET, 
                           result.message);
            return false;
        };
        
        cout << format("{}\n=== Running Random Games Tests ===\n{}", 
                       BOLD, RESET);

        for (size_t i = 0; i < nb_games; i++) {
            Yolah yolah;
            while (!yolah.game_over()) {
                yolah.moves(fast_moves);
                vector<Move> expected_moves = 
                    slow_moves_generation(yolah);
                if ( !run_test(
                         check_move_count(fast_moves, 
                                          expected_moves)) ) break;
                if ( !run_test(
                         check_move_lists_equal(fast_moves, 
                                                expected_moves, 
                                                yolah)) ) break;
                uniform_int_distribution<int> d(0,
                                                fast_moves.size()-1);
                Move m = fast_moves[d(mt)];
                Yolah before = yolah;
                yolah.play(m);
                if (m == Move::none()) {
                    if (!run_test(
                            check_none_move_execution(before, 
                                                      yolah))) break;
                } else {
                    if (!run_test(
                            check_regular_move_execution(before, 
                                                         yolah, 
                                                         m))) break;
                }
                yolah.undo(m);
                if (!run_test(check_undo(before, yolah))) break;
                yolah.play(m);
            }
            if (!yolah.game_over()) continue;
            yolah.moves(fast_moves);
            if (!run_test(
                    check_game_over_moves(yolah, 
                                          fast_moves))) continue;
            yolah.play(Move::none());
            yolah.moves(fast_moves);
            if (!run_test(
                    check_game_over_moves(yolah, 
                                          fast_moves))) continue;
        }

        cout << format("\n{}=== Test Summary ==={}\n", BOLD, RESET);
        cout << format("Total tests: {}\n", total_tests);
        cout << format("Passed: {}{}{}\n", GREEN, passed_tests, 
                       RESET);
        cout << format("Failed: {}{}{}\n", 
                        (passed_tests == total_tests ? GREEN : RED), 
                        total_tests - passed_tests, RESET);
        if (passed_tests == total_tests) {
            cout << format("{}All tests passed!{}\n", GREEN, RESET);
        } else {
            double pass_rate = 100.0 * passed_tests / total_tests;
            cout << format("{}Pass rate: {:.2f}%{}\n", YELLOW, 
                           pass_rate, RESET);
        }
    }
}

int main() {
    init_all_magics();    
    //play_random_games<true>(1000000, 42);
    play_random_games<false>(1000000, 42);
    //test::random_games(10000, 42);
    //play_random_games_fast(1000000, 42);
}
