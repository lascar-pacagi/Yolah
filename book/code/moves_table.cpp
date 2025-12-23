#include <bits/stdc++.h>

namespace magic {
    void init();
}

struct Magic {
    uint64_t  mask;
    uint64_t  magic;
    uint64_t* moves;
    uint32_t  shift;

    uint32_t index(uint64_t occupied) const {
        return uint32_t(((occupied & mask) * magic) >> shift);
    }
};

namespace {
    uint64_t orthogonalTable[102400];
    uint64_t diagonalTable[5248];
}

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

Magic orthogonalMagics[SQUARE_NB];
Magic diagonalMagics[SQUARE_NB];

uint64_t moves_bb(Square sq, uint64_t occupied) {
    return orthogonalMagics[sq].moves[orthogonalMagics[sq].index(occupied)] | 
            diagonalMagics[sq].moves[diagonalMagics[sq].index(occupied)];
}

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

constexpr bool is_ok(Square s) { return SQ_A1 <= s && s <= SQ_H8; }

constexpr File file_of(Square s) { return File(s & 7); }

constexpr Rank rank_of(Square s) { return Rank(s >> 3); }

constexpr uint64_t rank_bb(Rank r) { return Rank1BB << (8 * r); }

constexpr uint64_t rank_bb(Square s) { return rank_bb(rank_of(s)); }

constexpr uint64_t file_bb(File f) { return FileABB << f; }

constexpr uint64_t file_bb(Square s) { return file_bb(file_of(s)); }

constexpr uint64_t square_bb(Square s) {
    return uint64_t(1) << s;
}

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }

int manhattan_distance(Square sq1, Square sq2) {
    int d_rank = std::abs(rank_of(sq1) - rank_of(sq2));
    int d_file = std::abs(file_of(sq1) - file_of(sq2));
    return d_rank + d_file;
}

enum MoveType {
    ORTHOGONAL,
    DIAGONAL,
};

uint64_t sliding_moves(MoveType mt, Square sq, uint64_t occupied) {
    uint64_t  moves                    = 0;
    Direction orthogonal_directions[4] = {NORTH, SOUTH, EAST, WEST};
    Direction diagonal_directions[4]   = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
    for (Direction d : (mt == ORTHOGONAL ? orthogonal_directions : diagonal_directions)) {
        Square s = sq;
        while (true) {            
            Square to = s + d;
            if (!is_ok(to) || manhattan_distance(s, to) > 2) break;
            uint64_t bb = square_bb(to);            
            if ((square_bb(to) & occupied) != 0) break;
            s = to;
            moves |= bb;
        } 
    }
    return moves;
}

Square& operator++(Square& d) { return d = Square(int(d) + 1); }

void init_magics(MoveType mt, uint64_t table[], Magic magics[]) {
    static constexpr uint64_t O_MAGIC[64] = {0x8000873020c000,0x200102102008040,0x100100844200100,0x49001c5000090020,0x80040018008002,0x4080040021020080,0x4b80010001800600,0x600040200289041,0x8104800468864000,0x4a0802004c00284,0x2013001020010040,0x1201005000210008,0x8d000410880100,0x84808022000400,0x6001001419008200,0x800080004100,0x1020808001604001,0x5010020400080,0x6200808010002005,0x1405030010006028,0x48110008010004,0x808006000400,0x1e18140030420801,0x400812000400864b,0x112400880208000,0x3290300400080,0x2100280200080,0x6050008080080250,0x800800c0080,0x140120080800400,0xa584400071002,0x1222440200007083,0x8000804008800020,0x8d30012000c00040,0x1020052820020c1,0x60a2101001000,0xc02d0051000800,0x2008522001008,0x2090012c000802,0x802a50800300,0x94000c480688000,0x440003008006004,0x410002408002002,0x100203001010018,0x68000c00808048,0x820110080a0024,0x891000200010004,0x2400006400860001,0x1014020800100,0x6054400100208100,0x210100081200080,0x4100081080180,0xa040180080180,0x2005108040200,0x280802300120080,0x1481000040820100,0x80201580004109,0x10a1004080220112,0x33001040ca6001,0x9201000050009,0x20a002820041002,0x1100080c000201,0x2002000801208402,0x200414c00208106,};
    static constexpr uint64_t D_MAGIC[64] = {0x68103022410020,0x120280481818201,0x142008301000848,0x6028060040000002,0x87104012080000,0x1044240012310,0x424010108a24060,0x9806412204400,0x144060c410022040,0x802023404009212,0x80044400820000,0x250041242000180,0x8810091040066040,0x4c0020511080000,0x8020640101882001,0x4820201042ec6,0x121256401a820400,0xa0000809041280,0x906200e042004200,0x2002022024080,0x24000201210004,0xa812020300c50400,0x80400288080804,0x804028a4020808,0x1c80a4040261804,0xa060120080200,0x8404114040080,0x104400c040080,0x89004064044001,0x8081890002010084,0x385c8400c6010402,0x60960189a2008080,0x70042ca142042000,0x18081242044411,0x1600608820100060,0x320020080080280,0x40084100001100,0x881110200011801,0x10088500120100,0x9040020308200,0x2c100210c20840,0xd91050044800,0x408201540204040c,0x2031044012020,0x4052884300401400,0x801020081000202,0x20022c82080520,0x600800b081800200,0x8602020220450082,0x2004a08840000,0x1000020042082000,0x240040284041080,0x1040024150410040,0x8c829a2008008001,0x820010a061420,0x6024a04050024,0x804202490c012060,0x100004202012020,0x80012001840c8800,0xa0008000608814,0x8110000300a0203,0x4000a08911010602,0x440109182018400,0x20200180910040,};
    using namespace std;
    int32_t size = 0;
    vector<uint64_t> occupancies;
    vector<uint64_t> possible_moves;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Magic& m = magics[sq];
        uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) | ((FileABB | FileHBB) & ~file_bb(sq));
        uint64_t moves_bb = sliding_moves(mt, sq, 0) & ~edges;
        m.mask = moves_bb;
        m.shift = 64 - popcount(m.mask);
        m.magic = (mt == ORTHOGONAL ? O_MAGIC : D_MAGIC)[sq];
        occupancies.clear();
        possible_moves.clear();
        m.moves = sq == SQ_A1 ? table : magics[sq - 1].moves + size;
        size = 0;
        uint64_t b = 0;
        do {
            occupancies.push_back(b);
            possible_moves.push_back(sliding_moves(mt, sq, b));
            b = (b - moves_bb) & moves_bb;
            size++;
        } while (b);
        for (int32_t j = 0; j < size; j++) {
            int32_t index = m.index(occupancies[j]);
            m.moves[index] = possible_moves[j];
        }
    }
}

namespace magic {
    void init() {
        init_magics(ORTHOGONAL, orthogonalTable, orthogonalMagics);
        init_magics(DIAGONAL, diagonalTable, diagonalMagics);
    }
}

int main() {
    magic::init();
}
