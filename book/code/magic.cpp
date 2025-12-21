#include <bits/stdc++.h>
#include <random>
#include <format>

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

constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

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
    HORIZONTAL,
    DIAGONAL,
};

uint64_t sliding_moves(MoveType mt, Square sq, uint64_t occupied) {
    uint64_t  moves                    = 0;
    Direction horizontal_directions[4] = {NORTH, SOUTH, EAST, WEST};
    Direction diagonal_directions[4]   = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
    for (Direction d : (mt == HORIZONTAL ? horizontal_directions : diagonal_directions)) {
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

std::pair<int, uint64_t> magic_for_square(MoveType mt, Square sq) {    
    using namespace std;
    uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) | ((FileABB | FileHBB) & ~file_bb(sq));
    uint64_t moves_bb = sliding_moves(mt, sq, 0) & ~edges;
    vector<uint64_t> occupancies;
    vector<uint64_t> possible_moves;
    uint64_t b = 0;
    int size = 0;
    do {
        occupancies.push_back(b);
        possible_moves.push_back(sliding_moves(mt, sq, b));
        size++;
        b = (b - moves_bb) & moves_bb;
    } while (b);
    int k = popcount(moves_bb);
    int shift = 64 - k;                
    random_device rd;
    mt19937_64 twister(rd());
    uniform_int_distribution<uint64_t> d;
    vector<int> seen(1 << k);
    vector<uint64_t> moves(1 << k);                    
    for (int cnt = 0;; cnt++) {
        uint64_t magic = d(twister) & d(twister) & d(twister);
        bool found = true;                    
        for (size_t j = 0; j < occupancies.size(); j++) {
            uint64_t occ = occupancies[j];
            int index = magic * occ >> shift;
            if (seen[index] == cnt && moves[index] != possible_moves[j]) {
                found = false;
                break;
            }
            seen[index] = cnt;
            moves[index] = possible_moves[j];
        }
        if (found) {
            return {k, magic};
        }
    }
    unreachable();
}

int main() {
    using namespace std;
    for (MoveType mt : {HORIZONTAL, DIAGONAL}) {
        stringstream ss_k, ss_magic;
        ss_k << format("int {}_K[64] = {{", mt == HORIZONTAL ? "H" : "D");
        ss_magic << format("uint64_t {}_MAGIC[64] = {{", mt == HORIZONTAL ? "H" : "D");
        for (int sq = SQ_A1; sq <= SQ_H8; sq++) {
            const auto [k, magic] = magic_for_square(mt, Square(sq));
            ss_k << dec << k << ',';
            ss_magic << showbase << hex << magic << ',';
        }
        cout << ss_k.str() << "};\n";
        cout << ss_magic.str() << "};\n\n";
    }
}
