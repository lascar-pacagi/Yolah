#include <bits/stdc++.h>
#include <z3++.h>
#include "../../game/types.h"
#include <thread>
#include <random>
#include <atomic>
#include <format>

enum MoveType {
    HORIZONTAL,
    DIAGONAL,
};

int distance(Square sq1, Square sq2) {
    int d_rank = std::abs(rank_of(sq1) - rank_of(sq2));
    int d_file = std::abs(file_of(sq1) - file_of(sq2));
    return std::max(d_rank, d_file);
}

uint64_t sliding_moves(MoveType mt, Square sq, uint64_t occupied) {
    uint64_t  moves                    = 0;
    Direction horizontal_directions[4] = {NORTH, SOUTH, EAST, WEST};
    Direction diagonal_directions[4]   = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
    for (Direction d : (mt == HORIZONTAL ? horizontal_directions : diagonal_directions)) {
        Square s = sq;
        while (true) {            
            Square to = s + d;
            if (!is_ok(to) || distance(s, to) > 1) break;
            uint64_t bb = square_bb(to);            
            if ((square_bb(to) & occupied) != 0) break;
            s = to;
            moves |= bb;
        } 
    }
    return moves;
}

// std::pair<int, uint64_t> magic_for_square(MoveType mt, Square sq, size_t nb_iterations) {    
//     using namespace std;
//     uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) | ((FileABB | FileHBB) & ~file_bb(sq));
//     uint64_t moves_bb = sliding_moves(mt, sq, 0) & ~edges;
//     std::vector<uint64_t> occupancies;
//     std::vector<uint64_t> possible_moves;
//     uint64_t b = 0;
//     int size = 0;
//     set<uint64_t> moves;
//     do {
//         occupancies.push_back(b);
//         possible_moves.push_back(sliding_moves(HORIZONTAL, sq, b));
//         moves.insert(possible_moves.back());
//         size++;
//         b = (b - moves_bb) & moves_bb;
//     } while (b);
//     // for (int i = 0; i < size; i++) {
//     //     std::cout << Bitboard::pretty(occupancies[i]) << '\n';
//     //     std::cout << Bitboard::pretty(possible_moves[i]) << "\n\n\n";
//     //     string _; getline(cin, _);
//     // }
//     cout << size << '\n';
//     for (int k = bit_width(moves.size());; k++) {
//         cout << k << endl;
//         int shift = 64 - k;
//         vector<jthread> workers;
//         atomic<bool> found = false;
//         atomic<uint64_t> MAGIC = 0;
//         for (size_t i = 0; i < thread::hardware_concurrency(); i++) {
//             workers.emplace_back([&] {
//                 random_device rd;
//                 mt19937_64 mt(rd());
//                 uniform_int_distribution<uint64_t> d;
//                 vector<int> seen(16384);
//                 vector<uint64_t> moves(16384);                    
//                 int cnt = 0;    
//                 for (size_t i = 0; i < nb_iterations; i++, cnt++) {
//                     if (found) break;
//                     uint64_t magic;
//                     for (magic = 0; popcount((magic * moves_bb) >> 56) < 6;) {
//                         magic = d(mt) & d(mt) & d(mt);
//                     }
//                     bool ok = true;                    
//                     for (size_t j = 0; j < occupancies.size(); j++) {
//                         uint64_t occ = occupancies[j];
//                         int index = magic * occ >> shift;
//                         if (seen[index] == cnt && moves[index] != possible_moves[j]) {
//                             ok = false;
//                             break;
//                         }
//                         seen[index] = cnt;
//                         moves[index] = possible_moves[j];
//                     }
//                     if (ok) {
//                         found = true;
//                         MAGIC = magic;
//                     }
//                 }
//             });
//         }
//         if (found) {
//             return {k, MAGIC};
//         }
//     }
//     unreachable();
// }

std::pair<int, uint64_t> magic_for_square(MoveType mt, Square sq, size_t nb_iterations) {    
    using namespace std;
    uint64_t edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq)) | ((FileABB | FileHBB) & ~file_bb(sq));
    uint64_t moves_bb = sliding_moves(mt, sq, 0) & ~edges;
    std::vector<uint64_t> occupancies;
    std::vector<uint64_t> possible_moves;
    uint64_t b = 0;
    int size = 0;
    set<uint64_t> moves;
    do {
        occupancies.push_back(b);
        possible_moves.push_back(sliding_moves(HORIZONTAL, sq, b));
        moves.insert(possible_moves.back());
        size++;
        b = (b - moves_bb) & moves_bb;
    } while (b);
    // for (int i = 0; i < size; i++) {
    //     std::cout << Bitboard::pretty(occupancies[i]) << '\n';
    //     std::cout << Bitboard::pretty(possible_moves[i]) << "\n\n\n";
    //     string _; getline(cin, _);
    // }
    int k = popcount(moves_bb);
    int shift = 64 - k;            
    {
        random_device rd;
        mt19937_64 mt(rd());
        uniform_int_distribution<uint64_t> d;
        vector<int> seen(1 << k);
        vector<uint64_t> moves(1 << k);                    
        int cnt = 0;    
        for (size_t i = 0; i < nb_iterations; i++, cnt++) {
            uint64_t magic = d(mt) & d(mt) & d(mt);
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
    }
    unreachable();
}

int main() {
    using namespace std;
    for (MoveType mt : {HORIZONTAL, DIAGONAL}) {
        for (int sq = SQ_A1; sq <= SQ_H8; sq++) {
            const auto [k, magic] = magic_for_square(mt, Square(sq), 100000000);
            cout << format("MoveType: {} Square: {} K: {} Magic: {}\n", int(mt), int(sq), k, magic);
        }        
    }    
}
