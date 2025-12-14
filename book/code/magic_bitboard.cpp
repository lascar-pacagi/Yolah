#include <bits/stdc++.h>
#include <z3++.h>
#include "../../game/types.h"

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

void magic_for_square(Square sq) {    
    using namespace std;
    uint64_t moves_bb = sliding_moves(HORIZONTAL, sq, 0);
    std::vector<uint64_t> occupancies;
    std::vector<uint64_t> possible_moves;
    uint64_t b = 0;
    int size = 0;
    do {
        occupancies.push_back(b);
        possible_moves.push_back(sliding_moves(HORIZONTAL, sq, b));
        size++;
        b = (b - moves_bb) & moves_bb;
    } while (b);
    for (int i = 0; i < size; i++) {
        std::cout << Bitboard::pretty(occupancies[i]) << '\n';
        std::cout << Bitboard::pretty(possible_moves[i]) << "\n\n\n";
    }
    std::cout << size << '\n';
    auto index = [](z3::context& ctx, z3::expr magic, int k, uint64_t bitboard) -> z3::expr {
        z3::expr bb = ctx.bv_val(bitboard, 64);
        z3::expr shift = ctx.bv_val(64 - k, 64);
        return z3::lshr(magic * bb, shift);
    };
    for (int K = 6;; K++) {
        cout << format("K = {}\n", K);
        z3::context ctx;
        z3::solver solver(ctx);
        z3::expr MAGIC = ctx.bv_const("magic", 64);
        for (int i = 0; i < size; i++) {
            cout << i << '\n';
            for (int j = i + 1; j < size; j++) {
                if (possible_moves[i] != possible_moves[j]) {
                    z3::expr index1 = index(ctx, MAGIC, K, occupancies[i]);
                    z3::expr index2 = index(ctx, MAGIC, K, occupancies[j]);
                    solver.add(index1 != index2);
                }
            }
        }
        cout << "Done adding constraints" << endl;
        if (solver.check() == z3::sat) {
            z3::model model = solver.get_model();
            z3::expr magic_val = model.eval(MAGIC);
            uint64_t m = magic_val.get_numeral_uint64();
            cout << format("found magic for K = {}: {:#x}", K, m);
        }
    }
}

int main() {
    magic_for_square(SQ_A1);
}
