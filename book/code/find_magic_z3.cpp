#include <z3++.h>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>

int main() {
    // Create positions map: bitboard -> position
    std::map<uint64_t, int> positions;
    for (int i = 0; i < 64; i++) {
        positions[1ULL << i] = i;
    }

    // Create Z3 context and solver
    z3::context ctx;
    z3::solver solver(ctx);

    // Create 64-bit bitvector variable for magic number
    z3::expr MAGIC = ctx.bv_const("magic", 64);

    // Set K value
    const int K = 6;

    // Get all bitboards
    std::vector<uint64_t> bitboards;
    for (const auto& pair : positions) {
        bitboards.push_back(pair.first);
    }

    // Lambda function to compute index
    auto index = [&ctx](z3::expr magic, int k, uint64_t bitboard) -> z3::expr {
        z3::expr bb = ctx.bv_val(bitboard, 64);
        z3::expr shift_amount = ctx.bv_val(64 - k, 64);
        return z3::lshr(magic * bb, shift_amount);
    };

    // Add constraints: all indices must be different
    for (size_t i = 0; i < 64; i++) {
        z3::expr index1 = index(MAGIC, K, bitboards[i]);
        for (size_t j = i + 1; j < 64; j++) {
            z3::expr index2 = index(MAGIC, K, bitboards[j]);
            solver.add(index1 != index2);
        }
    }

    // Check satisfiability
    if (solver.check() == z3::sat) {
        z3::model model = solver.get_model();

        // Get magic number from model
        z3::expr magic_val = model.eval(MAGIC);
        uint64_t m = magic_val.get_numeral_uint64();

        std::cout << "found magic for K = " << K << ": 0x"
                  << std::hex << m << std::dec << std::endl;

        // Build lookup table
        int size = 1 << K;
        std::vector<int> table(size, -1);

        auto compute_index = [K](uint64_t magic, uint64_t bitboard) -> int {
            return (magic * bitboard) >> (64 - K);
        };

        for (const auto& pair : positions) {
            uint64_t bitboard = pair.first;
            int pos = pair.second;
            int idx = compute_index(m, bitboard) & (size - 1);
            table[idx] = pos;
        }

        // Print lookup table
        std::cout << "constexpr uint8_t bitscan[" << size << "] = {" << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << table[i] << ",";
        }
        std::cout << std::endl << "};" << std::endl;
    } else {
        std::cout << "No solution found!" << std::endl;
    }

    return 0;
}
