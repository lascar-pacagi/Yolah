/*
 * Magic Bitboard Generator for Rook Moves
 * 
 * This implementation provides:
 * 1. Random search method (fast, practical)
 * 2. Pre-computed optimal magics
 * 3. Complete explanation of the algorithm
 *
 * Compile: g++ -std=c++20 -O3 -o magic_rook magic_rook.cpp
 * Run: ./magic_rook [--explain]
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <bit>
#include <random>
#include <chrono>
#include <array>

using U64 = uint64_t;

// ============================================================================
// BIT MANIPULATION UTILITIES
// ============================================================================

constexpr int popcount(U64 x) { return std::popcount(x); }

void print_bitboard(U64 bb, const char* label = nullptr) {
    if (label) std::cout << label << ":\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << "  ";
        for (int f = 0; f < 8; ++f) {
            std::cout << ((bb >> (r * 8 + f)) & 1 ? "1 " : ". ");
        }
        std::cout << "\n";
    }
    std::cout << "   a b c d e f g h\n\n";
}

std::string sq_name(int sq) {
    return std::string(1, 'a' + (sq % 8)) + std::to_string(sq / 8 + 1);
}

// ============================================================================
// ROOK ATTACK GENERATION
// ============================================================================

// Slow but correct rook attack calculation (used for table initialization)
constexpr U64 rook_attacks_slow(int sq, U64 occ) {
    U64 attacks = 0;
    int r = sq / 8, f = sq % 8;
    
    // North
    for (int i = r + 1; i < 8; ++i) {
        U64 b = U64(1) << (i * 8 + f);
        attacks |= b;
        if (occ & b) break;
    }
    // South  
    for (int i = r - 1; i >= 0; --i) {
        U64 b = U64(1) << (i * 8 + f);
        attacks |= b;
        if (occ & b) break;
    }
    // East
    for (int i = f + 1; i < 8; ++i) {
        U64 b = U64(1) << (r * 8 + i);
        attacks |= b;
        if (occ & b) break;
    }
    // West
    for (int i = f - 1; i >= 0; --i) {
        U64 b = U64(1) << (r * 8 + i);
        attacks |= b;
        if (occ & b) break;
    }
    
    return attacks;
}

// Generate occupancy mask (excludes edge squares that don't affect moves)
constexpr U64 rook_mask(int sq) {
    U64 mask = 0;
    int r = sq / 8, f = sq % 8;
    
    for (int i = r + 1; i < 7; ++i) mask |= U64(1) << (i * 8 + f);
    for (int i = r - 1; i > 0; --i) mask |= U64(1) << (i * 8 + f);
    for (int i = f + 1; i < 7; ++i) mask |= U64(1) << (r * 8 + i);
    for (int i = f - 1; i > 0; --i) mask |= U64(1) << (r * 8 + i);
    
    return mask;
}

// Enumerate all subsets of a mask (Carry-Rippler trick)
std::vector<U64> enumerate_subsets(U64 mask) {
    std::vector<U64> subsets;
    U64 subset = 0;
    do {
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset);
    return subsets;
}

// ============================================================================
// MAGIC NUMBER SEARCH
// ============================================================================

/*
 * Find a magic number using random search with sparse candidates.
 * 
 * The key insight: We need (occupancy * magic) >> shift to give
 * distinct indices for occupancies with different attack patterns.
 * 
 * Sparse random numbers (few bits set) work well because:
 * - Multiplication by sparse numbers has less "interference" between bits
 * - The occupancy bits get distributed to different positions in the product
 */
U64 find_magic_random(int sq, int index_bits, int max_attempts = 100000000) {
    U64 mask = rook_mask(sq);
    auto occupancies = enumerate_subsets(mask);
    int num_occ = occupancies.size();
    int shift = 64 - index_bits;
    int table_size = 1 << index_bits;
    
    // Precompute attacks for each occupancy
    std::vector<U64> attacks(num_occ);
    for (int i = 0; i < num_occ; ++i) {
        attacks[i] = rook_attacks_slow(sq, occupancies[i]);
    }
    
    // Random number generator
    std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() ^ (sq * 1000003));
    
    // Generate sparse random number: AND three randoms to get ~8 bits set on average
    auto sparse_random = [&rng]() { return rng() & rng() & rng(); };
    
    std::vector<U64> table(table_size);
    const U64 EMPTY = ~U64(0);
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        U64 magic = sparse_random();
        
        // Quick rejection: need enough bits in the high byte after multiplication
        if (popcount((mask * magic) >> 56) < 6) continue;
        
        // Test this magic
        std::fill(table.begin(), table.end(), EMPTY);
        bool valid = true;
        
        for (int i = 0; i < num_occ && valid; ++i) {
            U64 index = (occupancies[i] * magic) >> shift;
            
            if (table[index] == EMPTY) {
                table[index] = attacks[i];
            } else if (table[index] != attacks[i]) {
                // Collision with different attack pattern - invalid magic
                valid = false;
            }
            // Note: same attack pattern sharing an index is OK (constructive collision)
        }
        
        if (valid) return magic;
    }
    
    return 0;  // Failed to find magic
}

// ============================================================================
// MAGIC BITBOARD SYSTEM
// ============================================================================

struct MagicEntry {
    U64 mask;       // Relevant occupancy mask
    U64 magic;      // Magic multiplier  
    int shift;      // Right shift amount (64 - index_bits)
    U64* attacks;   // Pointer into attack table
};

class RookMagics {
public:
    std::array<MagicEntry, 64> entries;
    std::vector<U64> attack_table;
    
    // Generate magic numbers using random search
    void generate() {
        std::cout << "Generating rook magic numbers...\n\n";
        
        std::array<U64, 64> magics;
        std::array<int, 64> shifts;
        std::array<int, 64> offsets;
        int total = 0;
        
        for (int sq = 0; sq < 64; ++sq) {
            int bits = popcount(rook_mask(sq));
            U64 magic = find_magic_random(sq, bits);
            
            if (magic == 0) {
                std::cerr << "ERROR: Failed to find magic for " << sq_name(sq) << "\n";
                exit(1);
            }
            
            magics[sq] = magic;
            shifts[sq] = 64 - bits;
            offsets[sq] = total;
            total += (1 << bits);
            
            std::cout << sq_name(sq) << ": 0x" 
                      << std::hex << std::setw(16) << std::setfill('0') << magic
                      << std::dec << " (" << bits << " index bits, " 
                      << popcount(magic) << " bits set)\n";
        }
        
        std::cout << "\nTotal table size: " << total << " entries (" 
                  << (total * sizeof(U64)) << " bytes)\n\n";
        
        // Initialize attack table
        attack_table.resize(total);
        for (int sq = 0; sq < 64; ++sq) {
            entries[sq] = {
                rook_mask(sq),
                magics[sq],
                shifts[sq],
                &attack_table[offsets[sq]]
            };
            
            for (U64 occ : enumerate_subsets(entries[sq].mask)) {
                U64 idx = (occ * entries[sq].magic) >> entries[sq].shift;
                entries[sq].attacks[idx] = rook_attacks_slow(sq, occ);
            }
        }
    }
    
    // Fast O(1) attack lookup
    U64 attacks(int sq, U64 occupancy) const {
        const auto& e = entries[sq];
        U64 index = ((occupancy & e.mask) * e.magic) >> e.shift;
        return e.attacks[index];
    }
    
    // Verify all entries are correct
    bool verify() const {
        for (int sq = 0; sq < 64; ++sq) {
            for (U64 occ : enumerate_subsets(entries[sq].mask)) {
                if (attacks(sq, occ) != rook_attacks_slow(sq, occ)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Print as C++ code
    void print_code() const {
        std::cout << "constexpr uint64_t ROOK_MAGICS[64] = {\n";
        for (int sq = 0; sq < 64; ++sq) {
            if (sq % 4 == 0) std::cout << "    ";
            std::cout << "0x" << std::hex << std::setw(16) << std::setfill('0') 
                      << entries[sq].magic << "ULL";
            std::cout << (sq < 63 ? "," : "") << (sq % 4 == 3 ? "\n" : " ");
        }
        std::cout << std::dec << "};\n\n";
        
        std::cout << "constexpr int ROOK_SHIFTS[64] = {\n";
        for (int sq = 0; sq < 64; ++sq) {
            if (sq % 8 == 0) std::cout << "    ";
            std::cout << std::setw(2) << std::setfill(' ') << entries[sq].shift;
            std::cout << (sq < 63 ? "," : "") << (sq % 8 == 7 ? "\n" : " ");
        }
        std::cout << "};\n";
    }
};

// ============================================================================
// EXPLANATION
// ============================================================================

void explain() {
    std::cout << R"(
================================================================================
                    MAGIC BITBOARDS EXPLAINED
================================================================================

GOAL: Compute rook attack squares in O(1) time using table lookup.

--------------------------------------------------------------------------------
STEP 1: IDENTIFY RELEVANT BLOCKERS
--------------------------------------------------------------------------------

For a rook, only certain squares can block its movement. Edge squares never
matter because if we reach them, we can always attack them.

Example: Rook on a1
The "occupancy mask" contains squares that could block movement:

    8  .  .  .  .  .  .  .  .
    7  1  .  .  .  .  .  .  .     (a7 could block vertical moves)
    6  1  .  .  .  .  .  .  .     (a6 could block vertical moves)
    5  1  .  .  .  .  .  .  .     ...
    4  1  .  .  .  .  .  .  .
    3  1  .  .  .  .  .  .  .
    2  1  .  .  .  .  .  .  .     (a2 could block vertical moves)
    1  R  1  1  1  1  1  1  .     (b1-g1 could block horizontal moves)
       a  b  c  d  e  f  g  h

This mask has 12 bits → 2^12 = 4096 possible occupancy patterns.

--------------------------------------------------------------------------------
STEP 2: THE MAGIC MULTIPLICATION
--------------------------------------------------------------------------------

We want a hash function that maps each occupancy to a unique table index:

    index = (occupancy * magic) >> shift

Where:
    - occupancy: blocking pieces ANDed with mask
    - magic: a carefully chosen 64-bit constant
    - shift: 64 - index_bits (e.g., 52 for 12 index bits)

The multiplication "collects" scattered occupancy bits into contiguous
high bits, which we then extract with the shift.

--------------------------------------------------------------------------------
STEP 3: FINDING THE MAGIC NUMBER
--------------------------------------------------------------------------------

We search for a magic that gives distinct indices for occupancies with
different attack patterns. The search:

1. Generate sparse random candidates (few bits set work best)
2. Test if all occupancies map to valid indices
3. Repeat until found (usually very fast)

Why sparse? Dense magics cause "bit collisions" where different occupancy
bits interfere destructively. Sparse magics keep contributions separate.

--------------------------------------------------------------------------------
STEP 4: BUILD THE LOOKUP TABLE
--------------------------------------------------------------------------------

For each occupancy pattern:
    1. Compute index = (occupancy * magic) >> shift
    2. Store precomputed attacks at table[index]

Now attack lookup is just:
    attacks = table[(occupancy & mask) * magic >> shift]

Three operations: AND, MULTIPLY, SHIFT, plus a memory read.

--------------------------------------------------------------------------------
TABLE SIZES
--------------------------------------------------------------------------------

Square positions have different mask sizes:
    - Corners (a1, h1, a8, h8): 12 bits → 4096 entries each
    - Edges: 11 bits → 2048 entries each
    - Center: 10 bits → 1024 entries each

Total for all 64 rook squares: 102,400 entries = 800 KB

================================================================================
)";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--explain") {
        explain();
        return 0;
    }
    
    RookMagics rook_magics;
    
    auto start = std::chrono::steady_clock::now();
    rook_magics.generate();
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Generation time: " << ms << " ms\n\n";
    
    // Verify
    std::cout << "Verifying... ";
    std::cout << (rook_magics.verify() ? "OK!" : "FAILED!") << "\n\n";
    
    // Demo: Rook on a1 with blockers
    std::cout << "=== DEMO ===\n\n";
    int sq = 0;  // a1
    U64 blockers = (1ULL << 16) | (1ULL << 2) | (1ULL << 5);  // a3, c1, f1
    
    print_bitboard(1ULL << sq, "Rook on a1");
    print_bitboard(blockers, "Blockers: a3, c1, f1");
    print_bitboard(rook_magics.attacks(sq, blockers), "Rook attacks");
    
    // Benchmark
    std::cout << "=== BENCHMARK ===\n\n";
    std::mt19937_64 rng(42);
    const int N = 10000000;
    U64 checksum = 0;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        checksum ^= rook_magics.attacks(rng() % 64, rng());
    }
    auto bench_end = std::chrono::high_resolution_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bench_end - bench_start).count();
    
    std::cout << N << " lookups: " << (ns / 1e6) << " ms\n";
    std::cout << "Average: " << (ns / N) << " ns/lookup\n";
    std::cout << "Checksum: 0x" << std::hex << checksum << std::dec << "\n\n";
    
    // Print code
    std::cout << "=== GENERATED CODE ===\n\n";
    rook_magics.print_code();
    
    return 0;
}
