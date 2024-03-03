#include "magic.h"
#include <bitset>
#include <bit>
#include "misc.h"
#include <iostream>

Magic rookMagics[SQUARE_NB];
Magic bishopMagics[SQUARE_NB];

namespace {
    uint8_t popCnt16[1 << 16];
    uint8_t squareDistance[SQUARE_NB][SQUARE_NB];

    uint64_t rookTable[0x19000];
    uint64_t bishopTable[0x1480];
 
    enum PieceType {
        ROOK,
        BISHOP,
    };

    template<typename T = Square>
    int distance(Square x, Square y);

    template<>
    inline int distance<File>(Square x, Square y) {
        return std::abs(file_of(x) - file_of(y));
    }

    template<>
    inline int distance<Rank>(Square x, Square y) {
        return std::abs(rank_of(x) - rank_of(y));
    }

    template<>
    inline int distance<Square>(Square x, Square y) {
        return squareDistance[x][y];
    }

    uint64_t safe_destination(Square s, int step) {
        Square to = Square(s + step);
        return is_ok(to) && distance(s, to) <= 2 ? square_bb(to) : uint64_t(0);
    }
    
    uint64_t sliding_attack(PieceType pt, Square sq, uint64_t occupied) {
        uint64_t  attacks             = 0;
        Direction rookDirections[4]   = {NORTH, SOUTH, EAST, WEST};
        Direction bishopDirections[4] = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
        for (Direction d : (pt == ROOK ? rookDirections : bishopDirections)) {
            Square s = sq;            
            while (safe_destination(s, d) && !(occupied & s)) {
                attacks |= (s += d);
            }                
        }
        return attacks;
    }

    // Computes all rook and bishop attacks at startup. Magic
    // bitboards are used to look up attacks of sliding pieces. As a reference see
    // www.chessprogramming.org/Magic_Bitboards. In particular, here we use the so
    // called "fancy" approach.
    void init_magics(PieceType pt, uint64_t table[], Magic magics[]) {

        // Optimal PRNG seeds to pick the correct magics in the shortest time
        int32_t seeds[RANK_NB] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};

        uint64_t occupancy[4096], reference[4096], edges, b;
        int32_t      epoch[4096] = {}, cnt = 0, size = 0;

        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            // Board edges are not considered in the relevant occupancies
            edges = ((Rank1BB | Rank8BB) & ~rank_bb(s)) | ((FileABB | FileHBB) & ~file_bb(s));

            // Given a square 's', the mask is the bitboard of sliding attacks from
            // 's' computed on an empty board. The index must be big enough to contain
            // all the attacks for each possible subset of the mask and so is 2 power
            // the number of 1s of the mask. Hence we deduce the size of the shift to
            // apply to the 64 bits word to get the index.
            Magic& m = magics[s];
            m.mask   = sliding_attack(pt, s, 0) & ~edges;
            m.shift  = 64 - std::popcount(m.mask);

            // Set the offset for the attacks table of the square. We have individual
            // table sizes for each square with "Fancy Magic Bitboards".
            m.attacks = s == SQ_A1 ? table : magics[s - 1].attacks + size;

            // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
            // store the corresponding sliding attack bitboard in reference[].
            b = size = 0;
            do {
                occupancy[size] = b;
                reference[size] = sliding_attack(pt, s, b);
                size++;
                b = (b - m.mask) & m.mask;
            } while (b);

            PRNG rng(seeds[rank_of(s)]);

            // Find a magic for square 's' picking up an (almost) random number
            // until we find the one that passes the verification test.
            for (int i = 0; i < size;) {
                for (m.magic = 0; std::popcount((m.magic * m.mask) >> 56) < 6;) {
                    m.magic = rng.sparse_rand<uint64_t>();                    
                }
                // A good magic must map every possible occupancy to an index that
                // looks up the correct sliding attack in the attacks[s] database.
                // Note that we build up the database for square 's' as a side
                // effect of verifying the magic. Keep track of the attempt count
                // and save it in epoch[], little speed-up trick to avoid resetting
                // m.attacks[] after every failed attempt.
                for (++cnt, i = 0; i < size; ++i) {
                    unsigned idx = m.index(occupancy[i]);                    
                    if (epoch[idx] < cnt) {
                        epoch[idx]     = cnt;
                        m.attacks[idx] = reference[i];
                    }
                    else if (m.attacks[idx] != reference[i]) {
                        break;
                    }                        
                }
            }
        }
    }
}

namespace magic {
    void init() {
        for (uint32_t i = 0; i < (1 << 16); ++i) {
            popCnt16[i] = uint8_t(std::bitset<16>(i).count());
        }
            
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
                squareDistance[s1][s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));
            }
        }    
        init_magics(ROOK, rookTable, rookMagics);
        init_magics(BISHOP, bishopTable, bishopMagics);
    }
}
