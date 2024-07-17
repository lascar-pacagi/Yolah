#include "zobrist.h"
#include "misc.h"

namespace zobrist {
    uint64_t psq[4][SQUARE_NB];
    uint64_t side;
    
    void init() {
        PRNG rng(1070372);
        for (auto content : { Yolah::BLACK, Yolah::WHITE, Yolah::EMPTY, Yolah::FREE }) {
            for (Square s = SQ_A1; s <= SQ_H8; ++s) {
                psq[content][s] = rng.rand<uint64_t>();
            }
        }
        side = rng.rand<uint64_t>();
    }

    uint64_t hash(const Yolah& yolah) {
        uint64_t res = yolah.current_player() == Yolah::BLACK ? 0 : side;
        for (Square s = SQ_A1; s <= SQ_H8; ++s) {
            res ^= psq[yolah.get(s)][s]; 
        }
        return res;
    }

    uint64_t update(uint64_t hash, uint8_t player, Move m) {
        if (m == Move::none()) [[unlikely]] {
            return hash ^ side;
        }
        return hash 
                ^ psq[player][m.from_sq()] 
                ^ psq[player][m.to_sq()] 
                ^ psq[Yolah::EMPTY][m.from_sq()]
                ^ psq[Yolah::FREE][m.to_sq()]
                ^ side;
    }
}
