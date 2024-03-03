#ifndef MAGIC_H
#define MAGIC_H
#include "types.h"

namespace magic {
    void init();
}

struct Magic {
    uint64_t  mask;
    uint64_t  magic;
    uint64_t* attacks;
    uint32_t  shift;

    unsigned index(uint64_t occupied) const {
        return unsigned(((occupied & mask) * magic) >> shift);
    }
};

extern Magic rookMagics[SQUARE_NB];
extern Magic bishopMagics[SQUARE_NB];

inline uint64_t attacks_bb(Square s, uint64_t occupied) {
    return rookMagics[s].attacks[rookMagics[s].index(occupied)] | 
            bishopMagics[s].attacks[bishopMagics[s].index(occupied)];
}

#endif
