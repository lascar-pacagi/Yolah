#ifndef ZOBRIST_H
#define ZOBRIST_H
#include "types.h"
#include "game.h"

namespace zobrist {
    extern uint64_t psq[4][SQUARE_NB];
    extern uint64_t side;
    void init();
    uint64_t hash(const Yolah& yolah);
    uint64_t update(uint64_t hash, uint8_t player, Move m);
}

#endif