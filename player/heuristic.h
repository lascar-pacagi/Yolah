#ifndef HEURISTIC_H
#define HEURISTIC_H
#include "game.h"

namespace heuristic {
    int32_t eval(uint8_t player, const Yolah&);
    void learn_weights();
}

#endif