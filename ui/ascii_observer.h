#ifndef ASCII_OBSERVER_H
#define ASCII_OBSERVER_H
#include "game.h"

struct AsciiObserver {
    void operator()(Yolah) const;
    void operator()(uint8_t player, Move) const;
};

#endif