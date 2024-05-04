#ifndef DO_NOTHING_OBSERVER_H
#define DO_NOTHING_OBSERVER_H
#include "game.h"

struct DoNothingObserver {
    void operator()(Yolah) const;
    void operator()(uint8_t player, Move) const;
};

#endif