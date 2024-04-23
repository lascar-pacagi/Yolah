#ifndef OBSERVER_H
#define OBSERVER_H
#include "game.h"

template<typename T>
concept Observer = requires(T display, Yolah yolah, uint8_t player, Move m)
{
    display(yolah);
    display(player, m);
};

#endif