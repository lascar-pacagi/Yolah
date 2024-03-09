#ifndef HUMAN_PLAYER_H
#define HUMAN_PLAYER_H
#include "player.h"

namespace test {
    class HumanPlayer : public Player {
        Move play(Yolah yolah) override;
    };
}

#endif
