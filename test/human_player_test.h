#ifndef HUMAN_PLAYER_TEST_H
#define HUMAN_PLAYER_TEST_H
#include "player.h"

namespace test {
    class HumanPlayer : public Player {
        Move play(Yolah yolah) override;
    };
}

#endif
