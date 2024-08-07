#ifndef RANDOM_PLAYER_H
#define RANDOM_PLAYER_H
#include "player.h"
#include "misc.h"

class RandomPlayer : public Player {
    PRNG prng;
    bool init_with_clock = true;
public:
    explicit RandomPlayer();
    explicit RandomPlayer(uint64_t seed);
    Move play(Yolah yolah) override;
    std::string info() override;
    json config() override;
};

#endif
