#ifndef MONTE_CARLO_PLAYER_H
#define MONTE_CARLO_PLAYER_H
#include "player.h"
#include "misc.h"
#include "BS_thread_pool.h"

class MonteCarloPlayer : public Player {
    const uint64_t thinking_time;
    BS::thread_pool pool;

    int16_t random_game(Yolah& yolah, uint8_t player);
public:
    explicit MonteCarloPlayer(uint64_t microseconds);
    explicit MonteCarloPlayer(uint64_t microseconds, std::size_t nb_threads);
    Move play(Yolah yolah) override;
    std::string info() override;
    json config() override;
};

#endif
