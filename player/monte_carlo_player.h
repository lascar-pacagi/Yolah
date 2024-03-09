#ifndef MONTE_CARLO_PLAYER_H
#define MONTE_CARLO_PLAYER_H
#include "player.h"
#include "misc.h"
#include "BS_thread_pool.h"

class MonteCarloPlayer : public Player {
    const std::size_t nb_iter;
    BS::thread_pool pool;

    uint64_t random_game(Yolah& yolah, uint8_t player);
public:
    MonteCarloPlayer(std::size_t nb_iter);
    MonteCarloPlayer(std::size_t nb_iter, std::size_t nb_threads);
    Move play(Yolah yolah) override;
};

#endif
