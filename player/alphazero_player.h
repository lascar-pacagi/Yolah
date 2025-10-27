#ifndef ALPHAZERO_PLAYER_H
#define ALPHAZERO_PLAYER_H

#include "player.h"
#include "../alphazero/alphazero_network.h"
#include "../alphazero/alphazero_mcts.h"
#include <memory>

/**
 * AlphaZero player for Yolah game
 *
 * This player uses Monte Carlo Tree Search (MCTS) guided by a deep neural network
 * to select moves. The neural network is trained through self-play using reinforcement
 * learning.
 *
 * Configuration format:
 * {
 *   "type": "alphazero",
 *   "network_weights": "path/to/weights.bin",
 *   "num_simulations": 800,
 *   "num_parallel_games": 8,
 *   "c_puct": 1.5,
 *   "temperature": 0.0,
 *   "use_gpu": true
 * }
 */
class AlphaZeroPlayer : public Player {
public:
    explicit AlphaZeroPlayer(const json& config);
    ~AlphaZeroPlayer() override;

    Move play(Yolah state) override;
    std::string info() override;
    void game_over(Yolah state) override;
    json config() override;

private:
    std::shared_ptr<alphazero::AlphaZeroNetwork> network_;
    std::unique_ptr<alphazero::AlphaZeroMCTS> mcts_;
    alphazero::AlphaZeroMCTS::Config mcts_config_;

    std::string weights_path_;
    bool use_gpu_;

    json config_;
};

#endif // ALPHAZERO_PLAYER_H
