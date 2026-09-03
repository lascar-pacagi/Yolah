#ifndef ALPHAZERO_MCTS_CHECK_H
#define ALPHAZERO_MCTS_CHECK_H
#include "game.h"
#include <cstdint>
#include <string>

namespace test {
    // Runs a fixed-budget search from the initial position (and from a few
    // random positions), checks the invariants of the tree statistics
    // (visits add up, π sums to 1, Q ∈ [-1, 1]) and prints the root stats.
    // `player_config` is the JSON of an AlphaZeroMCTSPlayer.
    bool alphazero_search_check(const json& player_config, uint64_t nb_simulations);
    // Checks that the subtree is reused across consecutive plays (ours, then a
    // random opponent reply): the second search must start with visits.
    bool alphazero_reuse_check(const json& player_config, uint64_t nb_simulations);
    // Plays `nb_games` games (alternating colours) against another player
    // config and prints the score. Returns the AlphaZero player's win rate.
    double alphazero_vs(const json& player_config, const json& opponent_config, size_t nb_games,
                        bool verbose = false);
}

#endif
