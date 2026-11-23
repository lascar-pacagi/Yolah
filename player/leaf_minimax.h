#ifndef LEAF_MINIMAX_H
#define LEAF_MINIMAX_H
// leaf_minimax.h — the minimax NNUE (MinMaxNNUE_DevPlayer) as the value of the
// MCTS ResNet's leaves, late in the game (campaign H2, nnue/leafmm_campaign.sh).
//
// The idea (user, 2026-11-23): campaign H1 switched from the MCTS ResNet to the
// minimax for the whole move; here the MCTS keeps the move choice and the
// ResNet's policy (the priors of the expanded node), but from some ply on the
// value of a new leaf is a short minimax search from that leaf instead of the
// network's value head. The network sees one position; the minimax a few
// thousand nodes under it, with an exact endgame when it gets there.
//
// Where it plugs in: az::Search calls LeafValue::value() for every leaf it
// has just evaluated with the network (player/alphazero_mcts.cpp, run_worker,
// step 3), in the search thread that gathered the leaf. Each search thread
// owns one minimax (single-threaded, its own small table: the leaves of one
// thread are close to each other, its table helps the next ones).
//
// JSON, under the MCTS player's config:
//   "leaf minimax": {
//     "min ply": 30,            // leaves with yolah.nb_plies() >= this get the minimax value
//     "nodes": 4000,            // node budget of each leaf search
//     "microseconds": 20000,    // and a time cap (0 = none), a guard against a slow search
//     "max depth": 64,          // iterative deepening up to this depth
//     "weight": 1.0,            // value = (1 − w)·network + w·minimax (1 = replace)
//     "concurrency": 4,         // at most this many leaf searches at the same time (the CPU
//                               //   budget: the opponent minimax gets 4 threads); 0 = every thread
//     "tt size": 16,            // MB of table per search thread
//     "player": { ...a MinMaxNNUE_DevPlayer config... }   // the dev's options and weights
//   }
// The minimax's value v (side to move, ±30000 = tanh ±1, see heuristic.h) is
// mapped to the MCTS scale [−1, 1] as v / 30000; a proven result (|v| >
// 30000: ±(WIN + margin)) gets the MCTS's terminal utility with that margin
// (score utility, alphazero_mcts.h), so a proven win weighs like a finished game.
#include "alphazero_mcts.h"
#include "player.h"
#include <atomic>
#include <memory>
#include <semaphore>
#include <vector>

class MinMaxNNUE_DevPlayer;

class LeafMinimax : public az::LeafValue {
public:
    // nb_threads: the MCTS's search threads (one minimax each);
    // score_utility / score_scale: the MCTS's, for the proven results.
    LeafMinimax(const json& j, size_t nb_threads, float score_utility, float score_scale);
    ~LeafMinimax() override;
    bool value(size_t thread, const Yolah& state, float net_value, float& value) override;
    void new_game();          // forget the tables (start of a game)
    std::string stats();      // the leaf searches since the last call (for "verbose")
    json config() const { return cfg; }

private:
    json cfg;
    uint16_t min_ply;
    uint64_t nodes, microseconds;
    uint8_t  max_depth;
    float    weight, score_utility, score_scale;
    std::vector<std::unique_ptr<Player>> engines;   // one per search thread
    std::unique_ptr<std::counting_semaphore<1024>> slots;   // null: no limit
    std::atomic<uint64_t> nb_searches{0}, nb_nodes{0}, nb_proven{0}, nb_us{0};
    std::atomic<double>   sum_diff{0.0};             // Σ |minimax − network|
    std::atomic<double>   sum_net{0.0}, sum_mm{0.0}; // Σ |network|, Σ |minimax| (the two scales)
};

#endif
