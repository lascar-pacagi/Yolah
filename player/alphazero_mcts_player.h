#ifndef ALPHAZERO_MCTS_PLAYER_H
#define ALPHAZERO_MCTS_PLAYER_H
// alphazero_mcts_player.h — Player wrapper around az::Search + a network backend.
//
// JSON configuration (keys follow the other players' conventions):
// {
//   "name": "AlphaZeroMCTSPlayer",
//   "microseconds": 1000000,          // thinking time per move (0 = use only "nb simulations")
//   "nb simulations": 0,              // simulation budget (0 = time only); 800 for self-play
//   "nb threads": 1,                  // search threads ("hardware concurrency" allowed)
//   "batch size": 32,                 // leaves gathered per search thread per network call
//   "weights": "../nnue/cnn_resnet_256x30_value_policy.bin",   // .bin (cpu) or .ts (torch)
//   "backend": "cpu",                 // "cpu" (Eigen/OpenMP) | "torch" (libtorch, -DENABLE_TORCH=ON)
//   "device": "cuda",                 // torch only
//   "fp16": true,                     // torch only
//   "nb eval threads": 0,             // cpu only: OpenMP threads inside the network (0 = all)
//   "c puct init": 1.25, "c puct base": 19652, "fpu reduction": 0.3, "policy temperature": 1.0,
//   "dirichlet alpha": 0.3, "dirichlet epsilon": 0.0,           // root noise (self-play)
//   "temperature": 0.0, "temperature cutoff": 0,                // move sampling (self-play)
//   "reuse tree": true,
//   "nn cache": 64,                   // MB of transposition cache for network outputs (0 = off)
//   "seed": 0,
//   "verbose": false                  // print the root statistics after every move
// }
// Only "microseconds" (or "nb simulations") and "weights" are mandatory.
//
// Threading model: the search threads share one tree and one evaluator. With
// more than one search thread the evaluator is wrapped in nn::BatchingEvaluator
// so that the backend sees merged batches (essential for a GPU, and it keeps
// the OpenMP CPU backend from being entered concurrently).
//
// Self-play / future learning: after play() the full root statistics of the
// last search are available through last_result() — the visit distribution is
// the policy target π and the game outcome gives z. Set "dirichlet epsilon"
// (0.25), "temperature" (1.0) and "temperature cutoff" (e.g. 20 plies) for
// exploration, "nb simulations" for a fixed budget, and swap networks with
// reload_weights().
#include "player.h"
#include "alphazero_mcts.h"
#include "nn_evaluator.h"
#include <memory>
#include <memory_resource>
#include <string>

class AlphaZeroMCTSPlayer : public Player {
public:
    explicit AlphaZeroMCTSPlayer(const json& config);
    ~AlphaZeroMCTSPlayer() override;

    Move play(Yolah) override;
    void game_over(Yolah) override;
    std::string info() override;
    json config() override;

    const az::SearchResult& last_result() const { return last; }
    az::Search& search() { return *searcher; }
    nn::Evaluator& evaluator() { return *net; }
    // Load new weights (same architecture) into the running backend.
    void reload_weights(const std::string& weights_filename);

private:
    json cfg;
    std::string weights_filename;
    bool verbose;
    std::pmr::synchronized_pool_resource memory;
    std::unique_ptr<nn::Evaluator> net;
    std::unique_ptr<az::Search> searcher;
    az::SearchResult last;
};

#endif
