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
//   // PUCT, KataGo style (see alphazero_mcts.h); the defaults are KataGo's
//   "cpuct exploration": 1.0, "cpuct exploration log": 0.45, "cpuct exploration base": 500,
//   "cpuct utility stdev scale": 0.85,   // 0 = plain AlphaZero exploration
//   "cpuct utility stdev prior": 0.40, "cpuct utility stdev prior weight": 2.0,
//   "fpu reduction": 0.2, "root fpu reduction": 0.1,
//   "fpu loss prop": 0.0, "root fpu loss prop": 0.0,
//   "fpu parent weight by visited policy": true,
//   "fpu parent weight by visited policy pow": 1.0, "fpu parent weight": 0.0,
//   "policy temperature": 1.0,
//   "graph search": true,             // one node per position (DAG) instead of a tree
//   // root move choice
//   "use lcb": true, "lcb stdevs": 5.0, "min visit prop for lcb": 0.2,
//   "policy target pruning": true,    // take the forced playouts back out of pi
//   "dirichlet alpha": 0.3, "dirichlet epsilon": 0.0,           // root noise (self-play)
//   "temperature": 0.0, "temperature cutoff": 0,                // move sampling (self-play)
//   "forced playouts k": 0.0,         // 2.0 during self-play, 0 for competitive play
//   "playout cap fast prob": 0.0,     // self-play: fraction of moves with a small budget
//   "nb simulations fast": 0,         //   ... and how small (needs "nb simulations")
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
// Self-play: after play() the full root statistics of the last search are
// available through last_result() — its play values are the policy target π
// and the game outcome gives z. The learning loop itself (player/
// alphazero_selfplay.h, nnue/alphazero_learn.py) does not use this class: it
// runs az::Search directly, many games on one shared evaluator, with the
// parameters built by search_params(). Self-play settings: "dirichlet
// epsilon" 0.25, "temperature" 1.0, "temperature cutoff" 20, "nb simulations"
// 800 — see config/alphazero_mcts_selfplay_player.cfg.
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
    // The search parameters described by a JSON config (the keys above);
    // missing keys keep az::SearchParams' defaults. Used by the self-play
    // driver, which runs az::Search directly on a shared evaluator.
    static az::SearchParams search_params(const json& config);

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
