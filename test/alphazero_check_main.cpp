// alphazero_check_main.cpp — self-check and benchmark of the AlphaZero MCTS
// player (test/alphazero_mcts_check.h) and its network backends
// (test/resnet_check.h). Run from the build directory:
//
//   ./alphazero_check                      # CPU backend, all checks
//   ./alphazero_check --backend torch --weights ../nnue/cnn_resnet_256x30_value_policy.ts
//   ./alphazero_check --bench-only         # network throughput at several batch sizes
//   ./alphazero_check --backend torch --fp32   # GPU in full precision (exact check)
//   ./alphazero_check --games 4 --time 1000000 --opponent ../config/mcts_player.cfg
#include "alphazero_mcts_player.h"
#include "alphazero_mcts_check.h"
#include "resnet_check.h"
#include "magic.h"
#include "zobrist.h"
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();

    std::string weights = "../nnue/cnn_resnet_256x30_value_policy.bin";
    std::string test_vectors = "../nnue/cnn_resnet_256x30_value_policy.test.bin";
    std::string backend = "cpu", device = "cuda", opponent;
    size_t threads = 1, batch = 0, games = 0;
    uint64_t sims = 400, time_us = 1'000'000;
    bool bench_only = false, no_bench = false, verbose = false, fp32 = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); } return std::string(argv[++i]); };
        if (a == "--weights") weights = next();
        else if (a == "--test-vectors") test_vectors = next();
        else if (a == "--backend") backend = next();
        else if (a == "--device") device = next();
        else if (a == "--threads") threads = std::stoul(next());
        else if (a == "--batch") batch = std::stoul(next());
        else if (a == "--sims") sims = std::stoull(next());
        else if (a == "--time") time_us = std::stoull(next());
        else if (a == "--games") games = std::stoul(next());
        else if (a == "--opponent") opponent = next();
        else if (a == "--bench-only") bench_only = true;
        else if (a == "--no-bench") no_bench = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--fp32") fp32 = true;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (backend == "torch" && weights.ends_with(".bin")) weights = weights.substr(0, weights.size() - 4) + ".ts";

    json cfg;
    cfg["name"] = "AlphaZeroMCTSPlayer";
    cfg["microseconds"] = time_us;
    cfg["nb simulations"] = 0;
    cfg["nb threads"] = threads;
    if (batch) cfg["batch size"] = batch;
    cfg["weights"] = weights;
    cfg["backend"] = backend;
    cfg["device"] = device;
    cfg["fp16"] = !fp32;
    cfg["verbose"] = verbose;
    // Half precision on the GPU deviates a little from the fp32 reference.
    const float tolerance = (backend == "torch" && !fp32) ? 2e-2f : 2e-3f;

    bool ok = true;
    {
        // Network check + throughput on the raw backend.
        std::unique_ptr<nn::Evaluator> net = nn::make_evaluator(cfg);
        if (!bench_only) ok &= test::resnet_check(*net, test_vectors, tolerance);
        if (!no_bench) {
            for (size_t bs : {size_t(1), size_t(16), size_t(64), size_t(256)}) test::resnet_bench(*net, bs, 2);
        }
    }
    if (bench_only) return ok ? 0 : 1;

    ok &= test::alphazero_search_check(cfg, sims);
    ok &= test::alphazero_reuse_check(cfg, sims);
    if (games > 0) {
        json opp;
        if (opponent.empty()) {
            opp = {{"name", "MCTSMemPlayer"}, {"microseconds", time_us}, {"nb threads", "hardware concurrency"}};
        } else {
            opp = json::parse(std::ifstream(opponent));
        }
        test::alphazero_vs(cfg, opp, games, verbose);
    }
    std::cout << (ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
    return ok ? 0 : 1;
}
