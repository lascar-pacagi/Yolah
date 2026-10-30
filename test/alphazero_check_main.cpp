// alphazero_check_main.cpp — self-check and benchmark of the AlphaZero MCTS
// player (test/alphazero_mcts_check.h) and its network backends
// (test/resnet_check.h). Run from the build directory:
//
//   ./alphazero_check                      # CPU backend, all checks
//   ./alphazero_check --backend torch --weights ../nnue/cnn_resnet_256x30_value_policy.ts
//   ./alphazero_check --bench-only         # network throughput at several batch sizes
//   ./alphazero_check --backend torch --fp32   # GPU in full precision (exact check)
//   ./alphazero_check --games 4 --time 1000000 --opponent ../config/mcts_player.cfg
//   ./alphazero_check --no-bench --set 'graph search=false'   # A/B a search parameter
//   ./alphazero_check --backend torch --weights az.ts --mcts-bench 4x32,16x64 \
//         --positions bench_positions.txt --limit 40 --time 1000000 [--ref-time 4000000]
//       search throughput per threads × batch (see mcts_bench below)
//
// --set takes any key of the player's JSON config (see alphazero_mcts_player.h)
// as 'key=value'; the value is parsed as JSON, so true/false/numbers work.
#include "alphazero_mcts_player.h"
#include "alphazero_mcts_check.h"
#include "resnet_check.h"
#include "magic.h"
#include "zobrist.h"
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// ─── MCTS throughput per threads × batch ─────────────────────────────────────
// The search keeps up to threads × batch descents in flight, each one
// steering the others away with a virtual loss: more in flight feeds the GPU
// better but makes the search less sequential. For every "TxB" of the list,
// a fresh player (no tree reuse, its own network cache) searches each
// position for --time: simulations per second, network evaluations per
// second, collisions. With --ref-time, the best move of a long 4x32 search
// (the evaluation matches' setting) is the reference, and each setting gets
// its rate of agreement with it: a cheap hint of search quality (a match
// decides).
static std::vector<Yolah> read_positions(const std::string& path, size_t limit) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    std::vector<Yolah> res;
    std::string line;
    while (res.size() < limit && std::getline(f, line)) {
        if (line.empty()) continue;
        Yolah y = Yolah::from_json(line);
        Yolah::MoveList moves;
        y.moves(moves);
        if (!y.game_over() && moves.size() > 1) res.push_back(y);
    }
    return res;
}

static int mcts_bench(json cfg, const std::string& list, const std::vector<Yolah>& positions, uint64_t ref_time_us) {
    cfg["reuse tree"] = false;
    auto run = [&](size_t threads, size_t batch, uint64_t time_us, std::vector<Move>* moves) {
        json c = cfg;
        c["nb threads"] = threads;
        c["batch size"] = batch;
        c["microseconds"] = time_us;
        AlphaZeroMCTSPlayer player(c);
        player.play(positions[0]);                  // warm-up (allocation, CUDA kernels)
        uint64_t sims = 0, evals = 0, collisions = 0;
        double seconds = 0;
        for (const Yolah& y : positions) {
            player.search().reset();
            const Move m = player.play(y);
            const az::SearchResult& r = player.last_result();
            sims += r.nb_simulations; evals += r.nb_evaluations; collisions += r.nb_collisions; seconds += r.seconds;
            if (moves) moves->push_back(m);
        }
        return std::tuple{sims / seconds, evals / seconds, double(collisions) / std::max<uint64_t>(sims, 1)};
    };
    std::vector<Move> ref;
    if (ref_time_us) {
        std::cout << std::format("reference: 4x32, {} µs per position ...", ref_time_us) << std::endl;
        run(4, 32, ref_time_us, &ref);
    }
    std::cout << std::format("{} positions, {} µs each, weights {}\n", positions.size(), cfg["microseconds"].get<uint64_t>(),
                             cfg["weights"].get<std::string>());
    std::cout << std::format("{:>8s} {:>10s} {:>12s} {:>12s} {:>12s}{}\n", "TxB", "in flight", "sims/s", "evals/s",
                             "collisions", ref_time_us ? "   same move as ref" : "");
    size_t start = 0;
    while (start < list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string::npos) end = list.size();
        const std::string item = list.substr(start, end - start);
        start = end + 1;
        const size_t x = item.find('x');
        if (x == std::string::npos) { std::cerr << "bad item " << item << " (expected TxB, e.g. 4x32)\n"; return 2; }
        const size_t t = std::stoul(item.substr(0, x)), b = std::stoul(item.substr(x + 1));
        std::vector<Move> moves;
        const auto [sps, eps, coll] = run(t, b, cfg["microseconds"].get<uint64_t>(), &moves);
        std::string agree;
        if (ref_time_us) {
            size_t same = 0;
            for (size_t i = 0; i < moves.size(); i++) same += moves[i] == ref[i];
            agree = std::format("   {:>6.1f} %", 100.0 * same / moves.size());
        }
        std::cout << std::format("{:>8s} {:>10d} {:>12.0f} {:>12.0f} {:>11.1f}%{}\n", item, t * b, sps, eps, 100 * coll, agree)
                  << std::flush;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();

    std::string weights = "../nnue/cnn_resnet_256x30_value_policy.bin";
    std::string test_vectors = "../nnue/cnn_resnet_256x30_value_policy.test.bin";
    std::string backend = "cpu", device = "cuda", opponent;
    size_t threads = 1, batch = 0, games = 0;
    uint64_t sims = 400, time_us = 1'000'000;
    bool bench_only = false, no_bench = false, verbose = false, fp32 = false;
    std::string mcts_list, positions_path = "bench_positions.txt";
    size_t limit = 40;
    uint64_t ref_time_us = 0;
    std::vector<std::pair<std::string, std::string>> overrides;
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
        else if (a == "--mcts-bench") mcts_list = next();
        else if (a == "--positions") positions_path = next();
        else if (a == "--limit") limit = std::stoul(next());
        else if (a == "--ref-time") ref_time_us = std::stoull(next());
        else if (a == "--set") {
            const std::string kv = next();
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) { std::cerr << "--set expects key=value\n"; return 2; }
            overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
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
    for (const auto& [key, value] : overrides) cfg[key] = json::parse(value);
    if (!mcts_list.empty()) return mcts_bench(cfg, mcts_list, read_positions(positions_path, limit), ref_time_us);
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
