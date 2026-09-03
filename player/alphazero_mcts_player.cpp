#include "alphazero_mcts_player.h"
#include "batched_evaluator.h"
#include <iostream>
#include <stdexcept>
#include <thread>

using std::string, std::invalid_argument;

namespace {
    size_t read_nb_threads(const json& j, const char* key, size_t default_value) {
        if (!j.contains(key)) return default_value;
        if (j[key].is_number()) return j[key].get<size_t>();
        if (j[key].is_string() && j[key].get<string>() == "hardware concurrency")
            return std::thread::hardware_concurrency();
        throw invalid_argument(string("number or \"hardware concurrency\" expected for ") + key);
    }
}

AlphaZeroMCTSPlayer::AlphaZeroMCTSPlayer(const json& j) : cfg(j) {
    if (!j.contains("weights")) throw invalid_argument("weights key expected");
    if (!j.contains("microseconds") && !j.contains("nb simulations"))
        throw invalid_argument("microseconds or nb simulations key expected");
    weights_filename = j["weights"].get<string>();
    verbose = j.value("verbose", false);

    az::SearchParams p;
    p.microseconds       = j.value("microseconds", uint64_t(0));
    p.nb_simulations     = j.value("nb simulations", uint64_t(0));
    p.nb_threads         = read_nb_threads(j, "nb threads", 1);
    p.batch_size         = j.value("batch size", size_t(0));
    p.max_collisions     = j.value("max collisions", size_t(0));
    p.nn_cache_mb        = j.value("nn cache", size_t(64));
    p.c_puct_init        = j.value("c puct init", p.c_puct_init);
    p.c_puct_base        = j.value("c puct base", p.c_puct_base);
    p.fpu_reduction      = j.value("fpu reduction", p.fpu_reduction);
    p.policy_temperature = j.value("policy temperature", p.policy_temperature);
    p.dirichlet_alpha    = j.value("dirichlet alpha", p.dirichlet_alpha);
    p.dirichlet_epsilon  = j.value("dirichlet epsilon", p.dirichlet_epsilon);
    p.temperature        = j.value("temperature", p.temperature);
    p.temperature_cutoff = j.value("temperature cutoff", p.temperature_cutoff);
    p.reuse_tree         = j.value("reuse tree", true);
    p.seed               = j.value("seed", uint64_t(0));

    // Backend, then the batching wrapper when several search threads share it.
    std::unique_ptr<nn::Evaluator> backend = nn::make_evaluator(j);
    if (p.batch_size == 0) p.batch_size = backend->preferred_batch_size();
    if (p.nb_threads > 1) {
        const size_t merged = j.value("merged batch size", p.batch_size * p.nb_threads);
        net = std::make_unique<nn::BatchingEvaluator>(std::move(backend), p.nb_threads, merged);
    } else {
        net = std::move(backend);
    }
    searcher = std::make_unique<az::Search>(*net, p, &memory);
}

AlphaZeroMCTSPlayer::~AlphaZeroMCTSPlayer() {
    // The tree lives in `memory`; destroy it before the pool goes away.
    searcher.reset();
}

Move AlphaZeroMCTSPlayer::play(Yolah yolah) {
    last = searcher->search(yolah);
    if (verbose) std::cout << last.to_string() << std::flush;
    return last.best_move;
}

void AlphaZeroMCTSPlayer::game_over(Yolah) {
    searcher->reset();
}

void AlphaZeroMCTSPlayer::reload_weights(const std::string& filename) {
    net->reload(filename);
    weights_filename = filename;
    cfg["weights"] = filename;
    searcher->reset();         // the old tree's priors/values came from the old network
    searcher->clear_cache();   // so did the cached evaluations
}

std::string AlphaZeroMCTSPlayer::info() {
    return "alphazero mcts player (PUCT + value/policy resnet, " + net->info() + ")";
}

json AlphaZeroMCTSPlayer::config() {
    json j = cfg;
    j["name"] = "AlphaZeroMCTSPlayer";
    const az::SearchParams& p = searcher->params();
    j["microseconds"] = p.microseconds;
    j["nb simulations"] = p.nb_simulations;
    if (p.nb_threads == std::thread::hardware_concurrency()) j["nb threads"] = "hardware concurrency";
    else                                                     j["nb threads"] = p.nb_threads;
    j["batch size"] = p.batch_size;
    j["weights"] = weights_filename;
    j["backend"] = cfg.value("backend", string("cpu"));
    j["c puct init"] = p.c_puct_init;
    j["c puct base"] = p.c_puct_base;
    j["fpu reduction"] = p.fpu_reduction;
    j["policy temperature"] = p.policy_temperature;
    j["dirichlet alpha"] = p.dirichlet_alpha;
    j["dirichlet epsilon"] = p.dirichlet_epsilon;
    j["temperature"] = p.temperature;
    j["temperature cutoff"] = p.temperature_cutoff;
    j["reuse tree"] = p.reuse_tree;
    j["nn cache"] = p.nn_cache_mb;
    j["verbose"] = verbose;
    return j;
}
