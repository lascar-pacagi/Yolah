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

az::SearchParams AlphaZeroMCTSPlayer::search_params(const json& j) {
    az::SearchParams p;
    p.microseconds       = j.value("microseconds", uint64_t(0));
    p.nb_simulations     = j.value("nb simulations", uint64_t(0));
    p.nb_threads         = read_nb_threads(j, "nb threads", 1);
    p.batch_size         = j.value("batch size", size_t(0));
    p.max_collisions     = j.value("max collisions", size_t(0));
    p.nn_cache_mb        = j.value("nn cache", size_t(64));
    p.cpuct_exploration      = j.value("cpuct exploration", p.cpuct_exploration);
    p.cpuct_exploration_log  = j.value("cpuct exploration log", p.cpuct_exploration_log);
    p.cpuct_exploration_base = j.value("cpuct exploration base", p.cpuct_exploration_base);
    p.cpuct_utility_stdev_scale        = j.value("cpuct utility stdev scale", p.cpuct_utility_stdev_scale);
    p.cpuct_utility_stdev_prior        = j.value("cpuct utility stdev prior", p.cpuct_utility_stdev_prior);
    p.cpuct_utility_stdev_prior_weight = j.value("cpuct utility stdev prior weight", p.cpuct_utility_stdev_prior_weight);
    p.fpu_reduction      = j.value("fpu reduction", p.fpu_reduction);
    p.root_fpu_reduction = j.value("root fpu reduction", p.root_fpu_reduction);
    p.fpu_loss_prop      = j.value("fpu loss prop", p.fpu_loss_prop);
    p.root_fpu_loss_prop = j.value("root fpu loss prop", p.root_fpu_loss_prop);
    p.fpu_parent_weight_by_visited_policy =
        j.value("fpu parent weight by visited policy", p.fpu_parent_weight_by_visited_policy);
    p.fpu_parent_weight_by_visited_policy_pow =
        j.value("fpu parent weight by visited policy pow", p.fpu_parent_weight_by_visited_policy_pow);
    p.fpu_parent_weight  = j.value("fpu parent weight", p.fpu_parent_weight);
    p.policy_temperature = j.value("policy temperature", p.policy_temperature);
    p.graph_search       = j.value("graph search", p.graph_search);
    p.use_lcb            = j.value("use lcb", p.use_lcb);
    p.lcb_stdevs         = j.value("lcb stdevs", p.lcb_stdevs);
    p.min_visit_prop_for_lcb = j.value("min visit prop for lcb", p.min_visit_prop_for_lcb);
    p.policy_target_pruning  = j.value("policy target pruning", p.policy_target_pruning);
    p.forced_playouts_k      = j.value("forced playouts k", p.forced_playouts_k);
    p.playout_cap_fast_prob  = j.value("playout cap fast prob", p.playout_cap_fast_prob);
    p.nb_simulations_fast    = j.value("nb simulations fast", p.nb_simulations_fast);
    p.dirichlet_alpha    = j.value("dirichlet alpha", p.dirichlet_alpha);
    p.dirichlet_epsilon  = j.value("dirichlet epsilon", p.dirichlet_epsilon);
    p.temperature        = j.value("temperature", p.temperature);
    p.temperature_cutoff = j.value("temperature cutoff", p.temperature_cutoff);
    p.reuse_tree         = j.value("reuse tree", true);
    p.seed               = j.value("seed", uint64_t(0));
    return p;
}

AlphaZeroMCTSPlayer::AlphaZeroMCTSPlayer(const json& j) : cfg(j) {
    if (!j.contains("weights")) throw invalid_argument("weights key expected");
    if (!j.contains("microseconds") && !j.contains("nb simulations"))
        throw invalid_argument("microseconds or nb simulations key expected");
    weights_filename = j["weights"].get<string>();
    verbose = j.value("verbose", false);

    az::SearchParams p = search_params(j);

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
    j["cpuct exploration"] = p.cpuct_exploration;
    j["cpuct exploration log"] = p.cpuct_exploration_log;
    j["cpuct exploration base"] = p.cpuct_exploration_base;
    j["cpuct utility stdev scale"] = p.cpuct_utility_stdev_scale;
    j["cpuct utility stdev prior"] = p.cpuct_utility_stdev_prior;
    j["cpuct utility stdev prior weight"] = p.cpuct_utility_stdev_prior_weight;
    j["fpu reduction"] = p.fpu_reduction;
    j["root fpu reduction"] = p.root_fpu_reduction;
    j["fpu loss prop"] = p.fpu_loss_prop;
    j["root fpu loss prop"] = p.root_fpu_loss_prop;
    j["fpu parent weight by visited policy"] = p.fpu_parent_weight_by_visited_policy;
    j["fpu parent weight by visited policy pow"] = p.fpu_parent_weight_by_visited_policy_pow;
    j["fpu parent weight"] = p.fpu_parent_weight;
    j["policy temperature"] = p.policy_temperature;
    j["graph search"] = p.graph_search;
    j["use lcb"] = p.use_lcb;
    j["lcb stdevs"] = p.lcb_stdevs;
    j["min visit prop for lcb"] = p.min_visit_prop_for_lcb;
    j["policy target pruning"] = p.policy_target_pruning;
    j["forced playouts k"] = p.forced_playouts_k;
    j["playout cap fast prob"] = p.playout_cap_fast_prob;
    j["nb simulations fast"] = p.nb_simulations_fast;
    j["dirichlet alpha"] = p.dirichlet_alpha;
    j["dirichlet epsilon"] = p.dirichlet_epsilon;
    j["temperature"] = p.temperature;
    j["temperature cutoff"] = p.temperature_cutoff;
    j["reuse tree"] = p.reuse_tree;
    j["nn cache"] = p.nn_cache_mb;
    j["verbose"] = verbose;
    return j;
}
