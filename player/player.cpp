#include "player.h"
#include <functional>
#include <map>
#include "random_player.h"
#include "MCTS_mem_player.h"
#include "MCTS_player.h"
#include "basic_minmax_player.h"
#include "minmax_player.h"
#include "human_player.h"
#include "monte_carlo_player.h"
#include "minmax_nnue_player.h"
#include "minmax_nnue_quantized_player.h"
#include "minmax_nnue_baseline_player.h"
#include "minmax_nnue_dev_player.h"
#include "MCTS_mem_nn_player.h"
#include "features_net_player.h"
#include "alphazero_mcts_player.h"
#ifdef ENABLE_CUDA
#include "alphazero_player.h"
#endif
#include <stdexcept>

using std::unique_ptr, std::string, std::make_unique, std::invalid_argument;

unique_ptr<Player> Player::create(const json& j) {
    const std::map<string, std::function<unique_ptr<Player>(const json&)>> m {
      {"RandomPlayer",
       [](const json &j) {
         if (!j.contains("seed")) {
           throw invalid_argument("seed key expected");
         }
         if (j["seed"].is_string()) {
           if (j["seed"].get<string>() != "clock") {
             throw invalid_argument("clock expected in seed");
           }
           return make_unique<RandomPlayer>();
         }
         return make_unique<RandomPlayer>(j["seed"].get<uint64_t>());
       }},
          {"MCTSMemPlayer",
           [](const json &j) {
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             return make_unique<MCTSMemPlayer<>>(
                 j["microseconds"].get<uint64_t>(), nb_threads);
           }},
          {"MCTSPlayer",
           [](const json &j) {
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             return make_unique<MCTSPlayer<>>(j["microseconds"].get<uint64_t>(),
                                              nb_threads);
           }},
          {"MCTSMemNNPlayer",
           [](const json &j) {
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             if (!j.contains("weights")) {
               throw invalid_argument("weights key expected");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             return make_unique<MCTSMemNNPlayer>(
                 j["microseconds"].get<uint64_t>(), j["weights"], nb_threads);
           }},
          {"MonteCarloPlayer",
           [](const json &j) {
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             return make_unique<MonteCarloPlayer>(
                 j["microseconds"].get<uint64_t>(), nb_threads);
           }},
          {"BasicMinMaxPlayer",
           [](const json &j) {
             if (!j.contains("depth")) {
               throw invalid_argument("depth key expected");
             }
             if (!j["depth"].is_number()) {
               throw invalid_argument("number expected for depth");
             }
             return make_unique<BasicMinMaxPlayer>(j["depth"].get<uint8_t>());
           }},
          {"MinMaxPlayer",
           [](const json &j) {
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             if (!j.contains("tt size")) {
               throw invalid_argument("tt size key expected");
             }
             if (!j.contains("nb moves at full depth")) {
               throw invalid_argument("nb moves at full depth key expected");
             }
             if (!j.contains("late move reduction")) {
               throw invalid_argument("late move reduction key expected");
             }
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             if (!j["tt size"].is_number()) {
               throw invalid_argument("number expected for tt size");
             }
             if (!j["nb moves at full depth"].is_number()) {
               throw invalid_argument(
                   "number expected for number of moves at full depth");
             }
             if (!j["late move reduction"].is_number()) {
               throw invalid_argument("number expected in late move reduction");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             return make_unique<MinMaxPlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), nb_threads);
           }},
          {"MinMaxPlayer2",
           [](const json &j) {
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             if (!j.contains("tt size")) {
               throw invalid_argument("tt size key expected");
             }
             if (!j.contains("nb moves at full depth")) {
               throw invalid_argument("nb moves at full depth key expected");
             }
             if (!j.contains("late move reduction")) {
               throw invalid_argument("late move reduction key expected");
             }
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             if (!j["tt size"].is_number()) {
               throw invalid_argument("number expected for tt size");
             }
             if (!j["nb moves at full depth"].is_number()) {
               throw invalid_argument(
                   "number expected for number of moves at full depth");
             }
             if (!j["late move reduction"].is_number()) {
               throw invalid_argument("number expected in late move reduction");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             return make_unique<MinMaxPlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), nb_threads);
           }},
          {"MinMaxNNUEPlayer",
           [](const json &j) {
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             if (!j.contains("tt size")) {
               throw invalid_argument("tt size key expected");
             }
             if (!j.contains("nb moves at full depth")) {
               throw invalid_argument("nb moves at full depth key expected");
             }
             if (!j.contains("late move reduction")) {
               throw invalid_argument("late move reduction key expected");
             }
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("weights")) {
               throw invalid_argument("weights key expected");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             if (!j["tt size"].is_number()) {
               throw invalid_argument("number expected for tt size");
             }
             if (!j["nb moves at full depth"].is_number()) {
               throw invalid_argument(
                   "number expected for number of moves at full depth");
             }
             if (!j["late move reduction"].is_number()) {
               throw invalid_argument("number expected in late move reduction");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             return make_unique<MinMaxNNUEPlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), j["weights"],
                 nb_threads);
           }},
          {"MinMaxNNUE_QuantizedPlayer",
           [](const json &j) {
             if (!j.contains("microseconds")) {
               throw invalid_argument("microseconds key expected");
             }
             if (!j.contains("tt size")) {
               throw invalid_argument("tt size key expected");
             }
             if (!j.contains("nb moves at full depth")) {
               throw invalid_argument("nb moves at full depth key expected");
             }
             if (!j.contains("late move reduction")) {
               throw invalid_argument("late move reduction key expected");
             }
             if (!j.contains("nb threads")) {
               throw invalid_argument("nb threads key expected");
             }
             if (!j.contains("weights")) {
               throw invalid_argument("weights key expected");
             }
             if (!j["microseconds"].is_number()) {
               throw invalid_argument("number expected for microseconds");
             }
             if (!j["tt size"].is_number()) {
               throw invalid_argument("number expected for tt size");
             }
             if (!j["nb moves at full depth"].is_number()) {
               throw invalid_argument(
                   "number expected for number of moves at full depth");
             }
             if (!j["late move reduction"].is_number()) {
               throw invalid_argument("number expected in late move reduction");
             }
             size_t nb_threads;
             if (j["nb threads"].is_number()) {
               nb_threads = j["nb threads"].get<size_t>();
             } else if (j["nb threads"].get<string>() ==
                        "hardware concurrency") {
               nb_threads = std::thread::hardware_concurrency();
             } else {
               throw invalid_argument(
                   "hardware concurrency expected in nb threads");
             }
             return make_unique<MinMaxNNUE_QuantizedPlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), j["weights"],
                 nb_threads);
           }},
          // The search experiments: the reference and the version being
          // improved (one thread; "nb threads" is ignored).
          {"MinMaxNNUE_BaselinePlayer",
           [](const json &j) {
             for (const char* k : {"microseconds", "tt size", "nb moves at full depth", "late move reduction", "weights"}) {
               if (!j.contains(k)) throw invalid_argument(string(k) + " key expected");
             }
             return make_unique<MinMaxNNUE_BaselinePlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), j["weights"].get<string>(),
                 j.value("verbose", false));
           }},
          {"MinMaxNNUE_DevPlayer",
           [](const json &j) {
             for (const char* k : {"microseconds", "tt size", "nb moves at full depth", "late move reduction", "weights"}) {
               if (!j.contains(k)) throw invalid_argument(string(k) + " key expected");
             }
             // The improvements can be switched off one by one (defaults: on).
             MinMaxNNUE_DevPlayer::Options options;
             options.pvs = j.value("pvs", options.pvs);
             options.aspiration_window = j.value("aspiration window", options.aspiration_window);
             options.history = j.value("history", options.history);
             options.countermove = j.value("countermove", options.countermove);
             options.root_ordering = j.value("root ordering", options.root_ordering);
             options.lazy_accumulator = j.value("lazy accumulator", options.lazy_accumulator);
             options.eval_cache_bits = j.value("eval cache", options.eval_cache_bits);
             options.lmr = j.value("lmr", options.lmr);
             options.lmr_base = j.value("lmr base", options.lmr_base);
             options.lmr_divisor = j.value("lmr divisor", options.lmr_divisor);
             options.rfp_depth = j.value("rfp depth", options.rfp_depth);
             options.rfp_margin = j.value("rfp margin", options.rfp_margin);
             options.null_move = j.value("null move", options.null_move);
             options.null_move_reduction = j.value("null move reduction", options.null_move_reduction);
             options.lmp_depth = j.value("lmp depth", options.lmp_depth);
             options.lmp_moves = j.value("lmp moves", options.lmp_moves);
             options.pass_rule = j.value("pass rule", options.pass_rule);
             options.yolah_table = j.value("yolah table", options.yolah_table);
             options.staged = j.value("staged", options.staged);
             options.eval_grain = j.value("eval grain", options.eval_grain);
             options.proxy_cut = j.value("proxy cut", options.proxy_cut);
             options.proxy_rank = j.value("proxy rank", options.proxy_rank);
             options.probcut_margin = j.value("probcut margin", options.probcut_margin);
             options.proxy_depth = j.value("proxy depth", options.proxy_depth);
             options.proxy_reduction = j.value("proxy reduction", options.proxy_reduction);
             options.proxy_delta = j.value("proxy delta", options.proxy_delta);
             options.proxy_witness = j.value("proxy witness", options.proxy_witness);
             options.proxy_multi_c = j.value("proxy multi c", options.proxy_multi_c);
             options.proxy_multi_m = j.value("proxy multi m", options.proxy_multi_m);
             options.proxy_diverse = j.value("proxy diverse", options.proxy_diverse);
             options.proxy_second = j.value("proxy second", options.proxy_second);
             options.proxy_prefilter = j.value("proxy prefilter", options.proxy_prefilter);
             options.proxy_weval = j.value("proxy weval", options.proxy_weval);
             options.proxy_verify = j.value("proxy verify", options.proxy_verify);
             options.proxy_lazy = j.value("proxy lazy", options.proxy_lazy);
             options.proxy_first = j.value("proxy first", options.proxy_first);
             options.proxy_tail = j.value("proxy tail", options.proxy_tail);
             options.proxy_tail_keep = j.value("proxy tail keep", options.proxy_tail_keep);
             options.proxy_tail_margin = j.value("proxy tail margin", options.proxy_tail_margin);
             options.proxy_tail_reduction = j.value("proxy tail reduction", options.proxy_tail_reduction);
             options.proxy_tail_step = j.value("proxy tail step", options.proxy_tail_step);
             options.proxy_tail_cap = j.value("proxy tail cap", options.proxy_tail_cap);
             options.mpc = j.value("mpc", options.mpc);
             options.mpc_depth = j.value("mpc depth", options.mpc_depth);
             options.mpc_ratio = j.value("mpc ratio", options.mpc_ratio);
             options.mpc_margin = j.value("mpc margin", options.mpc_margin);
             options.mpc_margin_per_ply = j.value("mpc margin per ply", options.mpc_margin_per_ply);
             options.probcut = j.value("probcut", options.probcut);
             options.probcut_depth = j.value("probcut depth", options.probcut_depth);
             options.probcut_reduction = j.value("probcut reduction", options.probcut_reduction);
             options.probcut_filter = j.value("probcut filter", options.probcut_filter);
             options.territory_ordering = j.value("territory ordering", options.territory_ordering);
             options.territory_depth = j.value("territory depth", options.territory_depth);
             options.territory_weight = j.value("territory weight", options.territory_weight);
             options.fast_influence = j.value("fast influence", options.fast_influence);
             options.articulation_ordering = j.value("articulation ordering", options.articulation_ordering);
             options.articulation_lmr = j.value("articulation lmr", options.articulation_lmr);
             options.endgame_root = j.value("endgame root", options.endgame_root);
             options.endgame_root_time = j.value("endgame root time", options.endgame_root_time);
             options.endgame_tree = j.value("endgame tree", options.endgame_tree);
             // L. Lazy SMP: "nb threads" (a number or "hardware concurrency", default 1).
             size_t nb_threads = 1;
             if (j.contains("nb threads")) {
               if (j["nb threads"].is_number()) nb_threads = j["nb threads"].get<size_t>();
               else if (j["nb threads"].get<string>() == "hardware concurrency")
                 nb_threads = std::thread::hardware_concurrency();
               else throw invalid_argument("number or hardware concurrency expected in nb threads");
             }
             return make_unique<MinMaxNNUE_DevPlayer>(
                 j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>(),
                 j["nb moves at full depth"].get<size_t>(),
                 j["late move reduction"].get<uint8_t>(), j["weights"].get<string>(),
                 j.value("verbose", false), options, nb_threads);
           }},
        {
            "FeaturesNetPlayer",
            [](const json& j) {
                if (!j.contains("microseconds")) {
                    throw invalid_argument("microseconds key expected");
                }
                if (!j.contains("tt size")) {
                    throw invalid_argument("tt size key expected");
                }
                if (!j.contains("nb moves at full depth")) {
                    throw invalid_argument("nb moves at full depth key expected");
                }
                if (!j.contains("late move reduction")) {
                    throw invalid_argument("late move reduction key expected");
                }
                 if (!j.contains("nb threads")) {
                    throw invalid_argument("nb threads key expected");
                }
                if (!j.contains("weights")) {
                    throw invalid_argument("weights key expected");
                }                
                if (!j["microseconds"].is_number()) {
                    throw invalid_argument("number expected for microseconds");
                }
                if (!j["tt size"].is_number()) {
                    throw invalid_argument("number expected for tt size");
                }                
                if (!j["nb moves at full depth"].is_number()) {
                    throw invalid_argument("number expected for number of moves at full depth");
                }
                if (!j["late move reduction"].is_number()) {
                    throw invalid_argument("number expected in late move reduction");
                }
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }                
                // "network": "wdl quantized" (default, the original networks),
                // "wdl float", "value quantized" or "value float" (see FeaturesEvaluator).
                return make_unique<FeaturesNetPlayer>(j["microseconds"].get<uint64_t>(), 
                                                     j["tt size"].get<size_t>(),
                                                     j["nb moves at full depth"].get<size_t>(),
                                                     j["late move reduction"].get<uint8_t>(),
                                                     j["weights"],
                                                     nb_threads,
                                                     j.value("network", std::string("wdl quantized")));
            }
        },          
        {
            "AlphaZeroMCTSPlayer",
            [](const json& j) {
                // Keys are validated by the constructor (see alphazero_mcts_player.h).
                return make_unique<AlphaZeroMCTSPlayer>(j);
            }
        },
#ifdef ENABLE_CUDA
        {
            "AlphaZeroPlayer",
            [](const json& j) {
                return make_unique<AlphaZeroPlayer>(j);
            }
        },
#endif
    };
    return m.at(j["name"].get<string>())(j);
}
