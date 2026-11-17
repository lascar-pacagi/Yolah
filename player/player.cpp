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
             options.aspiration_k = j.value("aspiration k", options.aspiration_k);
             options.aspiration_grow = j.value("aspiration grow", options.aspiration_grow);
             options.tempo = j.value("tempo", options.tempo);
             options.root_fix = j.value("root fix", options.root_fix);
             options.root_nodes_last = j.value("root nodes last", options.root_nodes_last);
             options.root_order_gap = j.value("root order gap", options.root_order_gap);
             options.root_close_full = j.value("root close full", options.root_close_full);
             options.aspiration_keep_beta = j.value("aspiration keep beta", options.aspiration_keep_beta);
             options.aspiration_stats_depth = j.value("aspiration stats depth", options.aspiration_stats_depth);
             options.aspiration_center2 = j.value("aspiration center 2", options.aspiration_center2);
             options.aspiration_last = j.value("aspiration last", options.aspiration_last);
             options.pvs_research_high = j.value("pvs research high", options.pvs_research_high);
             options.tt_cut_pv = j.value("tt cut pv", options.tt_cut_pv);
             options.tt_cut_pv_exact = j.value("tt cut pv exact", options.tt_cut_pv_exact);
             options.tt_pv = j.value("tt pv", options.tt_pv);
             options.tt_depth_asym = j.value("tt depth asym", options.tt_depth_asym);
             options.table_tiebreak = j.value("table tiebreak", options.table_tiebreak);
             options.proxy_after_tt = j.value("proxy after tt", options.proxy_after_tt);
             options.lmr_territory = j.value("lmr territory", options.lmr_territory);
             options.history = j.value("history", options.history);
             options.countermove = j.value("countermove", options.countermove);
             options.root_ordering = j.value("root ordering", options.root_ordering);
             options.lazy_accumulator = j.value("lazy accumulator", options.lazy_accumulator);
             options.eval_cache_bits = j.value("eval cache", options.eval_cache_bits);
             options.max_nodes = j.value("max nodes", options.max_nodes);
             options.record_depth = j.value("record depth", options.record_depth);
             options.sparse_h1 = j.value("sparse h1", options.sparse_h1);
             options.history_prior = j.value("history prior", options.history_prior);
             options.history_prior_mode = j.value("history prior mode", options.history_prior_mode);
             options.history_prior_weight = j.value("history prior weight", options.history_prior_weight);
             options.history_prior_threshold = j.value("history prior threshold", options.history_prior_threshold);
             options.history_prior_top = j.value("history prior top", options.history_prior_top);
             options.lmr = j.value("lmr", options.lmr);
             options.lmr_base = j.value("lmr base", options.lmr_base);
             options.lmr_divisor = j.value("lmr divisor", options.lmr_divisor);
             options.rfp_depth = j.value("rfp depth", options.rfp_depth);
             options.rfp_margin = j.value("rfp margin", options.rfp_margin);
             options.null_move = j.value("null move", options.null_move);
             options.null_move_reduction = j.value("null move reduction", options.null_move_reduction);
             options.fly_reply = j.value("fly reply", options.fly_reply);
             options.fly_reply_jumps = j.value("fly reply jumps", options.fly_reply_jumps);
             options.fly_reply_depth = j.value("fly reply depth", options.fly_reply_depth);
             options.fly_reply_reduction = j.value("fly reply reduction", options.fly_reply_reduction);
             options.fly_reply_div = j.value("fly reply div", options.fly_reply_div);
             options.fly_reply_from = j.value("fly reply from", options.fly_reply_from);
             options.fly_reply_order = j.value("fly reply order", options.fly_reply_order);
             options.fly_reply_margin = j.value("fly reply margin", options.fly_reply_margin);
             options.fly_reply_store = j.value("fly reply store", options.fly_reply_store);
             options.fly_reply_filter = j.value("fly reply filter", options.fly_reply_filter);
             options.fly_reply_filter_margin = j.value("fly reply filter margin", options.fly_reply_filter_margin);
             options.fly_prune = j.value("fly prune", options.fly_prune);
             options.fly_prune_jumps = j.value("fly prune jumps", options.fly_prune_jumps);
             options.fly_prune_depth = j.value("fly prune depth", options.fly_prune_depth);
             options.fly_prune_reduction = j.value("fly prune reduction", options.fly_prune_reduction);
             options.fly_prune_div = j.value("fly prune div", options.fly_prune_div);
             options.fly_prune_from = j.value("fly prune from", options.fly_prune_from);
             options.proxy_fly = j.value("proxy fly", options.proxy_fly);
             options.proxy_fly_jumps = j.value("proxy fly jumps", options.proxy_fly_jumps);
             options.fly_lmr = j.value("fly lmr", options.fly_lmr);
             options.fly_lmr_depth = j.value("fly lmr depth", options.fly_lmr_depth);
             options.fly_lmr_closed = j.value("fly lmr closed", options.fly_lmr_closed);
             options.fly_lmr_open = j.value("fly lmr open", options.fly_lmr_open);
             options.fly_lmr_jumps = j.value("fly lmr jumps", options.fly_lmr_jumps);
             options.region_bounds = j.value("region bounds", options.region_bounds);
             options.region_margin = j.value("region margin", options.region_margin);
             options.region_max_free = j.value("region max free", options.region_max_free);
             options.fly_order = j.value("fly order", options.fly_order);
             options.fly_order_depth = j.value("fly order depth", options.fly_order_depth);
             options.fly_order_jumps = j.value("fly order jumps", options.fly_order_jumps);
             options.refute = j.value("refute", options.refute);
             options.refute_replies = j.value("refute replies", options.refute_replies);
             options.refute_depth = j.value("refute depth", options.refute_depth);
             options.refute_from = j.value("refute from", options.refute_from);
             options.refute_reduction = j.value("refute reduction", options.refute_reduction);
             options.proxy_restrict = j.value("proxy restrict", options.proxy_restrict);
             options.proxy_restrict_deeper = j.value("proxy restrict deeper", options.proxy_restrict_deeper);
             options.pass_razor = j.value("pass razor", options.pass_razor);
             options.pass_razor_depth = j.value("pass razor depth", options.pass_razor_depth);
             options.pass_razor_margin = j.value("pass razor margin", options.pass_razor_margin);
             options.pass_razor_reduction = j.value("pass razor reduction", options.pass_razor_reduction);
             options.pass_razor_guard = j.value("pass razor guard", options.pass_razor_guard);
             options.nohole_razor = j.value("nohole razor", options.nohole_razor);
             options.nohole_razor_depth = j.value("nohole razor depth", options.nohole_razor_depth);
             options.nohole_razor_guard = j.value("nohole razor guard", options.nohole_razor_guard);
             options.nohole_razor_reduction = j.value("nohole razor reduction", options.nohole_razor_reduction);
             options.proxy_nohole = j.value("proxy nohole", options.proxy_nohole);
             options.verify_tests = j.value("verify tests", options.verify_tests);
             options.iir = j.value("iir", options.iir);
             options.iir_depth = j.value("iir depth", options.iir_depth);
             options.cutnode_lmr = j.value("cutnode lmr", options.cutnode_lmr);
             options.improving = j.value("improving", options.improving);
             options.corr_hist = j.value("corr hist", options.corr_hist);
             options.corr_scale = j.value("corr scale", options.corr_scale);
             options.se = j.value("se", options.se);
             options.se_depth = j.value("se depth", options.se_depth);
             options.se_margin = j.value("se margin", options.se_margin);
             options.se_double = j.value("se double", options.se_double);
             options.multicut = j.value("multicut", options.multicut);
             options.ext_max = j.value("ext max", options.ext_max);
             options.cont_hist = j.value("cont hist", options.cont_hist);
             options.piece_lmr = j.value("piece lmr", options.piece_lmr);
             options.family_lmr = j.value("family lmr", options.family_lmr);
             options.gate_lmr = j.value("gate lmr", options.gate_lmr);
             options.gate_min = j.value("gate min", options.gate_min);
             options.gate_depth = j.value("gate depth", options.gate_depth);
             options.gate_ext = j.value("gate ext", options.gate_ext);
             options.gate_ext_min = j.value("gate ext min", options.gate_ext_min);
             options.opp_lmr = j.value("opp lmr", options.opp_lmr);
             options.opp_lmr_k = j.value("opp lmr k", options.opp_lmr_k);
             options.split_solve = j.value("split solve", options.split_solve);
             options.split_max_free = j.value("split max free", options.split_max_free);
             options.split_budget = j.value("split budget", options.split_budget);
             options.pocket = j.value("pocket", options.pocket);
             options.pocket_depth = j.value("pocket depth", options.pocket_depth);
             options.qs = j.value("qs", options.qs);
             options.qs_depth = j.value("qs depth", options.qs_depth);
             options.qs_moves = j.value("qs moves", options.qs_moves);
             options.sibling_ref = j.value("sibling ref", options.sibling_ref);
             options.tt_fragile = j.value("tt fragile", options.tt_fragile);
             options.tt_fragile_ply = j.value("tt fragile ply", options.tt_fragile_ply);
             options.root_audit = j.value("root audit", options.root_audit);
             options.root_audit_depth = j.value("root audit depth", options.root_audit_depth);
             options.root_audit_frac = j.value("root audit frac", options.root_audit_frac);
             options.sym = j.value("sym", options.sym);
             options.sym_depth = j.value("sym depth", options.sym_depth);
             options.sym_window = j.value("sym window", options.sym_window);
             options.sym_thr = j.value("sym thr", options.sym_thr);
             options.reserve = j.value("reserve", options.reserve);
             options.reserve_max = j.value("reserve max", options.reserve_max);
             options.lmr_log = j.value("lmr log", options.lmr_log);
             options.lmr_table = j.value("lmr table", options.lmr_table);
             options.fly_reply_relative = j.value("fly reply relative", options.fly_reply_relative);
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
