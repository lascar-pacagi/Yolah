#include "player.h"
#include <functional>
#include <map>
#include "random_player.h"
#include "MCTS_mem_player.h"
#include "MCTS_player.h"
#include "basic_minmax_player.h"
#include "minmax_player_v1.h"
#include "minmax_player_v2.h"
#include "minmax_player_v3.h"
#include "minmax_player_v4.h"
#include "human_player.h"
#include "monte_carlo_player.h"
#include <stdexcept>

using std::unique_ptr, std::string, std::make_unique, std::invalid_argument;

unique_ptr<Player> Player::create(const json& j) {
    const std::map<string, std::function<unique_ptr<Player>(const json&)>> m {
        { 
            "RandomPlayer", 
            [](const json& j) {
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
            }
        },
        {
            "MCTSMemPlayer",
            [](const json& j) {
                if (!j.contains("nb threads")) {
                    throw invalid_argument("nb threads key expected");
                }
                if (!j.contains("microseconds")) {
                    throw invalid_argument("microseconds key expected");
                }
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                if (!j["microseconds"].is_number()) {
                    throw invalid_argument("number expected for microseconds");
                }
                return make_unique<MCTSMemPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MCTSPlayer",
            [](const json& j) {                
                if (!j.contains("nb threads")) {
                    throw invalid_argument("nb threads key expected");
                }
                if (!j.contains("microseconds")) {
                    throw invalid_argument("microseconds key expected");
                }
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                if (!j["microseconds"].is_number()) {
                    throw invalid_argument("number expected for microseconds");
                }
                return make_unique<MCTSPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MonteCarloPlayer",
            [](const json& j) {  
                if (!j.contains("nb threads")) {
                    throw invalid_argument("nb threads key expected");
                }
                if (!j.contains("microseconds")) {
                    throw invalid_argument("microseconds key expected");
                }              
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                if (!j["microseconds"].is_number()) {
                    throw invalid_argument("number expected for microseconds");
                }
                return make_unique<MonteCarloPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "BasicMinMaxPlayer",
            [](const json& j) {
                if (!j.contains("depth")) {
                    throw invalid_argument("depth key expected");
                }                
                if (!j["depth"].is_number()) {
                    throw invalid_argument("number expected for depth");
                } 
                return make_unique<BasicMinMaxPlayer>(j["depth"].get<uint8_t>());
            }
        },
        {
            "MinMaxPlayerV1",
            [](const json& j) {                                   
                if (!j.contains("microseconds")) {
                    throw invalid_argument("microseconds key expected");
                }             
                if (!j.contains("tt size")) {
                    throw invalid_argument("tt size key expected");
                }
                if (!j["microseconds"].is_number()) {
                    throw invalid_argument("number expected for microseconds");
                }
                if (!j["tt size"].is_number()) {
                    throw invalid_argument("number expected for tt size");
                }
                return make_unique<MinMaxPlayerV1>(j["microseconds"].get<uint64_t>(), j["tt size"].get<size_t>());
            }
        },
        {
            "MinMaxPlayerV2",
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
                return make_unique<MinMaxPlayerV2>(j["microseconds"].get<uint64_t>(), 
                                                   j["tt size"].get<size_t>(),
                                                   j["nb moves at full depth"].get<size_t>(),
                                                   j["late move reduction"].get<uint8_t>());
            }
        },
        {
            "MinMaxPlayerV3",
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
                if (!j.contains("null move reduction")) {
                    throw invalid_argument("null move reduction key expected");
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
                if (!j["null move reduction"].is_number()) {
                    throw invalid_argument("number expected in null move reduction");
                }
                return make_unique<MinMaxPlayerV3>(j["microseconds"].get<uint64_t>(), 
                                                   j["tt size"].get<size_t>(),
                                                   j["nb moves at full depth"].get<size_t>(),
                                                   j["late move reduction"].get<uint8_t>(),
                                                   j["null move reduction"].get<uint8_t>());
            }
        },
        {
            "MinMaxPlayerV4",
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
                return make_unique<MinMaxPlayerV4>(j["microseconds"].get<uint64_t>(), 
                                                   j["tt size"].get<size_t>(),
                                                   j["nb moves at full depth"].get<size_t>(),
                                                   j["late move reduction"].get<uint8_t>());
            }
        },
    };
    return m.at(j["name"].get<string>())(j);
}
