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
#include <stdexcept>

using std::unique_ptr, std::string, std::make_unique, std::invalid_argument;

unique_ptr<Player> Player::create(const json& j) {
    const std::map<string, std::function<unique_ptr<Player>(const json&)>> m {
        { 
            "RandomPlayer", 
            [](const json& j) {
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
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                return make_unique<MCTSMemPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MCTSPlayer",
            [](const json& j) {                
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                return make_unique<MCTSPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MonteCarloPlayer",
            [](const json& j) {                
                size_t nb_threads;
                if (j["nb threads"].is_number()) {
                    nb_threads = j["nb threads"].get<size_t>();
                } else if (j["nb threads"].get<string>() == "hardware concurrency") {
                    nb_threads = std::thread::hardware_concurrency();
                } else {
                    throw invalid_argument("hardware concurrency expected in nb threads");
                }
                return make_unique<MonteCarloPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "BasicMinMaxPlayer",
            [](const json& j) {
                if (!j["depth"].is_number()) {
                    throw invalid_argument("number expected for depth");
                } 
                return make_unique<BasicMinMaxPlayer>(j["depth"].get<uint16_t>());
            }
        },
    };
    return m.at(j["name"].get<string>())(j);
}