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

using std::unique_ptr, std::string, std::make_unique;

unique_ptr<Player> Player::create(const json& j) {
    const std::map<string, std::function<unique_ptr<Player>(const json&)>> m {
        { 
            "RandomPlayer", 
            [](const json& j) {
                if (j["seed"].get<string>() == "none") {
                    return make_unique<RandomPlayer>();
                } 
                return make_unique<RandomPlayer>(j["seed"].get<uint64_t>());
            }
        },
        {
            "MCTSMemPlayer",
            [](const json& j) {
                size_t nb_threads = std::thread::hardware_concurrency();
                if (j["nb threads"].get<string>() != "hardware concurrency") {
                    nb_threads = j["nb threads"].get<size_t>();
                } 
                return make_unique<MCTSMemPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MCTSPlayer",
            [](const json& j) {
                size_t nb_threads = std::thread::hardware_concurrency();
                if (j["nb threads"].get<string>() != "hardware concurrency") {
                    nb_threads = j["nb threads"].get<size_t>();
                } 
                return make_unique<MCTSPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "MonteCarloPlayer",
            [](const json& j) {
                size_t nb_threads = std::thread::hardware_concurrency();
                if (j["nb threads"].get<string>() != "hardware concurrency") {
                    nb_threads = j["nb threads"].get<size_t>();
                } 
                return make_unique<MonteCarloPlayer>(j["microseconds"].get<uint64_t>(), nb_threads);
            }
        },
        {
            "BasicMinMaxPlayer",
            [](const json& j) { 
                return make_unique<BasicMinMaxPlayer>(j["depth"].get<uint16_t>());
            }
        },
    };
    return m.at(j["name"].get<string>())(j);
}