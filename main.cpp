#include <iostream>
#include "magic.h"
#include "play.h"
#include "game.h"
#include "random_game.h"
#include "random_player.h"
#include "human_player.h"
#include "monte_carlo_player.h"
#include "MCTS_player.h"
#include "MCTS_mem_player.h"
#include "misc.h"
#include "ascii_observer.h"
#include "ws_observer.h"
#include "do_nothing_observer.h"
#include "cem_test.h"
#include "nelder_mead_test.h"
#include "basic_minmax_player.h"
#include "heuristic_weights_learner.h"
#include <iomanip>
#include "client_player.h"
#include <boost/program_options.hpp>
#include "json.hpp"
#include <fstream>
#include "zobrist.h"
#include "variability_timer_multithreads.h"
#include "minmax_player.h"
#include "generate_games.h"
#include "analyze_games.h"
#include "generate_symmetries.h"
#include <filesystem>
#include <regex>
#include "minmax_nnue_quantized_player.h"

namespace po = boost::program_options;
using std::cout, std::endl, std::string;

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();

    test::play(Player::create(nlohmann::json::parse(std::ifstream("../config/mcts_player.cfg"))),
                Player::create(nlohmann::json::parse(std::ifstream("../config/mm_nnue_quantized_player.cfg"))),
                4, 500);

    // auto input = std::ifstream("../data/games_7r_1s_bis_d.txt");
    // auto output = std::ofstream("../data/games_7r_1s_bis_d_symmetries.txt");
    //data::setify(input, output);
    //data::analyze_games(input, cout);
    // data::generate_games(cout, Player::create(nlohmann::json::parse(std::ifstream("../config/mm_player.cfg"))), 
    //                     Player::create(nlohmann::json::parse(std::ifstream("../config/mm_player.cfg"))), 6, 10000, 12);

    // data::generate_games2(cout, std::make_unique<MinMaxPlayer>(1000000, 100, 2, 3, 7), 
    //                         std::make_unique<MinMaxPlayer>(1000000, 100, 2, 3, 7), 2, 50000, 20);

    //data::cut("/home/elucterio/Yolah/data", "/home/elucterio/Yolah/data/games");
    //data::generate_symmetries("/home/elucterio/Yolah/data/games", "/home/elucterio/Yolah/data/games");
    // const std::filesystem::path data_dir("../data");
    // std::regex re_games("^games((?!.*symmetries.*))", std::regex_constants::ECMAScript|std::regex_constants::multiline);
    // for (auto const& dir_entry : std::filesystem::directory_iterator(data_dir)) {
    //     auto path = dir_entry.path();
    //     if (!std::regex_search(path.filename().string(), re_games)) continue;      
    //     cout << path << endl;
    //     auto input = std::ifstream(path);
    //     const std::filesystem::path ext("symmetries.txt");
    //     path.replace_extension(ext);
    //     auto output = std::ofstream(path);
    //     data::generate_symmetries(input, output);
    // }

    // po::options_description general("General options");
    // general.add_options()
    // ("help", "produce help message")
    // ("version", "output the version number");
    
    // po::options_description client("Client options");
    // client.add_options()
    // ("server,s", po::value<string>()->default_value("127.0.0.1"), "server ip adress")
    // ("port,p", po::value<uint16_t>()->default_value(8001), "server port")
    // ("key,k", po::value<string>(), "join key, if not present create a new game and get the join and watch keys by the server")
    // ("player", po::value<string>(), "configuration file for player");
    
    // po::options_description evaluate("Evaluate AI options");
    // evaluate.add_options()
    // ("player1,1", po::value<string>(), "configuration file for first AI player")
    // ("player2,2", po::value<string>(), "configuration file for second AI player")
    // ("nb-random-moves,r", po::value<size_t>()->default_value(0), "number of random moves at the beginning of the game")
    // ("nb-games,n", po::value<size_t>()->default_value(100), "number of games for the evaluation");

    // po::options_description all("Allowed options");
    // all.add(general).add(client).add(evaluate);
    
    // po::variables_map vm;
    // po::store(po::parse_command_line(argc, argv, all), vm);
    // po::notify(vm);
    // if (vm.count("help")) {
    //     cout << all << "\n";
    //     return EXIT_SUCCESS;
    // }
    // if (vm.count("version")) {
    //     cout << "Yolah 0.1 by Pascal Garcia\n";
    //     return EXIT_SUCCESS;
    // }
    // if (vm.count("player")) {
    //     std::ifstream f(vm["player"].as<string>());
    //     nlohmann::json j = nlohmann::json::parse(f);    
    //     cout << j << '\n';
    //     ClientPlayer player(Player::create(j),
    //                         WebsocketClientSync::create(vm["server"].as<string>(), vm["port"].as<uint16_t>()));
    //     player.run(vm.count("key") ? std::optional<string>(vm["key"].as<string>()) : std::nullopt);
    // } else if (vm.count("player1") && vm.count("player2")) {
    //     test::play(Player::create(nlohmann::json::parse(std::ifstream(vm["player1"].as<string>()))),
    //                Player::create(nlohmann::json::parse(std::ifstream(vm["player2"].as<string>()))),
    //                vm["nb-random-moves"].as<size_t>(),
    //                vm["nb-games"].as<size_t>());
    // } else {
    //     cout << "wrong mix of options\n";
    //     return EXIT_FAILURE;
    // }
    // return EXIT_SUCCESS;
}
