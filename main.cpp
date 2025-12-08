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
#include "tournament.h"
#include <format>
#include "logic_net_learning.h"
#include "compare_models.h"

namespace po = boost::program_options;
using std::cout, std::endl, std::string;

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();

    // test::play(Player::create(nlohmann::json::parse(std::ifstream("../config/mm_player.cfg"))),
    //             Player::create(nlohmann::json::parse(std::ifstream("../config/mm_nnue_quantized_player.cfg"))),
    //             2, 500);

    //test::tournament({"/Yolah/config/mc_player.cfg", "/Yolah/config/mcts_player.cfg", "/Yolah/config/mm_player.cfg", "/Yolah/config/mm_nnue_quantized_player.cfg", "/Yolah/config/mm_nnue_player.cfg"}, 4, 1000);
    //test::tournament({"../config/random_player.cfg", "../config/mc_player.cfg", "../config/mcts_player.cfg", "../config/mm_player.cfg", "../config/mm_nnue_quantized_player.cfg", "../config/mm_nnue_player.cfg"}, 4, 2);
    // auto input = std::ifstream("../data/games_7r_1s_bis_d.txt");
    // auto output = std::ofstream("../data/games_7r_1s_bis_d_symmetries.txt");
    //data::setify(input, output);
    //data::analyze_games(input, cout);
    //auto input = std::ifstream("../nnue/data/data_test/games.txt", std::ios::binary);
    const auto now = std::chrono::system_clock::now();
    const std::string timestamp = std::format("{:%Y_%m_%d_%H_%M_%S}", now);
    for (int i = 0; i < 4; i++) {
        std::string filename = "/mnt/games_" + timestamp + "_" + std::to_string(i) + ".txt";
        //std::string filename = "../nnue/data/data_test/games_" + timestamp + "_" + std::to_string(i) + ".txt";
        std::cout << filename << std::endl;
        std::ofstream output(filename, std::ios::binary);
        // data::generate_games(output, Player::create(nlohmann::json::parse(std::ifstream("/Yolah/config/mm_nnue_quantized_player.cfg"))), 
        //                     Player::create(nlohmann::json::parse(std::ifstream("/Yolah/config/mm_nnue_quantized_player.cfg"))), 
        //                     {0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, 300, 14);
        data::generate_games(output, Player::create(nlohmann::json::parse(std::ifstream("/Yolah/config/mm_nnue_quantized_player.cfg"))), 
                            Player::create(nlohmann::json::parse(std::ifstream("/Yolah/config/mm_nnue_quantized_player.cfg"))), 
                            {0, 1, 2, 3, 4, 5}, 300, 14);
        // data::generate_games(output, Player::create(nlohmann::json::parse(std::ifstream("../config/mm_nnue_quantized_player.cfg"))), 
        //                     Player::create(nlohmann::json::parse(std::ifstream("../config/mm_nnue_quantized_player.cfg"))), 
        //                     {2, 3, 4, 5, 10, 11, 16, 17}, 20, 4);
    }    
    // std::filesystem::path input("../data/games/games_2025_08_31_22_09_02.335269145_0.txt"); 
    // data::decode_games(input, cout);
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
    //data::generate_symmetries2("../data/games", "../nnue/data");
    //data::analyze_games2("../nnue/data", cout);
/*
    LogicNetLearning::Builder builder;
    try {
        builder
        .set_population_size(2000)
        .set_nb_iterations(100000)
        .set_network_depth(25)
        .set_crossover_rate(0.4)
        .set_mutation_rate(0.005)
        .set_selection_rate(0.02)
        // .set_logic_net_checkpoint_path("../nnue/model.txt")
        // .set_training_data_path("../nnue/data/data_test/games_2r.txt")
        .set_logic_net_checkpoint_path("/mnt/model.txt")
        .set_training_data_path("/Yolah/nnue/data/data_test/games_2r.txt")
        .build();
    } catch (const char* e) {
        std::cout << e << '\n';
    }
*/
    //compare_models("../nnue/nnue_quantized.txt", "../nnue/nnue_quantized2.txt", "../nnue/data", cout);

    //data::decode_games("../data/games/games_2025_08_31_22_09_02.335269145_0.txt", cout);
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
    //     cout << "Yolah 1.0 by Pascal Garcia\n";
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
    return EXIT_SUCCESS;
}
