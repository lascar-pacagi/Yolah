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
#include "minmax_player_v1.h"
#include "minmax_player_v2.h"
#include "minmax_player_v3.h"
#include "minmax_player_v4.h"

namespace po = boost::program_options;
using std::cout, std::string;

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    //test::play_random_game();
    //test::play_random_games(1000000000);
    // Yolah yolah;
    // cout << yolah.to_json() << '\n';
    // std::stringstream ss;
    // ss << yolah.to_json();
    // yolah = Yolah::from_json(ss);
    // cout << yolah.to_json() << '\n';
    // cout << yolah << '\n';   
    // test::play(std::make_unique<MCTSMemPlayer>(1000000),
    //            std::make_unique<HumanPlayer>(WebsocketServerSync::create("127.0.0.1", 4242)), 
    //            DoNothingObserver());
    // test::play(std::make_unique<MCTSMemPlayer>(100000),
    //             std::make_unique<MCTSMemPlayer>(2000000), 
    //             WSObserver("127.0.0.1", 4242));
    // test::play(std::make_unique<BasicMinMaxPlayer>(4),
    //             std::make_unique<BasicMinMaxPlayer>(4), 
    //             WSObserver("127.0.0.1", 4242));
    // test::cem_beale_function();
    // test::cem_sphere_function();
    // test::cem_rastrigin_function(); 
    // test::play(std::make_unique<BasicMinMaxPlayer>(4),
    //            std::make_unique<MCTSMemPlayer>(100000, 1),
    //            WSObserver("127.0.0.1", 4242));
    // test::play(std::make_unique<BasicMinMaxPlayer>(4),
    //            std::make_unique<MonteCarloPlayer>(500000),
    //            20);
    // test::nelder_mead_beale_function();
    // test::nelder_mead_sphere_function();
    // test::nelder_mead_rastrigin_function();
    heuristic::learn_weights(std::make_unique<heuristic::CrossEntropyMethodLearner>([](const std::vector<double>& weights) {
        return std::make_unique<MinMaxPlayerV2>(200000, 100, 4, 3, [&](uint8_t player, const Yolah& yolah) {
            assert(weights.size() == heuristic::NB_WEIGHTS);
            std::array<double, heuristic::NB_WEIGHTS> w;
            for (size_t i = 0; i < heuristic::NB_WEIGHTS; i++) {
                w[i] = weights[i];
            }
            return heuristic::eval(player, yolah, w);
        });
    }));
    // auto values = heuristic::sampling_heuristic_values(1000000);
    // for (int32_t v : values) {
    //     cout << v << ' ';
    // }
    // cout << '\n';
    // cout << values.size() << '\n';

    //test::variability_timer_multithreads(12, 2000000);

    // Yolah yolah;
    // Yolah::MoveList moves;
    // PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
    // auto print_score = [&]{
    //     const auto [black_score, white_score] = yolah.score();
    //     cout << "score: " << black_score << '/' << white_score << '\n';
    // };
    // while (!yolah.game_over()) {                 
    //     yolah.moves(moves);
    //     Move m = moves[prng.rand<size_t>() % moves.size()];        
    //     cout << yolah << '\n';
    //     print_score();
    //     cout << m << '\n';
    //     const auto [black_influence, white_influence] = heuristic::influence(yolah);
    //     cout << "black influence\n";
    //     cout << Bitboard::pretty(black_influence) << '\n';
    //     cout << "white influence\n";
    //     cout << Bitboard::pretty(white_influence) << '\n';
    //     std::string _;
    //     std::getline(std::cin, _);
    //     yolah.play(m);
    // }
    // cout << yolah << '\n';
    // print_score();

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
    return EXIT_SUCCESS;
}
