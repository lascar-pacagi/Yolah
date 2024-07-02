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

namespace po = boost::program_options;
using std::cout;

int main() {
    magic::init();
    //test::play_random_game();
    //test::play_random_games(1000000);
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
    // heuristic::learn_weights(std::make_unique<heuristic::NelderMeadLearner>([](const std::vector<double>& weights) {
    //     return std::make_unique<BasicMinMaxPlayer>(4, [&](uint8_t player, const Yolah& yolah) {
    //         assert(weights.size() == heuristic::NB_WEIGHTS);
    //         std::array<double, heuristic::NB_WEIGHTS> weights1;
    //         for (size_t i = 0; i < heuristic::NB_WEIGHTS; i++) {
    //             weights1[i] = weights[i];
    //         }
    //         return heuristic::eval(player, yolah, weights1);
    //     });
    // }));
    // ClientPlayer player(std::make_unique<MCTSMemPlayer>(400000),
    //                     WebsocketClientSync::create("127.0.0.1", 8001));
    // player.run();
}
