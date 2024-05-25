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
#include "html_observer.h"
#include "do_nothing_observer.h"
#include "cem_test.h"
#include "basic_minmax_player.h"
#include <iomanip>
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
    // test::play(std::make_unique<MCTSMemPlayer>(400000),
    //            std::make_unique<HumanPlayer>(WebsocketServerSync::create("127.0.0.1", 4242)), 
    //            DoNothingObserver());
    // test::play(std::make_unique<MCTSMemPlayer>(100000),
    //             std::make_unique<MCTSMemPlayer>(2000000), 
    //             HtmlObserver("127.0.0.1", 4242));
    // test::play(std::make_unique<BasicMinMaxPlayer>(4),
    //             std::make_unique<BasicMinMaxPlayer>(4), 
    //             HtmlObserver("127.0.0.1", 4242));
    test::cem_beale_function();
    test::cem_sphere_function();
    test::cem_rastrigin_function(); 
    // test::play(std::make_unique<BasicMinMaxPlayer>(4),
    //            std::make_unique<MCTSMemPlayer>(500000),                                             
    //            HtmlObserver("127.0.0.1", 4242));
    //BasicMinMaxPlayer::learn_weights();   
}
