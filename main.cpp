#include <iostream>
#include "magic.h"
#include "play.h"
#include "game.h"
#include "random_game.h"
#include "random_player.h"
#include "player/human_player.h"
#include "monte_carlo_player.h"
#include "MCTS_player.h"
#include "MCTS_mem_player.h"
#include "misc.h"
#include "ascii_observer.h"
#include "html_observer.h"

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
    test::play(std::make_unique<MCTSMemPlayer>(4000000),
               std::make_unique<HumanPlayer>(WebsocketServerSync::create("127.0.0.1", 4242)), 
               AsciiObserver());
    // test::play(std::make_unique<MCTSMemPlayer>(400000),
    //             std::make_unique<RandomPlayer>(), 
    //             HtmlObserver("127.0.0.1", 4242));
}
