#include <iostream>
#include "magic.h"
#include "random_game.h"
#include "game.h"

using std::cout;

int main() {
    magic::init();
    //play_random_game();
    //play_random_games(100000);
    Yolah yolah;
    cout << yolah.to_json() << '\n';
    std::stringstream ss;
    ss << yolah.to_json();
    yolah = Yolah::from_json(ss);
    cout << yolah.to_json() << '\n';
    cout << yolah << '\n';
}
