#include "human_player_test.h"
#include <iostream>

namespace test {
    Move HumanPlayer::play(Yolah yolah) {
        Move m;        
        while (std::cin >> m) {
            if (yolah.valid(m)) break;
            std::cout << "invalid move\n";
        } 
        return m;
    }
}
