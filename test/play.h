#ifndef PLAY_H
#define PLAY_H
#include "player.h"
#include <memory>
#include "observer.h"

namespace test {
    void play(std::unique_ptr<Player> player1, std::unique_ptr<Player> player2, Observer auto&& display) {
        using namespace std;
        Yolah yolah;  
        while (!yolah.game_over()) {
            display(yolah);
            auto current_player = yolah.current_player();
            Move m = (current_player == Yolah::BLACK ? player1 : player2)->play(yolah);
            display(current_player, m);
            yolah.play(m);
        }
        display(yolah);
        player1->game_over(yolah);
        player2->game_over(yolah);
    };
    void play(std::unique_ptr<Player> player1, std::unique_ptr<Player> player2, size_t nb_games);
}

#endif
