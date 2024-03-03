#include "random_game.h"
#include "game.h"
#include "misc.h"
#include <iostream>
#include <chrono>
#include <algorithm>

using std::cout;

void play_random_game() {
    Yolah yolah;
    Yolah::MoveList moves;
    PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
    auto print_score = [&]{
        const auto [black_score, white_score] = yolah.score();
        cout << "score: " << black_score << '/' << white_score << '\n';
    };
    cout << yolah << '\n';
    while (!yolah.game_over()) {                 
        yolah.moves(moves);
        if (moves.size() == 0) continue;
        Move m = moves[prng.rand<size_t>() % moves.size()];        
        cout << yolah << '\n';
        print_score();
        cout << m << '\n';
        std::string _;
        std::getline(std::cin, _);
        yolah.play(m);
    }
    cout << yolah << '\n';
    print_score();
}

void play_random_games(size_t n) {
    double black_scores = 0, white_scores = 0;
    size_t black_nb_victories = 0, white_nb_victories = 0;
    double game_lengths = 0;
    size_t max_nb_moves = 0;
    Yolah::MoveList moves;
    PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
    for (size_t i = 0; i < n; i++) {
        Yolah yolah;
        int k = 0;
        while (!yolah.game_over()) {                 
            yolah.moves(moves);
            k++;
            if (moves.size() == 0) continue;
            max_nb_moves = std::max(max_nb_moves, moves.size());
            Move m = moves[prng.rand<size_t>() % moves.size()];
            yolah.play(m);
        }
        game_lengths += k;
        const auto [black_score, white_score] = yolah.score();
        black_scores += black_score;
        white_scores += white_score;
        if (black_score > white_score) {
            black_nb_victories++;
        } else if (white_score > black_score) {
            white_nb_victories++;
        }
    }
    cout << "[ mean black score / mean white score ]: " << (black_scores / n) << "/" << (white_scores / n) << '\n';
    cout << "[  # of black wins / # of white wins  ]: " << black_nb_victories << "/" << white_nb_victories << '\n';
    cout << "[            # of draws               ]: " << (n - black_nb_victories - white_nb_victories) << '\n';
    cout << "[         mean game length            ]: " << (game_lengths / n) << '\n';
    cout << "[          max # of moves             ]: " << max_nb_moves << '\n';
}
