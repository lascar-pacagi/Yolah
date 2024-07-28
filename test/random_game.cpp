#include "random_game.h"
#include "game.h"
#include "misc.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <execution>
#include <thread>

using std::cout;

namespace test {
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
        using std::atomic;
        atomic<double> black_scores = 0; 
        atomic<double> white_scores = 0;
        atomic<size_t> black_nb_victories = 0;
        atomic<size_t> white_nb_victories = 0;
        atomic<double> game_lengths = 0;
        atomic<size_t> max_nb_moves = 0;
        size_t nb_threads = std::thread::hardware_concurrency();
        size_t nb_games_per_thread = (n + nb_threads - 1) / nb_threads;
        {
            std::vector<std::jthread> threads(nb_threads);
            for (size_t i = 0; i < nb_threads; i++) {
                threads.emplace_back([&](auto){
                    PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
                    double black_scores_ = 0; 
                    double white_scores_ = 0;
                    size_t black_nb_victories_ = 0;
                    size_t white_nb_victories_ = 0;
                    double game_lengths_ = 0;
                    size_t max_nb_moves_ = 0;
                    for (size_t i = 0; i < nb_games_per_thread; i++) {
                        Yolah yolah;
                        Yolah::MoveList moves;
                        size_t k = 0;
                        while (!yolah.game_over()) {                 
                            yolah.moves(moves);
                            k++;
                            max_nb_moves_ = std::max(max_nb_moves_, moves.size());
                            Move m = moves[prng.rand<size_t>() % moves.size()];
                            yolah.play(m);
                        }
                        game_lengths_ += k;
                        const auto [black_score, white_score] = yolah.score();
                        black_scores_ += black_score;
                        white_scores_ += white_score;
                        if (black_score > white_score) {
                            black_nb_victories_++;
                        } else if (white_score > black_score) {
                            white_nb_victories_++;
                        }
                    }
                    black_scores += black_scores_;
                    white_scores += white_scores_;
                    black_nb_victories += black_nb_victories_;
                    white_nb_victories += white_nb_victories_;
                    game_lengths += game_lengths_;
                    max_nb_moves = std::max(max_nb_moves.load(), max_nb_moves_);
                });
            }
        }
        n = nb_threads * nb_games_per_thread;
        cout << "[ mean black score / mean white score ]: " << (black_scores / n) << "/" << (white_scores / n) << '\n';
        cout << "[  # of black wins / # of white wins  ]: " << black_nb_victories << "/" << white_nb_victories << '\n';
        cout << "[            # of draws               ]: " << (n - black_nb_victories - white_nb_victories) << '\n';
        cout << "[         mean game length            ]: " << (game_lengths / n) << '\n';
        cout << "[          max # of moves             ]: " << max_nb_moves << '\n';
    }
}

