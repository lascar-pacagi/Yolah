#include "tournament.h"
#include <iostream>
#include <iomanip>
#include "misc.h"
#include <atomic>
#include <execution>
#include <mutex>
#include <thread>
#include <tuple>
#include "player.h"
#include <fstream>
#include <random>

namespace test {
    void tournament(const std::vector<std::string>& players_configs, size_t nb_random_moves, size_t nb_games) {
        using namespace std;
        auto first_n_moves_random = [](Yolah& yolah, uint64_t seed, size_t n) {
            PRNG prng(seed);
            Yolah::MoveList moves;
            size_t i = 0;
            while (!yolah.game_over()) {                 
                yolah.moves(moves);
                Move m = moves[prng.rand<size_t>() % moves.size()];
                yolah.play(m);
                if (++i >= n) break;
            }
        };
        vector<unique_ptr<Player>> players;
        for (const string& cfg : players_configs) {
            players.push_back(Player::create(nlohmann::json::parse(ifstream(cfg))));
        }
        {
            std::mutex mutex;          
            for (size_t p1 = 0; p1 < players.size(); p1++) {
                for (size_t p2 = p1 + 1; p2 < players.size(); p2++) {
                    jthread([&, p1, p2]{
                        double p1_black_victories = 0;
                        double p1_white_victories = 0;
                        double p2_black_victories = 0;
                        double p2_white_victories = 0;
                        double draws = 0;
                        vector<json> configs{players[p1]->config(), players[p2]->config()};
                        for (size_t side = 0; side < 2; side++) {
                            for (size_t j = 0; j < nb_games; j++) {                       
                                auto black = Player::create(configs[0]);
                                auto white = Player::create(configs[1]);
                                Yolah yolah;
                                if (nb_random_moves) {
                                    first_n_moves_random(yolah, j, nb_random_moves);
                                }
                                while (!yolah.game_over()) {                 
                                    Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                                    yolah.play(m);
                                }
                                black->game_over(yolah);
                                white->game_over(yolah);
                                const auto [black_score, white_score] = yolah.score();           
                                if (black_score > white_score) {
                                    (side == 0 ? p1_black_victories : p2_black_victories) += 1;
                                } else if (white_score > black_score) {
                                    (side == 0 ? p2_white_victories : p1_white_victories) += 1;
                                } else {
                                    draws += 1;
                                }   
                            }
                            swap(configs[0], configs[1]);
                        }
                        {
                            size_t n = nb_games * 2;
                            std::lock_guard lock(mutex);
                            cout << "player 1:\n";
                            cout << players[p1]->info() << '\n';
                            cout << "player 2:\n";
                            cout << players[p2]->info() << '\n';
                            cout << "[ player 1 % of black victories ]: " << (p1_black_victories / n * 100) << '\n';
                            cout << "[ player 2 % of black victories ]: " << (p2_black_victories / n * 100) << '\n';
                            cout << "[ player 1 % of white victories ]: " << (p1_white_victories / n * 100) << '\n';
                            cout << "[ player 2 % of white victories ]: " << (p2_white_victories / n * 100) << '\n';
                            cout << "[          % of draws           ]: " << (draws / n * 100) << '\n';
                            cout << "[   player 1 % of victories     ]: " << ((p1_black_victories + p1_white_victories + draws / 2) / n * 100) << '\n';
                            cout << "[   player 2 % of victories     ]: " << ((p2_black_victories + p2_white_victories + draws / 2) / n * 100) << '\n';
                        }
                    });
                }
            }
        }        
    }
}
