#include "play.h"
#include <iostream>
#include "BS_thread_pool.h"
#include <iomanip>
#include "misc.h"
#include "indicators.h"
#include <atomic>
#include <execution>
#include <mutex>
#include <thread>

using std::cout, std::size_t;

namespace {
    thread_local PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
}

namespace test {
    void play(std::unique_ptr<Player> p1, std::unique_ptr<Player> p2, size_t nb_games) {
        using namespace indicators;
        ProgressBar bar{
            option::BarWidth{50},
            option::Start{"["},
            option::Fill{"="},
            option::Lead{">"},
            option::Remainder{" "},
            option::End{"]"},
            option::PostfixText{""},
            option::ForegroundColor{Color::green},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
        };
        std::atomic<double> p1_black_victories = 0;
        std::atomic<double> p1_white_victories = 0;
        std::atomic<double> p2_black_victories = 0;
        std::atomic<double> p2_white_victories = 0;
        std::atomic<double> draws = 0;
        std::atomic<int> iter = 0;
        size_t nb_threads = std::thread::hardware_concurrency();
        size_t n = (nb_games + nb_threads - 1) / nb_threads;
        nb_games = 2 * n * nb_threads;
        {
            std::vector<std::jthread> threads;
            std::mutex mutex;            
            for (size_t i = 0; i < nb_threads; i++) {
                threads.emplace_back([&]{
                    double p1_black_victories_ = 0;
                    double p1_white_victories_ = 0;
                    double p2_black_victories_ = 0;
                    double p2_white_victories_ = 0;
                    double draws_ = 0;
                    std::vector<json> configs{p1->config(), p2->config()};
                    for (size_t i = 0; i < 2; i++) {
                        for (size_t j = 0; j < n; j++) {                       
                            auto black = Player::create(configs[0]);
                            auto white = Player::create(configs[1]);
                            Yolah yolah;
                            while (!yolah.game_over()) {                 
                                Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                                yolah.play(m);
                            }
                            black->game_over(yolah);
                            white->game_over(yolah);
                            const auto [black_score, white_score] = yolah.score();           
                            if (black_score > white_score) {
                                (i == 0 ? p1_black_victories_ : p2_black_victories_) += 1;
                            } else if (white_score > black_score) {
                                (i == 0 ? p2_white_victories_ : p1_white_victories_) += 1;
                            } else {
                                draws_ += 1;
                            }
                            ++iter;
                            {
                                std::lock_guard lock(mutex);
                                bar.set_progress(iter * 100 / nb_games);
                            }
                        }
                        std::swap(configs[0], configs[1]);
                    }
                    p1_black_victories += p1_black_victories_;
                    p1_white_victories += p1_white_victories_;
                    p2_black_victories += p2_black_victories_;
                    p2_white_victories += p2_white_victories_;
                    draws += draws_;
                });
            }
        }
        cout << "[ player 1 % of black victories ]: " << (p1_black_victories / nb_games * 100) << '\n';
        cout << "[ player 2 % of black victories ]: " << (p2_black_victories / nb_games * 100) << '\n';
        cout << "[ player 1 % of white victories ]: " << (p1_white_victories / nb_games * 100) << '\n';
        cout << "[ player 2 % of white victories ]: " << (p2_white_victories / nb_games * 100) << '\n';
        cout << "[          % of draws           ]: " << (draws / nb_games * 100) << '\n';
    }
}
