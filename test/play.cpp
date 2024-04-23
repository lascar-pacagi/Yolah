#include "play.h"
#include <iostream>
#include "BS_thread_pool.h"
#include <iomanip>
#include "misc.h"
#include "indicators.h"

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
        double black_victories = 0;
        double white_victories = 0;
        double draws = 0;
        for (size_t i = 0; i < nb_games; i++) {
            Yolah yolah;
            while (!yolah.game_over()) {                 
                Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2)->play(yolah);
                yolah.play(m);
            }
            const auto [black_score, white_score] = yolah.score();           
            if (black_score > white_score) {
                black_victories++;
            } else if (white_score > black_score) {
                white_victories++;
            } else {
                draws++;
            }
            bar.set_progress(i * 100 / nb_games);
        }
        bar.set_progress(100);
        cout << "[ % of black victories ]: " << (black_victories / nb_games * 100) << '\n';
        cout << "[ % of white victories ]: " << (white_victories / nb_games * 100) << '\n';
        cout << "[      % of draws      ]: " << (draws / nb_games * 100) << '\n';
    }
}
