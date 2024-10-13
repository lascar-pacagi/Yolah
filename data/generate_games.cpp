#include "generate_games.h"
#include "misc.h"
#include <thread>
#include <set>
#include <string>

namespace data {
    void generate_games(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, 
                        size_t nb_random_moves, size_t nb_games_per_thread, size_t nb_threads) {
        auto first_n_moves_random = [](Yolah& yolah, uint64_t seed, size_t n, std::vector<Move>& moves_history) {
            PRNG prng(seed);
            Yolah::MoveList moves;
            size_t i = 0;
            while (!yolah.game_over()) {                 
                yolah.moves(moves);
                Move m = moves[prng.rand<size_t>() % moves.size()];
                moves_history.push_back(m);
                yolah.play(m);
                if (++i >= n) break;
            }
        };
        {
            std::vector<std::jthread> threads;
            std::mutex mutex;            
            for (size_t i = 0; i < nb_threads; i++) {
                threads.emplace_back([&, i]{                    
                    std::vector<json> configs{black->config(), white->config()};                    
                    for (size_t j = 0; j < nb_games_per_thread; j++) {
                        auto black = Player::create(configs[0]);
                        auto white = Player::create(configs[1]);
                        Yolah yolah;
                        std::vector<Move> moves_history;
                        if (nb_random_moves) {
                            first_n_moves_random(yolah, i * nb_games_per_thread + j, nb_random_moves, moves_history);
                        }
                        while (!yolah.game_over()) {
                            Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                            moves_history.push_back(m);
                            yolah.play(m);
                        }
                        black->game_over(yolah);
                        white->game_over(yolah);
                        const auto [black_score, white_score] = yolah.score();                        
                        {
                            std::lock_guard lock(mutex);
                            for (auto m : moves_history) os << m << ' ';
                            os << '(' << black_score << '/' << white_score << ')' << std::endl;
                        }
                    }
                });
            }
        }
    }  

    void setify(std::istream& is, std::ostream& os) {
        using namespace std;
        set<string> games;
        size_t number_of_duplicates = 0;
        while (is) {
            string line;
            getline(is, line);
            if (const auto [ignore, inserted] = games.insert(line); !inserted) {
                number_of_duplicates++;
            }
        }
        cerr << "duplicates: " << number_of_duplicates << '\n';
        for (const auto& game : games) {
            os << game << '\n';
        }
    }
}