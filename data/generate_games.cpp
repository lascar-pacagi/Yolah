#include "generate_games.h"
#include "misc.h"
#include <thread>
#include <set>
#include <string>
#include <sstream>
#include <filesystem>

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

    void generate_games2(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, 
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
        constexpr size_t BLOCK = 1000;
        const size_t N = (nb_games_per_thread + BLOCK - 1) / BLOCK;
        {
            std::vector<std::jthread> threads;
            std::mutex mutex;            
            for (size_t i = 0; i < nb_threads; i++) {
                threads.emplace_back([&, i]{                    
                    std::vector<json> configs{black->config(), white->config()};                    
                    for (size_t j = 0; j < N; j++) {
                        std::vector<std::vector<Move>> block_history;
                        std::vector<std::pair<uint16_t, uint16_t>> block_scores;
                        for (size_t k = 0; k < BLOCK; k++) {
                            auto black = Player::create(configs[0]);
                            auto white = Player::create(configs[1]);
                            Yolah yolah;
                            std::vector<Move> moves_history;
                            if (nb_random_moves) {
                                first_n_moves_random(yolah, i * N * BLOCK + j * BLOCK + k, nb_random_moves, moves_history);
                            }
                            while (!yolah.game_over()) {
                                Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                                moves_history.push_back(m);
                                yolah.play(m);
                            }
                            black->game_over(yolah);
                            white->game_over(yolah);
                            const auto [black_score, white_score] = yolah.score();
                            block_history.push_back(moves_history);
                            block_scores.emplace_back(black_score, white_score);                        
                        }                        
                        {
                            std::lock_guard lock(mutex);
                            for (size_t i = 0; i < BLOCK; i++) {
                                for (auto m : block_history[i]) os << m << ' ';
                                const auto [black_score, white_score] = block_scores[i];
                                os << '(' << black_score << '/' << white_score << ')' << std::endl;
                            }
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

    void encode_game(const std::vector<Move>& moves, int nb_moves, int nb_random_moves, int black_score, int white_score, uint8_t* encoding) {
        encoding[0] = nb_moves;
        encoding[1] = nb_random_moves;
        for (int i = 0, k = 2; i < nb_moves; i++, k+=2) {
            encoding[k] = moves[i].from_sq();
            encoding[k + 1] = moves[i].to_sq();
        }
        encoding[2 * nb_moves + 2] = black_score;
        encoding[2 * nb_moves + 3] = white_score;        
    }

    void decode_game(uint8_t* encoding, std::vector<Move>& moves, int& nb_moves, int& nb_random_moves, int& black_score, int& white_score) {
        nb_moves = encoding[0];
        nb_random_moves = encoding[1];
        for (int i = 0, k = 2; i < nb_moves; i++, k+=2) {
            moves[i] = Move(Square(encoding[k]), Square(encoding[k + 1]));
        }
        black_score = encoding[nb_moves * 2 + 2];
        white_score = encoding[nb_moves * 2 + 3];
    }

    void generate_games(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, const std::vector<int>& nb_random_moves, 
                        int nb_games, int nb_threads) 
    {
        auto first_n_moves_random = [](Yolah& yolah, PRNG& prng, int n, std::vector<Move>& moves_history) {
            Yolah::MoveList moves;
            int i = 0;
            while (!yolah.game_over()) {                 
                yolah.moves(moves);
                Move m = moves[prng.rand<size_t>() % moves.size()];
                moves_history[i] = m;
                yolah.play(m);
                if (++i >= n) break;
            }
        };
        {
            std::vector<std::jthread> threads;
            std::mutex mutex;

            for (int i = 0; i < nb_threads; i++) {
                threads.emplace_back([&, i]{                    
                    const std::vector<json> configs{black->config(), white->config()};                                        
                    std::vector<Move> moves(Yolah::MAX_NB_MOVES);
                    std::vector<uint8_t> games(nb_games * nb_random_moves.size() * (2 * Yolah::MAX_NB_MOVES + 4));                             
                    uint64_t games_size = 0;
                    PRNG prng(i * 1000000000ULL + std::chrono::system_clock::now().time_since_epoch().count());
                    for (int j = 0; j < nb_games; j++) {
                        for (const int nb_random: nb_random_moves) {
                            auto black = Player::create(configs[0]);
                            auto white = Player::create(configs[1]);
                            Yolah yolah;
                            if (nb_random) {
                                first_n_moves_random(yolah, prng, nb_random, moves);
                            }
                            int k = nb_random;
                            while (!yolah.game_over()) {
                                Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                                moves[k++] = m;
                                yolah.play(m);
                            }
                            black->game_over(yolah);
                            white->game_over(yolah);
                            const auto [black_score, white_score] = yolah.score();
                            //std::cout << k << ' ' << nb_random << ' ';
                            // for (int i = 0; i < k; i++) {
                            //     std::cout << moves[i] << ' ';
                            // }
                            // std::cout << '(' << black_score << ',' << white_score << ")\n";
                            encode_game(moves, k, nb_random, black_score, white_score, &games.data()[games_size]);
                            games_size += 2 * k + 4;                            
                        }
                    }
                    {
                        std::lock_guard lock(mutex);
                        os.write(reinterpret_cast<const char*>(games.data()), games_size);            
                    }
                });
            }
        }
    }

    void decode_games(const std::filesystem::path& path, std::ostream& os) {
        using namespace std;
        auto size = filesystem::file_size(path);
        vector<uint8_t> encoding(size);
        ifstream ifs(path, ios::binary);
        ifs.read(reinterpret_cast<char*>(encoding.data()), size);
        stringbuf buffer;
        ostream bos(&buffer);
        size_t i = 0;
        while (i < encoding.size()) {
            int nb_moves = encoding[i];
            int nb_random_moves = encoding[i + 1];
            bos << nb_moves << ' ' << nb_random_moves << ' ';
            int k = i + 2;
            for (int j = 0; j < nb_moves; j++, k += 2) {
                bos << Move(Square(encoding[k]), Square(encoding[k+1])) << ' ';
            }
            int black_score = encoding[k];
            int white_score = encoding[k + 1];
            bos << '(' << black_score << ',' << white_score << ")\n";
            i += nb_moves * 2 + 4;
        }
        os << buffer.str();
    }
}
