#ifndef GENERATE_GAMES_H
#define GENERATE_GAMES_H
#include <iostream>
#include <memory>
#include "player.h"
#include <vector>
#include <fstream>
#include <filesystem>

namespace data {
    void generate_games(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, size_t nb_random_moves, size_t nb_games_per_thread, size_t nb_threads = 1);
    void generate_games2(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, size_t nb_random_moves, size_t nb_games_per_thread, size_t nb_threads = 1);
    void encode_game(const std::vector<Move>& moves, int nb_moves, int nb_random_moves, int black_score, int white_score, uint8_t* encoding);
    void decode_game(uint8_t* encoding, std::vector<Move>& moves, int& nb_moves, int& nb_random_moves, int& black_score, int& white_score);
    void generate_games(std::ostream& os, std::unique_ptr<Player> black, std::unique_ptr<Player> white, const std::vector<int>& nb_random_moves, 
                        int nb_games, int nb_threads = 1);
    void setify(std::istream& is, std::ostream& os);
    void decode_games(const std::filesystem::path& path, std::ostream& os);
}

#endif