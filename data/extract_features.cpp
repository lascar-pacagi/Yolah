/**
 * Extract heuristic features from binary game files.
 *
 * Reads games in binary format, replays each position, extracts
 * heuristic features, and outputs training data for small_nnue.
 *
 * Output format (text):
 *   f1 f2 ... f24 p_black p_draw p_white
 *
 * Usage:
 *   ./extract_features game_file1.txt [game_file2.txt ...] > training_data.txt
 *
 * Compile:
 *   g++ -std=c++23 -O3 -march=native -I../game -I../misc -I../player \
 *       extract_features.cpp ../game/game.cpp ../game/magic.cpp ../game/zobrist.cpp \
 *       ../player/heuristic_features.cpp -o extract_features
 */

#include "game.h"
#include "move.h"
#include "magic.h"
#include "heuristic_features.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <cmath>

// Decode a single game from binary encoding
// Returns: (moves, nb_random_moves, black_score, white_score)
struct GameData {
    std::vector<Move> moves;
    int nb_random_moves;
    int black_score;
    int white_score;
};

size_t decode_game(const uint8_t* encoding, GameData& game) {
    int nb_moves = encoding[0];
    game.nb_random_moves = encoding[1];
    game.moves.clear();
    game.moves.reserve(nb_moves);

    int k = 2;
    for (int i = 0; i < nb_moves; i++, k += 2) {
        game.moves.emplace_back(Square(encoding[k]), Square(encoding[k + 1]));
    }

    game.black_score = encoding[k];
    game.white_score = encoding[k + 1];

    return nb_moves * 2 + 4;  // size of this game record
}

// Convert final score to target probabilities
// Using soft labels based on score difference
std::array<float, 3> score_to_target(int black_score, int white_score) {
    int diff = black_score - white_score;

    if (diff > 0) {
        // Black wins
        return {1.0f, 0.0f, 0.0f};
    } else if (diff < 0) {
        // White wins
        return {0.0f, 0.0f, 1.0f};
    } else {
        // Draw
        return {0.0f, 1.0f, 0.0f};
    }
}

// Softer version: probability proportional to score margin
std::array<float, 3> score_to_soft_target(int black_score, int white_score) {
    int diff = black_score - white_score;

    // Temperature for softmax-like distribution
    constexpr float temperature = 5.0f;

    float p_black = std::exp(diff / temperature);
    float p_white = std::exp(-diff / temperature);
    float p_draw = 1.0f;  // baseline for draw

    float sum = p_black + p_draw + p_white;
    return {p_black / sum, p_draw / sum, p_white / sum};
}

void process_game(const GameData& game, std::ostream& os, bool soft_targets = false) {
    Yolah yolah;

    // Compute target based on final score
    auto target = soft_targets
        ? score_to_soft_target(game.black_score, game.white_score)
        : score_to_target(game.black_score, game.white_score);

    // Replay the game and extract features at each position
    for (size_t i = 0; i < game.moves.size(); i++) {
        // Skip some early random moves (less informative)
        if (static_cast<int>(i) < game.nb_random_moves / 2) {
            yolah.play(game.moves[i]);
            continue;
        }

        // Extract features for current position
        auto features = heuristic_features::extract(yolah);

        // Output: features + target
        os << std::fixed << std::setprecision(4);
        for (size_t j = 0; j < features.size(); j++) {
            os << features[j];
            if (j < features.size() - 1) os << ' ';
        }
        os << ' ' << target[0] << ' ' << target[1] << ' ' << target[2] << '\n';

        // Play the move
        yolah.play(game.moves[i]);
    }
}

void process_file(const std::filesystem::path& path, std::ostream& os, bool soft_targets) {
    auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> data(size);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Cannot open: " << path << std::endl;
        return;
    }

    ifs.read(reinterpret_cast<char*>(data.data()), size);

    size_t pos = 0;
    size_t game_count = 0;

    while (pos < data.size()) {
        GameData game;
        size_t consumed = decode_game(&data[pos], game);
        process_game(game, os, soft_targets);
        pos += consumed;
        game_count++;
    }

    std::cerr << "Processed " << game_count << " games from " << path.filename() << std::endl;
}

int main(int argc, char* argv[]) {
    magic::init();  // Initialize magic bitboards

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [-soft] game_file1 [game_file2 ...]\n";
        std::cerr << "  -soft: Use soft targets based on score margin\n";
        return 1;
    }

    bool soft_targets = false;
    int file_start = 1;

    if (std::string(argv[1]) == "-soft") {
        soft_targets = true;
        file_start = 2;
    }

    std::cerr << "Using " << (soft_targets ? "soft" : "hard") << " targets\n";

    for (int i = file_start; i < argc; i++) {
        process_file(argv[i], std::cout, soft_targets);
    }

    return 0;
}
