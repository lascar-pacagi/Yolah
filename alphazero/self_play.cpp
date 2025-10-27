#include "self_play.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <filesystem>

namespace alphazero {

// ============================================================================
// SelfPlayManager Implementation
// ============================================================================

SelfPlayManager::SelfPlayManager(std::shared_ptr<AlphaZeroNetwork> network,
                                 const AlphaZeroMCTS::Config& mcts_config,
                                 const SelfPlayConfig& selfplay_config)
    : network_(network), mcts_config_(mcts_config),
      selfplay_config_(selfplay_config), stop_workers_(false) {

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(selfplay_config_.output_dir);
}

SelfPlayManager::~SelfPlayManager() {
    stop_workers_ = true;
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void SelfPlayManager::generate_games() {
    std::cout << "Starting self-play generation: " << selfplay_config_.num_games
              << " games with " << selfplay_config_.num_threads << " threads\n";

    games_to_play_ = selfplay_config_.num_games;
    stop_workers_ = false;

    // Launch worker threads
    worker_threads_.clear();
    for (int i = 0; i < selfplay_config_.num_threads; ++i) {
        worker_threads_.emplace_back(&SelfPlayManager::worker_function, this);
    }

    // Wait for all games to complete
    for (auto& thread : worker_threads_) {
        thread.join();
    }

    std::cout << "\nSelf-play generation completed!\n";
    std::cout << "Games played: " << stats_.games_completed << "\n";
    std::cout << "Black wins: " << stats_.black_wins << " ("
              << (100.0 * stats_.black_wins / stats_.games_completed) << "%)\n";
    std::cout << "White wins: " << stats_.white_wins << " ("
              << (100.0 * stats_.white_wins / stats_.games_completed) << "%)\n";
    std::cout << "Draws: " << stats_.draws << " ("
              << (100.0 * stats_.draws / stats_.games_completed) << "%)\n";
    std::cout << "Average moves per game: "
              << (stats_.total_moves / stats_.games_completed) << "\n";
    std::cout << "Total training examples: " << training_examples_.size() << "\n";
}

void SelfPlayManager::worker_function() {
    while (games_to_play_.fetch_sub(1) > 0 && !stop_workers_) {
        play_game();
    }
}

void SelfPlayManager::play_game() {
    Yolah state;
    AlphaZeroMCTS mcts(network_, mcts_config_);

    std::vector<TrainingExample> game_examples;
    std::vector<float> recent_values;  // For resignation check

    int move_count = 0;
    bool resigned = false;
    uint8_t resigned_player = 0;

    while (!state.game_over() && move_count < selfplay_config_.max_moves_per_game) {
        // Adjust temperature based on move number
        float temperature = (move_count < selfplay_config_.temperature_threshold) ? 1.0f : 0.01f;
        mcts.config_.temperature = temperature;

        // Search for best move
        Move best_move = mcts.search(state);
        std::vector<std::pair<Move, float>> action_probs = mcts.get_action_probs();

        // Store training example
        TrainingExample example;
        example.state = state;
        example.policy = action_probs_to_policy(action_probs, state);
        // Value will be filled in after game completes
        game_examples.push_back(example);

        // Get evaluation for resignation check
        NetworkOutput output = network_->evaluate(state);
        recent_values.push_back(output.value);
        if (recent_values.size() > selfplay_config_.resign_consecutive_moves) {
            recent_values.erase(recent_values.begin());
        }

        // Check for resignation
        if (selfplay_config_.resign_enabled && should_resign(recent_values)) {
            resigned = true;
            resigned_player = state.current_player();
            break;
        }

        // Play the move
        state.play(best_move);
        mcts.advance_to_child(best_move);

        move_count++;
    }

    // Determine game outcome
    float game_result = 0.0f;
    bool is_draw = false;

    if (resigned) {
        // Resigned player loses
        game_result = -1.0f;  // From perspective of resigned player
        if (resigned_player == Yolah::BLACK) {
            stats_.white_wins++;
            stats_.black_resignations++;
        } else {
            stats_.black_wins++;
            stats_.white_resignations++;
        }
    } else if (state.game_over()) {
        auto [black_score, white_score] = state.score();
        if (black_score > white_score) {
            stats_.black_wins++;
        } else if (white_score > black_score) {
            stats_.white_wins++;
        } else {
            stats_.draws++;
            is_draw = true;
        }

        // Set game result from final position perspective
        if (black_score > white_score) {
            game_result = (state.current_player() == Yolah::BLACK) ? 1.0f : -1.0f;
        } else if (white_score > black_score) {
            game_result = (state.current_player() == Yolah::WHITE) ? 1.0f : -1.0f;
        } else {
            game_result = 0.0f;
        }
    } else {
        // Max moves reached - draw
        stats_.draws++;
        is_draw = true;
        game_result = 0.0f;
    }

    // Fill in values for training examples
    float value = game_result;
    for (auto it = game_examples.rbegin(); it != game_examples.rend(); ++it) {
        it->value = value;
        value = -value;  // Flip for alternating players
    }

    // Add to training data
    {
        std::lock_guard<std::mutex> lock(training_data_mutex_);
        training_examples_.insert(training_examples_.end(),
                                 game_examples.begin(), game_examples.end());
    }

    // Update statistics
    stats_.games_completed++;
    stats_.total_moves += move_count;

    // Print progress
    if (stats_.games_completed % 10 == 0) {
        std::cout << "Games completed: " << stats_.games_completed
                  << " / " << selfplay_config_.num_games << "\r" << std::flush;
    }
}

bool SelfPlayManager::should_resign(const std::vector<float>& recent_values) const {
    if (recent_values.size() < static_cast<size_t>(selfplay_config_.resign_consecutive_moves)) {
        return false;
    }

    // Check if all recent values are below threshold
    for (float value : recent_values) {
        if (value >= selfplay_config_.resign_threshold) {
            return false;
        }
    }

    return true;
}

std::vector<float> SelfPlayManager::action_probs_to_policy(
    const std::vector<std::pair<Move, float>>& action_probs,
    const Yolah& state) const {

    std::vector<float> policy(NetworkConfig::MAX_MOVES, 0.0f);

    // Map each move to its index in policy vector
    // This is simplified - in practice need proper move encoding
    Yolah::MoveList legal_moves;
    state.moves(legal_moves);

    for (size_t i = 0; i < action_probs.size() && i < legal_moves.size(); ++i) {
        policy[i] = action_probs[i].second;
    }

    return policy;
}

void SelfPlayManager::save_training_data(const std::string& filename) {
    std::cout << "Saving training data to " << filename << "...\n";

    TrainingDataWriter writer(filename);

    {
        std::lock_guard<std::mutex> lock(training_data_mutex_);
        for (const auto& example : training_examples_) {
            writer.write_example(example);
        }
    }

    writer.flush();
    std::cout << "Saved " << training_examples_.size() << " training examples.\n";
}

// ============================================================================
// TrainingDataWriter Implementation
// ============================================================================

TrainingDataWriter::TrainingDataWriter(const std::string& filename)
    : file_(filename, std::ios::binary) {
    if (!file_) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    buffer_.reserve(BUFFER_SIZE);
}

TrainingDataWriter::~TrainingDataWriter() {
    flush();
}

void TrainingDataWriter::write_example(const TrainingExample& example) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Binary format:
    // - Game state (compact representation)
    // - Policy vector (75 floats)
    // - Value (1 float)

    // Write game state
    uint64_t black = example.state.bitboard(Yolah::BLACK);
    uint64_t white = example.state.bitboard(Yolah::WHITE);
    uint64_t empty = example.state.empty_bitboard();
    uint16_t ply = example.state.nb_plies();

    auto write_data = [this](const void* data, size_t size) {
        const char* bytes = static_cast<const char*>(data);
        buffer_.insert(buffer_.end(), bytes, bytes + size);
    };

    write_data(&black, sizeof(black));
    write_data(&white, sizeof(white));
    write_data(&empty, sizeof(empty));
    write_data(&ply, sizeof(ply));

    // Write policy
    write_data(example.policy.data(), example.policy.size() * sizeof(float));

    // Write value
    write_data(&example.value, sizeof(example.value));

    // Flush if buffer is large enough
    if (buffer_.size() >= BUFFER_SIZE) {
        flush();
    }
}

void TrainingDataWriter::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffer_.empty()) {
        file_.write(buffer_.data(), buffer_.size());
        buffer_.clear();
    }
}

// ============================================================================
// TrainingDataReader Implementation
// ============================================================================

TrainingDataReader::TrainingDataReader(const std::string& filename)
    : file_(filename, std::ios::binary), num_examples_(0), current_index_(0) {

    if (!file_) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Calculate number of examples from file size
    file_.seekg(0, std::ios::end);
    size_t file_size = file_.tellg();
    file_.seekg(0, std::ios::beg);

    // Size per example: 3*8 + 2 + 75*4 + 4 = 332 bytes
    size_t example_size = 3 * sizeof(uint64_t) + sizeof(uint16_t) +
                         NetworkConfig::MAX_MOVES * sizeof(float) + sizeof(float);

    num_examples_ = file_size / example_size;
}

TrainingDataReader::~TrainingDataReader() = default;

bool TrainingDataReader::read_example(TrainingExample& example) {
    if (current_index_ >= num_examples_) {
        return false;
    }

    // Read game state
    uint64_t black, white, empty;
    uint16_t ply;

    file_.read(reinterpret_cast<char*>(&black), sizeof(black));
    file_.read(reinterpret_cast<char*>(&white), sizeof(white));
    file_.read(reinterpret_cast<char*>(&empty), sizeof(empty));
    file_.read(reinterpret_cast<char*>(&ply), sizeof(ply));

    // Reconstruct game state
    example.state.set_state(black, white, empty, 0, 0, ply);

    // Read policy
    example.policy.resize(NetworkConfig::MAX_MOVES);
    file_.read(reinterpret_cast<char*>(example.policy.data()),
              NetworkConfig::MAX_MOVES * sizeof(float));

    // Read value
    file_.read(reinterpret_cast<char*>(&example.value), sizeof(example.value));

    current_index_++;
    return true;
}

void TrainingDataReader::reset() {
    file_.clear();
    file_.seekg(0, std::ios::beg);
    current_index_ = 0;
}

} // namespace alphazero
