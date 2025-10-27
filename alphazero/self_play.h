#ifndef SELF_PLAY_H
#define SELF_PLAY_H

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <fstream>
#include "game.h"
#include "alphazero_mcts.h"
#include "alphazero_network.h"

namespace alphazero {

// Self-play configuration
struct SelfPlayConfig {
    int num_games = 1000;                    // Number of self-play games to generate
    int num_threads = 8;                      // Number of parallel game workers
    int max_moves_per_game = 200;            // Maximum moves before declaring draw
    float temperature_threshold = 15;         // Move number to switch from temp=1 to temp=0
    bool resign_enabled = true;               // Enable resignation
    float resign_threshold = -0.9f;          // Resign if value below this
    int resign_consecutive_moves = 5;        // Consecutive moves below threshold before resigning
    std::string output_dir = "./selfplay_data"; // Directory to save training examples
};

// Self-play manager for generating training data
class SelfPlayManager {
public:
    SelfPlayManager(std::shared_ptr<AlphaZeroNetwork> network,
                   const AlphaZeroMCTS::Config& mcts_config,
                   const SelfPlayConfig& selfplay_config);

    ~SelfPlayManager();

    // Generate training data through self-play
    void generate_games();

    // Get statistics
    struct Statistics {
        std::atomic<int> games_completed{0};
        std::atomic<int> black_wins{0};
        std::atomic<int> white_wins{0};
        std::atomic<int> draws{0};
        std::atomic<int> black_resignations{0};
        std::atomic<int> white_resignations{0};
        std::atomic<long long> total_moves{0};
        std::atomic<long long> total_simulations{0};
    };

    const Statistics& get_statistics() const { return stats_; }

    // Save training examples to file
    void save_training_data(const std::string& filename);

private:
    std::shared_ptr<AlphaZeroNetwork> network_;
    AlphaZeroMCTS::Config mcts_config_;
    SelfPlayConfig selfplay_config_;

    Statistics stats_;

    // Thread pool
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> stop_workers_;
    std::atomic<int> games_to_play_;

    // Training data buffer
    std::vector<TrainingExample> training_examples_;
    std::mutex training_data_mutex_;

    // Play a single game and generate training examples
    void play_game();

    // Worker thread function
    void worker_function();

    // Check if should resign based on recent evaluations
    bool should_resign(const std::vector<float>& recent_values) const;

    // Convert action probabilities to policy vector
    std::vector<float> action_probs_to_policy(
        const std::vector<std::pair<Move, float>>& action_probs,
        const Yolah& state) const;
};

// Training data writer for efficient storage
class TrainingDataWriter {
public:
    explicit TrainingDataWriter(const std::string& filename);
    ~TrainingDataWriter();

    // Write training example in binary format
    void write_example(const TrainingExample& example);

    // Flush buffer to disk
    void flush();

private:
    std::ofstream file_;
    std::vector<char> buffer_;
    std::mutex mutex_;

    static constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1MB buffer
};

// Training data reader for loading examples during training
class TrainingDataReader {
public:
    explicit TrainingDataReader(const std::string& filename);
    ~TrainingDataReader();

    // Read next training example
    bool read_example(TrainingExample& example);

    // Get total number of examples
    size_t size() const { return num_examples_; }

    // Reset to beginning
    void reset();

private:
    std::ifstream file_;
    size_t num_examples_;
    size_t current_index_;
};

} // namespace alphazero

#endif // SELF_PLAY_H
