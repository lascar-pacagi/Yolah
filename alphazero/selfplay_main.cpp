#include <iostream>
#include <memory>
#include <chrono>
#include "alphazero_network.h"
#include "alphazero_mcts.h"
#include "self_play.h"

int main(int argc, char* argv[]) {
    std::cout << "AlphaZero Self-Play Generator for Yolah\n";
    std::cout << "========================================\n\n";

    // Parse command line arguments
    std::string weights_file = "";
    int num_games = 100;
    int num_threads = 8;
    int num_simulations = 400;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_file = argv[++i];
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--simulations" && i + 1 < argc) {
            num_simulations = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --weights <file>      Path to network weights (default: random init)\n";
            std::cout << "  --games <n>           Number of games to generate (default: 100)\n";
            std::cout << "  --threads <n>         Number of parallel threads (default: 8)\n";
            std::cout << "  --simulations <n>     MCTS simulations per move (default: 400)\n";
            std::cout << "  --help                Show this help message\n";
            return 0;
        }
    }

    // Initialize neural network
    std::cout << "Initializing neural network...\n";
    auto network = std::make_shared<alphazero::AlphaZeroNetwork>();

    try {
        network->initialize(weights_file);
        if (!weights_file.empty()) {
            std::cout << "Loaded weights from: " << weights_file << "\n";
        } else {
            std::cout << "Initialized with random weights\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error initializing network: " << e.what() << "\n";
        return 1;
    }

    // Configure MCTS
    alphazero::AlphaZeroMCTS::Config mcts_config;
    mcts_config.num_simulations = num_simulations;
    mcts_config.num_parallel_games = 4;  // Parallel simulations within a game
    mcts_config.c_puct = 1.5f;
    mcts_config.temperature = 1.0f;
    mcts_config.add_dirichlet_noise = true;
    mcts_config.dirichlet_alpha = 0.3f;
    mcts_config.dirichlet_epsilon = 0.25f;

    // Configure self-play
    alphazero::SelfPlayConfig selfplay_config;
    selfplay_config.num_games = num_games;
    selfplay_config.num_threads = num_threads;
    selfplay_config.max_moves_per_game = 200;
    selfplay_config.temperature_threshold = 15;
    selfplay_config.resign_enabled = true;
    selfplay_config.resign_threshold = -0.9f;
    selfplay_config.resign_consecutive_moves = 5;
    selfplay_config.output_dir = "./selfplay_data";

    std::cout << "\nConfiguration:\n";
    std::cout << "  Games: " << num_games << "\n";
    std::cout << "  Threads: " << num_threads << "\n";
    std::cout << "  MCTS simulations: " << num_simulations << "\n";
    std::cout << "  Output directory: " << selfplay_config.output_dir << "\n\n";

    // Create self-play manager
    alphazero::SelfPlayManager manager(network, mcts_config, selfplay_config);

    // Generate games
    auto start_time = std::chrono::high_resolution_clock::now();

    manager.generate_games();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTime elapsed: " << duration.count() << " seconds\n";

    // Save training data
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string output_file = selfplay_config.output_dir + "/training_data_" +
                             std::to_string(timestamp) + ".bin";

    manager.save_training_data(output_file);

    // Print statistics
    const auto& stats = manager.get_statistics();
    std::cout << "\nFinal Statistics:\n";
    std::cout << "  Games completed: " << stats.games_completed << "\n";
    std::cout << "  Black wins: " << stats.black_wins
              << " (" << (100.0 * stats.black_wins / stats.games_completed) << "%)\n";
    std::cout << "  White wins: " << stats.white_wins
              << " (" << (100.0 * stats.white_wins / stats.games_completed) << "%)\n";
    std::cout << "  Draws: " << stats.draws
              << " (" << (100.0 * stats.draws / stats.games_completed) << "%)\n";
    std::cout << "  Average moves per game: "
              << (stats.total_moves / stats.games_completed) << "\n";

    std::cout << "\nSelf-play generation complete!\n";
    std::cout << "Training data saved to: " << output_file << "\n";

    return 0;
}
