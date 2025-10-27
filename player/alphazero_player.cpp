#include "alphazero_player.h"
#include <iostream>

AlphaZeroPlayer::AlphaZeroPlayer(const json& cfg)
    : config_(cfg) {

    // Parse configuration
    weights_path_ = cfg.value("network_weights", "alphazero_weights.bin");
    use_gpu_ = cfg.value("use_gpu", true);

    // MCTS configuration
    mcts_config_.num_simulations = cfg.value("num_simulations", 800);
    mcts_config_.num_parallel_games = cfg.value("num_parallel_games", 8);
    mcts_config_.c_puct = cfg.value("c_puct", 1.5f);
    mcts_config_.temperature = cfg.value("temperature", 0.0f);
    mcts_config_.virtual_loss = cfg.value("virtual_loss", 3);
    mcts_config_.batch_size = cfg.value("batch_size", 8);

    // Initialize neural network
    network_ = std::make_shared<alphazero::AlphaZeroNetwork>();

    try {
        network_->initialize(weights_path_);
        std::cout << "AlphaZero: Loaded network weights from " << weights_path_ << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "AlphaZero: Failed to load weights: " << e.what() << std::endl;
        std::cerr << "AlphaZero: Initializing with random weights" << std::endl;
        network_->initialize();
    }

    // Initialize MCTS
    mcts_ = std::make_unique<alphazero::AlphaZeroMCTS>(network_, mcts_config_);

    std::cout << "AlphaZero Player initialized:" << std::endl;
    std::cout << "  Simulations: " << mcts_config_.num_simulations << std::endl;
    std::cout << "  Parallel games: " << mcts_config_.num_parallel_games << std::endl;
    std::cout << "  C-PUCT: " << mcts_config_.c_puct << std::endl;
    std::cout << "  Temperature: " << mcts_config_.temperature << std::endl;
}

AlphaZeroPlayer::~AlphaZeroPlayer() = default;

Move AlphaZeroPlayer::play(Yolah state) {
    // Use MCTS to search for best move
    Move best_move = mcts_->search(state);

    // Get action probabilities for debugging
    auto action_probs = mcts_->get_action_probs();

    if (action_probs.size() > 0) {
        std::cout << "AlphaZero move distribution:" << std::endl;
        for (size_t i = 0; i < std::min(action_probs.size(), size_t(5)); ++i) {
            std::cout << "  " << action_probs[i].first
                     << ": " << (action_probs[i].second * 100.0f) << "%" << std::endl;
        }
    }

    // Don't reset tree - reuse for next move
    // mcts_->advance_to_child(best_move);

    return best_move;
}

std::string AlphaZeroPlayer::info() {
    std::ostringstream oss;
    oss << "AlphaZero Player\n";
    oss << "  Simulations: " << mcts_config_.num_simulations << "\n";
    oss << "  Parallel threads: " << mcts_config_.num_parallel_games << "\n";
    oss << "  Network: " << weights_path_ << "\n";
    return oss.str();
}

void AlphaZeroPlayer::game_over(Yolah state) {
    // Reset MCTS tree for next game
    mcts_->reset();
}

json AlphaZeroPlayer::config() {
    return config_;
}
