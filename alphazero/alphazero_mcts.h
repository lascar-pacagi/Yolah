#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H

#include <atomic>
#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <future>
#include "game.h"
#include "alphazero_network.h"

namespace alphazero {

// AlphaZero MCTS node
class MCTSNode {
public:
    MCTSNode(Move action, float prior_prob);
    ~MCTSNode() = default;

    // PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N)
    float get_value(float c_puct, float parent_visits) const;

    // Select best child according to PUCT
    MCTSNode* select_child(float c_puct);

    // Expand node with network policy
    void expand(const Yolah& state, const NetworkOutput& network_output);

    // Backpropagate value
    void backup(float value);

    // Add virtual loss for parallel search
    void add_virtual_loss(int virtual_loss = 1);
    void revert_virtual_loss(int virtual_loss = 1);

    // Getters
    bool is_leaf() const { return children_.empty(); }
    bool is_expanded() const { return expanded_; }
    int visit_count() const { return visit_count_.load(); }
    float q_value() const;
    Move action() const { return action_; }
    float prior() const { return prior_; }

    const std::vector<std::unique_ptr<MCTSNode>>& children() const { return children_; }

    // Get best action (highest visit count)
    Move best_action() const;

    // Get action probabilities proportional to visit counts
    std::vector<std::pair<Move, float>> get_action_probs(float temperature = 1.0f) const;

private:
    Move action_;                                      // Action that led to this node
    float prior_;                                      // Prior probability from policy network
    std::atomic<int> visit_count_;                     // Number of visits
    std::atomic<float> total_value_;                   // Sum of values from backpropagation
    std::atomic<int> virtual_loss_;                    // Virtual loss for parallelization
    std::atomic<bool> expanded_;                       // Whether node has been expanded

    std::vector<std::unique_ptr<MCTSNode>> children_; // Child nodes
    mutable std::mutex mutex_;                         // Mutex for thread-safe operations
};

// AlphaZero MCTS search
class AlphaZeroMCTS {
public:
    struct Config {
        int num_simulations = 800;          // Number of MCTS simulations per move
        int num_parallel_games = 8;          // Number of games to search in parallel
        float c_puct = 1.5f;                 // PUCT exploration constant
        float temperature = 1.0f;            // Temperature for action selection
        int virtual_loss = 3;                // Virtual loss value for parallel search
        bool add_dirichlet_noise = false;    // Add Dirichlet noise to root
        float dirichlet_alpha = 0.3f;        // Dirichlet noise parameter
        float dirichlet_epsilon = 0.25f;     // Dirichlet noise weight
        int batch_size = 8;                  // Batch size for neural network evaluation
    };

    AlphaZeroMCTS(std::shared_ptr<AlphaZeroNetwork> network, const Config& config);
    ~AlphaZeroMCTS();

    // Search for best move
    Move search(const Yolah& root_state);

    // Get action probabilities from search
    std::vector<std::pair<Move, float>> get_action_probs() const;

    // Reset search tree
    void reset();

    // Reuse subtree for next move
    void advance_to_child(Move action);

private:
    std::shared_ptr<AlphaZeroNetwork> network_;
    Config config_;
    std::unique_ptr<MCTSNode> root_;

    // Thread pool for parallel search
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> stop_search_;
    std::atomic<int> simulations_done_;

    // Batch evaluation queue
    struct EvaluationRequest {
        Yolah state;
        MCTSNode* node;
        std::promise<NetworkOutput> result;
    };

    std::vector<EvaluationRequest> eval_queue_;
    std::mutex eval_queue_mutex_;
    std::condition_variable eval_queue_cv_;

    // Run single simulation
    void simulate(const Yolah& root_state);

    // Selection phase: select leaf node using PUCT
    MCTSNode* select(const Yolah& state, Yolah& leaf_state, MCTSNode* node);

    // Evaluation phase: evaluate leaf node with neural network
    NetworkOutput evaluate(const Yolah& state, MCTSNode* node);

    // Backup phase: backpropagate value
    void backup(MCTSNode* node, float value);

    // Add Dirichlet noise to root for exploration
    void add_dirichlet_noise();

    // Batch evaluation worker
    void batch_evaluation_worker();

    // Parallel search worker
    void search_worker(const Yolah& root_state);
};

// Training data structure for self-play
struct TrainingExample {
    Yolah state;
    std::vector<float> policy;  // Policy vector (75 moves max)
    float value;                 // Game outcome from this player's perspective
};

} // namespace alphazero

#endif // ALPHAZERO_MCTS_H
