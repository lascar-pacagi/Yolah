#include "alphazero_mcts.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
#include <chrono>
#include <future>

namespace alphazero {

// ============================================================================
// MCTSNode Implementation
// ============================================================================

MCTSNode::MCTSNode(Move action, float prior_prob)
    : action_(action), prior_(prior_prob), visit_count_(0),
      total_value_(0.0f), virtual_loss_(0), expanded_(false) {}

float MCTSNode::get_value(float c_puct, float parent_visits) const {
    float q = q_value();
    int n = visit_count_.load();
    int vl = virtual_loss_.load();

    // PUCT formula with virtual loss
    float u = c_puct * prior_ * std::sqrt(parent_visits) / (1.0f + n);
    return q - vl + u;
}

float MCTSNode::q_value() const {
    int n = visit_count_.load();
    if (n == 0) return 0.0f;
    return total_value_.load() / n;
}

MCTSNode* MCTSNode::select_child(float c_puct) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (children_.empty()) return nullptr;

    float parent_visits = static_cast<float>(visit_count_.load());

    // Find child with highest PUCT value
    MCTSNode* best_child = nullptr;
    float best_value = -std::numeric_limits<float>::infinity();

    for (auto& child : children_) {
        float value = child->get_value(c_puct, parent_visits);
        if (value > best_value) {
            best_value = value;
            best_child = child.get();
        }
    }

    return best_child;
}

void MCTSNode::expand(const Yolah& state, const NetworkOutput& network_output) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (expanded_.load()) return;  // Already expanded by another thread

    // Get legal moves
    Yolah::MoveList legal_moves;
    state.moves(legal_moves);

    if (legal_moves.size() == 0) {
        expanded_ = true;
        return;  // Terminal node
    }

    // Create child nodes with prior probabilities from policy network
    children_.reserve(legal_moves.size());

    // Map moves to policy probabilities
    float policy_sum = 0.0f;
    std::vector<float> priors(legal_moves.size());

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        // Simple mapping: use move index
        // In full implementation, need proper move encoding
        priors[i] = std::max(network_output.policy[i], 1e-8f);
        policy_sum += priors[i];
    }

    // Normalize priors
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        priors[i] /= policy_sum;
        children_.push_back(std::make_unique<MCTSNode>(legal_moves[i], priors[i]));
    }

    expanded_ = true;
}

void MCTSNode::backup(float value) {
    visit_count_.fetch_add(1, std::memory_order_relaxed);

    float old_value = total_value_.load(std::memory_order_relaxed);
    while (!total_value_.compare_exchange_weak(old_value, old_value + value,
                                               std::memory_order_release,
                                               std::memory_order_relaxed));
}

void MCTSNode::add_virtual_loss(int virtual_loss) {
    virtual_loss_.fetch_add(virtual_loss, std::memory_order_relaxed);
}

void MCTSNode::revert_virtual_loss(int virtual_loss) {
    virtual_loss_.fetch_sub(virtual_loss, std::memory_order_relaxed);
}

Move MCTSNode::best_action() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (children_.empty()) return Move::none();

    auto best_child = std::max_element(children_.begin(), children_.end(),
        [](const auto& a, const auto& b) {
            return a->visit_count() < b->visit_count();
        });

    return (*best_child)->action();
}

std::vector<std::pair<Move, float>> MCTSNode::get_action_probs(float temperature) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::pair<Move, float>> action_probs;
    action_probs.reserve(children_.size());

    if (temperature < 1e-6f) {
        // Deterministic: return only best action
        Move best = best_action();
        action_probs.emplace_back(best, 1.0f);
        return action_probs;
    }

    // Apply temperature to visit counts
    std::vector<float> visit_counts;
    visit_counts.reserve(children_.size());

    for (const auto& child : children_) {
        float count = static_cast<float>(child->visit_count());
        visit_counts.push_back(std::pow(count, 1.0f / temperature));
    }

    // Normalize
    float sum = std::accumulate(visit_counts.begin(), visit_counts.end(), 0.0f);

    for (size_t i = 0; i < children_.size(); ++i) {
        action_probs.emplace_back(children_[i]->action(), visit_counts[i] / sum);
    }

    return action_probs;
}

// ============================================================================
// AlphaZeroMCTS Implementation
// ============================================================================

AlphaZeroMCTS::AlphaZeroMCTS(std::shared_ptr<AlphaZeroNetwork> network,
                             const Config& config)
    : network_(network), config_(config), root_(nullptr),
      stop_search_(false), simulations_done_(0) {}

AlphaZeroMCTS::~AlphaZeroMCTS() {
    stop_search_ = true;
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

Move AlphaZeroMCTS::search(const Yolah& root_state) {
    // Initialize root node
    if (!root_) {
        root_ = std::make_unique<MCTSNode>(Move::none(), 1.0f);

        // Expand root immediately
        NetworkOutput root_output = network_->evaluate(root_state);
        root_->expand(root_state, root_output);

        // Add Dirichlet noise if enabled
        if (config_.add_dirichlet_noise) {
            add_dirichlet_noise();
        }
    }

    simulations_done_ = 0;
    stop_search_ = false;

    // Launch batch evaluation worker thread
    worker_threads_.clear();
    worker_threads_.emplace_back(&AlphaZeroMCTS::batch_evaluation_worker, this);

    // Launch parallel search threads
    for (int i = 0; i < config_.num_parallel_games; ++i) {
        worker_threads_.emplace_back(&AlphaZeroMCTS::search_worker, this, root_state);
    }

    // Wait for all workers to complete
    for (auto& thread : worker_threads_) {
        thread.join();
    }

    // Select best move
    return root_->best_action();
}

void AlphaZeroMCTS::search_worker(const Yolah& root_state) {
    while (simulations_done_.load() < config_.num_simulations) {
        simulate(root_state);
        simulations_done_.fetch_add(1);
    }
}

void AlphaZeroMCTS::simulate(const Yolah& root_state) {
    Yolah state = root_state;

    // Selection: traverse tree to leaf
    MCTSNode* node = select(root_state, state, root_.get());

    if (!node) return;

    // Add virtual loss for parallel search
    node->add_virtual_loss(config_.virtual_loss);

    // Evaluation: get value and policy from network
    float value;
    if (state.game_over()) {
        // Terminal node: use actual game result
        auto [black_score, white_score] = state.score();
        if (black_score > white_score) {
            value = (state.current_player() == Yolah::BLACK) ? 1.0f : -1.0f;
        } else if (white_score > black_score) {
            value = (state.current_player() == Yolah::WHITE) ? 1.0f : -1.0f;
        } else {
            value = 0.0f;
        }
    } else {
        // Submit evaluation request to batch queue
        NetworkOutput output;
        {
            EvaluationRequest request;
            request.state = state;
            request.node = node;

            std::unique_lock<std::mutex> lock(eval_queue_mutex_);
            eval_queue_.push_back(std::move(request));

            // Get reference to the request we just added (before unlocking)
            auto& queued_request = eval_queue_.back();
            auto result_future = queued_request.result.get_future();

            // Notify batch worker
            eval_queue_cv_.notify_one();

            // Unlock and wait for result
            lock.unlock();
            output = result_future.get();
        }

        value = output.value;

        if (!node->is_expanded()) {
            node->expand(state, output);
        }

        // Flip value for opponent's perspective
        value = -value;
    }

    // Backup: propagate value up the tree
    backup(node, value);

    // Revert virtual loss
    node->revert_virtual_loss(config_.virtual_loss);
}

MCTSNode* AlphaZeroMCTS::select(const Yolah& root_state, Yolah& leaf_state,
                                MCTSNode* node) {
    leaf_state = root_state;

    while (node->is_expanded() && !node->is_leaf()) {
        MCTSNode* child = node->select_child(config_.c_puct);

        if (!child) break;

        // Play the move
        leaf_state.play(child->action());
        node = child;
    }

    return node;
}

void AlphaZeroMCTS::backup(MCTSNode* node, float value) {
    // Propagate value up the tree, flipping sign at each level
    while (node) {
        node->backup(value);
        value = -value;  // Flip for parent's perspective

        // Note: In full implementation, need to track parent pointers
        // For now, only backup leaf node
        break;
    }
}

void AlphaZeroMCTS::add_dirichlet_noise() {
    if (!root_ || root_->children().empty()) return;

    // Add Dirichlet noise to root node priors for exploration
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);

    size_t num_children = root_->children().size();
    std::vector<float> noise(num_children);
    float noise_sum = 0.0f;

    for (size_t i = 0; i < num_children; ++i) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }

    // Normalize noise
    for (size_t i = 0; i < num_children; ++i) {
        noise[i] /= noise_sum;
    }

    // Mix noise with priors
    // Note: This requires modifying child priors - simplified here
    // In full implementation, store original priors and mix during selection
}

std::vector<std::pair<Move, float>> AlphaZeroMCTS::get_action_probs() const {
    if (!root_) return {};
    return root_->get_action_probs(config_.temperature);
}

void AlphaZeroMCTS::reset() {
    root_.reset();
    simulations_done_ = 0;
}

void AlphaZeroMCTS::advance_to_child(Move action) {
    if (!root_) return;

    // Find child matching action and promote to root
    for (auto& child : root_->children()) {
        if (child->action() == action) {
            // Can't easily transfer ownership in this design
            // In practice, would need restructuring
            // For now, just reset
            reset();
            return;
        }
    }

    reset();
}

NetworkOutput AlphaZeroMCTS::evaluate(const Yolah& state, MCTSNode* node) {
    return network_->evaluate(state);
}

void AlphaZeroMCTS::batch_evaluation_worker() {
    while (!stop_search_.load()) {
        std::vector<EvaluationRequest*> batch;
        batch.reserve(config_.batch_size);

        {
            std::unique_lock<std::mutex> lock(eval_queue_mutex_);

            // Wait for requests or stop signal with timeout to allow batching
            auto timeout = std::chrono::milliseconds(10);
            eval_queue_cv_.wait_for(lock, timeout, [this]() {
                return eval_queue_.size() >= static_cast<size_t>(config_.batch_size) ||
                       stop_search_.load();
            });

            if (stop_search_.load() && eval_queue_.empty()) {
                break;
            }

            if (eval_queue_.empty()) {
                continue;
            }

            // Collect batch of requests (up to batch_size)
            size_t batch_count = std::min(static_cast<size_t>(config_.batch_size),
                                         eval_queue_.size());

            for (size_t i = 0; i < batch_count; ++i) {
                batch.push_back(&eval_queue_[i]);
            }
        }

        if (batch.empty()) {
            continue;
        }

        // Prepare batch of states
        std::vector<Yolah> states;
        states.reserve(batch.size());
        for (auto* request : batch) {
            states.push_back(request->state);
        }

        // Evaluate entire batch at once using GPU
        std::vector<NetworkOutput> outputs;
        network_->evaluate_batch(states, outputs, batch.size());

        // Deliver results to waiting threads
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i]->result.set_value(outputs[i]);
        }

        // Remove processed requests from queue
        {
            std::lock_guard<std::mutex> lock(eval_queue_mutex_);
            eval_queue_.erase(eval_queue_.begin(),
                            eval_queue_.begin() + batch.size());
        }
    }

    // Process any remaining requests before shutdown
    std::lock_guard<std::mutex> lock(eval_queue_mutex_);
    if (!eval_queue_.empty()) {
        std::vector<Yolah> states;
        states.reserve(eval_queue_.size());
        for (auto& request : eval_queue_) {
            states.push_back(request.state);
        }

        std::vector<NetworkOutput> outputs;
        network_->evaluate_batch(states, outputs, eval_queue_.size());

        for (size_t i = 0; i < eval_queue_.size(); ++i) {
            eval_queue_[i].result.set_value(outputs[i]);
        }
    }
    eval_queue_.clear();
}

} // namespace alphazero
