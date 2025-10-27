#ifndef ALPHAZERO_NETWORK_H
#define ALPHAZERO_NETWORK_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <memory>
#include <string>
#include "game.h"

namespace alphazero {

// Network configuration
struct NetworkConfig {
    static constexpr int INPUT_CHANNELS = 3;  // black, white, empty
    static constexpr int BOARD_SIZE = 64;
    static constexpr int RESIDUAL_BLOCKS = 10;
    static constexpr int FILTERS = 256;
    static constexpr int POLICY_HEAD_HIDDEN = 128;
    static constexpr int VALUE_HEAD_HIDDEN = 256;
    static constexpr int MAX_MOVES = 75;
};

// Policy and value output
struct NetworkOutput {
    float policy[NetworkConfig::MAX_MOVES];  // Move probabilities
    float value;                              // Position value [-1, 1]
    int num_valid_moves;
};

// CUDA neural network implementation
class AlphaZeroNetwork {
public:
    AlphaZeroNetwork();
    ~AlphaZeroNetwork();

    // Initialize network with random weights or load from file
    void initialize(const std::string& weights_file = "");

    // Evaluate batch of positions
    void evaluate_batch(const std::vector<Yolah>& positions,
                       std::vector<NetworkOutput>& outputs,
                       int batch_size);

    // Evaluate single position
    NetworkOutput evaluate(const Yolah& position);

    // Save/load weights
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);

    // Training interface
    void train_step(const std::vector<Yolah>& positions,
                   const std::vector<float>& policy_targets,
                   const std::vector<float>& value_targets,
                   float learning_rate);

private:
    // CUDA handles
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    // Device memory for network layers
    struct Layer {
        float* weights_device;
        float* bias_device;
        float* output_device;
        size_t weights_size;
        size_t bias_size;
        size_t output_size;
    };

    // Network layers
    std::vector<Layer> conv_layers_;        // Convolutional layers for residual blocks
    std::vector<Layer> residual_blocks_;    // Residual blocks
    Layer policy_head_;                      // Policy output head
    Layer value_head_;                       // Value output head

    // Temporary buffers
    float* input_device_;
    float* temp_device_;
    size_t input_size_;
    size_t max_batch_size_;

    // Encode game state to neural network input
    void encode_position(const Yolah& position, float* input);
    void encode_batch(const std::vector<Yolah>& positions, float* input_batch);

    // Forward pass implementation
    void forward_conv_block(const float* input, float* output,
                           const Layer& layer, int batch_size);
    void forward_residual_block(const float* input, float* output,
                               const Layer& layer, int batch_size);
    void forward_policy_head(const float* input, float* output, int batch_size);
    void forward_value_head(const float* input, float* output, int batch_size);

    // Helper functions
    void allocate_layer(Layer& layer, size_t weights, size_t bias, size_t output);
    void deallocate_layer(Layer& layer);
    void initialize_weights_xavier(Layer& layer);
};

// Utility functions for encoding game state
inline void bitboard_to_plane(uint64_t bitboard, float* plane) {
    for (int i = 0; i < 64; ++i) {
        plane[i] = ((bitboard >> i) & 1) ? 1.0f : 0.0f;
    }
}

// CUDA kernel declarations (implemented in .cu file)
void launch_conv2d_kernel(const float* input, const float* weights,
                         const float* bias, float* output,
                         int batch_size, int in_channels, int out_channels,
                         int height, int width, int kernel_size,
                         cudaStream_t stream = 0);

void launch_residual_block_kernel(const float* input, const float* weights,
                                 const float* bias, float* output,
                                 int batch_size, int channels,
                                 int height, int width,
                                 cudaStream_t stream = 0);

void launch_relu_kernel(float* data, int size, cudaStream_t stream = 0);

void launch_batch_norm_kernel(float* data, const float* mean, const float* variance,
                             const float* gamma, const float* beta,
                             int batch_size, int channels, int spatial_size,
                             cudaStream_t stream = 0);

void launch_softmax_kernel(float* data, int batch_size, int size,
                          cudaStream_t stream = 0);

} // namespace alphazero

#endif // ALPHAZERO_NETWORK_H
