#include "alphazero_network.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

namespace alphazero {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudnnGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernels
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void residual_add_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += input[idx];
    }
}

__global__ void softmax_kernel(float* data, int batch_size, int size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* batch_data = data + batch_idx * size;

    // Find max for numerical stability
    float max_val = batch_data[0];
    for (int i = 1; i < size; ++i) {
        max_val = fmaxf(max_val, batch_data[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        batch_data[i] = expf(batch_data[i] - max_val);
        sum += batch_data[i];
    }

    // Normalize
    for (int i = 0; i < size; ++i) {
        batch_data[i] /= sum;
    }
}

__global__ void tanh_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

// Launch wrappers for kernels
void launch_relu_kernel(float* data, int size, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size, 0, stream>>>(data, size);
}

void launch_softmax_kernel(float* data, int batch_size, int size, cudaStream_t stream) {
    softmax_kernel<<<batch_size, 1, 0, stream>>>(data, batch_size, size);
}

// AlphaZeroNetwork implementation
AlphaZeroNetwork::AlphaZeroNetwork()
    : input_device_(nullptr), temp_device_(nullptr),
      input_size_(0), max_batch_size_(256) {

    // Create CUDA handles
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

AlphaZeroNetwork::~AlphaZeroNetwork() {
    // Free device memory
    if (input_device_) CUDA_CHECK(cudaFree(input_device_));
    if (temp_device_) CUDA_CHECK(cudaFree(temp_device_));

    // Free layer memory
    for (auto& layer : conv_layers_) deallocate_layer(layer);
    for (auto& layer : residual_blocks_) deallocate_layer(layer);
    deallocate_layer(policy_head_);
    deallocate_layer(value_head_);

    // Destroy handles
    cudnnDestroy(cudnn_handle_);
    cublasDestroy(cublas_handle_);
}

void AlphaZeroNetwork::initialize(const std::string& weights_file) {
    // Allocate input buffer
    input_size_ = max_batch_size_ * NetworkConfig::INPUT_CHANNELS * NetworkConfig::BOARD_SIZE;
    CUDA_CHECK(cudaMalloc(&input_device_, input_size_ * sizeof(float)));

    // Allocate temporary buffer for intermediate computations
    size_t temp_size = max_batch_size_ * NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE;
    CUDA_CHECK(cudaMalloc(&temp_device_, temp_size * sizeof(float)));

    // Initialize network layers
    // Initial convolution: 3 channels -> 256 filters
    Layer initial_conv;
    allocate_layer(initial_conv,
                  NetworkConfig::INPUT_CHANNELS * NetworkConfig::FILTERS * 3 * 3,
                  NetworkConfig::FILTERS,
                  max_batch_size_ * NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE);
    conv_layers_.push_back(initial_conv);

    // Residual blocks
    for (int i = 0; i < NetworkConfig::RESIDUAL_BLOCKS; ++i) {
        Layer residual;
        allocate_layer(residual,
                      2 * NetworkConfig::FILTERS * NetworkConfig::FILTERS * 3 * 3,
                      2 * NetworkConfig::FILTERS,
                      max_batch_size_ * NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE);
        residual_blocks_.push_back(residual);
    }

    // Policy head
    allocate_layer(policy_head_,
                  NetworkConfig::FILTERS * NetworkConfig::POLICY_HEAD_HIDDEN +
                  NetworkConfig::POLICY_HEAD_HIDDEN * NetworkConfig::MAX_MOVES,
                  NetworkConfig::POLICY_HEAD_HIDDEN + NetworkConfig::MAX_MOVES,
                  max_batch_size_ * NetworkConfig::MAX_MOVES);

    // Value head
    allocate_layer(value_head_,
                  NetworkConfig::FILTERS * NetworkConfig::VALUE_HEAD_HIDDEN +
                  NetworkConfig::VALUE_HEAD_HIDDEN * 1,
                  NetworkConfig::VALUE_HEAD_HIDDEN + 1,
                  max_batch_size_);

    // Initialize weights
    if (weights_file.empty()) {
        // Xavier initialization
        for (auto& layer : conv_layers_) initialize_weights_xavier(layer);
        for (auto& layer : residual_blocks_) initialize_weights_xavier(layer);
        initialize_weights_xavier(policy_head_);
        initialize_weights_xavier(value_head_);
    } else {
        load_weights(weights_file);
    }
}

void AlphaZeroNetwork::allocate_layer(Layer& layer, size_t weights,
                                     size_t bias, size_t output) {
    layer.weights_size = weights;
    layer.bias_size = bias;
    layer.output_size = output;

    CUDA_CHECK(cudaMalloc(&layer.weights_device, weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.bias_device, bias * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.output_device, output * sizeof(float)));
}

void AlphaZeroNetwork::deallocate_layer(Layer& layer) {
    if (layer.weights_device) CUDA_CHECK(cudaFree(layer.weights_device));
    if (layer.bias_device) CUDA_CHECK(cudaFree(layer.bias_device));
    if (layer.output_device) CUDA_CHECK(cudaFree(layer.output_device));
}

void AlphaZeroNetwork::initialize_weights_xavier(Layer& layer) {
    // Xavier initialization on CPU then copy to GPU
    std::vector<float> weights(layer.weights_size);
    std::vector<float> bias(layer.bias_size, 0.0f);

    float stddev = std::sqrt(2.0f / layer.weights_size);

    // Use cuRAND for initialization
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, layer.weights_device, layer.weights_size, 0.0f, stddev);
    curandDestroy(gen);

    CUDA_CHECK(cudaMemcpy(layer.bias_device, bias.data(),
                         layer.bias_size * sizeof(float), cudaMemcpyHostToDevice));
}

void AlphaZeroNetwork::encode_position(const Yolah& position, float* input) {
    // Encode as 3 planes: black, white, empty
    float planes[3][64] = {0};

    for (int sq = 0; sq < 64; ++sq) {
        uint64_t mask = 1ULL << sq;
        if (position.bitboard(Yolah::BLACK) & mask) {
            planes[0][sq] = 1.0f;
        } else if (position.bitboard(Yolah::WHITE) & mask) {
            planes[1][sq] = 1.0f;
        } else if (position.empty_bitboard() & mask) {
            planes[2][sq] = 1.0f;
        }
    }

    std::memcpy(input, planes, 3 * 64 * sizeof(float));
}

void AlphaZeroNetwork::encode_batch(const std::vector<Yolah>& positions, float* input_batch) {
    for (size_t i = 0; i < positions.size(); ++i) {
        encode_position(positions[i], input_batch + i * NetworkConfig::INPUT_CHANNELS * NetworkConfig::BOARD_SIZE);
    }
}

NetworkOutput AlphaZeroNetwork::evaluate(const Yolah& position) {
    std::vector<Yolah> batch = {position};
    std::vector<NetworkOutput> outputs(1);
    evaluate_batch(batch, outputs, 1);
    return outputs[0];
}

void AlphaZeroNetwork::evaluate_batch(const std::vector<Yolah>& positions,
                                     std::vector<NetworkOutput>& outputs,
                                     int batch_size) {
    // Encode positions to input buffer
    std::vector<float> input_host(batch_size * NetworkConfig::INPUT_CHANNELS * NetworkConfig::BOARD_SIZE);
    encode_batch(positions, input_host.data());

    CUDA_CHECK(cudaMemcpy(input_device_, input_host.data(),
                         input_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass through network
    float* current_features = input_device_;

    // Initial convolution
    forward_conv_block(current_features, conv_layers_[0].output_device,
                      conv_layers_[0], batch_size);
    current_features = conv_layers_[0].output_device;

    // Residual blocks
    for (auto& residual_block : residual_blocks_) {
        forward_residual_block(current_features, residual_block.output_device,
                              residual_block, batch_size);
        current_features = residual_block.output_device;
    }

    // Policy head
    forward_policy_head(current_features, policy_head_.output_device, batch_size);

    // Value head
    forward_value_head(current_features, value_head_.output_device, batch_size);

    // Copy results back to host
    std::vector<float> policy_host(batch_size * NetworkConfig::MAX_MOVES);
    std::vector<float> value_host(batch_size);

    CUDA_CHECK(cudaMemcpy(policy_host.data(), policy_head_.output_device,
                         policy_host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(value_host.data(), value_head_.output_device,
                         value_host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Fill output structures
    outputs.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        std::memcpy(outputs[i].policy,
                   policy_host.data() + i * NetworkConfig::MAX_MOVES,
                   NetworkConfig::MAX_MOVES * sizeof(float));
        outputs[i].value = value_host[i];

        // Count valid moves
        Yolah::MoveList moves;
        positions[i].moves(moves);
        outputs[i].num_valid_moves = moves.size();
    }
}

void AlphaZeroNetwork::forward_conv_block(const float* input, float* output,
                                         const Layer& layer, int batch_size) {
    // Simplified: Use cuBLAS for matrix multiplication (treating convolution as matrix ops)
    // In full implementation, use cuDNN convolution operations

    int m = batch_size * NetworkConfig::BOARD_SIZE;
    int n = NetworkConfig::FILTERS;
    int k = NetworkConfig::INPUT_CHANNELS;

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            layer.weights_device, n,
                            input, k,
                            &beta,
                            output, n));

    // Add bias (simplified - should be done with CUDA kernel)
    // Apply ReLU
    launch_relu_kernel(output, batch_size * NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE);
}

void AlphaZeroNetwork::forward_residual_block(const float* input, float* output,
                                             const Layer& layer, int batch_size) {
    // First convolution
    forward_conv_block(input, temp_device_, layer, batch_size);

    // Second convolution
    forward_conv_block(temp_device_, output, layer, batch_size);

    // Residual connection: output += input
    int size = batch_size * NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    residual_add_kernel<<<grid_size, block_size>>>(input, output, size);

    // ReLU
    launch_relu_kernel(output, size);
}

void AlphaZeroNetwork::forward_policy_head(const float* input, float* output, int batch_size) {
    // Flatten and fully connected layers
    int m = batch_size;
    int n = NetworkConfig::MAX_MOVES;
    int k = NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE;

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            policy_head_.weights_device, n,
                            input, k,
                            &beta,
                            output, n));

    // Softmax
    launch_softmax_kernel(output, batch_size, NetworkConfig::MAX_MOVES);
}

void AlphaZeroNetwork::forward_value_head(const float* input, float* output, int batch_size) {
    // Fully connected layer
    int m = batch_size;
    int n = 1;
    int k = NetworkConfig::FILTERS * NetworkConfig::BOARD_SIZE;

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            value_head_.weights_device, n,
                            input, k,
                            &beta,
                            output, n));

    // Tanh activation
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    tanh_kernel<<<grid_size, block_size>>>(output, batch_size);
}

void AlphaZeroNetwork::save_weights(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    auto save_layer = [&](const Layer& layer) {
        std::vector<float> weights(layer.weights_size);
        std::vector<float> bias(layer.bias_size);

        CUDA_CHECK(cudaMemcpy(weights.data(), layer.weights_device,
                             layer.weights_size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bias.data(), layer.bias_device,
                             layer.bias_size * sizeof(float),
                             cudaMemcpyDeviceToHost));

        ofs.write(reinterpret_cast<const char*>(weights.data()),
                 layer.weights_size * sizeof(float));
        ofs.write(reinterpret_cast<const char*>(bias.data()),
                 layer.bias_size * sizeof(float));
    };

    for (const auto& layer : conv_layers_) save_layer(layer);
    for (const auto& layer : residual_blocks_) save_layer(layer);
    save_layer(policy_head_);
    save_layer(value_head_);
}

void AlphaZeroNetwork::load_weights(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    auto load_layer = [&](Layer& layer) {
        std::vector<float> weights(layer.weights_size);
        std::vector<float> bias(layer.bias_size);

        ifs.read(reinterpret_cast<char*>(weights.data()),
                layer.weights_size * sizeof(float));
        ifs.read(reinterpret_cast<char*>(bias.data()),
                layer.bias_size * sizeof(float));

        CUDA_CHECK(cudaMemcpy(layer.weights_device, weights.data(),
                             layer.weights_size * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layer.bias_device, bias.data(),
                             layer.bias_size * sizeof(float),
                             cudaMemcpyHostToDevice));
    };

    for (auto& layer : conv_layers_) load_layer(layer);
    for (auto& layer : residual_blocks_) load_layer(layer);
    load_layer(policy_head_);
    load_layer(value_head_);
}

void AlphaZeroNetwork::train_step(const std::vector<Yolah>& positions,
                                  const std::vector<float>& policy_targets,
                                  const std::vector<float>& value_targets,
                                  float learning_rate) {
    // Training would require implementing backpropagation
    // This is a placeholder - full implementation would need gradient computation
    // Recommend using PyTorch C++ API (libtorch) for training
    std::cerr << "Training step not fully implemented. Use Python training pipeline." << std::endl;
}

} // namespace alphazero
