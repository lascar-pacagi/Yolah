#ifndef RESNET_H
#define RESNET_H
// resnet.h — pure C++ (Eigen + OpenMP) inference of the two-headed ResNet
// trained by nnue/cnn_resnet_value_policy_chunked.py.
//
// Architecture (see the Python file for the training-side definition):
//   input (4 planes, 8x8) ─ 3x3 conv → BN → ReLU        (stem, BN folded)
//   ─ B × [ BN → ReLU → 3x3 conv → BN → ReLU → 3x3 conv, + skip ]
//   ─ BN → ReLU                                          (trunk output)
//   ├─ value : 1x1 conv → BN → ReLU → flatten(64) → FC → ReLU → FC → tanh
//   └─ policy: 1x1 conv → BN → ReLU → flatten(128) → FC → 4096 logits
//
// Implementation notes
// ────────────────────
// • Activations are kept pixel-major ("NHWC": [position][pixel 0..63][channel])
//   so that the 3x3 convolution becomes im2col + one big GEMM per layer:
//       Out (64n × C)  =  Col (64n × 9C)  ·  Wᵀ (9C × C)
//   Eigen's GEMM (with -march=native -mfma) runs close to peak on this shape.
//   The im2col copy itself is <5 % of the layer's cost.
// • BatchNorm is folded at export time (nnue/resnet_export.py): the stem and
//   the 1x1 head convs absorbed it into their weights; the pre-activation BNs
//   of the residual blocks become a per-channel affine map applied together
//   with the ReLU in a single pass over the activations.
// • Parallelism: one evaluate() call splits the batch into chunks of a few
//   positions and processes the chunks with an OpenMP team (each thread owns
//   its scratch buffers, Eigen runs single-threaded inside). The backend is
//   therefore most efficient with ONE search thread feeding it batches of
//   ~32-128 positions; several search threads should go through
//   nn::BatchingEvaluator so their requests are merged rather than nested.
// • Cost: ≈ 4.5 GFLOP per position for the 256×30 network — expect on the
//   order of 100 positions/s on a desktop CPU. The libtorch backend
//   (resnet_torch.h) is 20-50× faster on a GPU; this backend exists so the
//   player always works, and as the reference for correctness.
#include "nn_evaluator.h"
#include <Eigen/Core>
#include <string>
#include <vector>

namespace nn {

class ResNetCPU final : public Evaluator {
public:
    using AlignedVec = std::vector<float, Eigen::aligned_allocator<float>>;

    // weights_filename : .bin produced by resnet_export.py
    // nb_threads       : OpenMP threads per evaluate() call (0 = OpenMP default)
    // preferred_batch  : batch size hint reported to the search
    explicit ResNetCPU(const std::string& weights_filename, int nb_threads = 0,
                       size_t preferred_batch = 32);
    ~ResNetCPU() override;

    void evaluate(std::span<const Request> in, std::span<Result> out) override;
    size_t preferred_batch_size() const override { return preferred_batch; }
    std::string info() const override;
    void reload(const std::string& weights_filename) override;

    // Reference forward used by the self-test: `planes` holds n × 256 floats
    // (NCHW, see encode_planes); writes n values and n × NUM_ACTIONS logits.
    void forward(const float* planes, size_t n, float* values, float* logits) const;

    int channels()  const { return C; }
    int nb_blocks() const { return B; }
    int nb_threads() const { return threads; }

private:
    struct Block {
        AlignedVec bn1_scale, bn1_shift, conv1_w;   // conv: (C, 9, C) row-major
        AlignedVec bn2_scale, bn2_shift, conv2_w;
    };
    struct Scratch;   // per-thread work buffers, defined in resnet.cpp

    void load(const std::string& weights_filename);
    // Runs the trunk + heads on n ≤ max_chunk positions. Writes n values and
    // the n × 128 policy feature vectors (input of the last policy FC).
    void forward_chunk(Scratch& s, const float* planes, size_t n,
                       float* values, float* policy_vec) const;
    // Policy logit of action index a given the 128-d policy feature vector.
    float policy_logit(const float* policy_vec, int a) const;

    std::string filename;
    int threads;
    size_t preferred_batch;
    static constexpr size_t MAX_CHUNK = 8;   // positions per GEMM (64 rows each)

    int C = 0, B = 0, F = 0, A = 0;          // channels, blocks, value fc size, actions
    AlignedVec stem_w, stem_b;               // (C, 9, 4), (C)
    std::vector<Block> blocks;
    AlignedVec out_scale, out_shift;         // trunk output BN
    AlignedVec vconv_w, vconv_b;             // (C), (1)
    AlignedVec vfc1_w, vfc1_b;               // (F, 64), (F)
    AlignedVec vfc2_w, vfc2_b;               // (F), (1)
    AlignedVec pconv_w, pconv_b;             // (2, C), (2)
    AlignedVec pfc_w, pfc_b;                 // (A, 128), (A)
};

} // namespace nn

#endif
