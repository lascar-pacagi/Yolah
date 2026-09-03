#ifndef RESNET_TORCH_H
#define RESNET_TORCH_H
// resnet_torch.h — libtorch backend: runs the TorchScript module exported by
// nnue/resnet_export.py (<name>.ts) on the GPU (or CPU) through libtorch.
//
// Only compiled when the project is configured with -DENABLE_TORCH=ON; the
// build then needs a libtorch install (or the one that ships inside a pip
// torch package: -DCMAKE_PREFIX_PATH=<site-packages>/torch).
//
// The whole batch is one forward pass: (n, 4, 8, 8) → (n,), (n, 4096).
// In half precision on an RTX-class GPU the 256x30 network evaluates several
// thousand positions per second at batch 256, i.e. 20-50× the CPU backend.
#ifdef ENABLE_TORCH
#include "nn_evaluator.h"
#include <torch/script.h>
#include <torch/cuda.h>
#include <mutex>
#include <string>

namespace nn {

class ResNetTorch final : public Evaluator {
public:
    // device: "cuda", "cuda:1", "cpu" ... ; fp16: run the network in half precision
    ResNetTorch(const std::string& module_filename, const std::string& device = "cuda",
                bool fp16 = true, size_t preferred_batch = 256);

    void evaluate(std::span<const Request> in, std::span<Result> out) override;
    size_t preferred_batch_size() const override { return preferred_batch; }
    std::string info() const override;
    void reload(const std::string& module_filename) override;

private:
    void load(const std::string& module_filename);

    std::string filename;
    torch::Device device;
    bool fp16;
    size_t preferred_batch;
    torch::jit::script::Module module;
    std::mutex mutex;          // one forward at a time (callers are batched anyway)
    torch::Tensor input_cpu;   // pinned staging buffer, grown on demand
};

} // namespace nn

#endif // ENABLE_TORCH
#endif
