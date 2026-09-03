#ifdef ENABLE_TORCH
#include "resnet_torch.h"
#include <sstream>
#include <stdexcept>

namespace nn {

ResNetTorch::ResNetTorch(const std::string& module_filename, const std::string& device_name,
                         bool fp16, size_t preferred_batch)
    : filename(module_filename),
      device(device_name == "cuda" && !torch::cuda::is_available() ? torch::Device(torch::kCPU)
                                                                    : torch::Device(device_name)),
      fp16(fp16 && device.is_cuda()),   // half precision only makes sense on the GPU
      preferred_batch(preferred_batch) {
    load(module_filename);
}

void ResNetTorch::load(const std::string& module_filename) {
    std::lock_guard lock(mutex);
    // The default profiling executor re-specialises the graph for every new
    // batch shape (several seconds each, and the search uses many shapes):
    // use the plain executor. These are process-wide flags and must be set
    // before the module is loaded (a module frozen/optimised under the other
    // executor cannot be run by this one).
    torch::jit::getProfilingMode() = false;
    torch::jit::setGraphExecutorOptimize(false);
    try {
        module = torch::jit::load(module_filename, device);
    } catch (const c10::Error& e) {
        throw std::runtime_error("ResNetTorch: cannot load " + module_filename + ": " + e.what_without_backtrace());
    }
    module.eval();
    if (fp16) module.to(torch::kHalf);
    filename = module_filename;
    // Warm up cuDNN / the executor on the shapes the search is likely to use.
    torch::NoGradGuard no_grad;
    for (int64_t n : {int64_t(1), int64_t(8), int64_t(32), int64_t(128), int64_t(preferred_batch)}) {
        torch::Tensor x = torch::zeros({n, PLANES, 8, 8}, torch::TensorOptions().dtype(fp16 ? torch::kHalf : torch::kFloat32).device(device));
        module.forward({x});
    }
    if (device.is_cuda()) torch::cuda::synchronize();
}

void ResNetTorch::reload(const std::string& module_filename) {
    load(module_filename);
}

std::string ResNetTorch::info() const {
    std::ostringstream os;
    os << "ResNetTorch (" << filename << ", " << device << (fp16 ? ", fp16" : ", fp32") << ")";
    return os.str();
}

void ResNetTorch::evaluate(std::span<const Request> in, std::span<Result> out) {
    const int64_t n = static_cast<int64_t>(in.size());
    if (n == 0) return;
    if (out.size() < in.size()) throw std::invalid_argument("ResNetTorch::evaluate: output span too small");
    std::lock_guard lock(mutex);
    torch::NoGradGuard no_grad;
    c10::InferenceMode inference;

    if (!input_cpu.defined() || input_cpu.size(0) < n) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device.is_cuda());
        input_cpu = torch::empty({n, PLANES, 8, 8}, options);
    }
    float* dst = input_cpu.data_ptr<float>();
    for (int64_t i = 0; i < n; i++) encode_planes(*in[i].state, dst + i * INPUT_FLOATS);

    torch::Tensor x = input_cpu.narrow(0, 0, n).to(device, fp16 ? torch::kHalf : torch::kFloat32,
                                                   /*non_blocking=*/true);
    auto outputs = module.forward({x}).toTuple();
    // Both heads back to the host in fp32, one copy each.
    torch::Tensor values = outputs->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
    torch::Tensor logits = outputs->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
    const float* v = values.data_ptr<float>();
    const float* p = logits.data_ptr<float>();
    for (int64_t i = 0; i < n; i++) {
        const Request& rq = in[i];
        Result& rs = out[i];
        rs.value = v[i];
        const float* row = p + i * NUM_ACTIONS;
        for (uint16_t m = 0; m < rq.nb_moves; m++) rs.logits[m] = row[action_index(rq.moves[m])];
    }
}

} // namespace nn
#endif // ENABLE_TORCH
