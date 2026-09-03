#include "nn_evaluator.h"
#include "resnet.h"
#include "batched_evaluator.h"
#ifdef ENABLE_TORCH
#include "resnet_torch.h"
#endif
#include <stdexcept>
#include <thread>

namespace nn {

std::unique_ptr<Evaluator> make_evaluator(const json& j) {
    if (!j.contains("weights")) throw std::invalid_argument("weights key expected");
    const std::string weights = j["weights"].get<std::string>();
    const std::string backend = j.value("backend", std::string("cpu"));
    if (backend == "cpu") {
        const int eval_threads = j.value("nb eval threads", 0);
        const size_t batch = j.value("batch size", 32);
        return std::make_unique<ResNetCPU>(weights, eval_threads, batch);
    }
    if (backend == "torch") {
#ifdef ENABLE_TORCH
        const std::string device = j.value("device", std::string("cuda"));
        const bool fp16 = j.value("fp16", true);
        const size_t batch = j.value("batch size", 256);
        return std::make_unique<ResNetTorch>(weights, device, fp16, batch);
#else
        throw std::invalid_argument("backend \"torch\" requires a build with -DENABLE_TORCH=ON");
#endif
    }
    throw std::invalid_argument("unknown backend \"" + backend + "\" (expected \"cpu\" or \"torch\")");
}

} // namespace nn
