#ifndef RESNET_CHECK_H
#define RESNET_CHECK_H
#include "nn_evaluator.h"
#include <string>

namespace test {
    // Compare an evaluator against the PyTorch reference outputs recorded by
    // nnue/resnet_export.py (<name>.test.bin). Prints the max abs error on the
    // value and on the policy logits and returns true when both are within
    // `tolerance`.
    bool resnet_check(nn::Evaluator& evaluator, const std::string& test_vectors_filename,
                      float tolerance = 2e-3f);
    // Throughput of evaluate() on random positions for a given batch size.
    // Returns positions per second.
    double resnet_bench(nn::Evaluator& evaluator, size_t batch_size, size_t nb_batches = 4);
}

#endif
