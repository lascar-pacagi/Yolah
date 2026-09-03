#include "resnet_check.h"
#include "misc.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

namespace test {

namespace {
    struct Vector {
        Yolah state;
        float value;
        std::vector<float> logits;   // 4096
    };

    std::vector<Vector> read_vectors(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("cannot open " + filename);
        uint32_t n;
        ifs.read(reinterpret_cast<char*>(&n), 4);
        std::vector<Vector> res(n);
        for (Vector& v : res) {
            uint64_t black, white, empty;
            uint32_t turn;
            ifs.read(reinterpret_cast<char*>(&black), 8);
            ifs.read(reinterpret_cast<char*>(&white), 8);
            ifs.read(reinterpret_cast<char*>(&empty), 8);
            ifs.read(reinterpret_cast<char*>(&turn), 4);
            ifs.read(reinterpret_cast<char*>(&v.value), 4);
            v.logits.resize(nn::NUM_ACTIONS);
            ifs.read(reinterpret_cast<char*>(v.logits.data()), nn::NUM_ACTIONS * 4);
            // Scores are irrelevant to the network; ply parity encodes the turn.
            v.state.set_state(black, white, empty, 0, 0, static_cast<uint16_t>(turn));
        }
        if (!ifs) throw std::runtime_error("truncated " + filename);
        return res;
    }
}

bool resnet_check(nn::Evaluator& evaluator, const std::string& test_vectors_filename, float tolerance) {
    const auto vectors = read_vectors(test_vectors_filename);
    std::vector<Yolah::MoveList> moves(vectors.size());
    std::vector<nn::Request> requests(vectors.size());
    std::vector<nn::Result> results(vectors.size());
    for (size_t i = 0; i < vectors.size(); i++) {
        vectors[i].state.moves(moves[i]);
        requests[i] = { &vectors[i].state, moves[i].begin(), static_cast<uint16_t>(moves[i].size()) };
    }
    evaluator.evaluate(requests, results);
    float max_value_err = 0, max_logit_err = 0;
    size_t argmax_mismatches = 0;
    for (size_t i = 0; i < vectors.size(); i++) {
        max_value_err = std::max(max_value_err, std::fabs(results[i].value - vectors[i].value));
        size_t best_ref = 0, best_out = 0;
        for (size_t m = 0; m < moves[i].size(); m++) {
            const float ref = vectors[i].logits[nn::action_index(moves[i][m])];
            max_logit_err = std::max(max_logit_err, std::fabs(results[i].logits[m] - ref));
            if (ref > vectors[i].logits[nn::action_index(moves[i][best_ref])]) best_ref = m;
            if (results[i].logits[m] > results[i].logits[best_out]) best_out = m;
        }
        argmax_mismatches += best_ref != best_out;
    }
    std::cout << "[resnet check] " << evaluator.info() << "\n"
              << "  positions           : " << vectors.size() << "\n"
              << "  max |value error|   : " << max_value_err << "\n"
              << "  max |logit error|   : " << max_logit_err << "\n"
              << "  policy argmax diffs : " << argmax_mismatches << "\n";
    const bool ok = max_value_err <= tolerance && max_logit_err <= tolerance * 10;
    std::cout << "  " << (ok ? "OK" : "MISMATCH") << std::endl;
    return ok;
}

double resnet_bench(nn::Evaluator& evaluator, size_t batch_size, size_t nb_batches) {
    PRNG prng(42);
    std::vector<Yolah> states(batch_size);
    std::vector<Yolah::MoveList> moves(batch_size);
    std::vector<nn::Request> requests(batch_size);
    std::vector<nn::Result> results(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        // random non-terminal position
        const size_t plies = reduce(prng.rand<uint32_t>(), 50);
        for (size_t k = 0; k < plies && !states[i].game_over(); k++) {
            states[i].moves(moves[i]);
            states[i].play(moves[i][reduce(prng.rand<uint32_t>(), moves[i].size())]);
        }
        states[i].moves(moves[i]);
        requests[i] = { &states[i], moves[i].begin(), static_cast<uint16_t>(moves[i].size()) };
    }
    evaluator.evaluate(requests, results);   // warm-up
    const auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < nb_batches; k++) evaluator.evaluate(requests, results);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double pps = static_cast<double>(batch_size * nb_batches) / secs;
    std::cout << "[resnet bench] batch " << batch_size << ": " << pps << " positions/s ("
              << secs / nb_batches * 1e3 << " ms per batch)" << std::endl;
    return pps;
}

} // namespace test
