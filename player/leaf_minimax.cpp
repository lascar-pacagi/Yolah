#include "leaf_minimax.h"
#include "minmax_nnue_dev_player.h"
#include "heuristic.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using std::invalid_argument;

LeafMinimax::LeafMinimax(const json& j, size_t nb_threads, float su, float ss)
    : cfg(j), score_utility(su), score_scale(ss) {
    if (!j.contains("player")) throw invalid_argument("leaf minimax: player key expected");
    min_ply      = j.value("min ply", uint16_t(30));
    nodes        = j.value("nodes", uint64_t(4000));
    microseconds = j.value("microseconds", uint64_t(20000));
    max_depth    = j.value("max depth", uint8_t(64));
    weight       = j.value("weight", 1.0f);
    const size_t concurrency = j.value("concurrency", size_t(0));
    if (concurrency > 0 && concurrency < nb_threads)
        slots = std::make_unique<std::counting_semaphore<1024>>(std::ptrdiff_t(concurrency));
    json p = j["player"];
    p["name"] = "MinMaxNNUE_DevPlayer";
    p["nb threads"] = 1;                                   // one search thread per MCTS thread
    p["tt size"] = j.value("tt size", size_t(16));
    p["microseconds"] = microseconds;                      // unused (search() gets its own), but required
    p["verbose"] = false;
    for (size_t t = 0; t < nb_threads; t++) {
        auto e = Player::create(p);
        auto* dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(e.get());
        if (!dev) throw invalid_argument("leaf minimax: a MinMaxNNUE_DevPlayer expected");
        dev->set_max_nodes(nodes);
        engines.push_back(std::move(e));
    }
}

LeafMinimax::~LeafMinimax() = default;

bool LeafMinimax::value(size_t thread, const Yolah& state, float net_value, float& out) {
    if (state.nb_plies() < min_ply) return false;
    auto* dev = static_cast<MinMaxNNUE_DevPlayer*>(engines[thread].get());
    if (slots) slots->acquire();
    const auto start = std::chrono::steady_clock::now();
    const auto r = dev->search(state, max_depth, microseconds);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (slots) slots->release();
    // The minimax's scale → the MCTS's: a network value (|v| <= 30000) is the
    // tanh of the expected result; a proven result is ±(WIN + margin).
    const int v = r.value;
    float mm;
    if (std::abs(v) > heuristic::MAX_VALUE) {
        const int margin = std::abs(v) - heuristic::MAX_VALUE;
        const float s = v > 0 ? 1.0f : -1.0f;
        mm = score_utility > 0.0f ? s * ((1.0f - score_utility) + score_utility * std::tanh(float(margin) / score_scale)) : s;
        nb_proven.fetch_add(1, std::memory_order_relaxed);
    } else {
        mm = float(v) / float(heuristic::MAX_VALUE);
    }
    out = (1.0f - weight) * net_value + weight * mm;
    nb_searches.fetch_add(1, std::memory_order_relaxed);
    nb_nodes.fetch_add(r.nb_nodes, std::memory_order_relaxed);
    nb_us.fetch_add(uint64_t(us), std::memory_order_relaxed);
    double d = sum_diff.load(std::memory_order_relaxed);
    while (!sum_diff.compare_exchange_weak(d, d + std::abs(mm - net_value), std::memory_order_relaxed)) {}
    double a = sum_net.load(std::memory_order_relaxed);
    while (!sum_net.compare_exchange_weak(a, a + std::abs(net_value), std::memory_order_relaxed)) {}
    double b = sum_mm.load(std::memory_order_relaxed);
    while (!sum_mm.compare_exchange_weak(b, b + std::abs(mm), std::memory_order_relaxed)) {}
    return true;
}

void LeafMinimax::new_game() {
    for (auto& e : engines) static_cast<MinMaxNNUE_DevPlayer*>(e.get())->clear_table();
}

std::string LeafMinimax::stats() {
    const uint64_t n = nb_searches.exchange(0), nd = nb_nodes.exchange(0), pr = nb_proven.exchange(0), us = nb_us.exchange(0);
    const double diff = sum_diff.exchange(0.0), net = sum_net.exchange(0.0), mmv = sum_mm.exchange(0.0);
    if (n == 0) return "leaf minimax: no leaf searched\n";
    char buf[256];
    std::snprintf(buf, sizeof buf, "leaf minimax: %llu leaves, %.0f nodes and %.2f ms per leaf, %.1f %% proven, mean |minimax - net| %.3f, mean |net| %.3f, mean |minimax| %.3f\n",
                  (unsigned long long)n, double(nd) / n, us / 1000.0 / n, 100.0 * pr / n, diff / n, net / n, mmv / n);
    return buf;
}
