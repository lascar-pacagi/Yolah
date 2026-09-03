#ifndef NN_CACHE_H
#define NN_CACHE_H
// nn_cache.h — transposition cache of network evaluations for the AlphaZero
// MCTS (az::Search).
//
// In Yolah the four stones of a side move independently, so the same position
// is reached through many move orders: a large share of the leaves the search
// wants to expand are transpositions of positions already evaluated. The
// network only sees (board, side to move), so its output is a pure function of
// the Zobrist hash of that pair — we cache it: value + the priors of the legal
// moves (in Yolah::moves order, which is deterministic for a position). A hit
// expands the node without a network call, which is where >99 % of the time
// goes.
//
// Direct-mapped table (index = hash & mask, always overwrite), lock-striped so
// several search threads can use it concurrently; the cost of a lookup is
// negligible next to a network evaluation. The hash also verifies the number of
// legal moves, so a 64-bit collision cannot silently produce garbage priors.
// Priors are stored as 16-bit fixed point (1/65535 resolution).
//
// Statistics are shared by everything that evaluates the same network, so the
// cache survives across moves and games; it must be cleared when the weights
// change (az::Search::clear_cache / AlphaZeroMCTSPlayer::reload_weights).
#include "game.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace az {

class NNCache {
public:
    struct Entry {
        uint64_t key = 0;
        float    value = 0.0f;
        uint16_t nb_moves = 0;
        uint16_t priors[Yolah::MAX_NB_MOVES] = {};
    };

    // size_bytes = 0 disables the cache. The entry count is rounded down to a
    // power of two.
    explicit NNCache(size_t size_bytes) {
        size_t n = size_bytes / sizeof(Entry);
        size_t pow2 = 1;
        while (pow2 * 2 <= n) pow2 *= 2;
        if (n >= 2) {
            table.resize(pow2);
            mask = pow2 - 1;
        }
    }

    bool enabled() const { return !table.empty(); }
    size_t capacity() const { return table.size(); }
    size_t size_bytes() const { return table.size() * sizeof(Entry); }

    // Returns true and fills value / priors[0..nb_moves) on a hit.
    bool lookup(uint64_t key, size_t nb_moves, float& value, float* priors) {
        if (table.empty()) return false;
        const Entry& e = table[key & mask];
        std::lock_guard lock(locks[key & (locks.size() - 1)]);
        if (e.key != key || e.nb_moves != nb_moves) {
            misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        value = e.value;
        for (size_t m = 0; m < nb_moves; m++) priors[m] = e.priors[m] * (1.0f / 65535.0f);
        hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void store(uint64_t key, size_t nb_moves, float value, const float* priors) {
        if (table.empty()) return;
        Entry& e = table[key & mask];
        std::lock_guard lock(locks[key & (locks.size() - 1)]);
        e.key = key;
        e.value = value;
        e.nb_moves = static_cast<uint16_t>(nb_moves);
        for (size_t m = 0; m < nb_moves; m++)
            e.priors[m] = static_cast<uint16_t>(priors[m] * 65535.0f + 0.5f);
    }

    void clear() {
        for (size_t i = 0; i < locks.size(); i++) locks[i].lock();
        std::fill(table.begin(), table.end(), Entry{});
        hits = 0;
        misses = 0;
        for (size_t i = locks.size(); i-- > 0;) locks[i].unlock();
    }

    uint64_t nb_hits() const { return hits.load(std::memory_order_relaxed); }
    uint64_t nb_misses() const { return misses.load(std::memory_order_relaxed); }

private:
    std::vector<Entry> table;
    size_t mask = 0;
    std::array<std::mutex, 256> locks;
    std::atomic<uint64_t> hits{0}, misses{0};
};

} // namespace az

#endif
