#include "search_table.h"
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>

SearchTable::SearchTable(size_t mb_size) {
    // The largest power of two of clusters that fits in mb_size.
    const size_t n = std::bit_floor(std::max<size_t>(1, mb_size * 1024 * 1024 / sizeof(Cluster)));
    clusters = static_cast<Cluster*>(std::aligned_alloc(64, n * sizeof(Cluster)));
    if (!clusters) {
        std::cerr << "Failed to allocate " << mb_size << " MB for the search table\n";
        std::exit(EXIT_FAILURE);
    }
    mask = n - 1;
    clear();
}

SearchTable::~SearchTable() {
    std::free(clusters);
}

void SearchTable::clear() {
    std::memset(static_cast<void*>(clusters), 0, (mask + 1) * sizeof(Cluster));
}

const SearchTable::Entry* SearchTable::probe(uint64_t hash) const {
    const Cluster& c = clusters[hash & mask];
    const uint32_t k = key_of(hash);
    for (const Entry& e : c.entries) {
        if (e.key32 == k) return &e;
    }
    return nullptr;
}

void SearchTable::store(uint64_t hash, int free, int depth, int alpha, int beta, int value, Move move) {
    Cluster& c = clusters[hash & mask];
    const uint32_t k = key_of(hash);
    // The slot: this position's entry if there is one; otherwise the least
    // useful: an empty slot or an unreachable position (more free squares than
    // the root), else the shallowest entry.
    Entry* slot = nullptr;
    for (Entry& e : c.entries) {
        if (e.key32 == k) { slot = &e; break; }
    }
    bool same = slot != nullptr;
    if (!same) {
        auto worth = [&](const Entry& e) {
            if (e.key32 == 0 || e.free > root_free) return -1;   // empty or garbage
            return int(e.depth);
        };
        slot = &c.entries[0];
        for (Entry& e : c.entries) {
            if (worth(e) < worth(*slot)) slot = &e;
        }
    }
    // A deeper result of the same position replaces the bounds; a shallower
    // one does not overwrite them (it only brings a move if there was none).
    if (same && depth < slot->depth) {
        if (slot->move == Move::none()) slot->move = move;
        return;
    }
    if (!same || depth > slot->depth) {
        // A new position, or a deeper search of this one: fresh bounds. But
        // the best move of this position is kept (an all-node, where no move
        // beat alpha, has none to give, and the old one still orders well).
        if (!same) slot->move = Move::none();
        slot->key32 = k;
        slot->lower = -NO_BOUND;
        slot->upper = NO_BOUND;
        slot->depth = uint8_t(depth);
        slot->free = uint8_t(free);
    }
    // Same depth: merge (both bounds are true, the interval only shrinks).
    // Except that reductions and pruning make searches not perfectly
    // consistent with each other: if the bounds cross, the new result alone.
    const int lower = value > alpha ? std::max<int>(slot->lower, value) : slot->lower;
    const int upper = value < beta  ? std::min<int>(slot->upper, value) : slot->upper;
    if (lower <= upper) {
        slot->lower = int16_t(lower);
        slot->upper = int16_t(upper);
    } else {
        slot->lower = int16_t(value > alpha ? value : -NO_BOUND);
        slot->upper = int16_t(value < beta  ? value : NO_BOUND);
    }
    if (move != Move::none()) slot->move = move;
}

double SearchTable::load() const {
    size_t used = 0;
    for (uint64_t i = 0; i <= mask; i++) {
        for (const Entry& e : clusters[i].entries) used += e.key32 != 0 && e.free <= root_free;
    }
    return double(used) / double((mask + 1) * CLUSTER_SIZE);
}
