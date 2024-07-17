#include "tt.h"

TranspositionTableSlow::TranspositionTableSlow(size_t bits) : table(1 << bits), mask((1 << bits) - 1) {
}

TranspositionTableSlow::Entry* TranspositionTableSlow::get_entry(uint64_t k) {
    Entry* entry = &table[k & mask];
    if (entry->bound == BOUND_NONE || entry->key != k) {
        return nullptr;
    }
    return entry;
}

Move TranspositionTableSlow::get_move(uint64_t k) {
    Entry* entry = get_entry(k);
    if (!entry) {
        return Move::none();
    }
    return entry->move;
}

void TranspositionTableSlow::update(uint64_t k, int32_t v, Bound b, uint8_t d, Move m) {
    Entry* entry = &table[k & mask];
    if (entry->bound == BOUND_NONE || entry->key != k || d >= entry->depth) {
        entry->key = k;
        entry->value = v;
        entry->move  = m;
        entry->depth = d;
        entry->bound = b;
    }
}

void TranspositionTableSlow::new_search() {
    table = std::vector<Entry>(table.size());
}
