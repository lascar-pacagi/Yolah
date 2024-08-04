#include "transposition_table.h"
#include <thread>

void TranspositionTableEntry::save(uint64_t k, int16_t v, Bound b, uint8_t d, Move m, uint8_t generation) {
    uint32_t key32 = uint32_t(k);
    if (uint16_t(key32) != key_lo || uint16_t(key32 >> 16) != key_hi || 
        d > depth8 || (d == depth8 && b == BOUND_EXACT) || relative_age(generation)) {
        key_lo     = uint16_t(key32);
        key_hi     = uint16_t(key32 >> 16);
        depth8     = d;
        gen_bound8 = generation | b;
        if (m != Move::none()) move16 = m;
        value16    = v;
    }
}

uint8_t TranspositionTableEntry::relative_age(uint8_t generation) const {
    return (TranspositionTable::GENERATION_CYCLE + generation - gen_bound8)
            & TranspositionTable::GENERATION_MASK;
}

TranspositionTable::TranspositionTable(size_t mb_size, size_t thread_count) {
    resize(mb_size, thread_count);
}

TranspositionTable::~TranspositionTable() {
    std::free(table);
}

void TranspositionTable::new_search() {
    generation8 += GENERATION_DELTA;
}

TranspositionTableEntry* TranspositionTable::probe(uint64_t key, bool& found) const {
    TranspositionTableEntry* tte = first_entry(key);    
    uint32_t key32 = uint32_t(key);
    for (size_t i = 0; i < CLUSTER_SIZE; ++i) {
        if (tte[i].key_lo == uint16_t(key32) && tte[i].key_hi == uint16_t(key32 >> 16) || !tte[i].depth8) {
            return found = bool(tte[i].depth8), &tte[i];
        }
    }
    TranspositionTableEntry* replace = tte;
    for (size_t i = 1; i < CLUSTER_SIZE; ++i) {
        if (replace->depth8 - replace->relative_age(generation8) * 2
            > tte[i].depth8 - tte[i].relative_age(generation8) * 2) {
            replace = &tte[i];
        }
    }
    return found = false, replace;
}

Move TranspositionTable::get_move(uint64_t key) const {
    bool found;
    TranspositionTableEntry* entry = probe(key, found);
    if (!found) {
        return Move::none();
    }
    return entry->move16;
}

void TranspositionTable::set_move(uint64_t key, Move m) {
    bool found;
    TranspositionTableEntry* entry = probe(key, found);
    if (found && entry->move() == Move::none()) {
        entry->move16 = m;
    }
}

void TranspositionTable::resize(size_t mb_size, size_t thread_count) {
    std::free(table);
    cluster_count = mb_size * 1024 * 1024 / sizeof(Cluster);
    //std::cout << "cluster count: " << cluster_count << std::endl;
    table = static_cast<Cluster*>(aligned_pages_alloc(cluster_count * sizeof(Cluster)));
    if (!table) {
        std::cerr << "Failed to allocate " << mb_size << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }
    clear(thread_count);
}

void TranspositionTable::clear(size_t thread_count) {
    std::vector<std::jthread> threads;
    for (size_t idx = 0; idx < size_t(thread_count); ++idx) {
        threads.emplace_back([this, idx, thread_count] {            
            const size_t stride = size_t(cluster_count / thread_count);
            const size_t start  = size_t(stride * idx);
            const size_t len = idx != thread_count - 1 ? stride : cluster_count - start;
            //std::cout << start << ' ' << len << std::endl;
            std::memset(&table[start], 0, len * sizeof(Cluster));
        });
    }
}

void TranspositionTable::update(uint64_t k, int16_t v, Bound b, uint8_t d, Move m) {
    bool found;
    TranspositionTableEntry* entry = probe(k, found);
    entry->save(k, v, b, d, m, generation8);
}

void TranspositionTable::update(uint64_t k, int16_t v, Bound b, uint8_t d) {
    update(k, v, b, d, Move::none());
}
