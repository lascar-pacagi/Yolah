#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H
#include "move.h"
#include "misc.h"

enum Bound {
    BOUND_NONE,
    BOUND_UPPER,
    BOUND_LOWER,
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

class TranspositionTableEntry {
    friend class TranspositionTable;
    uint16_t key_lo;
    uint16_t key_hi;
    uint8_t  depth8;
    uint8_t  gen_bound8;
    Move     move16;
    int16_t  value16;

public:
    Move    move() const { return move16; }
    int16_t value() const { return value16; }
    uint8_t depth() const { return depth8; }
    Bound   bound() const { return Bound(gen_bound8 & 3); }
    void    save(uint64_t k, int16_t v, Bound b, uint8_t d, Move m, uint8_t generation);
    uint8_t relative_age(uint8_t generation) const;
};

class TranspositionTable {
    friend struct TranspositionTableEntry;
    static constexpr size_t CLUSTER_SIZE = 3;
    struct Cluster {
        TranspositionTableEntry entries[CLUSTER_SIZE];
        char padding[2];
    };
    static_assert(sizeof(Cluster) == 32, "Unexpected Cluster size");
    static constexpr unsigned GENERATION_BITS = 2;
    static constexpr int GENERATION_DELTA = (1 << GENERATION_BITS);
    static constexpr int GENERATION_CYCLE = 255 + GENERATION_DELTA;
    static constexpr int GENERATION_MASK = (0xFF << GENERATION_BITS) & 0xFF;
    size_t   cluster_count = 0;
    Cluster* table         = nullptr;
    uint8_t  generation8   = 0;  // Size must be not bigger than TranspositionTableEntry::gen_bound8
public:
    TranspositionTable(size_t mb_size, size_t thread_count = 1);
    ~TranspositionTable();
    void new_search();
    TranspositionTableEntry* probe(uint64_t key, bool& found) const;
    void resize(size_t mb_size, size_t thread_count);
    void clear(size_t thread_count);
    TranspositionTableEntry* first_entry(uint64_t key) const {
        return &table[mul_hi64(key, cluster_count)].entries[0];
    }
    uint8_t generation() const {
        return generation8;
    }
};

#endif