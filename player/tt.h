#ifndef TT_H 
#define TT_H 
#include "transposition_table.h"

class TranspositionTableSlow {
public:
    struct Entry {
        uint64_t key  = 0;
        int16_t value = 0;
        Move move     = Move::none();
        uint8_t depth = 0;
        Bound bound   = BOUND_NONE;
    };
private:
    std::vector<Entry> table;
    uint64_t mask;
public:
    TranspositionTableSlow(size_t bits);
    Entry* get_entry(uint64_t k);
    Move get_move(uint64_t k);
    void update(uint64_t k, int16_t v, Bound b, uint8_t d, Move m);
    void new_search();
};

#endif
