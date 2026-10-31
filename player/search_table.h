#ifndef SEARCH_TABLE_H
#define SEARCH_TABLE_H
#include "move.h"
#include <cstddef>
#include <cstdint>

// The transposition table of MinMaxNNUE_DevPlayer (step H), designed for Yolah.
//
// What it remembers of a position: a LOWER and an UPPER bound of its value,
// both proven by searches of the same depth, and the best move found. The
// reference's table (transposition_table.h) keeps one value and one bound
// type: with PVS, a position often gets a lower bound (a null-window search
// that cut) and later an upper bound (one that failed low), and each store
// erased the other. Here they are merged while the depth is the same.
//
// Replacement. Yolah has no cycles and a free square never comes back, so a
// position with MORE free squares than the current root can never be reached
// again: its entry is garbage, for sure. Each entry keeps the number of free
// squares of its position; when a cluster is full, such entries go first,
// then the shallowest one. (The reference ages entries by generations, an
// approximation of the same thing.)
//
// Layout: clusters of 4 entries of 16 bytes = 64 bytes, one cache line,
// aligned: a probe touches a single line. The cluster is chosen by the low
// bits of the hash, the entry is checked with bits 63..16 (key32, key16).
// Only the key bits above the cluster bits tell the positions of a cluster
// apart (and key_of forces bit 32): 39 of them with the 2^24 clusters of a
// 1 GB table, 4 entries per probe → a wrong position about once in 1.4e11
// probes (31 bits before key16: once in 5e8, every 20 moves or so at 16
// threads and 1 s).
// A wrong entry (also a torn one: lazy SMP writes entries without locks) can
// bring the move of another position: users check that it is legal.
class SearchTable {
public:
    static constexpr int16_t NO_BOUND = 32767;   // "no lower/upper bound known"

    struct Entry {
        uint32_t key32;     // bits 63..32 of the hash (0 = empty slot, see store)
        Move     move;      // best move found, or Move::none()
        int16_t  lower;     // value ≥ lower (−NO_BOUND: nothing known)
        int16_t  upper;     // value ≤ upper (+NO_BOUND: nothing known)
        uint8_t  depth;     // depth of the searches that proved the bounds
        uint8_t  free;      // number of free squares of the position
        uint16_t key16;     // bits 31..16 of the hash
        uint8_t  padding[2];
    };
    static_assert(sizeof(Entry) == 16);
    static constexpr int CLUSTER_SIZE = 4;
    struct alignas(64) Cluster {
        Entry entries[CLUSTER_SIZE];
    };
    static_assert(sizeof(Cluster) == 64);

    explicit SearchTable(size_t mb_size);
    ~SearchTable();
    SearchTable(const SearchTable&) = delete;
    SearchTable& operator=(const SearchTable&) = delete;

    void clear();
    // Each search: positions with more free squares than `root_free` are
    // unreachable from now on (see above).
    void new_search(int root_free) { this->root_free = uint8_t(root_free); }
    // The entry of this position, or nullptr.
    const Entry* probe(uint64_t hash) const;
    // What a search of `depth` with the window (alpha, beta) found: a
    // fail-soft value v ≤ alpha is an upper bound, v ≥ beta a lower bound,
    // and in between both.
    void store(uint64_t hash, int free, int depth, int alpha, int beta, int value, Move move);
    // The slot store() writes for this position: its entry, or the one it
    // evicts (for the search debugger: the entry before and after a store).
    const Entry* slot(uint64_t hash) const;
    // Among entries of equal worth (same depth), replace the one with the most
    // free squares — the oldest: written at a shallower ply, an earlier
    // iteration or move — instead of always the first ("table tiebreak").
    bool tiebreak = false;
    void prefetch(uint64_t hash) const { __builtin_prefetch(&clusters[hash & mask]); }
    Move get_move(uint64_t hash) const {
        const Entry* e = probe(hash);
        return e ? e->move : Move::none();
    }
    size_t size_mb() const { return (mask + 1) * sizeof(Cluster) / (1024 * 1024); }
    double load() const;    // share of the entries in use and reachable

private:
    Cluster* clusters = nullptr;
    uint64_t mask = 0;      // number of clusters − 1 (a power of two)
    uint8_t  root_free = 64;
    static uint32_t key_of(uint64_t hash) { return uint32_t(hash >> 32) | 1; }  // never 0
    static uint16_t key16_of(uint64_t hash) { return uint16_t(hash >> 16); }
};

#endif
