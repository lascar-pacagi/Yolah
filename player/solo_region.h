#ifndef SOLO_REGION_H
#define SOLO_REGION_H
#include "game.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

// Private regions of free squares, solved exactly (the retrograde analysis
// project, doc/solving.qmd "Regions").
//
// A REGION is an 8-connected group of free squares. A queen move only crosses
// free squares, and two consecutive squares of a line are neighbours, so a
// piece only ever moves inside the regions it touches, and regions never merge
// (squares only stop being free). A region touched by the pieces of ONE side
// only is private for good: an opponent piece can only move to free squares of
// its own regions, and a square next to ours would belong to ours.
//
// A CLUSTER is the unit that is really independent: regions linked by the
// pieces that touch several of them (a piece can go into one or the other, not
// both), with those pieces. A private cluster is a one-player puzzle: its VALUE
// is the largest number of moves its side can still make in it. It depends
// neither on the rest of the board nor on when the moves are made (nobody else
// can touch it), so a side with private clusters of total value M owns M
// points and M tempi it can spend whenever it likes.
//
// value() solves a cluster by a search with memory: after a move the cluster
// may fall apart into smaller clusters, whose values add up. The memory is
// keyed by the cluster's CANONICAL form: queen lines do not depend on where a
// shape stands nor on the board's 8 symmetries (rotations, reflections), so the
// key is the smallest of the 8 transformed shapes, translated to the corner.
namespace solo {

// A cluster: its free squares and the pieces (of one side) next to them.
struct Cluster {
    uint64_t free = 0, pieces = 0;
};

// The canonical form of (free, pieces): two bitboards, compared as a pair.
struct Key {
    uint64_t free = 0, pieces = 0;
    bool operator==(const Key& o) const { return free == o.free && pieces == o.pieces; }
};
struct KeyHash {
    size_t operator()(const Key& k) const { return size_t(k.free * 0x9E3779B97F4A7C15ull ^ (k.pieces + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full); }
};

Key canonical(uint64_t free, uint64_t pieces);

// The 8-connected regions of `free`.
int regions(uint64_t free, uint64_t out[64]);

// The clusters of `free` with `pieces` (one side's pieces): regions joined by
// the pieces touching several of them; pieces touching no region are left out
// (no move: value 0). Returns the number of clusters.
int clusters(uint64_t free, uint64_t pieces, Cluster out[64]);

// For a position: the free squares and pieces of the clusters private to each
// side (regions touched by no piece of the other side), and the squares of the
// contested regions (touched by both).
struct Split {
    std::vector<Cluster> priv[2];   // BLACK, WHITE
    uint64_t contested = 0;
};
Split split(const Yolah& y);

// Most clusters can be filled entirely: their value is their number of free
// squares, the most possible (each move fills one). The search stops as soon as
// it finds such a sequence, and the memory can keep only the EXCEPTIONS (value
// below the number of free squares): a cluster not in the memory is searched
// again, which is cheap when it can be filled (the first lines found usually
// fill it).
class Solver {
public:
    // false: only the exceptions are kept in `memo` (much less memory).
    bool keep_full = true;
    // The largest number of moves `pieces` can make in `free` (any clusters).
    int value(uint64_t free, uint64_t pieces);
    // The same within `budget` new clusters solved, −1 beyond (nothing wrong is
    // kept: the clusters left unfinished are not stored).
    int value_budget(uint64_t free, uint64_t pieces, int64_t budget) {
        budget_left = budget;
        over = false;
        const int v = value(free, pieces);
        budget_left = -1;
        return over ? -1 : v;
    }
    size_t size() const { return memo.size(); }
    uint64_t nodes() const { return nb_nodes; }
    // The table: canonical cluster → value (for saving, or filling from a file).
    std::unordered_map<Key, int8_t, KeyHash> memo;
private:
    uint64_t nb_nodes = 0;
    int64_t budget_left = -1;   // value_budget: new clusters still allowed (−1: no budget)
    bool over = false;          //   the budget ran out
    // A small direct-mapped cache of clusters as they stand on the board (no
    // canonical form to compute: the 8 symmetries cost more than a lookup). From a
    // node to the next, the private clusters are often the very same.
    struct Raw { uint64_t free = 0, pieces = 0; int8_t value = -1; };
    static constexpr int RAW_BITS = 16;
    std::vector<Raw> raw = std::vector<Raw>(size_t(1) << RAW_BITS);
    int cluster_value(uint64_t free, uint64_t pieces);
};

}  // namespace solo

#endif
