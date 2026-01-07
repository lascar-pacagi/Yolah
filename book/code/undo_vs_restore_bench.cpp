// Benchmark: undo() vs save/restore state
// Compares the performance of using undo() versus saving and restoring the
// complete game state in a recursive tree traversal.

#include <benchmark/benchmark.h>
#include <bits/stdc++.h>

using namespace std;

// =============================================================================
// BASIC TYPES AND CONSTANTS
// =============================================================================

enum Square : int8_t {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_ZERO = 0,
    SQUARE_NB   = 64
};

enum Direction : int8_t {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,
    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

constexpr uint8_t BLACK = 0;
constexpr uint8_t WHITE = 1;

constexpr uint64_t FileABB = 0x0101010101010101;
constexpr uint64_t FileHBB = FileABB << 7;

constexpr uint64_t BLACK_INITIAL_POSITION =
    0b10000000'00000000'00000000'00001000'00010000'00000000'00000000'00000001;
constexpr uint64_t WHITE_INITIAL_POSITION =
    0b00000001'00000000'00000000'00010000'00001000'00000000'00000000'10000000;

// =============================================================================
// BIT MANIPULATION
// =============================================================================

inline Square lsb(uint64_t b) {
    return Square(std::countr_zero(b));
}

inline Square pop_lsb(uint64_t& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr uint64_t square_bb(Square s) {
    return uint64_t(1) << s;
}

template<Direction D>
constexpr uint64_t shift(uint64_t b) {
    if constexpr (D == NORTH)           return b << 8;
    else if constexpr (D == SOUTH)      return b >> 8;
    else if constexpr (D == EAST)       return (b & ~FileHBB) << 1;
    else if constexpr (D == WEST)       return (b & ~FileABB) >> 1;
    else if constexpr (D == NORTH_EAST) return (b & ~FileHBB) << 9;
    else if constexpr (D == NORTH_WEST) return (b & ~FileABB) << 7;
    else if constexpr (D == SOUTH_EAST) return (b & ~FileHBB) >> 7;
    else if constexpr (D == SOUTH_WEST) return (b & ~FileABB) >> 9;
    else return 0;
}

// =============================================================================
// MOVE
// =============================================================================

struct Move {
    uint16_t data;
    Move() = default;
    constexpr explicit Move(uint16_t d) : data(d) {}
    constexpr explicit Move(Square from, Square to) : data((to << 6) + from) {}
    constexpr Square to_sq() const { return Square((data >> 6) & 0x3F); }
    constexpr Square from_sq() const { return Square(data & 0x3F); }
    static constexpr Move none() { return Move(0); }
    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }
};

// =============================================================================
// MAGIC BITBOARDS
// =============================================================================

struct Magic {
    uint64_t  mask;
    uint64_t  magic;
    uint64_t* moves;
    uint32_t  shift;

    uint32_t index(uint64_t occupied) const {
        return uint32_t(((occupied & mask) * magic) >> shift);
    }
};

namespace {
    uint64_t orthogonalTable[102400];
    uint64_t diagonalTable[5248];
}

Magic orthogonalMagics[SQUARE_NB];
Magic diagonalMagics[SQUARE_NB];

uint64_t moves_bb(Square sq, uint64_t occupied) {
    return orthogonalMagics[sq].moves[orthogonalMagics[sq].index(occupied)] |
           diagonalMagics[sq].moves[diagonalMagics[sq].index(occupied)];
}

// =============================================================================
// MOVELIST
// =============================================================================

static constexpr uint16_t MAX_NB_MOVES = 75;

struct MoveList {
    Move move_list[MAX_NB_MOVES], *last;

    MoveList() : last(move_list) {}
    const Move* begin() const { return move_list; }
    const Move* end() const { return last; }
    size_t size() const { return last - move_list; }
};

// =============================================================================
// YOLAH CLASS
// =============================================================================

class Yolah {
public:
    uint64_t black = BLACK_INITIAL_POSITION;
    uint64_t white = WHITE_INITIAL_POSITION;
    uint64_t holes = 0;
    uint8_t black_score = 0;
    uint8_t white_score = 0;
    uint8_t ply = 0;

    bool game_over() const noexcept {
        uint64_t possible = ~holes & ~black & ~white;
        uint64_t players  = black | white;
        uint64_t around_players = shift<NORTH>(players) |
            shift<SOUTH>(players) | shift<EAST>(players) |
            shift<WEST>(players) | shift<NORTH_EAST>(players) |
            shift<NORTH_WEST>(players) | shift<SOUTH_EAST>(players) |
            shift<SOUTH_WEST>(players);
        return (around_players & possible) == 0;
    }

    uint8_t current_player() const noexcept {
        return ply & 1;
    }

    void moves(MoveList& moves) const noexcept {
        Move* ml = moves.move_list;
        uint64_t occupied = black | white | holes;
        uint64_t bb = (ply & 1) ? white : black;
        while (bb) {
            Square from = pop_lsb(bb);
            uint64_t b = moves_bb(from, occupied) & ~occupied;
            while (b) {
                *ml++ = Move(from, pop_lsb(b));
            }
        }
        if (ml == moves.move_list) [[unlikely]] {
            *ml++ = Move::none();
        }
        moves.last = ml;
    }

    void play(Move m) noexcept {
        if (m != Move::none()) [[likely]] {
            uint64_t pos1 = square_bb(m.from_sq());
            uint64_t pos2 = square_bb(m.to_sq());
            if (ply & 1) {
                white ^= pos1 | pos2;
                white_score++;
            } else {
                black ^= pos1 | pos2;
                black_score++;
            }
            holes |= pos1;
        }
        ply++;
    }

    void undo(Move m) noexcept {
        ply--;
        if (m != Move::none()) [[likely]] {
            uint64_t pos1 = square_bb(m.from_sq());
            uint64_t pos2 = square_bb(m.to_sq());
            if (ply & 1) {
                white ^= pos1 | pos2;
                white_score--;
            } else {
                black ^= pos1 | pos2;
                black_score--;
            }
            holes ^= pos1;
        }
    }
};

// =============================================================================
// MAGIC BITBOARD INITIALIZATION
// =============================================================================

uint64_t sliding_attack(Square sq, uint64_t occupied, const Direction deltas[4]) {
    uint64_t attack = 0;
    for (int i = 0; i < 4; ++i) {
        int s = sq;
        while (true) {
            int prev_file = s & 7;
            int prev_rank = s >> 3;
            s += deltas[i];
            int new_file = s & 7;
            int new_rank = s >> 3;
            if (s < 0 || s >= 64) break;
            if (abs(new_file - prev_file) > 1 || abs(new_rank - prev_rank) > 1) break;
            attack |= uint64_t(1) << s;
            if (occupied & (uint64_t(1) << s)) break;
        }
    }
    return attack;
}

uint64_t make_mask(Square sq, const Direction deltas[4]) {
    uint64_t edges = ((FileABB | FileHBB) & ~(FileABB << (sq & 7)) & ~(FileABB << (sq & 7))) |
                     ((0xFFULL | 0xFFULL << 56) & ~(0xFFULL << (8 * (sq >> 3))));
    uint64_t attack = sliding_attack(sq, 0, deltas);
    return attack & ~edges;
}

void init_magics(Magic magics[], uint64_t* table, const Direction deltas[4]) {
    uint64_t seeds[8] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};
    uint64_t occupancy[4096], reference[4096];
    int size = 0;

    for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(sq + 1)) {
        Magic& m = magics[sq];
        m.mask = make_mask(sq, deltas);
        m.shift = 64 - __builtin_popcountll(m.mask);
        m.moves = (sq == SQ_A1) ? table : magics[sq - 1].moves + size;

        uint64_t b = 0;
        size = 0;
        do {
            occupancy[size] = b;
            reference[size] = sliding_attack(sq, b, deltas);
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

        mt19937_64 rng(seeds[sq >> 3]);
        for (int i = 0; i < size; ) {
            for (m.magic = 0; __builtin_popcountll((m.mask * m.magic) >> 56) < 6; )
                m.magic = rng() & rng() & rng();

            for (int j = 0; j < size; j++)
                m.moves[j] = 0;

            for (i = 0; i < size; i++) {
                uint32_t idx = m.index(occupancy[i]);
                if (m.moves[idx] && m.moves[idx] != reference[i])
                    break;
                m.moves[idx] = reference[i];
            }
        }
    }
}

void init_all_magics() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    const Direction orthogonal[4] = {NORTH, SOUTH, EAST, WEST};
    const Direction diagonal[4] = {NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST};
    init_magics(orthogonalMagics, orthogonalTable, orthogonal);
    init_magics(diagonalMagics, diagonalTable, diagonal);
}

// =============================================================================
// RECURSIVE TRAVERSAL FUNCTIONS
// =============================================================================

uint64_t count_nodes_with_undo(Yolah& yolah, int depth) {
    if (depth == 0 || yolah.game_over()) {
        return 1;
    }
    uint64_t count = 0;
    MoveList moves;
    yolah.moves(moves);
    for (const Move& m : moves) {
        yolah.play(m);
        count += count_nodes_with_undo(yolah, depth - 1);
        yolah.undo(m);
    }
    return count;
}

uint64_t count_nodes_with_restore(Yolah& yolah, int depth) {
    if (depth == 0 || yolah.game_over()) {
        return 1;
    }
    uint64_t count = 0;
    MoveList moves;
    yolah.moves(moves);
    // Save the complete state
    Yolah saved = yolah;
    for (const Move& m : moves) {
        yolah.play(m);
        count += count_nodes_with_restore(yolah, depth - 1);
        // Restore the complete state
        yolah = saved;
    }
    return count;
}

// =============================================================================
// BENCHMARKS
// =============================================================================

static void BM_recursive_with_undo(benchmark::State& state) {
    int depth = state.range(0);
    for (auto _ : state) {
        Yolah yolah;
        uint64_t nodes = count_nodes_with_undo(yolah, depth);
        benchmark::DoNotOptimize(nodes);
    }
    // Calculate nodes for reporting
    Yolah yolah;
    uint64_t nodes = count_nodes_with_undo(yolah, depth);
    state.SetItemsProcessed(state.iterations() * nodes);
    state.counters["nodes"] = nodes;
}

static void BM_recursive_with_restore(benchmark::State& state) {
    int depth = state.range(0);
    for (auto _ : state) {
        Yolah yolah;
        uint64_t nodes = count_nodes_with_restore(yolah, depth);
        benchmark::DoNotOptimize(nodes);
    }
    // Calculate nodes for reporting
    Yolah yolah;
    uint64_t nodes = count_nodes_with_restore(yolah, depth);
    state.SetItemsProcessed(state.iterations() * nodes);
    state.counters["nodes"] = nodes;
}

// Test at various depths
BENCHMARK(BM_recursive_with_undo)->Arg(3)->Arg(4)->Arg(5);
BENCHMARK(BM_recursive_with_restore)->Arg(3)->Arg(4)->Arg(5);
int main(int argc, char** argv) {
    init_all_magics();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
