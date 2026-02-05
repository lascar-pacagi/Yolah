#ifndef HEURISTIC_FEATURES_H
#define HEURISTIC_FEATURES_H

#include "game.h"
#include <array>
#include <cstdint>
#include <bit>

namespace heuristic_features {

    // Feature indices for the small NN
    enum FeatureIndex {
        // Material and score (2 features)
        MATERIAL_DIFF,          // popcount(black) - popcount(white)
        SCORE_DIFF,             // black_score - white_score

        // Mobility (2 features)
        MOBILITY_DIFF,          // nb_black_moves - nb_white_moves
        FREE_SQUARES,           // popcount(free) - game phase indicator

        // Positional - center control (3 features)
        CENTER_CONTROL,         // pieces in c3-f6 area
        INNER_CENTER,           // pieces in d4-e5 area
        EXTENDED_CENTER,        // pieces in b2-g7 area

        // Positional - edges and corners (3 features)
        EDGE_PIECES,            // pieces on edges (bad)
        CORNER_PIECES,          // pieces in corners (very bad)
        NEAR_CORNER,            // pieces adjacent to corners

        // Structural (4 features)
        BLOCKED_PIECES,         // pieces with 0 freedom
        FRONTIER_PIECES,        // pieces adjacent to free squares
        GROUPS_DIFF,            // connected components difference
        TENSION,                // min distance between opponents

        // Territorial - from existing heuristics (5 features)
        CONNECTIVITY_DIFF,      // sum of connected squares per piece
        CONNECTIVITY_SET_DIFF,  // unique connected squares
        ALONE_DIFF,             // squares only we can reach
        FIRST_DIFF,             // squares we reach first (1 move)
        INFLUENCE_DIFF,         // Voronoi-like territory

        // Freedom distribution (3 features - simplified from 9x3)
        FREEDOM_LOW,            // pieces with 0-2 free neighbors
        FREEDOM_MID,            // pieces with 3-5 free neighbors
        FREEDOM_HIGH,           // pieces with 6-8 free neighbors

        // Game phase (1 feature)
        GAME_PHASE,             // 0=opening, 0.5=middle, 1=end

        // Turn (1 feature)
        TURN,                   // 0=black to play, 1=white to play

        NB_FEATURES             // Total: 24 features
    };

    // Precomputed masks for fast feature extraction
    namespace masks {
        // Center masks
        constexpr uint64_t CENTER_4x4    = 0x00003C3C3C3C0000ULL; // c3-f6
        constexpr uint64_t INNER_2x2     = 0x0000001818000000ULL; // d4-e5
        constexpr uint64_t EXTENDED_6x6  = 0x007E7E7E7E7E7E00ULL; // b2-g7

        // Edge and corner masks
        constexpr uint64_t EDGES         = 0xFF818181818181FFULL; // all edges
        constexpr uint64_t CORNERS       = 0x8100000000000081ULL; // a1, h1, a8, h8
        constexpr uint64_t NEAR_CORNERS  = 0x42C300000000C342ULL; // adjacent to corners

        // Quadrant masks (for potential future use)
        constexpr uint64_t QUADRANT_NW   = 0xF0F0F0F000000000ULL;
        constexpr uint64_t QUADRANT_NE   = 0x0F0F0F0F00000000ULL;
        constexpr uint64_t QUADRANT_SW   = 0x00000000F0F0F0F0ULL;
        constexpr uint64_t QUADRANT_SE   = 0x000000000F0F0F0FULL;

        // Diagonal masks
        constexpr uint64_t MAIN_DIAG     = 0x8040201008040201ULL;
        constexpr uint64_t ANTI_DIAG     = 0x0102040810204080ULL;
    }

    // Feature vector type
    using Features = std::array<float, NB_FEATURES>;

    // Fast feature extraction
    Features extract(const Yolah& yolah);

    // Extract features with precomputed moves (faster if you already have them)
    Features extract(const Yolah& yolah,
                     const Yolah::MoveList& black_moves,
                     const Yolah::MoveList& white_moves);

    // Individual fast features (O(1) or O(popcount))
    inline int material_diff(uint64_t black, uint64_t white) {
        return std::popcount(black) - std::popcount(white);
    }

    inline int center_control(uint64_t black, uint64_t white) {
        return std::popcount(black & masks::CENTER_4x4) -
               std::popcount(white & masks::CENTER_4x4);
    }

    inline int inner_center(uint64_t black, uint64_t white) {
        return std::popcount(black & masks::INNER_2x2) -
               std::popcount(white & masks::INNER_2x2);
    }

    inline int edge_pieces(uint64_t black, uint64_t white) {
        return std::popcount(black & masks::EDGES) -
               std::popcount(white & masks::EDGES);
    }

    inline int corner_pieces(uint64_t black, uint64_t white) {
        return std::popcount(black & masks::CORNERS) -
               std::popcount(white & masks::CORNERS);
    }

    inline int blocked_pieces(uint64_t player, uint64_t free) {
        // Pieces with no adjacent free squares
        uint64_t expandable = shift_all_directions(free);
        return std::popcount(player & ~expandable);
    }

    inline int frontier_pieces(uint64_t player, uint64_t free) {
        // Pieces adjacent to at least one free square
        uint64_t adjacent_to_free = shift_all_directions(free);
        return std::popcount(player & adjacent_to_free);
    }

    // Count connected groups (components) of pieces
    int count_groups(uint64_t pieces);

    // Minimum distance between player pieces and opponent
    int min_distance_to_opponent(uint64_t us, uint64_t them, uint64_t free);

    // Freedom counts: how many pieces have 0-2, 3-5, 6-8 free neighbors
    std::array<int, 3> freedom_buckets(uint64_t player, uint64_t free);

    // Floodfill for connectivity
    uint64_t floodfill(uint64_t start, uint64_t allowed);

    // Print features for debugging
    void print_features(const Features& f);
}

#endif
