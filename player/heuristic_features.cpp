#include "heuristic_features.h"
#include "misc.h"
#include <iostream>
#include <iomanip>

namespace heuristic_features {

    // Game phase thresholds (based on free squares)
    constexpr int MIDDLE_GAME_THRESHOLD = 36;
    constexpr int END_GAME_THRESHOLD = 18;

    uint64_t floodfill(uint64_t start, uint64_t allowed) {
        uint64_t prev = 0;
        uint64_t flood = start;
        while (prev != flood) {
            prev = flood;
            flood |= shift_all_directions(flood) & allowed;
        }
        return flood;
    }

    int count_groups(uint64_t pieces) {
        int groups = 0;
        while (pieces) {
            uint64_t piece = pieces & -pieces;  // isolate LSB
            // Floodfill on the pieces themselves (not free squares)
            uint64_t group = floodfill(piece, pieces);
            groups++;
            pieces &= ~group;  // remove this group
        }
        return groups;
    }

    int min_distance_to_opponent(uint64_t us, uint64_t them, uint64_t free) {
        if (!us || !them) return 64;  // max distance if one side has no pieces

        uint64_t frontier = us;
        int dist = 0;
        const int max_dist = 16;  // early exit

        while (dist < max_dist) {
            if (frontier & them) return dist;
            uint64_t next = shift_all_directions(frontier) & (free | them);
            if (next == frontier) return max_dist;  // no progress
            frontier = next;
            dist++;
        }
        return max_dist;
    }

    std::array<int, 3> freedom_buckets(uint64_t player, uint64_t free) {
        std::array<int, 3> buckets = {0, 0, 0};
        while (player) {
            uint64_t piece = player & -player;
            int sq = std::countr_zero(piece);
            int neighbors = std::popcount(AROUND[sq] & free);

            if (neighbors <= 2) buckets[0]++;
            else if (neighbors <= 5) buckets[1]++;
            else buckets[2]++;

            player &= player - 1;  // clear LSB
        }
        return buckets;
    }

    // Connectivity: sum of reachable squares from each piece
    static int connectivity(uint64_t player, uint64_t free) {
        int total = 0;
        while (player) {
            uint64_t piece = player & -player;
            uint64_t reachable = floodfill(piece, free);
            total += std::popcount(reachable);
            player &= player - 1;
        }
        return total;
    }

    // Connectivity set: unique reachable squares (no double counting)
    static int connectivity_set(uint64_t player, uint64_t free) {
        return std::popcount(floodfill(player, free));
    }

    // Alone: squares only reachable by us (opponent can't reach)
    static int alone(uint64_t us, uint64_t them, uint64_t free) {
        uint64_t opponent_reach = floodfill(them, free);
        uint64_t our_reach = floodfill(us, free);
        return std::popcount(our_reach & ~opponent_reach);
    }

    // First: squares we can reach in 1 move that opponent can't
    static int first(const Yolah::MoveList& our_moves, const Yolah::MoveList& their_moves) {
        uint64_t our_targets = 0;
        uint64_t their_targets = 0;

        for (const Move& m : our_moves) {
            if (m == Move::none()) break;
            our_targets |= square_bb(m.to_sq());
        }
        for (const Move& m : their_moves) {
            if (m == Move::none()) break;
            their_targets |= square_bb(m.to_sq());
        }

        return std::popcount(our_targets & ~their_targets);
    }

    // Influence: Voronoi-like territory based on expansion distance
    static std::pair<int, int> influence(uint64_t black, uint64_t white, uint64_t free) {
        uint64_t black_influence = black;
        uint64_t white_influence = white;
        uint64_t black_frontier = black;
        uint64_t white_frontier = white;
        uint64_t neutral = 0;

        uint64_t prev_black = 0;
        uint64_t prev_white = 0;

        while (prev_black != black_influence || prev_white != white_influence) {
            prev_black = black_influence;
            prev_white = white_influence;

            uint64_t next_black = shift_all_directions(black_frontier) & free & ~white_influence;
            uint64_t next_white = shift_all_directions(white_frontier) & free & ~black_influence;

            // Squares reached by both become neutral
            neutral |= next_black & next_white;
            black_frontier = next_black & ~neutral;
            white_frontier = next_white & ~neutral;

            black_influence |= black_frontier;
            white_influence |= white_frontier;
        }

        return {std::popcount(black_influence), std::popcount(white_influence)};
    }

    Features extract(const Yolah& yolah) {
        Yolah::MoveList black_moves, white_moves;
        yolah.moves(Yolah::BLACK, black_moves);
        yolah.moves(Yolah::WHITE, white_moves);
        return extract(yolah, black_moves, white_moves);
    }

    Features extract(const Yolah& yolah,
                     const Yolah::MoveList& black_moves,
                     const Yolah::MoveList& white_moves) {
        Features f;

        const uint64_t black = yolah.bitboard(Yolah::BLACK);
        const uint64_t white = yolah.bitboard(Yolah::WHITE);
        const uint64_t free = yolah.free_squares();
        const int free_count = std::popcount(free);

        // Material and score
        f[MATERIAL_DIFF] = static_cast<float>(material_diff(black, white));
        const auto [black_score, white_score] = yolah.score();
        f[SCORE_DIFF] = static_cast<float>(black_score) - static_cast<float>(white_score);

        // Mobility
        f[MOBILITY_DIFF] = static_cast<float>(
            static_cast<int>(black_moves.size()) - static_cast<int>(white_moves.size())
        );
        f[FREE_SQUARES] = static_cast<float>(free_count);

        // Center control
        f[CENTER_CONTROL] = static_cast<float>(center_control(black, white));
        f[INNER_CENTER] = static_cast<float>(inner_center(black, white));
        f[EXTENDED_CENTER] = static_cast<float>(
            std::popcount(black & masks::EXTENDED_6x6) -
            std::popcount(white & masks::EXTENDED_6x6)
        );

        // Edges and corners
        f[EDGE_PIECES] = static_cast<float>(edge_pieces(black, white));
        f[CORNER_PIECES] = static_cast<float>(corner_pieces(black, white));
        f[NEAR_CORNER] = static_cast<float>(
            std::popcount(black & masks::NEAR_CORNERS) -
            std::popcount(white & masks::NEAR_CORNERS)
        );

        // Structural
        f[BLOCKED_PIECES] = static_cast<float>(
            blocked_pieces(black, free) - blocked_pieces(white, free)
        );
        f[FRONTIER_PIECES] = static_cast<float>(
            frontier_pieces(black, free) - frontier_pieces(white, free)
        );
        f[GROUPS_DIFF] = static_cast<float>(
            count_groups(black) - count_groups(white)
        );
        f[TENSION] = static_cast<float>(min_distance_to_opponent(black, white, free));

        // Territorial (from existing heuristics)
        f[CONNECTIVITY_DIFF] = static_cast<float>(
            connectivity(black, free) - connectivity(white, free)
        );
        f[CONNECTIVITY_SET_DIFF] = static_cast<float>(
            connectivity_set(black, free) - connectivity_set(white, free)
        );
        f[ALONE_DIFF] = static_cast<float>(
            alone(black, white, free) - alone(white, black, free)
        );
        f[FIRST_DIFF] = static_cast<float>(
            first(black_moves, white_moves) - first(white_moves, black_moves)
        );

        const auto [black_infl, white_infl] = influence(black, white, free);
        f[INFLUENCE_DIFF] = static_cast<float>(black_infl - white_infl);

        // Freedom distribution
        auto black_freedom = freedom_buckets(black, free);
        auto white_freedom = freedom_buckets(white, free);
        f[FREEDOM_LOW] = static_cast<float>(black_freedom[0] - white_freedom[0]);
        f[FREEDOM_MID] = static_cast<float>(black_freedom[1] - white_freedom[1]);
        f[FREEDOM_HIGH] = static_cast<float>(black_freedom[2] - white_freedom[2]);

        // Game phase: 0 = opening, 0.5 = middle, 1 = endgame
        if (free_count > MIDDLE_GAME_THRESHOLD) {
            f[GAME_PHASE] = 0.0f;
        } else if (free_count > END_GAME_THRESHOLD) {
            f[GAME_PHASE] = 0.5f;
        } else {
            f[GAME_PHASE] = 1.0f;
        }

        // Turn
        f[TURN] = (yolah.current_player() == Yolah::WHITE) ? 1.0f : 0.0f;

        return f;
    }

    void print_features(const Features& f) {
        const char* names[] = {
            "MATERIAL_DIFF", "SCORE_DIFF", "MOBILITY_DIFF", "FREE_SQUARES",
            "CENTER_CONTROL", "INNER_CENTER", "EXTENDED_CENTER",
            "EDGE_PIECES", "CORNER_PIECES", "NEAR_CORNER",
            "BLOCKED_PIECES", "FRONTIER_PIECES", "GROUPS_DIFF", "TENSION",
            "CONNECTIVITY_DIFF", "CONNECTIVITY_SET_DIFF", "ALONE_DIFF",
            "FIRST_DIFF", "INFLUENCE_DIFF",
            "FREEDOM_LOW", "FREEDOM_MID", "FREEDOM_HIGH",
            "GAME_PHASE", "TURN"
        };

        std::cout << "=== Heuristic Features ===" << std::endl;
        for (int i = 0; i < NB_FEATURES; i++) {
            std::cout << std::setw(22) << std::left << names[i]
                      << ": " << std::setw(8) << std::right << f[i] << std::endl;
        }
    }
}
