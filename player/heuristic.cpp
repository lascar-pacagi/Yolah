#include "heuristic.h"
#include <utility>
#include <bitset>
#include <bit>

/*
    1. No move
    2. Mobility: sum of squares reachable by each piece.
    3. Mobility set: sum of squares reachable by each piece without counting the same square twice.
    4. Connectivity: sum of squares connected to each piece.
    5. Connectivity set: sum of squares connected to each piece without counting the same square twice.
    6. Alone: number of squares owns by the player.
*/

namespace heuristic {
    namespace {
        using pair = std::pair<int32_t, int32_t>;
        pair operator+(const pair& p1, const pair& p2) {
            return { p1.first + p2.first, p1.second + p2.second };
        }
        pair operator*(int32_t coeff, const pair& p) {
            return { coeff * p.first, coeff * p.second };
        }
        pair operator*(const pair& p, int32_t coeff) {
            return coeff * p;
        }

        constexpr uint32_t NO_MOVE_WEIGHT = -1000;
        constexpr uint32_t NB_MOVES_WEIGHT = 1;
        constexpr uint32_t MOBILITY_WEIGHT = 1;
        constexpr uint32_t CONNECTIVITY_WEIGHT = 1;
        constexpr uint32_t CONNECTIVITY_SET_WEIGHT = 1;
        constexpr uint32_t ALONE_WEIGHT = 1;

        int32_t mobility(const Yolah::MoveList& moves) {
            uint64_t seen = 0;        
            int32_t res = 0;
            for (Move m : moves) {
                uint64_t bb = square_bb(m.to_sq());
                if (!(seen & bb)) {
                    res++;
                    seen |= bb;
                }
            }
            return res;
        }
        uint64_t floodfill(uint64_t player_bb, uint64_t free) {
            uint64_t prev_flood = 0;   
            uint64_t flood = player_bb;            
            //std::cout << Bitboard::pretty(flood) << std::endl;
            int count = 0;
            while (prev_flood ^ flood) {
                prev_flood = flood;
                uint64_t flood1 = shift<NORTH>(flood) & free;
                uint64_t flood2 = shift<SOUTH>(flood) & free;
                uint64_t flood3 = shift<EAST>(flood) & free;
                uint64_t flood4 = shift<WEST>(flood) & free;
                uint64_t flood5 = shift<NORTH_EAST>(flood) & free;
                uint64_t flood6 = shift<SOUTH_EAST>(flood) & free;
                uint64_t flood7 = shift<NORTH_WEST>(flood) & free;
                uint64_t flood8 = shift<SOUTH_WEST>(flood) & free;
                flood |= flood1 | flood2 | flood3 | flood4 | flood5 | flood6 | flood7 | flood8;
                count++;
            }
            flood ^= player_bb;
            // std::cout << count << std::endl;
            // std::cout << Bitboard::pretty(flood) << std::endl;
            return flood;
        }
        int32_t connectivity(uint64_t player_bb, uint64_t free) {
            return std::popcount(floodfill(player_bb, free));
        }
        int32_t connectivity(uint8_t player, const Yolah& yolah) {
             uint64_t player_bb = yolah.bitboard(player);
             uint64_t free = yolah.free_squares();             
             int32_t res = 0;
             while (player_bb) {
                uint64_t bb = player_bb & -player_bb;
                res += connectivity(bb, free);
                player_bb &= ~bb;
             }
             return res;
        }
        int32_t connectivity_set(uint8_t player, const Yolah& yolah) {
            return connectivity(yolah.bitboard(player), yolah.free_squares());
        }
        int32_t alone(uint8_t player, const Yolah& yolah) {
            uint64_t player_bb = yolah.bitboard(player);
            uint64_t opponent_bb = yolah.bitboard(Yolah::other_player(player));
            uint64_t free = yolah.free_squares();
            uint64_t flood_opponent = floodfill(opponent_bb, free);
            std::cout << Bitboard::pretty(flood_opponent) << std::endl;
            int32_t res = 0;
            while (player_bb) {
                uint64_t bb = player_bb & -player_bb;
                std::cout << Bitboard::pretty(bb) << std::endl;
                uint64_t flood = floodfill(bb, free);
                std::cout << Bitboard::pretty(flood) << std::endl;
                if (!(flood & flood_opponent)) {
                    std::cout << std::popcount(flood) << std::endl;
                    res += std::popcount(flood);
                }
                player_bb &= ~bb;
            }
            return res;
        }
        int32_t nb_connected_components(const Yolah& yolah) {
            uint64_t free = yolah.free_squares();
            int32_t res = 0;
            while (free) {
                uint64_t bb = free & -free;
                uint64_t flood = floodfill(bb, free);                
                res++;
                free &= ~(bb | flood);
            }
            return res;
        }
    }
    int32_t eval(uint8_t player, const Yolah& yolah) {
        std::cout << yolah << std::endl;
        std::cout << Bitboard::pretty(yolah.free_squares()) << std::endl;
        Yolah::MoveList black_moves, white_moves;
        yolah.moves(Yolah::BLACK, black_moves);
        yolah.moves(Yolah::WHITE, white_moves);
        std::cout << nb_connected_components(yolah) << std::endl;
        pair res{};
        /*res = //NO_MOVE_WEIGHT * pair{ black_moves[0] == Move::none(), white_moves[0] == Move::none() } +
              //NB_MOVES_WEIGHT * pair{ black_moves.size(), white_moves.size() } +
              //MOBILITY_WEIGHT * pair{ mobility(black_moves), mobility(white_moves) } +
              //CONNECTIVITY_WEIGHT * pair{ connectivity(Yolah::BLACK, yolah), connectivity(Yolah::WHITE, yolah) };
              //CONNECTIVITY_SET_WEIGHT * pair{ connectivity_set(Yolah::BLACK, yolah), connectivity_set(Yolah::WHITE, yolah) };
              ALONE_WEIGHT * pair{ alone(Yolah::BLACK, yolah), alone(Yolah::WHITE, yolah) };
        */
        return (res.first - res.second) * (player == Yolah::BLACK ? 1 : -1);
    }

    void learn_weights() {

    }
}