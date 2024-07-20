#include "heuristic.h"
#include <utility>
#include <bitset>
#include <bit>
#include "misc.h"

// namespace heuristic {
//     namespace {
//         using pair = std::pair<double, double>;
//         pair operator+(const pair& p1, const pair& p2) {
//             return { p1.first + p2.first, p1.second + p2.second };
//         }
//         pair operator*(double coeff, const pair& p) {
//             return { coeff * p.first, coeff * p.second };
//         }
//         pair operator*(const pair& p, double coeff) {
//             return coeff * p;
//         }    
//         uint64_t floodfill(uint64_t player_bb, uint64_t free) {
//             uint64_t prev_flood = 0;   
//             uint64_t flood = player_bb;            
//             //std::cout << Bitboard::pretty(flood) << std::endl;
//             //int count = 0;
//             while (prev_flood ^ flood) {
//                 prev_flood = flood;
//                 uint64_t flood1 = shift<NORTH>(flood) & free;
//                 uint64_t flood2 = shift<SOUTH>(flood) & free;
//                 uint64_t flood3 = shift<EAST>(flood) & free;
//                 uint64_t flood4 = shift<WEST>(flood) & free;
//                 uint64_t flood5 = shift<NORTH_EAST>(flood) & free;
//                 uint64_t flood6 = shift<SOUTH_EAST>(flood) & free;
//                 uint64_t flood7 = shift<NORTH_WEST>(flood) & free;
//                 uint64_t flood8 = shift<SOUTH_WEST>(flood) & free;
//                 flood |= flood1 | flood2 | flood3 | flood4 | flood5 | flood6 | flood7 | flood8;
//                 //count++;
//             }
//             flood ^= player_bb;
//             // std::cout << count << std::endl;
//             // std::cout << Bitboard::pretty(flood) << std::endl;
//             return flood;
//         }
//         int16_t connectivity(uint64_t player_bb, uint64_t free) {
//             return std::popcount(floodfill(player_bb, free));
//         }
//         int16_t nb_connected_components(const Yolah& yolah) {
//             uint64_t free = yolah.free_squares();
//             int16_t res = 0;
//             while (free) {
//                 uint64_t bb = free & -free;
//                 uint64_t flood = floodfill(bb, free);                
//                 res++;
//                 free &= ~(bb | flood);
//             }
//             return res;
//         }
//         int16_t reachable_first(uint8_t player, const Yolah& yolah, auto&& one_step) {
//             uint64_t player_flood = yolah.bitboard(player);
//             uint64_t opponent_flood = yolah.bitboard(Yolah::other_player(player));
//             uint64_t free = yolah.free_squares();            
//             int16_t res = 0;
//             uint64_t prev_player_flood = 0;
//             while (player_flood ^ prev_player_flood) {
//                 prev_player_flood = player_flood;
//                 opponent_flood |= one_step(opponent_flood, free);
//                 uint64_t frontier = one_step(player_flood, free) & ~player_flood;
//                 //std::cout << Bitboard::pretty(player_flood) << std::endl;
//                 //std::cout << Bitboard::pretty(frontier) << std::endl;
//                 res += std::popcount(frontier & ~opponent_flood);
//                 player_flood |= frontier;
//             }
//             return res;
//         }
//         pair first_fast(const Yolah::MoveList& player_moves, const Yolah::MoveList& opponent_moves) {
//             auto moves_to_bitboard = [](const Yolah::MoveList& moves) {
//                 uint64_t res = 0;
//                 for (Move m : moves) {
//                     res |= square_bb(m.to_sq());
//                 }
//                 return res;
//             };
//             uint64_t player_moves_bb = moves_to_bitboard(player_moves);
//             uint64_t opponent_moves_bb = moves_to_bitboard(opponent_moves);            
//             return { std::popcount(player_moves_bb & ~opponent_moves_bb), 
//                      std::popcount(opponent_moves_bb & ~player_moves_bb) };
//         }
//     }
//     int16_t mobility(const Yolah::MoveList& moves) {
//         uint64_t seen = 0;        
//         int16_t res = 0;
//         for (Move m : moves) {
//             uint64_t bb = square_bb(m.to_sq());
//             if (!(seen & bb)) {
//                 res++;
//                 seen |= bb;
//             }
//         }
//         return res;
//     }
//     int16_t connectivity(uint8_t player, const Yolah& yolah) {
//         uint64_t player_bb = yolah.bitboard(player);
//         uint64_t free = yolah.free_squares();             
//         int16_t res = 0;
//         while (player_bb) {
//             uint64_t bb = player_bb & -player_bb;
//             res += connectivity(bb, free);
//             player_bb &= ~bb;
//         }
//         return res;
//     }
//     int16_t connectivity_set(uint8_t player, const Yolah& yolah) {
//         return connectivity(yolah.bitboard(player), yolah.free_squares());
//     }
//     int16_t alone(uint8_t player, const Yolah& yolah) {
//         uint64_t player_bb = yolah.bitboard(player);
//         uint64_t opponent_bb = yolah.bitboard(Yolah::other_player(player));
//         uint64_t free = yolah.free_squares();
//         uint64_t flood_opponent = floodfill(opponent_bb, free);
//         //std::cout << Bitboard::pretty(flood_opponent) << std::endl;
//         int16_t res = 0;
//         while (player_bb) {
//             uint64_t bb = player_bb & -player_bb;
//             //std::cout << Bitboard::pretty(bb) << std::endl;
//             uint64_t flood = floodfill(bb, free);
//             //std::cout << Bitboard::pretty(flood) << std::endl;
//             if (!(flood & flood_opponent)) {
//                 //std::cout << std::popcount(flood) << std::endl;
//                 res += std::popcount(flood);
//             }
//             player_bb &= ~bb;
//         }
//         return res;
//     }
//     int16_t closer(uint8_t player, const Yolah& yolah) {
//         auto one_step = [&](uint64_t flood, uint64_t free) {
//             uint64_t flood1 = shift<NORTH>(flood) & free;
//             uint64_t flood2 = shift<SOUTH>(flood) & free;
//             uint64_t flood3 = shift<EAST>(flood) & free;
//             uint64_t flood4 = shift<WEST>(flood) & free;
//             uint64_t flood5 = shift<NORTH_EAST>(flood) & free;
//             uint64_t flood6 = shift<SOUTH_EAST>(flood) & free;
//             uint64_t flood7 = shift<NORTH_WEST>(flood) & free;
//             uint64_t flood8 = shift<SOUTH_WEST>(flood) & free;
//             return flood1 | flood2 | flood3 | flood4 | flood5 | flood6 | flood7 | flood8;            
//         };
//         return reachable_first(player, yolah, one_step);
//     }
//     int16_t first(const Yolah::MoveList& player_moves, const Yolah::MoveList& opponent_moves) {
//         return first_fast(player_moves, opponent_moves).first;
//     }
//     int16_t blocked(uint8_t player, const Yolah& yolah) {
//         uint64_t player_bb = yolah.bitboard(player);
//         uint64_t free = yolah.free_squares();
//         int16_t res = 0;
//         while (player_bb) {
//             uint64_t bb = player_bb & -player_bb;
//             //std::cout << Bitboard::pretty(bb) << std::endl;
//             uint64_t north = shift<NORTH>(bb);
//             uint64_t south = shift<SOUTH>(bb);
//             uint64_t east = shift<EAST>(bb);
//             uint64_t west = shift<WEST>(bb);
//             uint64_t north_east = shift<NORTH_EAST>(bb);
//             uint64_t south_east = shift<SOUTH_EAST>(bb);
//             uint64_t north_west = shift<NORTH_WEST>(bb);
//             uint64_t south_west = shift<SOUTH_WEST>(bb);
//             res += ((north | south | east | west | north_east | south_east | north_west | south_west) & free) == 0;
//             //std::cout << res << std::endl;
//             player_bb &= ~bb;
//         }
//         return res;
//     }
//     int16_t eval(uint8_t player, const Yolah& yolah, const std::array<double, NB_WEIGHTS>& weights) {
//         //std::cout << yolah << std::endl;
//         //std::cout << Bitboard::pretty(yolah.free_squares()) << std::endl;
//         Yolah::MoveList black_moves, white_moves;
//         yolah.moves(Yolah::BLACK, black_moves);
//         yolah.moves(Yolah::WHITE, white_moves);
//         //std::cout << nb_connected_components(yolah) << std::endl;
//         pair res{};
//         res = weights[NO_MOVE_WEIGHT] * pair{ black_moves[0] == Move::none(), white_moves[0] == Move::none() } +
//               weights[NB_MOVES_WEIGHT] * pair{ black_moves.size(), white_moves.size() } +
//               weights[MOBILITY_WEIGHT] * pair{ mobility(black_moves), mobility(white_moves) } +
//               weights[CONNECTIVITY_WEIGHT] * pair{ connectivity(Yolah::BLACK, yolah), connectivity(Yolah::WHITE, yolah) } +
//               weights[CONNECTIVITY_SET_WEIGHT] * pair{ connectivity_set(Yolah::BLACK, yolah), connectivity_set(Yolah::WHITE, yolah) } +
//               weights[ALONE_WEIGHT] * pair{ alone(Yolah::BLACK, yolah), alone(Yolah::WHITE, yolah) } +
//               weights[CLOSER_WEIGHT] * pair{ closer(Yolah::BLACK, yolah), closer(Yolah::WHITE, yolah) } +
//               weights[FIRST_WEIGHT] * first_fast(black_moves, white_moves) +
//               weights[BLOCKED_WEIGHT] * pair{ blocked(Yolah::BLACK, yolah), blocked(Yolah::WHITE, yolah) };
//         int16_t value = static_cast<int16_t>((res.first - res.second) * (player == Yolah::BLACK ? 1 : -1));
//         return std::max(MIN_VALUE, std::min(MAX_VALUE, value));
//     }
//     int16_t evaluation(uint8_t player, const Yolah& yolah) {
//         //return 0;
//         return eval(player, yolah);
//     }
//     std::set<int16_t> sampling_heuristic_values(size_t nb_random_games) {
//         std::set<int16_t> res;
//         PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
//         for (size_t i = 0; i < nb_random_games; i++) {
//             Yolah yolah;
//             res.insert(evaluation(yolah.current_player(), yolah));
//             Yolah::MoveList moves;
//             while (!yolah.game_over()) {
//                 yolah.moves(moves);
//                 if (moves.size() == 0) continue;
//                 Move m = moves[prng.rand<size_t>() % moves.size()];
//                 yolah.play(m);
//                 res.insert(evaluation(yolah.current_player(), yolah));
//             }
//         }
//         return res;
//     }
// }
namespace heuristic {
    namespace {
        uint64_t floodfill(uint64_t player_bb, uint64_t free) {
            uint64_t prev_flood = 0;   
            uint64_t flood = player_bb;            
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
            }
            flood ^= player_bb;
            return flood;
        }
    }

    int16_t blocked(uint8_t player, const Yolah& yolah) {
        uint64_t player_bb = yolah.bitboard(player);
        uint64_t free = yolah.free_squares();
        int16_t res = 0;
        while (player_bb) {
            uint64_t bb = player_bb & -player_bb;
            uint64_t north = shift<NORTH>(bb);
            uint64_t south = shift<SOUTH>(bb);
            uint64_t east = shift<EAST>(bb);
            uint64_t west = shift<WEST>(bb);
            uint64_t north_east = shift<NORTH_EAST>(bb);
            uint64_t south_east = shift<SOUTH_EAST>(bb);
            uint64_t north_west = shift<NORTH_WEST>(bb);
            uint64_t south_west = shift<SOUTH_WEST>(bb);
            res += ((north | south | east | west | north_east | south_east | north_west | south_west) & free) == 0;
            player_bb &= ~bb;
        }
        return res;
    }

    int16_t first(const Yolah::MoveList& player_moves, const Yolah::MoveList& opponent_moves) {
        auto moves_to_bitboard = [](const Yolah::MoveList& moves) {
            uint64_t res = 0;
            for (Move m : moves) {
                res |= square_bb(m.to_sq());
            }
            return res;
        };
        uint64_t player_moves_bb = moves_to_bitboard(player_moves);
        uint64_t opponent_moves_bb = moves_to_bitboard(opponent_moves);            
        return std::popcount(player_moves_bb & ~opponent_moves_bb) - 
                std::popcount(opponent_moves_bb & ~player_moves_bb);
    }

    int16_t connectivity_set(uint64_t player_bb, uint64_t free) {
        return std::popcount(floodfill(player_bb, free));
    }

    int16_t eval(uint8_t player, const Yolah& yolah, const std::array<double, NB_WEIGHTS>& weights) {
        Yolah::MoveList black_moves, white_moves;
        yolah.moves(Yolah::BLACK, black_moves);
        yolah.moves(Yolah::WHITE, white_moves);
        double res = 0;
        res += weights[NO_MOVE_WEIGHT] * ((black_moves[0] == Move::none()) - (white_moves[0] == Move::none()));
        res += weights[NB_MOVES_WEIGHT] * int(black_moves.size() - white_moves.size());
        res += weights[BLOCKED_WEIGHT] * (blocked(Yolah::BLACK, yolah) - blocked(Yolah::WHITE, yolah));
        res += weights[FIRST_WEIGHT] * first(black_moves, white_moves);
        res += weights[CONNECTIVITY_SET_WEIGHT] * int(connectivity_set(yolah.bitboard(Yolah::BLACK), yolah.free_squares()) - 
                                                        connectivity_set(yolah.bitboard(Yolah::WHITE), yolah.free_squares()));
        int16_t value = static_cast<int16_t>(res * (player == Yolah::BLACK ? 1 : -1));
        return std::max(MIN_VALUE, std::min(MAX_VALUE, value));
    }

    int16_t evaluation(uint8_t player, const Yolah& yolah) {
        return eval(player, yolah);
    }

    std::set<int16_t> sampling_heuristic_values(size_t nb_random_games) {
        std::set<int16_t> res;
        PRNG prng(std::chrono::system_clock::now().time_since_epoch().count());
        for (size_t i = 0; i < nb_random_games; i++) {
            Yolah yolah;
            res.insert(evaluation(yolah.current_player(), yolah));
            Yolah::MoveList moves;
            while (!yolah.game_over()) {
                yolah.moves(moves);
                if (moves.size() == 0) continue;
                Move m = moves[prng.rand<size_t>() % moves.size()];
                yolah.play(m);
                res.insert(evaluation(yolah.current_player(), yolah));
            }
        }
        return res;
    }
}