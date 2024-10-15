#include "heuristic.h"
#include <utility>
#include <bitset>
#include <bit>
#include "misc.h"

namespace heuristic {
    uint64_t floodfill(uint64_t player_bb, uint64_t free) {
        uint64_t prev_flood = 0;   
        uint64_t flood = player_bb;            
        while (prev_flood != flood) {
            prev_flood = flood;
            flood |= shift_all_directions(flood) & free;
        }
        flood ^= player_bb;
        return flood;
    }

    int16_t connectivity_set(uint64_t player_bb, uint64_t free) {
        return std::popcount(floodfill(player_bb, free));
    }

    int16_t connectivity(uint8_t player, const Yolah& yolah) {
        uint64_t player_bb = yolah.bitboard(player);
        uint64_t free = yolah.free_squares();
        int32_t res = 0;
        while (player_bb) {
            uint64_t bb = player_bb & -player_bb;
            res += connectivity_set(bb, free);
            player_bb &= ~bb;
        }
        return res;
    }

    int16_t alone(uint8_t player, const Yolah& yolah) {
        uint64_t player_bb = yolah.bitboard(player);
        uint64_t opponent_bb = yolah.bitboard(Yolah::other_player(player));
        uint64_t free = yolah.free_squares();
        uint64_t flood_opponent = floodfill(opponent_bb, free);
        uint64_t total_player_flood = 0;
        int16_t res = 0;
        while (player_bb) {
            uint64_t bb = player_bb & -player_bb;
            uint64_t flood = floodfill(bb, free);
            if (!(flood & flood_opponent)) {
                res += std::popcount(flood & ~total_player_flood);
            }
            total_player_flood |= flood;
            player_bb &= ~bb;
        }
        return res;
    }

    std::pair<uint64_t, uint64_t> first(const Yolah::MoveList& player_moves, const Yolah::MoveList& opponent_moves) {
        auto moves_to_bitboard = [](const Yolah::MoveList& moves) {
            uint64_t res = 0;
            for (Move m : moves) {
                res |= square_bb(m.to_sq());
            }
            return res;
        };
        uint64_t player_moves_bb = moves_to_bitboard(player_moves);
        uint64_t opponent_moves_bb = moves_to_bitboard(opponent_moves);            
        return { player_moves_bb & ~opponent_moves_bb, opponent_moves_bb & ~player_moves_bb };
    }

    std::pair<uint64_t, uint64_t> influence(const Yolah& yolah) {
        auto one_step = [&](uint64_t flood, uint64_t free) {
            return (shift<NORTH>(flood) | shift<SOUTH>(flood) | shift<EAST>(flood) | 
                    shift<WEST>(flood) | shift<NORTH_EAST>(flood) | shift<SOUTH_EAST>(flood) | 
                    shift<NORTH_WEST>(flood) | shift<SOUTH_WEST>(flood)) & free;            
        };
        uint64_t prev_black_influence = 0;
        uint64_t prev_white_influence = 0;
        uint64_t black_influence = yolah.bitboard(Yolah::BLACK);
        uint64_t white_influence = yolah.bitboard(Yolah::WHITE);
        uint64_t black_frontier = black_influence;
        uint64_t white_frontier = white_influence;
        uint64_t free = yolah.free_squares();
        uint64_t neutral = 0;
        while ((prev_black_influence != black_influence) || 
                (prev_white_influence != white_influence)) {
            black_frontier = one_step(black_frontier, free) & ~white_influence;
            white_frontier = one_step(white_frontier, free) & ~black_influence;
            neutral |= one_step(neutral, free) | (black_frontier & white_frontier);
            black_frontier &= ~neutral;
            white_frontier &= ~neutral;
            prev_black_influence = black_influence;
            prev_white_influence = white_influence;
            black_influence |= black_frontier;
            white_influence |= white_frontier;
        }
        return { black_influence, white_influence };
    }

    namespace {
        size_t phase(const Yolah& yolah) {
            auto free = std::popcount(yolah.free_squares());
            if (free <= END_GAME) return 2;
            if (free <= MIDDLE_GAME)  return 1;
            return 0;
        }
    }

    double freedom(uint8_t player, const Yolah& yolah, const std::array<double, NB_WEIGHTS>& weights) {
        double count[9]{};
        uint64_t player_bb = yolah.bitboard(player);
        uint64_t free = yolah.free_squares();
        while (player_bb) {
            uint64_t b = player_bb & -player_bb;
            ++count[std::popcount(AROUND[std::countr_zero(b)] & free)];
            player_bb &= ~b;
        }
        size_t p = 9 * phase(yolah);
        return weights[FREEDOM_0_OPENING_WEIGTH + p]  * count[0] + 
                weights[FREEDOM_1_OPENING_WEIGTH + p] * count[1] +
                weights[FREEDOM_2_OPENING_WEIGTH + p] * count[2] +
                weights[FREEDOM_3_OPENING_WEIGTH + p] * count[3] +
                weights[FREEDOM_4_OPENING_WEIGTH + p] * count[4] +
                weights[FREEDOM_5_OPENING_WEIGTH + p] * count[5] +
                weights[FREEDOM_6_OPENING_WEIGTH + p] * count[6] +
                weights[FREEDOM_7_OPENING_WEIGTH + p] * count[7] +
                weights[FREEDOM_8_OPENING_WEIGTH + p] * count[8];
    }

    int16_t eval(uint8_t player, const Yolah& yolah, const std::array<double, NB_WEIGHTS>& weights) {
        Yolah::MoveList black_moves, white_moves;
        yolah.moves(Yolah::BLACK, black_moves);
        yolah.moves(Yolah::WHITE, white_moves);
        double res = 0;
        size_t p = phase(yolah);
        res += weights[NO_MOVE_WEIGHT] * ((black_moves[0] == Move::none()) - (white_moves[0] == Move::none()));
     
        res += weights[NB_MOVES_OPENING_WEIGHT + p] * (int(black_moves.size()) - int(white_moves.size()));
          
        const auto [black_first, white_first] = first(black_moves, white_moves);
        res += weights[FIRST_OPENING_WEIGHT + p] * (std::popcount(black_first) - std::popcount(white_first));
     
        res += weights[CONNECTIVITY_OPENING_WEIGHT + p] * (connectivity(Yolah::BLACK, yolah) - connectivity(Yolah::WHITE, yolah));

        res += weights[CONNECTIVITY_SET_OPENING_WEIGHT + p] * (connectivity_set(yolah.bitboard(Yolah::BLACK), yolah.free_squares()) - 
                                                    connectivity_set(yolah.bitboard(Yolah::WHITE), yolah.free_squares()));
        
        res += weights[ALONE_OPENING_WEIGHT + p] * (alone(Yolah::BLACK, yolah) - alone(Yolah::WHITE, yolah));                                                        
     
        const auto [black_influence, white_influence] = influence(yolah);
        res += weights[INFLUENCE_OPENING_WEIGHT + p] * (std::popcount(black_influence) - std::popcount(white_influence));
     
        res += freedom(Yolah::BLACK, yolah, weights) - freedom(Yolah::WHITE, yolah, weights);

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