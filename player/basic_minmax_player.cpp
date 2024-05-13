#include "basic_minmax_player.h"
#include <cstring>
#include "cem.h"
#include <random>
#include <chrono>
#include <array>
#include "monte_carlo_player.h"

BasicMinMaxPlayer::BasicMinMaxPlayer(uint16_t depth, heuristic_eval h) : depth(depth), heuristic(h) {
}

void BasicMinMaxPlayer::sort_moves(Yolah& yolah, Yolah::MoveList& moves) {
    #define XOR_SWAP(a, b) a = a ^ b; b = a ^ b; a = a ^ b
    int32_t scores[Yolah::MAX_NB_MOVES];
    uint8_t indexes[Yolah::MAX_NB_MOVES];
    size_t nb_moves = moves.size();
    auto current_player = yolah.current_player();
    for (size_t i = 0; i < nb_moves; i++) {
        yolah.play(moves[i]);
        scores[i] = heuristic(current_player, yolah);
        yolah.undo(moves[i]);
        indexes[i] = i;
    }
    for (size_t i = 1; i < nb_moves; i++) {
        int j = i;
        while (j > 0 && scores[j - 1] < scores[j]) {
            XOR_SWAP(scores[j - 1], scores[j]);
            XOR_SWAP(indexes[j - 1], indexes[j]);
            j--;
        }
    }
    Yolah::MoveList tmp;
    std::memcpy(tmp.data(), moves.data(), sizeof(Move) * nb_moves);
    for (size_t i = 0; i < nb_moves; i++) {
        moves.data()[i] = tmp[indexes[i]];
    }
    #undef XOR_SWAP
}

int32_t BasicMinMaxPlayer::negamax(Yolah& yolah, int32_t alpha, int32_t beta, uint16_t depth) {
    if (yolah.game_over()) {
        return yolah.score(yolah.current_player()) * heuristic::MAX_VALUE;
    }
    if (depth == 0) {
        return heuristic(yolah.current_player(), yolah);
    }
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, moves);
    for (const Move& m : moves) {
        yolah.play(m);
        int32_t v = -negamax(yolah, -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (v >= beta) {
            return v;
        }
        if (v > alpha) {
            alpha = v;
        }
    }
    return alpha;
}

int32_t BasicMinMaxPlayer::search(Yolah& yolah, Move& res) {
    res = Move::none();
    if (yolah.game_over()) {
        return yolah.score(yolah.current_player()) * heuristic::MAX_VALUE;
    }
    Yolah::MoveList moves;
    yolah.moves(moves);
    int32_t alpha = -std::numeric_limits<int32_t>::max();
    int32_t beta = std::numeric_limits<int32_t>::max();
    sort_moves(yolah, moves);
    for (const Move& m : moves) {
        yolah.play(m);
        int32_t v = -negamax(yolah, alpha, beta, depth - 1);
        yolah.undo(m);        
        if (v > alpha) {
            alpha = v;
            res = m;
        }
    }
    return alpha;
}

Move BasicMinMaxPlayer::play(Yolah yolah) {
    Move m;
    auto value = search(yolah, m);
    //std::cout << "BasicMinMax value: " << value << std::endl;
    return m;
}

void BasicMinMaxPlayer::learn_weights() {
    using std::vector, std::array;
    NoisyCrossEntropyMethod::Builder builder;
    builder
    .population_size(30)
    .nb_iterations(100)
    .elite_fraction(0.2)
    .stddev(5)
    .extra_stddev(10)
    .weights(vector<double>(heuristic::NB_WEIGHTS))
    .fitness([](const vector<double>& w, const vector<vector<double>>& population) {             
        std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
        array<double, heuristic::NB_WEIGHTS> me_weights;
        for (size_t i = 0; double x : w) {
            me_weights[i++] = x;
        }
        auto play = [&](Player& p1, Player& p2) {
            Yolah yolah;
            while (!yolah.game_over()) {                 
                Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2).play(yolah);
                yolah.play(m);
            }
            return yolah.score(Yolah::BLACK);
        };
        double res = 0;        
        BasicMinMaxPlayer me(4, [&](uint8_t player, const Yolah& yolah) {
            return heuristic::eval(player, yolah, me_weights);
        });            
        MonteCarloPlayer opponent(100000, 1);
        res += play(me, opponent);
        res -= play(opponent, me);
        return res;
    });
    // .fitness([](const vector<double>& w, const vector<vector<double>>& population) {             
    //     size_t tournament_size = population.size() * 0.1;
    //     std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    //     std::uniform_int_distribution<size_t> choose(0, tournament_size - 1);
    //     array<double, heuristic::NB_WEIGHTS> me_weights;
    //     for (size_t i = 0; double x : w) {
    //         me_weights[i++] = x;
    //     }
    //     auto play = [&](BasicMinMaxPlayer& p1, BasicMinMaxPlayer& p2) {
    //         Yolah yolah;
    //         while (!yolah.game_over()) {                 
    //             Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2).play(yolah);
    //             yolah.play(m);
    //         }
    //         return yolah.score(Yolah::BLACK);
    //     };
    //     double res = 0;
    //     for (size_t i = 0; i < tournament_size; i++) {
    //         array<double, heuristic::NB_WEIGHTS> opponent_weights;
    //         for (size_t i = 0; double x : population[choose(generator)]) {
    //             opponent_weights[i] = x;
    //         }
    //         BasicMinMaxPlayer me(4, [&](uint8_t player, const Yolah& yolah) {
    //             return heuristic::eval(player, yolah, me_weights);
    //         });
    //         BasicMinMaxPlayer opponent(4, [&](uint8_t player, const Yolah& yolah) {
    //             return heuristic::eval(player, yolah, opponent_weights);
    //         });
    //         res += play(me, opponent);
    //         res -= play(opponent, me);
    //     }       
    //     return res / (2 * tournament_size);
    // });
    NoisyCrossEntropyMethod cem = builder.build();
    cem.run();
}