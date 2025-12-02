#include "heuristic_weights_learner.h"
#include "heuristic.h"
#include "MCTS_mem_player.h"
#include "misc.h"

namespace heuristic {
    CrossEntropyMethodLearner::CrossEntropyMethodLearner(CreatePlayer factory) {
        using std::vector, std::array;
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(120)
        .nb_iterations(500)
        .elite_fraction(0.15)
        .keep_overall_best(true)
        .stddev(5)
        .extra_stddev(1)
        .weights({-255.745, 81.62132957543031, 91.57609768985716, -177.6773616734466,
        68.16486241472865, 26.70495498502099, 154.383379630677,
        -530.8242077186487, -313.2023757332665, -43.87849657315125,
        172.0314980997644, 326.4480489128537, 310.909906521125,
        230.0669166683616, 220.9179293732371, 289.5340889519555,
        387.3041149887017, 825.3446537589967, 1028.305233286851,
        -180.1927551775643, -1069.837309882286, -14.56689415550916,
        -345.1710839652004, 375.3764302746833, 291.2553310488145,
        -137.9246980219349, -288.4070807306961, -289.8783023241542,
        -666.1384943743948, -112.2225426176193, 369.0052859270172,
        552.4179521303455, 517.3756779669266, -2.68655764395052,
        -725.3576607550269, -784.0368752945366, -1084.969212245822,
        -563.2524584917178, 237.5401061901083, -108.1002928009579,
        177.788676880064, 277.559496857502, -48.35322675721582,
        -342.6502992564743, -288.0894324261568, -705.161239285647})
        // .transform([](size_t i, double w) {
        //     if (i == heuristic::NO_MOVE_WEIGHT) {
        //         return std::min(0.0, w);
        //     }
        //     return w;    
        // })
        .fitness([&](const vector<double>& w, const vector<vector<double>>& population) {   
            auto first_n_moves_random = [](Yolah& yolah, uint64_t seed, size_t n) {
                if (n == 0) return;
                PRNG prng(seed);
                Yolah::MoveList moves;
                size_t i = 0;
                while (!yolah.game_over()) {                 
                    yolah.moves(moves);
                    Move m = moves[prng.rand<size_t>() % moves.size()];
                    yolah.play(m);
                    if (++i >= n) break;
                }
            }; 
            auto play = [&](const auto& p1, const auto& p2, uint64_t seed, size_t nb_random) {
                Yolah yolah;
                first_n_moves_random(yolah, seed, nb_random);
                while (!yolah.game_over()) {
                    Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2)->play(yolah);                
                    yolah.play(m);
                }
                return yolah.score(Yolah::BLACK);
            };
            double res = 0;                                        
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer<>>(300000, 1);
            auto update = [&](uint64_t seed, size_t nb_random) {
                constexpr double W1 = 1e5;
                constexpr double W2 = 1;
                auto me = factory(w);
                auto score = play(me, opponent, seed, nb_random);
                res += W1 * ((score > 0) + (score < 0) * -1);
                res += W2 * score;
                me = factory(w);
                score = play(opponent, me, seed, nb_random);
                res += W1 * ((score > 0) * -1 + (score < 0));
                res -= W2 * score;
            };
            for (uint64_t i = 0; i < 40; i++) {
                update(i, 0);
            }
            for (uint64_t i = 0; i < 50; i++) {
                update(i, 2);
            }
            for (uint64_t i = 0; i < 200; i++) {
                update(i, 4);
            }
            return res;
        });
        cem = builder.build();
    }
    
    CrossEntropyMethodLearner::CrossEntropyMethodLearner(const NoisyCrossEntropyMethod& cem) : cem(cem) {
    }

    std::vector<double> CrossEntropyMethodLearner::learn() {
        cem.run();
        return cem.best_weights();
    }

    NelderMeadLearner::NelderMeadLearner(CreatePlayer factory) {
        using std::vector, std::array;
        vector<NelderMead::BoundingBox> bounding_boxes;
        for (size_t i = 0; i < heuristic::NB_WEIGHTS; i++) {
            bounding_boxes.emplace_back(-1000, 1000);
        }
        NelderMead::Builder builder;
        builder
        .bounding_boxes(bounding_boxes)
        .nb_iterations(100)
        .fitness([&](const vector<double>& w) {    
            auto play = [&](const auto& p1, const auto& p2) {
                Yolah yolah;
                while (!yolah.game_over()) {    
                    Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2)->play(yolah);                
                    yolah.play(m);
                }            
                return yolah.score(Yolah::BLACK);
            };
            double res = 0;        
            auto me = factory(w);
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer<>>(1000000, 1);
            auto update = [&](const auto& me, const auto& opponent) {
                constexpr double W1 = 1e5;
                constexpr double W2 = 1;
                auto score = play(me, opponent);
                res += W1 * ((score > 0) + (score < 0) * -1);
                res += W2 * score;
                score = play(opponent, me);
                res += W1 * ((score > 0) * -1 + (score < 0));
                res -= W2 * score;
            };
            update(me, opponent);        
            update(me, opponent);
            update(me, opponent);
            update(me, opponent);        
            update(me, opponent);
            update(me, opponent);
            update(me, opponent);        
            update(me, opponent);
            update(me, opponent);
            update(me, opponent);
            return res;
        });
        nelder_mead = builder.build();
    }
    
    NelderMeadLearner::NelderMeadLearner(const NelderMead& nelder_mead) : nelder_mead(nelder_mead) {
    }
    
    std::vector<double> NelderMeadLearner::learn() {
        nelder_mead.run();
        
        return nelder_mead.best_weights();
    }

    std::vector<double> learn_weights(std::unique_ptr<HeuristicWeightsLearner> learner) {
        return learner->learn();
    }
}
