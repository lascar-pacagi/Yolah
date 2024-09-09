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
        .nb_iterations(200)
        .elite_fraction(0.15)
        .keep_overall_best(true)
        .stddev(10)
        .extra_stddev(5)
        .weights({-277.108, 142.193893489657, 142.584277814301, -117.6717883751547, 107.1563469489577, 40.59476758916619, 128.4925478414285, -518.4020193040536, -224.0209505404123, -17.08744459127947, 136.2468402086275, 261.0075939686621, 314.0783572892569, 271.9917630182447, 149.4302588014587, 126.7587335947779, 474.4009403142998, 715.1417958422134, 913.5612685590646, -171.1934085907059, -976.4789027288631, 63.24880988234823, -386.1701040708617, 343.4002881041962, 282.4490642686596, -134.7402628511411, -298.0041038380612, -371.1386407953464, -641.6281879825533, -105.2183813613418, 278.9687264075095, 592.7081279057317, 377.4933744872052, 225.8286921286923, -613.2933887521617, -736.0381986195573, -991.1993361036077, -386.5213437968281, 268.9699704536527, -110.2022748673798, 171.0462338115501, 133.0766969881873, 47.13055579253391, -307.0114227524336, -270.5042971473602, -270.9437469877694})
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
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer>(200000, 1);
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
            for (uint64_t i = 0; i < 160; i++) {
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
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer>(1000000, 1);
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
