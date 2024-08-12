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
        .nb_iterations(600)
        .elite_fraction(0.15)
        .keep_overall_best(false)
        .stddev(20)
        .extra_stddev(5)
        .weights({-6.32397, 96.29134579720809, 47.49216469401797, 99.53749501299016, 33.96878163896622, 27.08388797097006, 73.18216253233302, -21.21460943406819, -54.42621095436699, -55.10510023804331, 13.47806836791044, 138.6067401548363, -11.78116160894561, 186.1628271616771, 150.7249786018566, 53.54199602550555, 301.5105120906174, 371.8421408591228, 319.7397503539172, -29.07885149830955, 170.1048207494621, -347.4041673225432, 96.12541061236001, 59.47066706628728, 131.1794425801229, 55.34812950731414, -194.0707621439058, -178.4466106109931, -118.3438488417585, -114.302352937307, 76.69680754067346, 187.4609005582567, 140.6406954284249, 121.6688036955905, -130.4798127929584, -398.4935105863272, -391.6225673460466, -197.1740001239349, 97.13265102203192, 249.7516687427828, -98.374250668304, 142.6831013526544, 10.0563385936091, -97.07465868560097, -76.50106482119998, 62.41608463759165})
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
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer>(300000, 1);
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
            for (uint64_t i = 0; i < 20; i++) {
                update(i, 0);
            }
            for (uint64_t i = 0; i < 80; i++) {
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
