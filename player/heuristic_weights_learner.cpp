#include "heuristic_weights_learner.h"
#include "heuristic.h"
#include "MCTS_mem_player.h"
#include "misc.h"

namespace heuristic {
    CrossEntropyMethodLearner::CrossEntropyMethodLearner(CreatePlayer factory) {
        using std::vector, std::array;
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(42)
        .nb_iterations(300)
        .elite_fraction(0.2)
        .keep_overall_best(false)
        .stddev(10)
        .extra_stddev(1)
        .weights({-261.299, 105.4702532154874, -48.32193088939683, 115.9949267398348, 195.8689144128958, -354.2752260588214, 360.3292424090679})
        .transform([](size_t i, double w) {
            if (i == heuristic::NO_MOVE_WEIGHT || i == heuristic::BLOCKED_WEIGHT) {
                return std::min(0.0, w);
            }
            return w;    
        })
        .fitness([&](const vector<double>& w, const vector<vector<double>>& population) {   
            auto first_n_moves_random = [](Yolah& yolah, uint64_t seed, size_t n) {
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
            auto play = [&](const auto& p1, const auto& p2, uint64_t seed) {
                Yolah yolah;
                first_n_moves_random(yolah, seed, 4);
                while (!yolah.game_over()) {
                    Move m = (yolah.current_player() == Yolah::BLACK ? p1 : p2)->play(yolah);                
                    yolah.play(m);
                }
                return yolah.score(Yolah::BLACK);
            };
            double res = 0;                                        
            std::unique_ptr<Player> opponent = std::make_unique<MCTSMemPlayer>(1000000, 1);
            auto update = [&](uint64_t seed) {
                constexpr double W1 = 1e5;
                constexpr double W2 = 1;
                auto me = factory(w);
                auto score = play(me, opponent, seed);
                res += W1 * ((score > 0) + (score < 0) * -1);
                res += W2 * score;
                me = factory(w);
                score = play(opponent, me, seed);
                res += W1 * ((score > 0) * -1 + (score < 0));
                res -= W2 * score;
            };
            for (uint64_t i = 0; i < 30; i++) {
                update(i);
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
