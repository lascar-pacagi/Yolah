#ifndef HEURISTIC_WEIGHTS_LEARNER_H
#define HEURISTIC_WEIGHTS_LEARNER_H
#include <vector>
#include <memory>
#include "cem.h"
#include "nelder_mead.h"
#include <functional>
#include "player.h"

namespace heuristic {
    class HeuristicWeightsLearner {
    public:
        virtual std::vector<double> learn() = 0;
        virtual ~HeuristicWeightsLearner() = default;
    };

    using CreatePlayer = std::function<std::unique_ptr<Player>(const std::vector<double>&)>;

    class CrossEntropyMethodLearner : public HeuristicWeightsLearner {
        NoisyCrossEntropyMethod cem;
    public:
        CrossEntropyMethodLearner(CreatePlayer);
        CrossEntropyMethodLearner(const NoisyCrossEntropyMethod&);
        std::vector<double> learn() override;
    };

    class NelderMeadLearner : public HeuristicWeightsLearner {
        NelderMead nelder_mead;
    public:
        NelderMeadLearner(CreatePlayer);
        NelderMeadLearner(const NelderMead&);
        std::vector<double> learn() override;
    };

    std::vector<double> learn_weights(std::unique_ptr<HeuristicWeightsLearner> learner);
}

#endif