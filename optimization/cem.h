#ifndef CEM_H
#define CEM_H
#include <vector>
#include <functional>
#include <cstdint>

class NoisyCrossEntropyMethod {
    using FitnessFunction = std::function<double(const std::vector<double>& weights, 
                                                 const std::vector<std::vector<double>>& population)>;
    using TransformWeight = std::function<double(size_t i, double weight)>;
    std::vector<double> optimized_weights;
    uint32_t nb_iterations;
    uint32_t population_size;
    double stddev;
    double extra_stddev;
    FitnessFunction fitness;
    uint32_t elite_size;
    bool keep_overall_best;
    TransformWeight transform;
public:
    class Builder {
        std::vector<double> weights_;
        uint32_t nb_iterations_ = 100;
        uint32_t population_size_ = 30;
        double stddev_ = 1.0;
        double extra_stddev_ = 1.0;
        FitnessFunction fitness_ = [](const std::vector<double>&, const std::vector<std::vector<double>>&) {
            return 0;
        };
        double elite_fraction_ = 0.2;
        bool keep_overall_best_ = true;
        TransformWeight transform_ = [](size_t, double weight) {
            return weight;
        };
    public:
        Builder() = default;
        Builder& weights(const std::vector<double>&);
        Builder& nb_weights(uint32_t n);
        Builder& nb_iterations(uint32_t);
        Builder& population_size(uint32_t);
        Builder& stddev(double);
        Builder& extra_stddev(double);
        Builder& fitness(FitnessFunction);
        Builder& elite_fraction(double);
        Builder& keep_overall_best(bool);
        Builder& transform(TransformWeight);
        NoisyCrossEntropyMethod build() const;
    };
    NoisyCrossEntropyMethod() = default;
    NoisyCrossEntropyMethod(const std::vector<double>& weights, uint32_t nb_iterations, uint32_t population_size, double elite_fraction, 
                            bool keep_overall_best, double stddev, double extra_stddev, FitnessFunction fitness, TransformWeight transform);
    void run();
    std::vector<double> best_weights() const;
};

#endif
