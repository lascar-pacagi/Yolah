#ifndef CEM_H
#define CEM_H
#include <vector>
#include <functional>
#include <cstdint>

class NoisyCrossEntropyMethod {
    using fitness_function = std::function<double(const std::vector<double>& weights, 
                                                  const std::vector<std::vector<double>>& population)>; 
    std::vector<double> optimized_weights;
    uint32_t nb_iterations;
    uint32_t population_size;
    double stddev;
    double extra_stddev;
    fitness_function fitness;
    uint32_t elite_size;

public:
    class Builder {
        std::vector<double> weights_;
        uint32_t nb_iterations_ = 100;
        uint32_t population_size_ = 30;
        double stddev_ = 1.0;
        double extra_stddev_ = 1.0;
        fitness_function fitness_ = [](const std::vector<double>&, const std::vector<std::vector<double>>&) {
            return 0;
        };
        double elite_fraction_ = 0.2;
    public:
        Builder() = default;
        Builder& weights(const std::vector<double>&);
        Builder& nb_weights(uint32_t n);
        Builder& nb_iterations(uint32_t);
        Builder& population_size(uint32_t);
        Builder& stddev(double);
        Builder& extra_stddev(double);
        Builder& fitness(fitness_function);
        Builder& elite_fraction(double);
        NoisyCrossEntropyMethod build() const;
    };
    NoisyCrossEntropyMethod(const std::vector<double>& weights, uint32_t nb_iterations, uint32_t population_size, double elite_fraction, 
                            double stddev, double extra_stddev, fitness_function fitness);
    void run();
    std::vector<double> best_weights() const;
};

#endif
