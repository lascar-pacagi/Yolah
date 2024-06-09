
#ifndef NELDER_MEAD_H
#define NELDER_MEAD_H
#include <vector>
#include <functional>
#include <cstdint>
#include <utility>
#include "indicators.h"
#include <thread>

class NelderMead {
    using FitnessFunction = std::function<double(const std::vector<double>& weights)>;
public:
    struct BoundingBox {
        double lo;
        double hi;
    };
    class Builder {
        std::vector<BoundingBox> bounding_boxes_;
        size_t nb_iterations_   = 100;
        double reflexion_       = 1.0;
        double expansion_       = 2.0;
        double contraction_     = 0.5;        
        double shrinkage_       = 0.5;          
        FitnessFunction fitness_ = [](const std::vector<double>&) {
            return 0;
        };
    public:
        Builder() = default;
        Builder& bounding_boxes(const std::vector<BoundingBox>& bb);
        Builder& nb_iterations(size_t);
        Builder& reflexion(double);
        Builder& expansion(double);
        Builder& contraction(double);        
        Builder& shrinkage(double);
        Builder& fitness(FitnessFunction);
        NelderMead build() const;
    };
    NelderMead() = default;
    NelderMead(const std::vector<BoundingBox>& bounding_boxes, size_t nb_iterations, 
               double reflexion, double expansion, double contraction, double shrinkage, FitnessFunction fitness);
    void run();
    std::vector<double> best_weights() const;
private:
    std::vector<BoundingBox> bounding_boxes;
    size_t nb_iterations;
    double reflexion;
    double expansion;
    double contraction;
    double shrinkage;
    FitnessFunction fitness;
    std::vector<double> best_weights_;
};

#endif
