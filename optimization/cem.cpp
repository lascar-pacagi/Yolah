#include "cem.h"
#include <limits>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>
#include <iterator>
#include <iostream>
#include "indicators.h"
#include <sstream>
#include <iomanip>

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::weights(const std::vector<double>& w) {
    weights_ = w;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::nb_weights(uint32_t n) {
    weights_ = std::vector<double>(n);
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::nb_iterations(uint32_t n) {
    nb_iterations_ = n;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::population_size(uint32_t n) {
    population_size_ = n;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::stddev(double v) {
    stddev_ = v;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::extra_stddev(double v) {
    extra_stddev_ = v;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::fitness(FitnessFunction f) {
    fitness_ = f;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::elite_fraction(double v) {
    elite_fraction_ = v;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::keep_overall_best(bool b) {
    keep_overall_best_ = b;
    return *this;
}

NoisyCrossEntropyMethod::Builder& NoisyCrossEntropyMethod::Builder::transform(TransformWeight t) {
    transform_ = t;
    return *this;
}

NoisyCrossEntropyMethod NoisyCrossEntropyMethod::Builder::build() const {
    return { weights_, nb_iterations_, population_size_, elite_fraction_, keep_overall_best_, stddev_, extra_stddev_, fitness_, transform_ };
}

NoisyCrossEntropyMethod::NoisyCrossEntropyMethod(const std::vector<double>& weights, uint32_t nb_iterations, 
                                                 uint32_t population_size, double elite_fraction, bool keep_overall_best,
                                                 double stddev, double extra_stddev, 
                                                 FitnessFunction fitness, TransformWeight transform)
        : optimized_weights(weights), nb_iterations(nb_iterations), population_size(population_size), 
          stddev(stddev), extra_stddev(extra_stddev), fitness(fitness), elite_size(population_size * elite_fraction),
          keep_overall_best(keep_overall_best), transform(transform) {
}

void NoisyCrossEntropyMethod::run() {
    using std::vector, std::pair, std::numeric_limits;
    using std::normal_distribution, std::cout, std::endl;
    const size_t weights_size = optimized_weights.size();
    vector<double> weights_mean = optimized_weights;
    vector<double> weights_std = vector<double>(weights_size, stddev);
    vector<vector<double>> population(population_size, vector<double>(weights_size));
    vector<pair<double, size_t>> fitness_index(population_size, {numeric_limits<double>::lowest(), 0});  
    vector<normal_distribution<double>> distributions(weights_size);
    double best_fitness = numeric_limits<double>::lowest();
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    // using namespace indicators;
    // ProgressBar bar{
    //     option::BarWidth{50},
    //     option::Start{"["},
    //     option::Fill{"="},
    //     option::Lead{">"},
    //     option::Remainder{" "},
    //     option::End{"]"},
    //     option::PostfixText{""},
    //     option::ForegroundColor{Color::green},
    //     option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    // };
    for (size_t iter = 0; iter < nb_iterations; ++iter) {
        for (size_t i = 0; i < weights_size; ++i) {
            distributions[i] = normal_distribution<double>(weights_mean[i],
                                                            weights_std[i]
                                                            + extra_stddev
                                                            * std::max(0.0, 1.0 - iter / (nb_iterations / 2.0)));
        }
        for (size_t i = 0; i < population_size; ++i) {
            fitness_index[i].second = i;
            for (size_t j = 0; j < weights_size; ++j) {
                population[i][j] = transform(j, distributions[j](generator));
            }
        }
        std::for_each(std::execution::par_unseq, begin(fitness_index), end(fitness_index), [&](pair<double, size_t>& p) {
            auto i = std::distance(&fitness_index.data()[0], &p);
            p.first = fitness(population[i], population);
        });
        sort(begin(fitness_index), end(fitness_index), std::greater<pair<double, size_t>>());      
        if (!keep_overall_best || fitness_index[0].first > best_fitness) {
            best_fitness = fitness_index[0].first;
            optimized_weights = population[fitness_index[0].second];
            std::stringbuf buf;
            std::ostream os(&buf);
            os << iter << '/' << nb_iterations << '\n';
            os << "Best fitness: " << best_fitness;
            os << " {" << optimized_weights[0];
            for (size_t i = 1; i < optimized_weights.size(); i++) {
                os << ", " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << optimized_weights[i];
            }
            os << "}";
            cout << buf.str() << endl;
            //bar.set_option(option::PostfixText{buf.str()});            
        }
        for (size_t i = 0; i < weights_size; ++i) {
            weights_mean[i] = 0;
            weights_std[i] = 0;
        }

        // Mean
        for (size_t i = 0; i < elite_size; ++i) {
            for (size_t j = 0; j < weights_size; ++j) {
                weights_mean[j] += population[fitness_index[i].second][j];
            }
        }
        for (size_t i = 0; i < weights_size; ++i) {
            weights_mean[i] /= elite_size;
        }

        // Stdev
        for (size_t i = 0; i < elite_size; ++i) {
            for (int j = 0; j < weights_size; ++j) {
                double v = population[fitness_index[i].second][j] - weights_mean[j];
                weights_std[j] += v * v;
            }
        }
        for (int i = 0; i < weights_size; ++i) {
            weights_std[i] = std::sqrt(weights_std[i] / elite_size);
        }
        //bar.set_progress(iter * 100 / nb_iterations);
    } 
    //bar.set_progress(100);
    cout << "best fitness: " << best_fitness << endl;
    cout << "{" << optimized_weights[0];
    for (size_t i = 1; i < optimized_weights.size(); i++) {
        cout << ", " << optimized_weights[i];
    }
    cout << "}" << endl;
}

std::vector<double> NoisyCrossEntropyMethod::best_weights() const {
    return optimized_weights;
}
