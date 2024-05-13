#include "cem_test.h"
#include "cem.h"
#include <iostream>
#include <numeric>
#include <ranges>
#include <numbers>
#include <cmath>

namespace test {
    void cem_beale_function() {
        // f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
        // f(3, 0.5) = 0
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(50)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-1000, 1000})
        .fitness([](const std::vector<double>& w, auto) {
            double x = w[0];
            double y = w[1];
            double first_part = 1.5 - x + x * y;
            first_part *= first_part;
            double second_part = 2.25 - x + x * y * y;
            second_part *= second_part;
            double third_part = 2.625 - x + x * y * y * y;
            third_part *= third_part;
            return -(first_part + second_part + third_part); 
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << "Beale function: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2\n";
        std::cout << "Expected value is 0 for (3, 0.5)" << std::endl;
    }
    void cem_sphere_function() {
        // f(x) = sum_0^n x_i^2
        // f(x_0, ..., x_n) = f(0, ..., 0) = 0
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(50)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-1000, 1000, -1000, 1000, -1000, 1000, -1000, 1000})
        .fitness([](const std::vector<double>& w, auto) {
            double res = 0;
            for (const double v : w) {
                res += v * v;
            }
            return -res;
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << "Sphere function: f(x) = sum_0^n x_i^2\n";
        std::cout << "Expected value is 0 for (0, ..., 0)" << std::endl;
    }
    void cem_rastrigin_function() {
        // f(x) = 10(n+1) + sum_0^n (x_i^2 - 10cos(2pi * x_i))
        // f(x_0, ..., x_n) = f(0, ..., 0) = 0
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(150)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-5, 5, -5, 5, -5, 5, -5, 5})
        .fitness([](const std::vector<double>& w, auto) {
            double res = 10 * w.size();
            for (const double v : w) {
                res += v * v - 10 * std::cos(2 * std::numbers::pi * v);
            }
            return -std::abs(res);
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << "Rastrigin function: f(x) = 10(n+1) + sum_0^n (x_i^2 - 10cos(2pi * x_i))\n";
        std::cout << "Expected value is 0 for (0, ..., 0)" << std::endl;
    }
}