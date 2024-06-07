#include "cem_test.h"
#include "cem.h"
#include <iostream>
#include "beale_function.h"
#include "sphere_function.h"
#include "rastrigin_function.h"

namespace test {
    void cem_beale_function() {
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(50)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-1000, 1000})
        .fitness([](const std::vector<double>& w, auto) {
            return beale_function(w[0], w[1]);
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << beale_function_info() << std::endl;
    }
    void cem_sphere_function() {
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(50)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-1000, 1000, -1000, 1000, -1000, 1000, -1000, 1000})
        .fitness([](const std::vector<double>& w, auto) {
            return sphere_function(w);
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << sphere_function_info() << std::endl;
    }
    void cem_rastrigin_function() {
        NoisyCrossEntropyMethod::Builder builder;
        builder
        .population_size(150)
        .nb_iterations(10000)
        .elite_fraction(0.2)
        .stddev(5)
        .extra_stddev(10)
        .weights({-5, 5, -5, 5, -5, 5, -5, 5})
        .fitness([](const std::vector<double>& w, auto) {
            return rastrigin_function(w);
        });
        NoisyCrossEntropyMethod cem = builder.build();
        cem.run();
        std::cout << rastrigin_function_info() << std::endl;
    }
}