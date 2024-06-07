#include "nelder_mead.h"
#include "beale_function.h"
#include "sphere_function.h"
#include "rastrigin_function.h"
#include "nelder_mead.h"

namespace test {
    void nelder_mead_beale_function() {
        NelderMead::Builder builder;
        builder
        .bounding_boxes({{-10, 10}, {-10, 10}})
        .nb_iterations(1000)
        .fitness([](const std::vector<double>& w) {
            return beale_function(w[0], w[1]);
        });
        NelderMead nelder_mead = builder.build();
        nelder_mead.run();
        std::cout << beale_function_info() << std::endl;
    }

    void nelder_mead_sphere_function() {
        NelderMead::Builder builder;
        builder
        .bounding_boxes({{-100, 100}, {-100, 100}, {-100, 100}, {-100, 100}, {-100, 100}, {-100, 100}, {-100, 100}})
        .nb_iterations(1000)
        .fitness([](const std::vector<double>& w) {
            return sphere_function(w);
        });
        NelderMead nelder_mead = builder.build();
        nelder_mead.run();
        std::cout << sphere_function_info() << std::endl;
    }

    void nelder_mead_rastrigin_function() {
        NelderMead::Builder builder;
        builder
        .bounding_boxes({{-5, 5}, {-5, 5}, {-5, 5}, {-5, 5}, {-5, 5}, {-5, 5}, {-5, 5}})
        .nb_iterations(1000)
        .fitness([](const std::vector<double>& w) {
            return rastrigin_function(w);
        });
        NelderMead nelder_mead = builder.build();
        nelder_mead.run();
        std::cout << rastrigin_function_info() << std::endl;
    }
}