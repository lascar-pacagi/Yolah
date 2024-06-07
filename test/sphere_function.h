#ifndef SPHERE_FUNCTION_H
#define SPHERE_FUNCTION_H
#include <vector>
#include <string>

namespace test {
    // f(x) = sum_0^n x_i^2
    // f(x_0, ..., x_n) = f(0, ..., 0) = 0
    double sphere_function(const std::vector<double>&);
    std::string sphere_function_info();
}

#endif
