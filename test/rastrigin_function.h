#ifndef RASTRIGIN_FUNCTION_H
#define RASTRIGIN_FUNCTION_H
#include <vector>
#include <string>

namespace test {
    // f(x) = 10(n+1) + sum_0^n (x_i^2 - 10cos(2pi * x_i))
    // f(x_0, ..., x_n) = f(0, ..., 0) = 0
    double rastrigin_function(const std::vector<double>&);
    std::string rastrigin_function_info();
}

#endif
