#include "rastrigin_function.h"
#include <numbers>
#include <cmath>

namespace test {
    double rastrigin_function(const std::vector<double>& w) {
        double res = 10 * w.size();
        for (const double v : w) {
            res += v * v - 10 * std::cos(2 * std::numbers::pi * v);
        }
        return -std::abs(res);
    }
    std::string rastrigin_function_info() {
        return "Rastrigin function: f(x) = 10(n+1) + sum_0^n (x_i^2 - 10cos(2pi * x_i))\nExpected value is 0 for (0, ..., 0)";
    }
}
