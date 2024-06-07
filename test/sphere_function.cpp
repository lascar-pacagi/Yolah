#include "sphere_function.h"

namespace test {
    double sphere_function(const std::vector<double>& w) {
        double res = 0;
        for (const double v : w) {
            res += v * v;
        }
        return -res;
    }
    std::string sphere_function_info() {
        return "Sphere function: f(x) = sum_0^n x_i^2\nExpected value is 0 for (0, ..., 0)";
    }
}
