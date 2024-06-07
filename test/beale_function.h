#ifndef BEALE_FUNCTION_H
#define BEALE_FUNCTION_H
#include <string>

namespace test {
    // f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    // f(3, 0.5) = 0
    double beale_function(double x, double y);
    std::string beale_function_info();
}

#endif
