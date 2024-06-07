#include "beale_function.h"

namespace test {
    double beale_function(double x, double y) {
        double first_part = 1.5 - x + x * y;
        first_part *= first_part;
        double second_part = 2.25 - x + x * y * y;
        second_part *= second_part;
        double third_part = 2.625 - x + x * y * y * y;
        third_part *= third_part;
        return -(first_part + second_part + third_part);
    }
    std::string beale_function_info() {
        return "Beale function: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2\nExpected value is 0 for (3, 0.5)";
    }
}
