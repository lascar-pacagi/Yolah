#pragma once
#include <random>

inline void rinit(float* a, int n) {
    std::default_random_engine g(42);
    std::uniform_real_distribution<float> d;
    for (int i = 0; i < n; i++) {
        a[i] = d(g); 
    }
}

inline void rinit_int8(int8_t* a, int n) {
    std::default_random_engine g(42);
    std::uniform_int_distribution<int8_t> d;
    for (int i = 0; i < n; i++) {
        a[i] = d(g); 
    }
}
