#pragma once

void matvec1(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec2(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec3(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec4(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec5(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec6(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec7(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec8(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec9(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec10(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matvec3x64_1(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void addvec1(int n, float* a, const float* __restrict__ b);
void addvec2(int n, float* a, const float* __restrict__ b);
void matvec_int1(int m, int inner, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c);
void matvec_int2(int m, int inner, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c);

struct nnue {  
    static constexpr int H1 = 4096;
    static constexpr int H2 = 64;
    static constexpr int H3 = 64;
    static constexpr int OUTPUT = 3;  
    float* acc;
    float* h1_to_h2;
    float* h2_to_h3;
    float* h3_to_output;
    nnue();
    float output();
    ~nnue();
};

