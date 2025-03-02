#pragma once

void rinit(float* a, int n);
void matmul1(int m, int n, int inner, const float* a, const float* b, float* c);
void matmul2(int m, int n, int inner, const float* a, const float* b, float* c);
//void kernel4x4_3(int m, int n, int inner, const float* a, const float* b, float* c);
void matmul3(int m, int n, int inner, const float* a, const float* b, float* c);
//void kernel4x4_4(int m, int n, int inner, const float* a, const float* b, float* c);
void matmul4(int m, int n, int inner, const float* a, const float* b, float* c);
//void kernel4x4_5(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul5(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
//void kernel4x4_6(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul6(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
//void kernel4x4_7(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul7(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
//void kernel4x4_8(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul8(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
//void kernel8x8_9(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul9(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
// void kernel7x8_10(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
// void matmul10(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
//void kernel8x8_11(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul11(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul12(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul13(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
void matmul14(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
