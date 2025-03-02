#include <bits/stdc++.h>
#include "matmul.h"
using namespace std;

void rinit(float* a, int n) {
    std::default_random_engine g(42);
    std::uniform_real_distribution<float> d;
    for (int i = 0; i < n; i++) {
        a[i] = d(g); 
    }
}


// c += a * b
void matmul1(int m, int n, int inner, const float* a, const float* b, float* c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < inner; k++) {
                c[i * n + j] += a[i * inner + k] * b[k * n + j];
            }
        }
    }   
}

void matmul2(int m, int n, int inner, const float* a, const float* b, float* c) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < inner; k++) {
            for (int j = 0; j < n; j++) {            
                c[i * n + j] += a[i * inner + k] * b[k * n + j];
            }
        }
    }   
}

void kernel4x4_3(int m, int n, int inner, const float* a, const float* b, float* c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    for (int k = 0; k < inner; k++) {
        c[0 * n + 0] += a[0 * inner + k] * b[k * n + 0]; // c(0, 0) += a(0, k) * b(k, 0)
        c[0 * n + 1] += a[0 * inner + k] * b[k * n + 1]; // c(0, 1) += a(0, k) * b(k, 1)
        c[0 * n + 2] += a[0 * inner + k] * b[k * n + 2]; // c(0, 2) += a(0, k) * b(k, 2)
        c[0 * n + 3] += a[0 * inner + k] * b[k * n + 3]; // c(0, 3) += a(0, k) * b(k, 3)

        c[1 * n + 0] += a[1 * inner + k] * b[k * n + 0]; // c(1, 0) += a(1, k) * b(k, 0)
        c[1 * n + 1] += a[1 * inner + k] * b[k * n + 1]; // c(1, 1) += a(1, k) * b(k, 1)
        c[1 * n + 2] += a[1 * inner + k] * b[k * n + 2]; // c(1, 2) += a(1, k) * b(k, 2)
        c[1 * n + 3] += a[1 * inner + k] * b[k * n + 3]; // c(1, 3) += a(1, k) * b(k, 3)

        c[2 * n + 0] += a[2 * inner + k] * b[k * n + 0]; // c(2, 0) += a(2, k) * b(k, 0)
        c[2 * n + 1] += a[2 * inner + k] * b[k * n + 1]; // c(2, 1) += a(2, k) * b(k, 1)
        c[2 * n + 2] += a[2 * inner + k] * b[k * n + 2]; // c(2, 2) += a(2, k) * b(k, 2)
        c[2 * n + 3] += a[2 * inner + k] * b[k * n + 3]; // c(2, 3) += a(2, k) * b(k, 3)

        c[3 * n + 0] += a[3 * inner + k] * b[k * n + 0]; // c(3, 0) += a(3, k) * b(k, 0)
        c[3 * n + 1] += a[3 * inner + k] * b[k * n + 1]; // c(3, 1) += a(3, k) * b(k, 1)
        c[3 * n + 2] += a[3 * inner + k] * b[k * n + 2]; // c(3, 2) += a(3, k) * b(k, 2)
        c[3 * n + 3] += a[3 * inner + k] * b[k * n + 3]; // c(3, 3) += a(3, k) * b(k, 3)
    }
}

void matmul3(int m, int n, int inner, const float* a, const float* b, float* c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_3(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

void kernel4x4_4(int m, int n, int inner, const float* a, const float* b, float* c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    for (int k = 0; k < inner; k++) {
        float a0k = a[0 * inner + k];
        float a1k = a[1 * inner + k];
        float a2k = a[2 * inner + k];
        float a3k = a[3 * inner + k];

        c00 += a0k * b[k * n + 0]; // c(0, 0) += a(0, k) * b(k, 0)
        c01 += a0k * b[k * n + 1]; // c(0, 1) += a(0, k) * b(k, 1)
        c02 += a0k * b[k * n + 2]; // c(0, 2) += a(0, k) * b(k, 2)
        c03 += a0k * b[k * n + 3]; // c(0, 3) += a(0, k) * b(k, 3)

        c10 += a1k * b[k * n + 0]; // c(1, 0) += a(1, k) * b(k, 0)
        c11 += a1k * b[k * n + 1]; // c(1, 1) += a(1, k) * b(k, 1)
        c12 += a1k * b[k * n + 2]; // c(1, 2) += a(1, k) * b(k, 2)
        c13 += a1k * b[k * n + 3]; // c(1, 3) += a(1, k) * b(k, 3)

        c20 += a2k * b[k * n + 0]; // c(2, 0) += a(2, k) * b(k, 0)
        c21 += a2k * b[k * n + 1]; // c(2, 1) += a(2, k) * b(k, 1)
        c22 += a2k * b[k * n + 2]; // c(2, 2) += a(2, k) * b(k, 2)
        c23 += a2k * b[k * n + 3]; // c(2, 3) += a(2, k) * b(k, 3)

        c30 += a3k * b[k * n + 0]; // c(3, 0) += a(3, k) * b(k, 0)
        c31 += a3k * b[k * n + 1]; // c(3, 1) += a(3, k) * b(k, 1)
        c32 += a3k * b[k * n + 2]; // c(3, 2) += a(3, k) * b(k, 2)
        c33 += a3k * b[k * n + 3]; // c(3, 3) += a(3, k) * b(k, 3)
    }
    c[0 * n + 0] += c00; c[0 * n + 1] += c01; c[0 * n + 2] += c02; c[0 * n + 3] += c03;
    c[1 * n + 0] += c10; c[1 * n + 1] += c11; c[1 * n + 2] += c12; c[1 * n + 3] += c13;
    c[2 * n + 0] += c20; c[2 * n + 1] += c21; c[2 * n + 2] += c22; c[2 * n + 3] += c23;
    c[3 * n + 0] += c30; c[3 * n + 1] += c31; c[3 * n + 2] += c32; c[3 * n + 3] += c33;
}

void matmul4(int m, int n, int inner, const float* a, const float* b, float* c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_4(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

void kernel4x4_5(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    
    for (int k = 0; k < inner; k++) {
        float a0k = a[0 * inner + k];
        float a1k = a[1 * inner + k];
        float a2k = a[2 * inner + k];
        float a3k = a[3 * inner + k];

        c00 += a0k * b[k * n + 0]; // c(0, 0) += a(0, k) * b(k, 0)
        c01 += a0k * b[k * n + 1]; // c(0, 1) += a(0, k) * b(k, 1)
        c02 += a0k * b[k * n + 2]; // c(0, 2) += a(0, k) * b(k, 2)
        c03 += a0k * b[k * n + 3]; // c(0, 3) += a(0, k) * b(k, 3)

        c10 += a1k * b[k * n + 0]; // c(1, 0) += a(1, k) * b(k, 0)
        c11 += a1k * b[k * n + 1]; // c(1, 1) += a(1, k) * b(k, 1)
        c12 += a1k * b[k * n + 2]; // c(1, 2) += a(1, k) * b(k, 2)
        c13 += a1k * b[k * n + 3]; // c(1, 3) += a(1, k) * b(k, 3)

        c20 += a2k * b[k * n + 0]; // c(2, 0) += a(2, k) * b(k, 0)
        c21 += a2k * b[k * n + 1]; // c(2, 1) += a(2, k) * b(k, 1)
        c22 += a2k * b[k * n + 2]; // c(2, 2) += a(2, k) * b(k, 2)
        c23 += a2k * b[k * n + 3]; // c(2, 3) += a(2, k) * b(k, 3)

        c30 += a3k * b[k * n + 0]; // c(3, 0) += a(3, k) * b(k, 0)
        c31 += a3k * b[k * n + 1]; // c(3, 1) += a(3, k) * b(k, 1)
        c32 += a3k * b[k * n + 2]; // c(3, 2) += a(3, k) * b(k, 2)
        c33 += a3k * b[k * n + 3]; // c(3, 3) += a(3, k) * b(k, 3)
    }
    c[0 * n + 0] += c00; c[0 * n + 1] += c01; c[0 * n + 2] += c02; c[0 * n + 3] += c03;
    c[1 * n + 0] += c10; c[1 * n + 1] += c11; c[1 * n + 2] += c12; c[1 * n + 3] += c13;
    c[2 * n + 0] += c20; c[2 * n + 1] += c21; c[2 * n + 2] += c22; c[2 * n + 3] += c23;
    c[3 * n + 0] += c30; c[3 * n + 1] += c31; c[3 * n + 2] += c32; c[3 * n + 3] += c33;
}

void matmul5(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_5(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

void kernel4x4_6(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    
    for (int k = 0; k < inner; k++) {
        float a0k = a[0 * inner + k];
        float a1k = a[1 * inner + k];
        float a2k = a[2 * inner + k];
        float a3k = a[3 * inner + k];

        float bk0 = b[k * n + 0];
        float bk1 = b[k * n + 1];
        float bk2 = b[k * n + 2];
        float bk3 = b[k * n + 3];

        c00 += a0k * bk0; // c(0, 0) += a(0, k) * b(k, 0)
        c01 += a0k * bk1; // c(0, 1) += a(0, k) * b(k, 1)
        c02 += a0k * bk2; // c(0, 2) += a(0, k) * b(k, 2)
        c03 += a0k * bk3; // c(0, 3) += a(0, k) * b(k, 3)

        c10 += a1k * bk0; // c(1, 0) += a(1, k) * b(k, 0)
        c11 += a1k * bk1; // c(1, 1) += a(1, k) * b(k, 1)
        c12 += a1k * bk2; // c(1, 2) += a(1, k) * b(k, 2)
        c13 += a1k * bk3; // c(1, 3) += a(1, k) * b(k, 3)

        c20 += a2k * bk0; // c(2, 0) += a(2, k) * b(k, 0)
        c21 += a2k * bk1; // c(2, 1) += a(2, k) * b(k, 1)
        c22 += a2k * bk2; // c(2, 2) += a(2, k) * b(k, 2)
        c23 += a2k * bk3; // c(2, 3) += a(2, k) * b(k, 3)

        c30 += a3k * bk0; // c(3, 0) += a(3, k) * b(k, 0)
        c31 += a3k * bk1; // c(3, 1) += a(3, k) * b(k, 1)
        c32 += a3k * bk2; // c(3, 2) += a(3, k) * b(k, 2)
        c33 += a3k * bk3; // c(3, 3) += a(3, k) * b(k, 3)
    }
    c[0 * n + 0] += c00; c[0 * n + 1] += c01; c[0 * n + 2] += c02; c[0 * n + 3] += c03;
    c[1 * n + 0] += c10; c[1 * n + 1] += c11; c[1 * n + 2] += c12; c[1 * n + 3] += c13;
    c[2 * n + 0] += c20; c[2 * n + 1] += c21; c[2 * n + 2] += c22; c[2 * n + 3] += c23;
    c[3 * n + 0] += c30; c[3 * n + 1] += c31; c[3 * n + 2] += c32; c[3 * n + 3] += c33;
}

void matmul6(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_6(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

void kernel4x4_7(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    const float* a0k_p = &a[0 * inner];
    const float* a1k_p = &a[1 * inner];
    const float* a2k_p = &a[2 * inner];
    const float* a3k_p = &a[3 * inner];

    for (int k = 0; k < inner; k++) {
        float a0k = *a0k_p++;
        float a1k = *a1k_p++;
        float a2k = *a2k_p++;
        float a3k = *a3k_p++;

        float bk0 = b[k * n + 0];
        float bk1 = b[k * n + 1];
        float bk2 = b[k * n + 2];
        float bk3 = b[k * n + 3];

        c00 += a0k * bk0; // c(0, 0) += a(0, k) * b(k, 0)
        c01 += a0k * bk1; // c(0, 1) += a(0, k) * b(k, 1)
        c02 += a0k * bk2; // c(0, 2) += a(0, k) * b(k, 2)
        c03 += a0k * bk3; // c(0, 3) += a(0, k) * b(k, 3)

        c10 += a1k * bk0; // c(1, 0) += a(1, k) * b(k, 0)
        c11 += a1k * bk1; // c(1, 1) += a(1, k) * b(k, 1)
        c12 += a1k * bk2; // c(1, 2) += a(1, k) * b(k, 2)
        c13 += a1k * bk3; // c(1, 3) += a(1, k) * b(k, 3)

        c20 += a2k * bk0; // c(2, 0) += a(2, k) * b(k, 0)
        c21 += a2k * bk1; // c(2, 1) += a(2, k) * b(k, 1)
        c22 += a2k * bk2; // c(2, 2) += a(2, k) * b(k, 2)
        c23 += a2k * bk3; // c(2, 3) += a(2, k) * b(k, 3)

        c30 += a3k * bk0; // c(3, 0) += a(3, k) * b(k, 0)
        c31 += a3k * bk1; // c(3, 1) += a(3, k) * b(k, 1)
        c32 += a3k * bk2; // c(3, 2) += a(3, k) * b(k, 2)
        c33 += a3k * bk3; // c(3, 3) += a(3, k) * b(k, 3)
    }
    c[0 * n + 0] += c00; c[0 * n + 1] += c01; c[0 * n + 2] += c02; c[0 * n + 3] += c03;
    c[1 * n + 0] += c10; c[1 * n + 1] += c11; c[1 * n + 2] += c12; c[1 * n + 3] += c13;
    c[2 * n + 0] += c20; c[2 * n + 1] += c21; c[2 * n + 2] += c22; c[2 * n + 3] += c23;
    c[3 * n + 0] += c30; c[3 * n + 1] += c31; c[3 * n + 2] += c32; c[3 * n + 3] += c33;
}

void matmul7(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_7(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

typedef float vec4 __attribute__ (( vector_size(4 * 4) ));

void kernel4x4_8(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    /*
        c( 0, 0 ), c( 0, 1 ), c( 0, 2 ), c( 0, 3 )  
        c( 1, 0 ), c( 1, 1 ), c( 1, 2 ), c( 1, 3 )  
        c( 2, 0 ), c( 2, 1 ), c( 2, 2 ), c( 2, 3 )  
        c( 3, 0 ), c( 3, 1 ), c( 3, 2 ), c( 3, 3 )
    */
    vec4 c0 = vec4{} + 0;
    vec4 c1 = vec4{} + 0;
    vec4 c2 = vec4{} + 0;
    vec4 c3 = vec4{} + 0;

    const float* a0k_p = &a[0 * inner];
    const float* a1k_p = &a[1 * inner];
    const float* a2k_p = &a[2 * inner];
    const float* a3k_p = &a[3 * inner];

    for (int k = 0; k < inner; k++) {
        vec4 a0k = vec4{} + *a0k_p++;        
        vec4 a1k = vec4{} + *a1k_p++;
        vec4 a2k = vec4{} + *a2k_p++;
        vec4 a3k = vec4{} + *a3k_p++;

        vec4 bk = *((vec4*)&b[k * n + 0]);

        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
    }
    *((vec4*)&c[0 * n]) += c0;
    *((vec4*)&c[1 * n]) += c1;
    *((vec4*)&c[2 * n]) += c2;
    *((vec4*)&c[3 * n]) += c3;
}

void matmul8(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            kernel4x4_8(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

typedef float vec8 __attribute__ (( vector_size(8 * 4) ));

void kernel8x8_9(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 c0 = vec8{} + 0;
    vec8 c1 = vec8{} + 0;
    vec8 c2 = vec8{} + 0;
    vec8 c3 = vec8{} + 0;
    vec8 c4 = vec8{} + 0;
    vec8 c5 = vec8{} + 0;
    vec8 c6 = vec8{} + 0;
    vec8 c7 = vec8{} + 0;

    const float* a0k_p = &a[0 * inner];
    const float* a1k_p = &a[1 * inner];
    const float* a2k_p = &a[2 * inner];
    const float* a3k_p = &a[3 * inner];
    const float* a4k_p = &a[4 * inner];
    const float* a5k_p = &a[5 * inner];
    const float* a6k_p = &a[6 * inner];
    const float* a7k_p = &a[7 * inner];

    for (int k = 0; k < inner; k++) {
        vec8 a0k = vec8{} + *a0k_p++;        
        vec8 a1k = vec8{} + *a1k_p++;
        vec8 a2k = vec8{} + *a2k_p++;
        vec8 a3k = vec8{} + *a3k_p++;
        vec8 a4k = vec8{} + *a4k_p++;        
        vec8 a5k = vec8{} + *a5k_p++;
        vec8 a6k = vec8{} + *a6k_p++;
        vec8 a7k = vec8{} + *a7k_p++;

        vec8 bk = *((vec8*)&b[k * n + 0]);

        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
        c4 += a4k * bk;
        c5 += a5k * bk;
        c6 += a6k * bk;
        c7 += a7k * bk;

    }
    *((vec8*)&c[0 * n]) += c0;
    *((vec8*)&c[1 * n]) += c1;
    *((vec8*)&c[2 * n]) += c2;
    *((vec8*)&c[3 * n]) += c3;
    *((vec8*)&c[4 * n]) += c4;
    *((vec8*)&c[5 * n]) += c5;
    *((vec8*)&c[6 * n]) += c6;
    *((vec8*)&c[7 * n]) += c7;
}

void matmul9(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m; i += 8) {
            kernel8x8_9(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

// void kernel7x8_10(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
//     vec8 c0 = vec8{} + 0;
//     vec8 c1 = vec8{} + 0;
//     vec8 c2 = vec8{} + 0;
//     vec8 c3 = vec8{} + 0;
//     vec8 c4 = vec8{} + 0;
//     vec8 c5 = vec8{} + 0;
//     vec8 c6 = vec8{} + 0;

//     const float* a0k_p = &a[0 * inner];
//     const float* a1k_p = &a[1 * inner];
//     const float* a2k_p = &a[2 * inner];
//     const float* a3k_p = &a[3 * inner];
//     const float* a4k_p = &a[4 * inner];
//     const float* a5k_p = &a[5 * inner];
//     const float* a6k_p = &a[6 * inner];

//     for (int k = 0; k < inner; k++) {
//         vec8 a0k = vec8{} + *a0k_p++;        
//         vec8 a1k = vec8{} + *a1k_p++;
//         vec8 a2k = vec8{} + *a2k_p++;
//         vec8 a3k = vec8{} + *a3k_p++;
//         vec8 a4k = vec8{} + *a4k_p++;        
//         vec8 a5k = vec8{} + *a5k_p++;
//         vec8 a6k = vec8{} + *a6k_p++;

//         vec8 bk = *((vec8*)&b[k * n + 0]);

//         c0 += a0k * bk;
//         c1 += a1k * bk;
//         c2 += a2k * bk;
//         c3 += a3k * bk;
//         c4 += a4k * bk;
//         c5 += a5k * bk;
//         c6 += a6k * bk;

//     }
//     *((vec8*)&c[0 * n]) += c0;
//     *((vec8*)&c[1 * n]) += c1;
//     *((vec8*)&c[2 * n]) += c2;
//     *((vec8*)&c[3 * n]) += c3;
//     *((vec8*)&c[4 * n]) += c4;
//     *((vec8*)&c[5 * n]) += c5;
//     *((vec8*)&c[6 * n]) += c6;
// }

// void matmul10(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
//     for (int j = 0; j < n; j += 8) {
//         for (int i = 0; i < m; i += 7) {
//             kernel7x8_10(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
//         }
//     }
// }

void kernel8x8_11(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 c0 = vec8{} + 0;
    vec8 c1 = vec8{} + 0;
    vec8 c2 = vec8{} + 0;
    vec8 c3 = vec8{} + 0;
    vec8 c4 = vec8{} + 0;
    vec8 c5 = vec8{} + 0;
    vec8 c6 = vec8{} + 0;
    vec8 c7 = vec8{} + 0;

    const float* a0k_p = &a[0 * inner];
    const float* a1k_p = &a[1 * inner];
    const float* a2k_p = &a[2 * inner];
    const float* a3k_p = &a[3 * inner];
    const float* a4k_p = &a[4 * inner];
    const float* a5k_p = &a[5 * inner];
    const float* a6k_p = &a[6 * inner];
    const float* a7k_p = &a[7 * inner];

    for (int k = 0; k < inner; k++) {
        vec8 bk = *((vec8*)&b[k * n + 0]);
        
        vec8 a0k = vec8{} + *a0k_p++;        
        vec8 a1k = vec8{} + *a1k_p++;
        vec8 a2k = vec8{} + *a2k_p++;
        vec8 a3k = vec8{} + *a3k_p++;
        
        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
        
        
        vec8 a4k = vec8{} + *a4k_p++;        
        vec8 a5k = vec8{} + *a5k_p++;
        vec8 a6k = vec8{} + *a6k_p++;
        vec8 a7k = vec8{} + *a7k_p++;
        
        c4 += a4k * bk;
        c5 += a5k * bk;
        c6 += a6k * bk;
        c7 += a7k * bk;

    }
    *((vec8*)&c[0 * n]) += c0;
    *((vec8*)&c[1 * n]) += c1;
    *((vec8*)&c[2 * n]) += c2;
    *((vec8*)&c[3 * n]) += c3;
    *((vec8*)&c[4 * n]) += c4;
    *((vec8*)&c[5 * n]) += c5;
    *((vec8*)&c[6 * n]) += c6;
    *((vec8*)&c[7 * n]) += c7;
}

void matmul11(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m; i += 8) {
            kernel8x8_11(m, n, inner, &a[i * inner], &b[j], &c[i * n + j]);
        }
    }
}

constexpr int M_BLOCK = 256;
constexpr int INNER_BLOCK = 128;

void micro_kernel(int n, int inner_block, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 c0 = vec8{} + 0;
    vec8 c1 = vec8{} + 0;
    vec8 c2 = vec8{} + 0;
    vec8 c3 = vec8{} + 0;
    vec8 c4 = vec8{} + 0;
    vec8 c5 = vec8{} + 0;
    vec8 c6 = vec8{} + 0;
    vec8 c7 = vec8{} + 0;

    const float* a0k_p = &a[0 * INNER_BLOCK];
    const float* a1k_p = &a[1 * INNER_BLOCK];
    const float* a2k_p = &a[2 * INNER_BLOCK];
    const float* a3k_p = &a[3 * INNER_BLOCK];
    const float* a4k_p = &a[4 * INNER_BLOCK];
    const float* a5k_p = &a[5 * INNER_BLOCK];
    const float* a6k_p = &a[6 * INNER_BLOCK];
    const float* a7k_p = &a[7 * INNER_BLOCK];

    for (int k = 0; k < inner_block; k++) {
        vec8 bk = *((vec8*)&b[k * n]);
        
        vec8 a0k = vec8{} + *a0k_p++;    
        vec8 a1k = vec8{} + *a1k_p++;
        vec8 a2k = vec8{} + *a2k_p++;
        vec8 a3k = vec8{} + *a3k_p++;
        
        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
        
        
        vec8 a4k = vec8{} + *a4k_p++;        
        vec8 a5k = vec8{} + *a5k_p++;
        vec8 a6k = vec8{} + *a6k_p++;
        vec8 a7k = vec8{} + *a7k_p++;
        
        c4 += a4k * bk;
        c5 += a5k * bk;
        c6 += a6k * bk;
        c7 += a7k * bk;
    }
    *((vec8*)&c[0 * n]) += c0;
    *((vec8*)&c[1 * n]) += c1;
    *((vec8*)&c[2 * n]) += c2;
    *((vec8*)&c[3 * n]) += c3;
    *((vec8*)&c[4 * n]) += c4;
    *((vec8*)&c[5 * n]) += c5;
    *((vec8*)&c[6 * n]) += c6;
    *((vec8*)&c[7 * n]) += c7;
}

void pack_matrix_a(int inner, int inner_block, const float* __restrict__ a, float* __restrict__ packed_a) {    
    for (int i = 0; i < 8; i++) {
        const float* a_ptr = &a[i * inner];        
        for (int j = 0; j < inner_block; j++) {
            *(packed_a + j) = *(a_ptr + j);
        }
        packed_a += INNER_BLOCK;
    }
}

void macro_kernel(int m_block, int n, int inner, int inner_block, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    alignas(32) float packed_a[M_BLOCK * INNER_BLOCK];
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m_block; i += 8) {
            if (j == 0) {
                pack_matrix_a(inner, inner_block, &a[i * inner], &packed_a[i * INNER_BLOCK]);
            }
            micro_kernel(n, inner_block, &packed_a[i * INNER_BLOCK], &b[j], &c[i * n + j]);
        }
    }
}

void matmul12(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int k = 0; k < inner; k += INNER_BLOCK) {
        int inner_block = min(inner - k, INNER_BLOCK);
        for (int i = 0; i < m; i += M_BLOCK) {
            int m_block = min(m - i, M_BLOCK);
            macro_kernel(m_block, n, inner, inner_block, &a[i * inner + k], &b[k * n], &c[i * n]);
        }
    }
}

void micro_kernel13(int n, int inner_block, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 c0 = vec8{} + 0;
    vec8 c1 = vec8{} + 0;
    vec8 c2 = vec8{} + 0;
    vec8 c3 = vec8{} + 0;
    vec8 c4 = vec8{} + 0;
    vec8 c5 = vec8{} + 0;
    vec8 c6 = vec8{} + 0;
    vec8 c7 = vec8{} + 0;

    const float* a0k_p = &a[0 * INNER_BLOCK];
    const float* a1k_p = &a[1 * INNER_BLOCK];
    const float* a2k_p = &a[2 * INNER_BLOCK];
    const float* a3k_p = &a[3 * INNER_BLOCK];
    const float* a4k_p = &a[4 * INNER_BLOCK];
    const float* a5k_p = &a[5 * INNER_BLOCK];
    const float* a6k_p = &a[6 * INNER_BLOCK];
    const float* a7k_p = &a[7 * INNER_BLOCK];

    for (int k = 0; k < inner_block; k++) {
        vec8 bk = *((vec8*)&b[k * n]);
        
        vec8 a0k = vec8{} + *a0k_p++;    
        vec8 a1k = vec8{} + *a1k_p++;
        vec8 a2k = vec8{} + *a2k_p++;
        vec8 a3k = vec8{} + *a3k_p++;
        
        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
        
        
        vec8 a4k = vec8{} + *a4k_p++;        
        vec8 a5k = vec8{} + *a5k_p++;
        vec8 a6k = vec8{} + *a6k_p++;
        vec8 a7k = vec8{} + *a7k_p++;
        
        c4 += a4k * bk;
        c5 += a5k * bk;
        c6 += a6k * bk;
        c7 += a7k * bk;
    }
    *((vec8*)&c[0 * n]) += c0;
    *((vec8*)&c[1 * n]) += c1;
    *((vec8*)&c[2 * n]) += c2;
    *((vec8*)&c[3 * n]) += c3;
    *((vec8*)&c[4 * n]) += c4;
    *((vec8*)&c[5 * n]) += c5;
    *((vec8*)&c[6 * n]) += c6;
    *((vec8*)&c[7 * n]) += c7;
}

void pack_matrix_b(int n, int inner_block, const float* __restrict__ b, float* __restrict__ packed_b) {    
    for (int i = 0; i < inner_block; i++) {
        const float* b_ptr = &b[i * n];
        for (int j = 0; j < 8; j++) {
            *(packed_b + j) = *(b_ptr + j);
        }
        packed_b += 4096;
    }
}

void macro_kernel13(int m_block, int n, int inner, int inner_block, bool first, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    alignas(32) float packed_a[M_BLOCK * INNER_BLOCK];
    alignas(32) static float packed_b[INNER_BLOCK * 4096];
    for (int j = 0; j < n; j += 8) {
        if (first) {
            pack_matrix_b(n, inner_block, &b[j], &packed_b[j]);
        }
        for (int i = 0; i < m_block; i += 8) {
            if (j == 0) {
                pack_matrix_a(inner, inner_block, &a[i * inner], &packed_a[i * INNER_BLOCK]);
            }
            micro_kernel13(n, inner_block, &packed_a[i * INNER_BLOCK], &packed_b[j], &c[i * n + j]);
        }
    }
}

void matmul13(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int k = 0; k < inner; k += INNER_BLOCK) {
        int inner_block = min(inner - k, INNER_BLOCK);
        for (int i = 0; i < m; i += M_BLOCK) {
            int m_block = min(m - i, M_BLOCK);
            macro_kernel13(m_block, n, inner, inner_block, i == 0, &a[i * inner + k], &b[k * n], &c[i * n]);
        }
    }
}

constexpr int N_BLOCK = 512;

void micro_kernel14(int n, int inner_block, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 c0 = vec8{} + 0;
    vec8 c1 = vec8{} + 0;
    vec8 c2 = vec8{} + 0;
    vec8 c3 = vec8{} + 0;
    vec8 c4 = vec8{} + 0;
    vec8 c5 = vec8{} + 0;
    vec8 c6 = vec8{} + 0;
    vec8 c7 = vec8{} + 0;

    const float* a0k_p = &a[0 * INNER_BLOCK];
    const float* a1k_p = &a[1 * INNER_BLOCK];
    const float* a2k_p = &a[2 * INNER_BLOCK];
    const float* a3k_p = &a[3 * INNER_BLOCK];
    const float* a4k_p = &a[4 * INNER_BLOCK];
    const float* a5k_p = &a[5 * INNER_BLOCK];
    const float* a6k_p = &a[6 * INNER_BLOCK];
    const float* a7k_p = &a[7 * INNER_BLOCK];

    for (int k = 0; k < inner_block; k++) {
        vec8 bk = *((vec8*)&b[k * N_BLOCK]);
        
        vec8 a0k = vec8{} + *a0k_p++;    
        vec8 a1k = vec8{} + *a1k_p++;
        vec8 a2k = vec8{} + *a2k_p++;
        vec8 a3k = vec8{} + *a3k_p++;
        
        c0 += a0k * bk;
        c1 += a1k * bk;
        c2 += a2k * bk;
        c3 += a3k * bk;
        
        
        vec8 a4k = vec8{} + *a4k_p++;        
        vec8 a5k = vec8{} + *a5k_p++;
        vec8 a6k = vec8{} + *a6k_p++;
        vec8 a7k = vec8{} + *a7k_p++;
        
        c4 += a4k * bk;
        c5 += a5k * bk;
        c6 += a6k * bk;
        c7 += a7k * bk;
    }
    *((vec8*)&c[0 * n]) += c0;
    *((vec8*)&c[1 * n]) += c1;
    *((vec8*)&c[2 * n]) += c2;
    *((vec8*)&c[3 * n]) += c3;
    *((vec8*)&c[4 * n]) += c4;
    *((vec8*)&c[5 * n]) += c5;
    *((vec8*)&c[6 * n]) += c6;
    *((vec8*)&c[7 * n]) += c7;
}

void pack_matrix_a14(int inner, int inner_block, const float* __restrict__ a, float* __restrict__ packed_a) {    
    for (int i = 0; i < 8; i++) {
        const float* a_ptr = &a[i * inner];        
        for (int j = 0; j < inner_block; j++) {
            *(packed_a + j) = *(a_ptr + j);
        }
        packed_a += INNER_BLOCK;
    }
}

void pack_matrix_b14(int n, int inner_block, const float* __restrict__ b, float* __restrict__ packed_b) {    
    for (int i = 0; i < inner_block; i++) {
        const float* b_ptr = &b[i * n];
        for (int j = 0; j < 8; j++) {
            *(packed_b + j) = *(b_ptr + j);
        }
        packed_b += N_BLOCK;
    }
}

void macro_kernel14(int m_block, int n, int inner, int inner_block, bool first, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    alignas(32) float packed_a[M_BLOCK * INNER_BLOCK];
    alignas(32) static float packed_b[INNER_BLOCK * N_BLOCK];
    for (int j = 0; j < N_BLOCK; j += 8) {
        if (first) {
            pack_matrix_b14(n, inner_block, &b[j], &packed_b[j]);
        }
        for (int i = 0; i < m_block; i += 8) {
            if (j == 0) {
                pack_matrix_a14(inner, inner_block, &a[i * inner], &packed_a[i * INNER_BLOCK]);
            }
            micro_kernel14(n, inner_block, &packed_a[i * INNER_BLOCK], &packed_b[j], &c[i * n + j]);
        }
    }
}

void matmul14(int m, int n, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int k = 0; k < inner; k += INNER_BLOCK) {
        int inner_block = min(inner - k, INNER_BLOCK);
        for (int j = 0; j < n; j += N_BLOCK) {
            for (int i = 0; i < m; i += M_BLOCK) {
                int m_block = min(m - i, M_BLOCK);
                macro_kernel14(m_block, n, inner, inner_block, i == 0, &a[i * inner + k], &b[k * n + j], &c[i * n + j]);
            }
        }        
    }
}

// int main() {
//     const int M = 2048;
//     const int INNER = 2048;
//     const int N = 4096;
//     float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//     float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//     float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//     rinit(a, M * INNER);
//     rinit(b, INNER * N);
//     memset(c, 0, sizeof(float) * M * N);
//     matmul14(M, N, INNER, a, b, c);
//     cout << c[0] << ' ' << c[M * N - 1] << endl;
// }
