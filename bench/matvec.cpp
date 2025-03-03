#include <bits/stdc++.h>
#include "matvec.h"
#include "util.h"

using namespace std;

void matvec1(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < inner; j++) {
            sum += a[i * inner + j] * b[j];
        }
        c[i] = sum;
    }
}

typedef float vec8 __attribute__ (( vector_size(8 * 4) ));

void matvec2(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 res[m]{};
    for (int j = 0; j < inner; j += 64) {
        const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
        const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
        const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
        const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
        const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
        const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
        const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
        const vec8 b7 = *((vec8*)&b[j + 7 * 8]);
        for (int i = 0; i < m; i++) {
            const vec8* aa = (vec8*)&a[i * inner + j];
            res[i] += aa[0] * b0 + aa[1] * b1 + aa[2] * b2 + aa[3] * b3 +
                    aa[4] * b4 + aa[5] * b5 + aa[6] * b6 + aa[7] * b7; 
        }
    }
    for (int i = 0; i < m; i++) {
        c[i] = res[i][0] + res[i][1] + res[i][2] + res[i][3] + res[i][4] + res[i][5] + res[i][6] + res[i][7];
    }
}

void matvec3(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 res[m]{};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < inner; j += 8) {
            const vec8 aa = *((vec8*)&a[i * inner + j]);
            const vec8 bb = *((vec8*)&b[j]);
            res[i] += aa * bb;
        }
    }
    for (int i = 0; i < m; i++) {
        c[i] = res[i][0] + res[i][1] + res[i][2] + res[i][3] + res[i][4] + res[i][5] + res[i][6] + res[i][7];
    }
}

void matvec4(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 res[m]{};
    for (int j = 0; j < inner; j += 64) {
        const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
        const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
        const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
        const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
        const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
        const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
        const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
        const vec8 b7 = *((vec8*)&b[j + 7 * 8]);
        const float* a_ptr = &a[j];
        vec8 sum0;
        vec8 sum1;
        vec8 sum2;
        vec8 sum3;
        vec8 sum4;
        vec8 sum5;
        vec8 sum6;
        vec8 sum7;        
        for (int i = 0; i < m; i += 8) {
            const vec8* aa0 = (vec8*)&a[i * inner + j];
            const vec8* aa1 = (vec8*)&a[(i + 1) * inner + j];
            const vec8* aa2 = (vec8*)&a[(i + 2) * inner + j];
            const vec8* aa3 = (vec8*)&a[(i + 3) * inner + j];
            const vec8* aa4 = (vec8*)&a[(i + 4) * inner + j];
            const vec8* aa5 = (vec8*)&a[(i + 5) * inner + j];
            const vec8* aa6 = (vec8*)&a[(i + 6) * inner + j];
            const vec8* aa7 = (vec8*)&a[(i + 7) * inner + j];

            sum0 = aa0[0] * b0 +
                aa0[1] * b1 +
                aa0[2] * b2 +
                aa0[3] * b3 + 
                aa0[4] * b4 +
                aa0[5] * b5 +
                aa0[6] * b6 +
                aa0[7] * b7;

            sum1 = aa1[0] * b0 +
                aa1[1] * b1 +
                aa1[2] * b2 +
                aa1[3] * b3 + 
                aa1[4] * b4 +
                aa1[5] * b5 +
                aa1[6] * b6 +
                aa1[7] * b7;
            
            sum2 = aa2[0] * b0 +
                aa2[1] * b1 +
                aa2[2] * b2 +
                aa2[3] * b3 + 
                aa2[4] * b4 +
                aa2[5] * b5 +
                aa2[6] * b6 +
                aa2[7] * b7;

            sum3 = aa3[0] * b0 +
                aa3[1] * b1 +
                aa3[2] * b2 +
                aa3[3] * b3 + 
                aa3[4] * b4 +
                aa3[5] * b5 +
                aa3[6] * b6 +
                aa3[7] * b7;
                
            sum4 = aa4[0] * b0 +
                aa4[1] * b1 +
                aa4[2] * b2 +
                aa4[3] * b3 + 
                aa4[4] * b4 +
                aa4[5] * b5 +
                aa4[6] * b6 +
                aa4[7] * b7;
            
            sum5 = aa5[0] * b0 +
                aa5[1] * b1 +
                aa5[2] * b2 +
                aa5[3] * b3 + 
                aa5[4] * b4 +
                aa5[5] * b5 +
                aa5[6] * b6 +
                aa5[7] * b7;

            sum6 = aa6[0] * b0 +
                aa6[1] * b1 +
                aa6[2] * b2 +
                aa6[3] * b3 + 
                aa6[4] * b4 +
                aa6[5] * b5 +
                aa6[6] * b6 +
                aa6[7] * b7;

            sum7 = aa7[0] * b0 +
                aa7[1] * b1 +
                aa7[2] * b2 +
                aa7[3] * b3 + 
                aa7[4] * b4 +
                aa7[5] * b5 +
                aa7[6] * b6 +
                aa7[7] * b7;

            res[i] += sum0;
            res[i + 1] += sum1;
            res[i + 2] += sum2;
            res[i + 3] += sum3;
            res[i + 4] += sum4;
            res[i + 5] += sum5;
            res[i + 6] += sum6;
            res[i + 7] += sum7;
        }        
    }
    for (int i = 0; i < m; i++) {
        c[i] = res[i][0] + res[i][1] + res[i][2] + res[i][3] + res[i][4] + res[i][5] + res[i][6] + res[i][7];
    }
}

void matvec5(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    vec8 res[m]{};
    for (int j = 0; j < inner; j += 128) {
        const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
        const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
        const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
        const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
        const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
        const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
        const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
        const vec8 b7 = *((vec8*)&b[j + 7 * 8]);
        const vec8 b8 = *((vec8*)&b[j + 8 * 8]);
        const vec8 b9 = *((vec8*)&b[j + 9 * 8]);
        const vec8 b10 = *((vec8*)&b[j + 10 * 8]);
        const vec8 b11 = *((vec8*)&b[j + 11 * 8]);
        const vec8 b12 = *((vec8*)&b[j + 12 * 8]);
        const vec8 b13 = *((vec8*)&b[j + 13 * 8]);
        const vec8 b14 = *((vec8*)&b[j + 14 * 8]);
        const vec8 b15 = *((vec8*)&b[j + 15 * 8]);
        for (int i = 0; i < m; i++) {
            const vec8* aa = (vec8*)&a[i * inner + j];
            res[i] += aa[0] * b0 + aa[1] * b1 + aa[2] * b2 + aa[3] * b3 +
                    aa[4] * b4 + aa[5] * b5 + aa[6] * b6 + aa[7] * b7 +
                    aa[8] * b8 + aa[9] * b9 + aa[10] * b10 + aa[11] * b11 +
                    aa[12] * b12 + aa[13] * b13 + aa[14] * b14 + aa[15] * b15; 
        }
    }
    for (int i = 0; i < m; i++) {
        c[i] = res[i][0] + res[i][1] + res[i][2] + res[i][3] + res[i][4] + res[i][5] + res[i][6] + res[i][7];
    }
}

void matvec6(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < m; i += 8) {
        vec8 sum[8]{};
        for (int j = 0; j < inner; j += 64) {
            const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
            const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
            const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
            const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
            const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
            const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
            const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
            const vec8 b7 = *((vec8*)&b[j + 7 * 8]);
            for (int k = 0; k < 8; k++) {
                const vec8* aa = (vec8*)&a[(i + k) * inner + j];
                sum[k] += aa[0] * b0 + aa[1] * b1 + aa[2] * b2 + aa[3] * b3 +
                    aa[4] * b4 + aa[5] * b5 + aa[6] * b6 + aa[7] * b7;
            }            
        }
        for (int k = 0; k < 8; k++) {
            c[i + k] = sum[k][0] + sum[k][1] + sum[k][2] + sum[k][3] +
                        sum[k][4] + sum[k][5] + sum[k][6] + sum[k][7];
        }
    }
}

// int main() {
//     const size_t M = 8192;
//     const size_t INNER = 2048 * 4;
//     float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//     float* b = (float*)std::aligned_alloc(32, 32 * INNER);
//     float* c = (float*)std::aligned_alloc(32, 32 * M);
//     rinit(a, M * INNER);
//     rinit(b, INNER);
//     memset(c, 0, sizeof(float) * M);
//     matvec6(M, INNER, a, b, c);
//     cout << c[0] << ' ' << c[M - 1] << endl;
// }
