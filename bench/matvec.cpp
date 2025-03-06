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

// void matvec6(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
//     for (int i = 0; i < m; i += 8) {
//         vec8 sum[8]{};
//         for (int j = 0; j < inner; j += 64) {
//             const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
//             const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
//             const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
//             const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
//             const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
//             const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
//             const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
//             const vec8 b7 = *((vec8*)&b[j + 7 * 8]);
//             for (int k = 0; k < 8; k++) {
//                 const vec8* aa = (vec8*)&a[(i + k) * inner + j];
//                 sum[k] += aa[0] * b0 + aa[1] * b1 + aa[2] * b2 + aa[3] * b3 +
//                     aa[4] * b4 + aa[5] * b5 + aa[6] * b6 + aa[7] * b7;
//             }            
//         }  
//         for (int k = 0; k < 8; k++) {
//             c[i + k] = sum[k][0] + sum[k][1] + sum[k][2] + sum[k][3] +
//                         sum[k][4] + sum[k][5] + sum[k][6] + sum[k][7];
//         }        
//     }
// }

void matvec6(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < m; i += 8) {
        vec8 sum0{}, sum1{}, sum2{}, sum3{}, sum4{}, sum5{}, sum6{}, sum7{};
        for (int j = 0; j < inner; j += 64) {
            const vec8 b0 = *((vec8*)&b[j + 0 * 8]);
            const vec8 b1 = *((vec8*)&b[j + 1 * 8]);
            const vec8 b2 = *((vec8*)&b[j + 2 * 8]);
            const vec8 b3 = *((vec8*)&b[j + 3 * 8]);
            const vec8 b4 = *((vec8*)&b[j + 4 * 8]);
            const vec8 b5 = *((vec8*)&b[j + 5 * 8]);
            const vec8 b6 = *((vec8*)&b[j + 6 * 8]);
            const vec8 b7 = *((vec8*)&b[j + 7 * 8]);

            const vec8* aa0 = (vec8*)&a[(i + 0) * inner + j];
            sum0 += aa0[0] * b0 + aa0[1] * b1 + aa0[2] * b2 + aa0[3] * b3 + aa0[4] * b4 + aa0[5] * b5 + aa0[6] * b6 + aa0[7] * b7;

            const vec8* aa1 = (vec8*)&a[(i + 1) * inner + j];
            sum1 += aa1[0] * b0 + aa1[1] * b1 + aa1[2] * b2 + aa1[3] * b3 + aa1[4] * b4 + aa1[5] * b5 + aa1[6] * b6 + aa1[7] * b7;

            const vec8* aa2 = (vec8*)&a[(i + 2) * inner + j];
            sum2 += aa2[0] * b0 + aa2[1] * b1 + aa2[2] * b2 + aa2[3] * b3 + aa2[4] * b4 + aa2[5] * b5 + aa2[6] * b6 + aa2[7] * b7;

            const vec8* aa3 = (vec8*)&a[(i + 3) * inner + j];
            sum3 += aa3[0] * b0 + aa3[1] * b1 + aa3[2] * b2 + aa3[3] * b3 + aa3[4] * b4 + aa3[5] * b5 + aa3[6] * b6 + aa3[7] * b7;

            const vec8* aa4 = (vec8*)&a[(i + 4) * inner + j];
            sum4 = aa4[0] * b0;
            sum4 += aa4[1] * b1 + aa4[2] * b2 + aa4[3] * b3 + aa4[4] * b4 + aa4[5] * b5 + aa4[6] * b6 + aa4[7] * b7;

            const vec8* aa5 = (vec8*)&a[(i + 5) * inner + j];
            sum5 += aa5[0] * b0 + aa5[1] * b1 + aa5[2] * b2 + aa5[3] * b3 + aa5[4] * b4 + aa5[5] * b5 + aa5[6] * b6 + aa5[7] * b7;

            const vec8* aa6 = (vec8*)&a[(i + 6) * inner + j];
            sum6 += aa6[0] * b0 + aa6[1] * b1 + aa6[2] * b2 + aa6[3] * b3 + aa6[4] * b4 + aa6[5] * b5 + aa6[6] * b6 + aa6[7] * b7;

            const vec8* aa7 = (vec8*)&a[(i + 7) * inner + j];
            sum7 += aa7[0] * b0 + aa7[1] * b1 + aa7[2] * b2 + aa7[3] * b3 + aa7[4] * b4 + aa7[5] * b5 + aa7[6] * b6 + aa7[7] * b7;
        }  
        c[i] = sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum0[4] + sum0[5] + sum0[6] + sum0[7];
        c[i + 1] = sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
        c[i + 2] = sum2[0] + sum2[1] + sum2[2] + sum2[3] + sum2[4] + sum2[5] + sum2[6] + sum2[7];
        c[i + 3] = sum3[0] + sum3[1] + sum3[2] + sum3[3] + sum3[4] + sum3[5] + sum3[6] + sum3[7];
        c[i + 4] = sum4[0] + sum4[1] + sum4[2] + sum4[3] + sum4[4] + sum4[5] + sum4[6] + sum4[7];
        c[i + 5] = sum5[0] + sum5[1] + sum5[2] + sum5[3] + sum5[4] + sum5[5] + sum5[6] + sum5[7];
        c[i + 6] = sum6[0] + sum6[1] + sum6[2] + sum6[3] + sum6[4] + sum6[5] + sum6[6] + sum6[7];
        c[i + 7] = sum7[0] + sum7[1] + sum7[2] + sum7[3] + sum7[4] + sum7[5] + sum7[6] + sum7[7];                
    }
}

#include <immintrin.h>

void matvec7(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < m; i += 8) {
        __m256 sum[8]{};
        for (int j = 0; j < inner; j += 64) {
            const __m256 b0 = *((__m256*)&b[j + 0 * 8]);
            const __m256 b1 = *((__m256*)&b[j + 1 * 8]);
            const __m256 b2 = *((__m256*)&b[j + 2 * 8]);
            const __m256 b3 = *((__m256*)&b[j + 3 * 8]);
            const __m256 b4 = *((__m256*)&b[j + 4 * 8]);
            const __m256 b5 = *((__m256*)&b[j + 5 * 8]);
            const __m256 b6 = *((__m256*)&b[j + 6 * 8]);
            const __m256 b7 = *((__m256*)&b[j + 7 * 8]);
            for (int k = 0; k < 8; k++) {
                const __m256* aa = (__m256*)&a[(i + k) * inner + j];
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

void matvec8(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {    
    for (int i = 0; i < m; i += 64) {        
        vec8 sum[8]{};
        for (int j = 0; j < inner; j++) {
            vec8 bb = vec8{} + b[j];
            const vec8* aa = (vec8*)&a[j * m + i];
            for (int k = 0; k < 8; k++) {
                sum[k] += aa[k] * bb;
            }
        }
        for (int k = 0; k < 8; k++) {
            *((vec8*)&c[i + k * 8]) = sum[k];
        }
    }
}

// void matvec8(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {        
//     vec8 sum[8]{};
//     for (int j = 0; j < inner; j++) {
//         vec8 bb = vec8{} + b[j];
//         const vec8* aa = (vec8*)&a[j * m];
//         for (int k = 0; k < 8; k++) {
//             sum[k] += aa[k] * bb;
//         }
//     }
//     for (int k = 0; k < 8; k++) {
//         *((vec8*)&c[k * 8]) = sum[k];
//     }    
// }

void matvec9(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {    
    for (int i = 0; i < m; i += 64) {        
        vec8 sum[8]{};
        for (int j = 0; j < inner; j += 8) {
            vec8 bb0 = vec8{} + b[j];
            vec8 bb1 = vec8{} + b[j + 1];
            vec8 bb2 = vec8{} + b[j + 2];
            vec8 bb3 = vec8{} + b[j + 3];
            vec8 bb4 = vec8{} + b[j + 4];
            vec8 bb5 = vec8{} + b[j + 5];
            vec8 bb6 = vec8{} + b[j + 6];
            vec8 bb7 = vec8{} + b[j + 7];
            
            sum[0] += *((vec8*)&a[j * m + i + 0 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 0 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 0 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 0 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 0 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 0 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 0 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 0 * 8]) * bb7;
            sum[1] += *((vec8*)&a[j * m + i + 1 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 1 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 1 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 1 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 1 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 1 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 1 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 1 * 8]) * bb7;
            sum[2] += *((vec8*)&a[j * m + i + 2 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 2 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 2 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 2 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 2 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 2 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 2 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 2 * 8]) * bb7;
            sum[3] += *((vec8*)&a[j * m + i + 3 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 3 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 3 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 3 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 3 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 3 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 3 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 3 * 8]) * bb7;
            sum[4] += *((vec8*)&a[j * m + i + 4 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 4 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 4 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 4 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 4 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 4 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 4 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 4 * 8]) * bb7;
            sum[5] += *((vec8*)&a[j * m + i + 5 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 5 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 5 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 5 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 5 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 5 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 5 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 5 * 8]) * bb7;
            sum[6] += *((vec8*)&a[j * m + i + 6 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 6 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 6 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 6 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 6 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 6 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 6 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 6 * 8]) * bb7;
            sum[7] += *((vec8*)&a[j * m + i + 7 * 8]) * bb0 + *((vec8*)&a[(j + 1) * m + i + 7 * 8]) * bb1 + *((vec8*)&a[(j + 2) * m + i + 7 * 8]) * bb2 + *((vec8*)&a[(j + 3) * m + i + 7 * 8]) * bb3 
                        + *((vec8*)&a[(j + 4) * m + i + 7 * 8]) * bb4 + *((vec8*)&a[(j + 5) * m + i + 7 * 8]) * bb5 + *((vec8*)&a[(j + 6) * m + i + 7 * 8]) * bb6 + *((vec8*)&a[(j + 7) * m + i + 7 * 8]) * bb7;
        }
        for (int k = 0; k < 8; k++) {
            *((vec8*)&c[i + k * 8]) = sum[k];
        }
    }
}

void matvec10(int m, int inner, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {    
    for (int i = 0; i < m; i += 64) {        
        vec8 sum[8]{};
        for (int j = 0; j < inner; j++) {
            vec8 bb = vec8{} + b[j];
            const vec8* aa = (vec8*)&a[j * m + i];            
            sum[0] += aa[0] * bb;
            sum[1] += aa[1] * bb;
            sum[2] += aa[2] * bb;
            sum[3] += aa[3] * bb;
            sum[4] += aa[4] * bb;
            sum[5] += aa[5] * bb;
            sum[6] += aa[6] * bb;
            sum[7] += aa[7] * bb;
        }
        for (int k = 0; k < 8; k++) {
            *((vec8*)&c[i + k * 8]) = sum[k];
        }
    }
}

void matvec3x64_1(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < 64; j++) {
            sum += a[i * 64 + j] * b[j];
        }
        c[i] = sum;
    }
}

void matvec_int1(int m, int inner, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c) {
    for (int i = 0; i < m; i++) {
        int8_t sum = 0;
        for (int j = 0; j < inner; j++) {
            sum += a[i * inner + j] * b[j];
        }
        c[i] = sum;
    }
}

typedef int8_t vec32 __attribute__ (( vector_size(32) ));

void matvec_int2(int m, int inner, const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ c) {
    for (int i = 0; i < m; i += 256) { 
        vec32 sum[8]{};
        for (int j = 0; j < inner; j++) {
            vec32 bb = vec32{} + b[j];
            const vec32* aa = (vec32*)&a[j * m + i];
            for (int k = 0; k < 8; k++) {
                sum[k] += aa[k] * bb;
            }
        }
        for (int k = 0; k < 8; k++) {
            *((vec32*)&c[i + k * 32]) = sum[k];
        }
    }
}

void addvec1(int n, float* a, const float* __restrict__ b) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void addvec2(int n, float* a, const float* __restrict__ b) {
    vec8* aa = (vec8*)a;
    const vec8* bb = (vec8*)b;
    int m = n / 8;
    for (int i = 0; i < m; i++) {
        aa[i] += bb[i];
    }
}

nnue::nnue() {
    acc = (float*)std::aligned_alloc(32, 32 * (H1 + H2));
    rinit(acc, H1 + H2);
    h1_to_h2 = (float*)std::aligned_alloc(32, 32 * H1 * H2);
    rinit(h1_to_h2, H1 * H2);
    h2_to_h3 = (float*)std::aligned_alloc(32, 32 * H2 * H3);
    rinit(h2_to_h3, H2 * H3);
    h3_to_output = (float*)std::aligned_alloc(32, 32 * H3 * OUTPUT);
    rinit(h3_to_output, H3 * OUTPUT);
}

nnue::~nnue() {    
    delete[] acc;
    delete[] h1_to_h2;
    delete[] h2_to_h3;
    delete[] h3_to_output;    
}

float nnue::output() {
    matvec8(H2, H1, h1_to_h2, acc, acc + H1);
    matvec8(H3, H2, h2_to_h3, acc + H1, acc);
    matvec3x64_1(h3_to_output, acc, acc + H2);
    const float* output = acc + H2;
    float e1 = std::exp(output[0]);
    float e2 = std::exp(output[1]);
    float e3 = std::exp(output[2]);
    float sum = e1 + e2 + e3;
    float res = e1 / sum + e2 / sum + e3 / sum;
    return res;
}

//int main() {
//     // const size_t M = 8192;
//     // const size_t INNER = 2048 * 4;
//     // float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//     // float* a_t = (float*)std::aligned_alloc(32, 32 * INNER * M);
//     // float* b = (float*)std::aligned_alloc(32, 32 * INNER);
//     // float* c = (float*)std::aligned_alloc(32, 32 * M);
//     // rinit(a, M * INNER);
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < INNER; j++) {
//     //         a_t[j * M + i] = a[i * INNER + j];
//     //     }
//     // }
//     // rinit(b, INNER);
//     // memset(c, 0, sizeof(float) * M);
//     // //matvec6(M, INNER, a, b, c);
//     // matvec10(M, INNER, a_t, b, c);
//     // cout << c[0] << ' ' << c[M - 1] << endl;

//     // float* a = (float*)std::aligned_alloc(32, 32 * 3 * 64);
//     // float* b = (float*)std::aligned_alloc(32, 32 * 64);
//     // float* c = (float*)std::aligned_alloc(32, 32 * 3);
//     // rinit(a, 64 * 3);
//     // rinit(b, 64);
//     // memset(c, 0, sizeof(float) * M);
//     // matvec3x64_1(a, b, c);
//     // cout << c[0] << ' ' << c[1] << ' ' << c[2] << '\n';

//     // const size_t N = 2048 * 2;
//     // float* a = (float*)std::aligned_alloc(32, 32 * N);
//     // float* b = (float*)std::aligned_alloc(32, 32 * N);
//     // rinit(a, N);
//     // rinit(b, N);
//     // addvec2(N, a, b);
//     // cout << a[0] << ' ' << a[N - 1] << '\n';
//     nnue net;
//     cout << net.output() << '\n';
//     const size_t M = 8192;
//     const size_t INNER = 2048 * 4;
//     int8_t* a = (int8_t*)std::aligned_alloc(32, M * INNER);
//     int8_t* a_t = (int8_t*)std::aligned_alloc(32, M * INNER);
//     int8_t* b = (int8_t*)std::aligned_alloc(32, INNER);
//     int8_t* c = (int8_t*)std::aligned_alloc(32, M);
//     rinit_int8(a, M * INNER);
//     rinit_int8(b, INNER);
//     memset(c, 0, M);
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < INNER; j++) {
//             a_t[j * M + i] = a[i * INNER + j];
//         }
//     }
//     matvec_int2(M, INNER, a_t, b, c);
//     cout << (int)c[0] << ' ' << (int)c[M - 1] << '\n';
// }
