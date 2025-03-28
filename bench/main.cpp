#include <chrono>
#include <cmath>
#include <benchmark/benchmark.h>
#include "nnue.h"
#include "game.h"
#include "misc.h"
#include "zobrist.h"
#include "magic.h"
#include <tuple>
#include "Eigen/Dense"
#include "matmul.h"
#include "matvec.h"
#include "util.h"

using namespace std;

static void duration_helper(uint32_t mask, uint64_t n) {
    using namespace std::chrono;
    const steady_clock::time_point start = steady_clock::now();
    duration<uint64_t, std::micro> mu;
    double res = 0;
    for (size_t iter = 0; iter < n; iter++) {
        for (size_t i = 0; i < 10; i++) {
            res = res + sqrt(i);
        }          
        if ((iter & mask) == 0) {
            mu = duration_cast<microseconds>(steady_clock::now() - start);
            res += mu.count();
        }
    }
}

static void duration0(benchmark::State& state) {
  for (auto _ : state) {
    duration_helper(0, 1000000);
  }
}

static void duration0x3(benchmark::State& state) {
  for (auto _ : state) {
    duration_helper(0x3, 1000000);
  }
}

static void duration0xf(benchmark::State& state) {
  for (auto _ : state) {
    duration_helper(0xf, 1000000);
  }
}

static void duration0xffff(benchmark::State& state) {
  for (auto _ : state) {
    duration_helper(0xffff, 1000000);
  }
}

// BENCHMARK(duration0);
// BENCHMARK(duration0x3);
// BENCHMARK(duration0xf);
// BENCHMARK(duration0xffff);

// template <size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE>
// static float nnue_helper(NNUE<H1_SIZE, H2_SIZE, H3_SIZE>& nnue, 
//                         typename NNUE<H1_SIZE, H2_SIZE, H3_SIZE>::Accumulator& acc, 
//                         size_t nb_games = 1000) {
//   float res = 0;
//   PRNG prng(42);
//   for (size_t i = 0; i < nb_games; i++) {
//     Yolah yolah;
//     nnue.init(yolah, acc);
//     Yolah::MoveList moves;
//     while (!yolah.game_over()) {
//         yolah.moves(moves);        
//         Move m = moves[prng.rand<size_t>() % moves.size()];
//         const auto [black_proba, ignore, white_proba] = nnue.output_softmax(acc);
//         res += black_proba - white_proba;
//         nnue.play(yolah.current_player(), m, acc);
//         yolah.play(m);
//     }
//   }
//   return res;
// }

// static void nnue1(benchmark::State& state) {
//   zobrist::init();
//   magic::init();  
//   NNUE<4096, 64, 64> nnue;
//   auto acc = nnue.make_accumulator();
//   nnue.load("../../nnue/nnue_parameters.txt");
//   float res = 0;
//   benchmark::DoNotOptimize(res);
//   for (auto _ : state) {    
//     res += nnue_helper(nnue, acc, 1000);
//   }
// }

static void matmul_eigen(benchmark::State& state) {
  constexpr size_t N = 1000;
  auto m1 = Eigen::MatrixXf::Random(N, N);
  auto m2 = Eigen::MatrixXf::Random(N, N);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> res(N, N);
  for (auto _ : state) {
    res += m1 * m2;
  }
}

template<typename T>
using matrix = vector<vector<T>>;

template<typename T>
matrix<T> make_matrix(size_t n, size_t m) {
  return matrix<T>(n, vector<T>(m));
}

template<typename T>
matrix<T> random_matrix(size_t n, size_t m) {
  auto res = make_matrix<T>(n, m);
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      res[i][j] = dis(gen);
    }
  }
  return res;
}

template<typename T>
matrix<T> operator+(const matrix<T>& m1, const matrix<T>& m2) {
  auto N = m1.size();
  auto M = m1[0].size();
  auto res = make_matrix<T>(N, M);     
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      res[i][j] = m1[i][j] + m2[i][j];
    }
  }
  return res;
}

template<typename T>
matrix<T> operator*(const matrix<T>& m1, const matrix<T>& m2) {
  auto N1 = m1.size();
  auto M1 = m1[0].size();
  auto N2 = m2.size();
  assert(M1 == N2);
  auto M2 = m2[0].size();
  auto res = make_matrix<T>(N1, M2);
  for (size_t i = 0; i < N1; i++) {
    for (size_t j = 0; j < M2; j++) {
      T sum{};  
      for (size_t k = 0; k < M1; k++) {
        sum += m1[i][k] * m2[k][j];  
      }
      res[i][j] = sum;
    }
  }
  return res;
}

static void matmul_basic(benchmark::State& state) {
  constexpr size_t N = 1000;
  auto m1 = random_matrix<float>(N, N);
  auto m2 = random_matrix<float>(N, N);
  auto res = make_matrix<float>(N, N);
  for (auto _ : state) {
    res = res + m1 * m2;
  }
}

// constexpr size_t M = 64;
// constexpr size_t INNER = 2048 * 2;
// constexpr size_t N = 8;

// static void mm1(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul1(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm2(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul2(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm3(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul3(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm4(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul4(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm5(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul5(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm6(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul6(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm7(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul7(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm8(benchmark::State& state) {
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul8(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm9(benchmark::State& state) {  
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul9(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// // static void mm10(benchmark::State& state) {  
// //   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
// //   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
// //   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
// //   rinit(a, M * INNER);
// //   rinit(b, INNER * N);
// //   memset(c, 0, sizeof(float) * M * N);
// //   //float res = 0;
// //   for (auto _ : state) {
// //     matmul10(M, N, INNER, a, b, c);
// //     //res += c[0];
// //   }
// // }

// static void mm11(benchmark::State& state) {  
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul11(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm12(benchmark::State& state) {  
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul12(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm13(benchmark::State& state) {  
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul13(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

// static void mm14(benchmark::State& state) {  
//   float* a = (float*)std::aligned_alloc(32, 32 * M * INNER);
//   float* b = (float*)std::aligned_alloc(32, 32 * INNER * N);
//   float* c = (float*)std::aligned_alloc(32, 32 * M * N);
//   rinit(a, M * INNER);
//   rinit(b, INNER * N);
//   memset(c, 0, sizeof(float) * M * N);
//   //float res = 0;
//   for (auto _ : state) {
//     matmul14(M, N, INNER, a, b, c);
//     //res += c[0];
//   }
// }

constexpr size_t M = 64;
constexpr size_t INNER = 4096;

static void mv1(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;

  for (auto _ : state) {
    matvec1(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv2(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec2(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv3(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec3(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv4(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec4(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv5(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec5(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv6(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec6(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv7(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec7(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv8(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec8(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv9(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec9(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv10(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * M * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  float* c = (float*)aligned_alloc(32, 32 * M);
  rinit(a, M * INNER);
  rinit(b, INNER);
  memset(c, 0, sizeof(float) * M);
  //float res = 0;
  for (auto _ : state) {
    matvec10(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv3x64_1(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * 3 * 64);
  float* b = (float*)aligned_alloc(32, 32 * 64);
  float* c = (float*)aligned_alloc(32, 32 * 3);
  rinit(a, 3 * 64);
  rinit(b, 64);
  memset(c, 0, sizeof(float) * 3);
  //float res = 0;
  for (auto _ : state) {
    matvec3x64_1(a, b, c);
    //res += c[0];
  }
}

static void mv_int1(benchmark::State& state) {  
  int8_t* a = (int8_t*)aligned_alloc(32, M * INNER);
  int8_t* b = (int8_t*)aligned_alloc(32, INNER);
  int8_t* c = (int8_t*)aligned_alloc(32, M);
  rinit_int8(a, M * INNER);
  rinit_int8(b, INNER);
  memset(c, 0, M);
  //float res = 0;
  for (auto _ : state) {
    matvec_int1(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void mv_int2(benchmark::State& state) {  
  int8_t* a = (int8_t*)aligned_alloc(32, M * INNER);
  int8_t* b = (int8_t*)aligned_alloc(32, INNER);
  int8_t* c = (int8_t*)aligned_alloc(32, M);
  rinit_int8(a, M * INNER);
  rinit_int8(b, INNER);
  memset(c, 0, M);
  //float res = 0;
  for (auto _ : state) {
    matvec_int2(M, INNER, a, b, c);
    //res += c[0];
  }
}

static void addvec1(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  rinit(a, INNER);
  rinit(b, INNER);
  //float res = 0;
  for (auto _ : state) {
    addvec1(INNER, a, b);
    //res += c[0];
  }
}

static void addvec2(benchmark::State& state) {  
  float* a = (float*)aligned_alloc(32, 32 * INNER);
  float* b = (float*)aligned_alloc(32, 32 * INNER);
  rinit(a, INNER);
  rinit(b, INNER);
  //float res = 0;
  for (auto _ : state) {
    addvec2(INNER, a, b);
    //res += c[0];
  }
}

static void nnue2(benchmark::State& state) {  
  nnue net;
  for (auto _ : state) {
    net.output();
  }
}

constexpr int NB_RANDOM_NUMBERS = 500;

static void modulo1(benchmark::State& state) {
  PRNG prng(42);
  int n = prng.rand<uint16_t>() + 1;
  uint32_t random_numbers[NB_RANDOM_NUMBERS];
  for (int i = 0; i < NB_RANDOM_NUMBERS; i++) {
    random_numbers[i] = prng.rand<uint32_t>();
  }
  for (auto _ : state) {
    for (int i = 0; i < NB_RANDOM_NUMBERS; i++) {
      benchmark::DoNotOptimize(random_numbers[i] % n);      
    }    
  }
}

static void modulo2(benchmark::State& state) {
  PRNG prng(42);
  int n = prng.rand<uint16_t>() + 1;
  uint32_t random_numbers[NB_RANDOM_NUMBERS];
  for (int i = 0; i < NB_RANDOM_NUMBERS; i++) {
    random_numbers[i] = prng.rand<uint32_t>();
  }
  for (auto _ : state) {    
    for (int i = 0; i < NB_RANDOM_NUMBERS; i++) {
      benchmark::DoNotOptimize((random_numbers[i] * (uint64_t)n) >> 32);      
    }
  }
}

// BENCHMARK(nnue1);
// BENCHMARK(matmul_eigen);
// BENCHMARK(matmul_basic);
// BENCHMARK(mm1);
// BENCHMARK(mm2);
// BENCHMARK(mm3);
// BENCHMARK(mm4);
// BENCHMARK(mm5);
// BENCHMARK(mm6);
// BENCHMARK(mm7);
// BENCHMARK(mm8);
//BENCHMARK(mm9);
//BENCHMARK(mm10);
//BENCHMARK(mm11);
//BENCHMARK(mm12);
//BENCHMARK(mm13);
//BENCHMARK(mm14);
//BENCHMARK(mv1);
// BENCHMARK(mv2);
// BENCHMARK(mv3);
// BENCHMARK(mv4);
// BENCHMARK(mv5);
// BENCHMARK(mv6);
// BENCHMARK(mv7);
//BENCHMARK(mv8);
// BENCHMARK(mv9);
// BENCHMARK(mv10);
// BENCHMARK(mv3x64_1);
// BENCHMARK(addvec1);
// BENCHMARK(addvec2);
// BENCHMARK(nnue2);
// BENCHMARK(mv_int1);
// BENCHMARK(mv_int2);
BENCHMARK(modulo1);
BENCHMARK(modulo2);

BENCHMARK_MAIN();
