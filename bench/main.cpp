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
using namespace std;


static void duration_helper(uint32_t mask, uint64_t n) {
    using namespace std::chrono;
    const steady_clock::time_point start = steady_clock::now();
    duration<uint64_t, std::micro> mu;
    double res = 0;
    benchmark::DoNotOptimize(res);
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

template <size_t H1_SIZE, size_t H2_SIZE, size_t H3_SIZE>
static float nnue_helper(NNUE<H1_SIZE, H2_SIZE, H3_SIZE>& nnue, 
                        typename NNUE<H1_SIZE, H2_SIZE, H3_SIZE>::Accumulator& acc, 
                        size_t nb_games = 1000) {
  float res = 0;
  PRNG prng(42);
  for (size_t i = 0; i < nb_games; i++) {
    Yolah yolah;
    nnue.init(yolah, acc);
    Yolah::MoveList moves;
    while (!yolah.game_over()) {
        yolah.moves(moves);        
        Move m = moves[prng.rand<size_t>() % moves.size()];
        const auto [black_proba, ignore, white_proba] = nnue.output_softmax(acc);
        res += black_proba - white_proba;
        nnue.play(yolah.current_player(), m, acc);
        yolah.play(m);
    }
  }
  return res;
}

static void nnue(benchmark::State& state) {
  zobrist::init();
  magic::init();  
  NNUE<4096, 64, 64> nnue;
  auto acc = nnue.make_accumulator();
  nnue.load("../../nnue/nnue_parameters.txt");
  float res = 0;
  benchmark::DoNotOptimize(res);
  for (auto _ : state) {    
    res += nnue_helper(nnue, acc, 1000);
  }
}

static void matmul_eigen(benchmark::State& state) {
  constexpr size_t N = 1000;
  auto m1 = Eigen::MatrixXf::Random(N, N);
  auto m2 = Eigen::MatrixXf::Random(N, N);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> res(N, N);
  benchmark::DoNotOptimize(res);
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
  benchmark::DoNotOptimize(res);
  for (auto _ : state) {
    res = res + m1 * m2;
  }
}

BENCHMARK(nnue);
// BENCHMARK(matmul_eigen);
// BENCHMARK(matmul_basic);

BENCHMARK_MAIN();
