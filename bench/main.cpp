#include <chrono>
#include <cmath>
#include <benchmark/benchmark.h>

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

BENCHMARK(duration0);
BENCHMARK(duration0x3);
BENCHMARK(duration0xf);
BENCHMARK(duration0xffff);

BENCHMARK_MAIN();
