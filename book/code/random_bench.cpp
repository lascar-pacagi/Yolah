#include <benchmark/benchmark.h>
#include <bits/stdc++.h>
using namespace std;

constexpr uint32_t NB_ITERATIONS = 1000;
constexpr uint64_t SEED = 42;

class PRNG {
    uint64_t s;
    uint64_t rand64() {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
   public:
    PRNG(uint64_t seed) :
        s(seed) {
    }
    uint64_t seed() {
        return s;
    }
    template<typename T>
    T rand() {
        return T(rand64());
    }
};

constexpr uint32_t reduce(uint32_t x, uint32_t N) {
    return ((uint64_t) x * (uint64_t) N) >> 32;
}

uint32_t random_prng_modulo(uint32_t nb_iterations) {
    uint32_t res = 0;
    PRNG prng(SEED);
    for (uint32_t i = 1; i <= nb_iterations; i++) {
        res += prng.rand<uint32_t>() % i;
    }
    return res;
}

static void BM_random_prng_modulo(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(random_prng_modulo(NB_ITERATIONS));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_random_prng_modulo);

uint32_t random_prng_reduce(uint32_t nb_iterations) {
    uint32_t res = 0;
    PRNG prng(SEED);
    for (uint32_t i = 1; i <= nb_iterations; i++) {
        res += reduce(prng.rand<uint32_t>(), i);
    }
    return res;
}

static void BM_random_prng_reduce(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(random_prng_reduce(NB_ITERATIONS));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_random_prng_reduce);

uint32_t random_mt19937(uint32_t nb_iterations) {
    uint32_t res = 0;
    mt19937 mt(SEED);
    for (uint32_t i = 1; i <= nb_iterations; i++) {        
        res += uniform_int_distribution<uint32_t>{0, i - 1}(mt);
    }
    return res;
}

static void BM_random_mt19937(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(random_mt19937(NB_ITERATIONS));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_random_mt19937);

uint32_t random_mt19937_modulo(uint32_t nb_iterations) {
    uint32_t res = 0;
    mt19937 mt(SEED);
    for (uint32_t i = 1; i <= nb_iterations; i++) {        
        res += uniform_int_distribution<uint32_t>{}(mt) % i;
    }
    return res;
}

static void BM_random_mt19937_modulo(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(random_mt19937_modulo(NB_ITERATIONS));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_random_mt19937_modulo);

uint32_t random_mt19937_reduce(uint32_t nb_iterations) {
    uint32_t res = 0;
    mt19937 mt(SEED);
    for (uint32_t i = 1; i <= nb_iterations; i++) {        
        res += reduce(uniform_int_distribution<uint32_t>{}(mt), i);
    }
    return res;
}

static void BM_random_mt19937_reduce(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(random_mt19937_reduce(NB_ITERATIONS));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_random_mt19937_reduce);

BENCHMARK_MAIN();