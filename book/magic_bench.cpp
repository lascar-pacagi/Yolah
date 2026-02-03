#include <benchmark/benchmark.h>
#include <unordered_map>
#include <vector>
#include <bit>

using ll = uint64_t;

std::vector<ll> generate_isolated_bits() {
    std::vector<ll> samples;
    for (int i = 0; i < 64; i++) {
        samples.push_back(1ULL << i);
    }
    return samples;
}

static void BM_magic_positions(benchmark::State& state) {
    constexpr ll MAGIC = 0x2643c51ab9dfa5b;
    constexpr int K = 6;
    constexpr uint8_t positions[64] = {
        0,1,2,14,3,22,28,15,11,4,23,55,7,29,41,16,12,26,53,5,24,33,56,35,61,8,30,58,37,42,17,46,63,13,21,27,10,54,6,40,25,52,32,34,60,57,36,45,62,20,9,39,51,31,59,44,19,38,50,43,18,49,48,47,
    };
    auto samples = generate_isolated_bits();
    size_t idx = 0;
    for (auto _ : state) {
        ll bitboard = samples[idx++ & 63];
        benchmark::DoNotOptimize(positions[bitboard * MAGIC >> (64 - K)]);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_magic_positions);

static void BM_unordered_map_positions(benchmark::State& state) {
    std::unordered_map<ll, uint8_t> map;
    for (uint8_t i = 0; i < 64; i++) {
        map[1ULL << i] = i;
    }
    auto samples = generate_isolated_bits();
    size_t idx = 0;
    for (auto _ : state) {
        ll bitboard = samples[idx++ & 63];
        benchmark::DoNotOptimize(map[bitboard]);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_unordered_map_positions);

static void BM_countr_zero_positions(benchmark::State& state) {
    auto samples = generate_isolated_bits();
    size_t idx = 0;
    for (auto _ : state) {
        ll bitboard = samples[idx++ & 63];
        benchmark::DoNotOptimize(std::countr_zero(bitboard));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_countr_zero_positions);

BENCHMARK_MAIN();



























// #include <benchmark/benchmark.h>
// #include <random>
// #include <unordered_map>
// #include <vector>

// using ll = uint64_t;

// // Magic number and bit shift found by Z3
// constexpr ll MAGIC = 0x2643c51ab9dfa5b;
// constexpr int K = 6;

// // Pre-computed lookup table (array-based)
// constexpr uint8_t bitscan[64] = {
//     0,1,2,14,3,22,28,15,11,4,23,55,7,29,41,16,12,26,53,5,24,33,56,35,61,8,30,58,37,42,17,46,63,13,21,27,10,54,6,40,25,52,32,34,60,57,36,45,62,20,9,39,51,31,59,44,19,38,50,43,18,49,48,47,
// };

// // Generate test data: isolated bit positions (powers of 2)
// std::vector<ll> generate_isolated_bits() {
//     std::vector<ll> samples;
//     for (int i = 0; i < 64; i++) {
//         samples.push_back(1ULL << i);
//     }
//     return samples;
// }

// // Generate test data: random bitboards with varying popcount
// std::vector<ll> generate_random_bitboards(int count, int seed = 42) {
//     std::vector<ll> samples;
//     std::mt19937_64 rng(seed);
//     std::uniform_int_distribution<ll> dist;

//     for (int i = 0; i < count; i++) {
//         samples.push_back(dist(rng));
//     }
//     return samples;
// }

// // Generate test data: sparse bitboards (fewer bits set)
// std::vector<ll> generate_sparse_bitboards(int count, int seed = 42) {
//     std::vector<ll> samples;
//     std::mt19937_64 rng(seed);
//     std::uniform_int_distribution<ll> dist;

//     for (int i = 0; i < count; i++) {
//         // AND multiple randoms to get sparse bitboards
//         ll value = dist(rng) & dist(rng) & dist(rng);
//         if (value != 0) {
//             samples.push_back(value);
//         }
//     }
//     return samples;
// }

// // Create hash map for comparison
// std::unordered_map<ll, int> create_bitscan_map() {
//     std::unordered_map<ll, int> map;
//     for (int i = 0; i < 64; i++) {
//         map[1ULL << i] = i;
//     }
//     return map;
// }

// //=============================================================================
// // Benchmark: Array-based lookup with magic multiplication (isolated bits)
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_MagicBitscan_IsolatedBits(benchmark::State& state) {
//     auto samples = generate_isolated_bits();
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[idx++ & 63];
//             sum += bitscan[bitboard * MAGIC >> (64 - K)];
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_MagicBitscan_IsolatedBits);

// //=============================================================================
// // Benchmark: Hash map lookup (isolated bits)
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_UnorderedMap_IsolatedBits(benchmark::State& state) {
//     auto samples = generate_isolated_bits();
//     auto map = create_bitscan_map();
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[idx++ & 63];
//             sum += map[bitboard];
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_UnorderedMap_IsolatedBits);

// //=============================================================================
// // Benchmark: Builtin CTZ (Count Trailing Zeros) - baseline for isolated bits
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_BuiltinCTZ_IsolatedBits(benchmark::State& state) {
//     auto samples = generate_isolated_bits();
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[idx++ & 63];
//             sum += __builtin_ctzll(bitboard);
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_BuiltinCTZ_IsolatedBits);

// //=============================================================================
// // Benchmark: Extract LSB and use magic (for random bitboards)
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_MagicBitscan_RandomBitboards(benchmark::State& state) {
//     auto samples = generate_random_bitboards(10000);
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[(idx++) % samples.size()];
//             ll lsb = bitboard & -bitboard;
//             sum += bitscan[lsb * MAGIC >> (64 - K)];
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_MagicBitscan_RandomBitboards);

// //=============================================================================
// // Benchmark: Extract LSB and use hash map (for random bitboards)
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_UnorderedMap_RandomBitboards(benchmark::State& state) {
//     auto samples = generate_random_bitboards(10000);
//     auto map = create_bitscan_map();
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[(idx++) % samples.size()];
//             ll lsb = bitboard & -bitboard;
//             sum += map[lsb];
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_UnorderedMap_RandomBitboards);

// //=============================================================================
// // Benchmark: Builtin CTZ (for random bitboards)
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_BuiltinCTZ_RandomBitboards(benchmark::State& state) {
//     auto samples = generate_random_bitboards(10000);
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[(idx++) % samples.size()];
//             sum += __builtin_ctzll(bitboard);
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_BuiltinCTZ_RandomBitboards);

// //=============================================================================
// // Benchmark: Magic bitscan with sparse bitboards
// // Multiple lookups per iteration to amortize overhead
// //=============================================================================
// static void BM_MagicBitscan_SparseBitboards(benchmark::State& state) {
//     auto samples = generate_sparse_bitboards(10000);
//     constexpr int LOOKUPS_PER_ITER = 100;
//     size_t idx = 0;

//     for (auto _ : state) {
//         int sum = 0;
//         for (int i = 0; i < LOOKUPS_PER_ITER; i++) {
//             ll bitboard = samples[(idx++) % samples.size()];
//             ll lsb = bitboard & -bitboard;
//             sum += bitscan[lsb * MAGIC >> (64 - K)];
//         }
//         benchmark::DoNotOptimize(sum);
//     }

//     state.SetItemsProcessed(state.iterations() * LOOKUPS_PER_ITER);
// }
// BENCHMARK(BM_MagicBitscan_SparseBitboards);

// //=============================================================================
// // Benchmark: Full bitboard scan - iterate all set bits using magic
// //=============================================================================
// static void BM_FullScan_Magic(benchmark::State& state) {
//     auto samples = generate_random_bitboards(1000);
//     size_t idx = 0;
//     int result = 0;

//     for (auto _ : state) {
//         ll bitboard = samples[idx % samples.size()];

//         // Iterate over all set bits
//         while (bitboard) {
//             ll lsb = bitboard & -bitboard;
//             int pos = bitscan[lsb * MAGIC >> (64 - K)];
//             benchmark::DoNotOptimize(result += pos);
//             bitboard &= bitboard - 1;  // Clear LSB
//         }
//         idx++;
//     }

//     state.SetItemsProcessed(state.iterations());
//     benchmark::DoNotOptimize(result);
// }
// BENCHMARK(BM_FullScan_Magic);

// //=============================================================================
// // Benchmark: Full bitboard scan - iterate all set bits using CTZ
// //=============================================================================
// static void BM_FullScan_CTZ(benchmark::State& state) {
//     auto samples = generate_random_bitboards(1000);
//     size_t idx = 0;
//     int result = 0;

//     for (auto _ : state) {
//         ll bitboard = samples[idx % samples.size()];

//         // Iterate over all set bits
//         while (bitboard) {
//             int pos = __builtin_ctzll(bitboard);
//             benchmark::DoNotOptimize(result += pos);
//             bitboard &= bitboard - 1;  // Clear LSB
//         }
//         idx++;
//     }

//     state.SetItemsProcessed(state.iterations());
//     benchmark::DoNotOptimize(result);
// }
// BENCHMARK(BM_FullScan_CTZ);

// //=============================================================================
// // Benchmark: Memory access pattern - sequential array access
// //=============================================================================
// static void BM_SequentialArrayAccess(benchmark::State& state) {
//     int result = 0;

//     for (auto _ : state) {
//         for (int i = 0; i < 64; i++) {
//             benchmark::DoNotOptimize(result += bitscan[i]);
//         }
//     }

//     state.SetItemsProcessed(state.iterations() * 64);
//     benchmark::DoNotOptimize(result);
// }
// BENCHMARK(BM_SequentialArrayAccess);

// //=============================================================================
// // Benchmark: Memory access pattern - random array access
// //=============================================================================
// static void BM_RandomArrayAccess(benchmark::State& state) {
//     std::vector<int> indices(1000);
//     std::mt19937 rng(42);
//     std::uniform_int_distribution<int> dist(0, 63);
//     for (auto& idx : indices) {
//         idx = dist(rng);
//     }

//     size_t idx = 0;
//     int result = 0;

//     for (auto _ : state) {
//         int i = indices[idx % indices.size()];
//         benchmark::DoNotOptimize(result += bitscan[i]);
//         idx++;
//     }

//     state.SetItemsProcessed(state.iterations());
//     benchmark::DoNotOptimize(result);
// }
// BENCHMARK(BM_RandomArrayAccess);

// BENCHMARK_MAIN();
