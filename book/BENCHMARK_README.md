# Magic Number Benchmark Suite

Comprehensive benchmark comparing different bit scanning methods for bitboard operations.

## Overview

This benchmark compares three approaches to finding the position of set bits in 64-bit bitboards:

1. **Magic Number + Array Lookup**: Multiplicative hashing with pre-computed lookup table
2. **Hash Map (unordered_map)**: Standard C++ hash map lookup
3. **Builtin CTZ (`__builtin_ctzll`)**: Hardware instruction (Count Trailing Zeros)

## Quick Start

### Compile and Run

```bash
# Using Make (recommended)
make bench-run

# Or compile manually
g++ -std=c++23 -O3 -march=native -Wall -Wextra magic_bench.cpp -o magic_bench -lbenchmark -lpthread

# Run benchmark
./magic_bench
```

### Using CMake

```bash
cd book
mkdir build && cd build
cmake ..
make
./magic_bench
```

## Benchmark Tests

### 1. Isolated Bits (Powers of 2)
Tests single-bit positions (1ULL << i for i in 0..63).

- `BM_MagicBitscan_IsolatedBits` - Magic number lookup
- `BM_UnorderedMap_IsolatedBits` - Hash map lookup
- `BM_BuiltinCTZ_IsolatedBits` - CTZ instruction

**Use case**: Finding the position of an isolated bit (e.g., `1ULL << 42`)

### 2. Random Bitboards
Tests extracting LSB from random 64-bit values.

- `BM_MagicBitscan_RandomBitboards` - Magic + extract LSB
- `BM_UnorderedMap_RandomBitboards` - HashMap + extract LSB
- `BM_BuiltinCTZ_RandomBitboards` - CTZ instruction

**Use case**: Finding first set bit in arbitrary bitboards

### 3. Sparse Bitboards
Tests bitboards with fewer bits set (sparse data).

- `BM_MagicBitscan_SparseBitboards`

**Use case**: Bitboards representing specific game states

### 4. Full Bitboard Scan
Iterates over ALL set bits in a bitboard.

- `BM_FullScan_Magic` - Magic number method
- `BM_FullScan_CTZ` - CTZ instruction

**Use case**: Processing all pieces on a board

### 5. Memory Access Patterns
Tests cache behavior and memory access.

- `BM_SequentialArrayAccess` - Sequential array reads
- `BM_RandomArrayAccess` - Random array lookups

**Use case**: Understanding cache effects

## Expected Results

Based on typical hardware (x86_64 with BMI1/BMI2):

### Fastest to Slowest (typical)

1. **`__builtin_ctzll` (CTZ)** - ~1-2 ns per operation
   - Hardware instruction (TZCNT on modern CPUs)
   - Most efficient for finding first set bit
   - No memory access required

2. **Magic Number + Array** - ~2-4 ns per operation
   - One multiplication + one array lookup
   - Small, cache-friendly lookup table (64 bytes)
   - Predictable performance

3. **Hash Map (unordered_map)** - ~10-50 ns per operation
   - Hash computation + memory access
   - Larger memory footprint
   - Cache misses more common

### Why Use Magic Numbers?

If CTZ is faster, why use magic numbers?

1. **Portability**: Works on all platforms (CTZ requires BMI1)
2. **Historical**: Used before widespread BMI support
3. **Educational**: Demonstrates multiplicative hashing
4. **Predictable**: No branch mispredictions
5. **Multiple uses**: Same technique works for other bit operations

### Real-World Performance

For Yolah (board game engine):
- **For bit scanning**: Use `__builtin_ctzll` on modern hardware
- **For magic bitboards**: Use magic numbers for move generation
- **For hash tables**: Use `unordered_map` when exact match needed

## Understanding the Output

Example output:
```
Benchmark                              Time             CPU   Iterations
------------------------------------------------------------------------
BM_MagicBitscan_IsolatedBits        2.45 ns         2.45 ns    285714285
BM_UnorderedMap_IsolatedBits         42.3 ns         42.3 ns     16528925
BM_BuiltinCTZ_IsolatedBits          1.89 ns         1.89 ns    370370370
```

- **Time/CPU**: Average time per operation in nanoseconds
- **Iterations**: Number of times the benchmark ran
- Lower time = better performance

## Key Techniques Demonstrated

### 1. Magic Number Formula
```cpp
int pos = bitscan[bitboard * MAGIC >> (64 - K)];
```
- Multiply by magic constant
- Take top K bits (shift right)
- Use as index into lookup table

### 2. LSB Extraction
```cpp
uint64_t lsb = bitboard & -bitboard;
```
- Isolates the least significant bit
- Example: `0b10110 & -0b10110 = 0b00010`

### 3. Clear LSB
```cpp
bitboard &= bitboard - 1;
```
- Clears the least significant bit
- Used for iterating over all set bits

### 4. DoNotOptimize
```cpp
benchmark::DoNotOptimize(result += pos);
```
- Prevents compiler from optimizing away the computation
- Essential for accurate benchmarks

## Customizing Benchmarks

### Adjust Sample Size
```cpp
auto samples = generate_random_bitboards(10000);  // Change 10000
```

### Adjust Magic Number
```cpp
constexpr ll MAGIC = 0x2643c51ab9dfa5b;  // Your magic number
constexpr int K = 6;                      // Hash table size (2^K)
```

### Add New Test
```cpp
static void BM_MyTest(benchmark::State& state) {
    // Setup
    auto samples = generate_test_data();
    int result = 0;

    for (auto _ : state) {
        // Benchmark code here
        benchmark::DoNotOptimize(result += compute());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MyTest);
```

## Running Specific Benchmarks

```bash
# Run only magic-related benchmarks
./magic_bench --benchmark_filter=Magic

# Run only CTZ benchmarks
./magic_bench --benchmark_filter=CTZ

# Show more statistics
./magic_bench --benchmark_repetitions=10

# Output to JSON
./magic_bench --benchmark_format=json --benchmark_out=results.json

# Output to CSV
./magic_bench --benchmark_format=csv --benchmark_out=results.csv
```

## Interpreting Results for Your Use Case

### For Yolah Game Engine

**Recommended approach**:
```cpp
// Use CTZ for bit scanning (fastest)
int pos = __builtin_ctzll(bitboard);

// Use magic numbers for move generation (different use case)
uint64_t attacks = attack_table[square * MAGIC >> shift];
```

**When magic numbers make sense**:
- Move generation with magic bitboards
- When you need hash-based perfect hashing
- When CTZ unavailable (very old hardware)

**When to use unordered_map**:
- Complex keys (not just powers of 2)
- Dynamic key sets
- When memory not critical

## Profiling Tips

1. **Run with high optimization**: Always use `-O3 -march=native`
2. **Watch for cache effects**: Small arrays (64 bytes) fit in L1 cache
3. **Consider instruction count**: CTZ is 1 instruction, magic is ~3
4. **Profile in context**: Benchmark isolated operations AND full game loops

## Common Pitfalls

❌ **Don't**:
- Benchmark in debug mode
- Use `volatile` (unless testing specific behavior)
- Ignore compiler optimizations
- Test with trivial workloads

✅ **Do**:
- Use `-O3 -march=native`
- Use `DoNotOptimize`
- Test with realistic data
- Run multiple iterations

## References

- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [Magic Bitboards in Chess](https://www.chessprogramming.org/Magic_Bitboards)
- [Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html)
- [Multiplicative Hashing](https://en.wikipedia.org/wiki/Hash_function#Multiplicative_hashing)

## See Also

- `find_magic_z3.py` - Find magic numbers using Z3
- `test.py` - Simple Z3 example
- `test.cpp` - Random search for magic numbers
- `MAGIC_NUMBERS.md` - Complete guide to magic numbers
