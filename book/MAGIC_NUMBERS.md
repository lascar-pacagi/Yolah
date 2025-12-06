# Magic Number Finder for Perfect Hashing

This directory contains tools for finding magic numbers that provide perfect hashing for a given set of keys.

## What are Magic Numbers?

A magic number `M` is a multiplier used in multiplicative hashing:
```
hash(key) = (key * M) >> shift
```

For perfect hashing, we need `M` such that all keys produce unique hash values.

## Files

### Python (Z3-based - Exhaustive Search)

1. **find_magic_z3.py** - Full-featured Z3-based magic finder
   - Uses SMT solver for exhaustive search
   - Guaranteed to find all solutions (if they exist)
   - Can find multiple solutions
   - Includes verification

2. **find_magic_simple.py** - Minimal Z3 example
   - Simple demonstration of Z3 usage
   - Good for learning

### C++ (Random Search)

1. **test.cpp** - Your original version
   - **Bug**: `seen` map should be inside the while loop (line 34)
   - Random search approach

2. **test_improved.cpp** - Fixed and enhanced version
   - Bug fixes
   - Verification function
   - Distribution analysis
   - Progress indicator
   - Timing information

## Usage

### Python (Recommended for exhaustive search)

```bash
# Install Z3 (if not already installed)
pip install z3-solver

# Find magic for 500 keys, 13-bit hash
python find_magic_z3.py --keys 500 --bits 13

# Use sparse keys (fewer bits set, often better)
python find_magic_z3.py --keys 500 --bits 13 --sparse

# Find multiple solutions
python find_magic_z3.py --keys 500 --bits 13 --solutions 10

# Verify a known magic number
python find_magic_z3.py --verify 0x1234567890ABCDEF --keys 500 --bits 13

# Simple example with 4 keys
python find_magic_simple.py
```

### C++

```bash
# Compile
g++ -O3 -std=c++20 test_improved.cpp -o find_magic

# Run
./find_magic
```

## Key Differences: Z3 vs Random Search

### Z3 (Exhaustive Search)
**Pros:**
- Exhaustive - finds ALL solutions
- Can prove no solution exists
- Faster for small problem sizes
- Can handle additional constraints

**Cons:**
- May timeout for large problems (many keys)
- Requires Z3 installation
- More complex to understand

### Random Search
**Pros:**
- Simple to implement
- No dependencies
- Works well with sparse magic generation
- Fast for problems with many solutions

**Cons:**
- Not exhaustive
- May never find solution
- No guarantee of optimality
- Slower for problems with few solutions

## Understanding the Code

### The Hash Function
```cpp
ll hash_val = (key * magic) >> (64 - K);
```
- Multiply key by magic number
- Take the top K bits (shift right by 64-K)
- This gives a value in range [0, 2^K)

### Perfect Hashing Constraint
For all pairs of different keys k1, k2:
```
(k1 * magic) >> shift != (k2 * magic) >> shift
```

### Sparse Magic Generation
```cpp
magic = d(mt) & d(mt) & d(mt) & d(mt);
```
- ANDing multiple random numbers produces sparse values
- Fewer bits set often works better for multiplicative hashing
- This is the technique used in Stockfish for magic bitboards

## Theoretical Background

For N keys and K-bit hash:
- Hash table size: 2^K entries
- For perfect hashing: 2^K >= N
- Probability of success increases with 2^K / N ratio

For your setup:
- N = 500 keys
- K = 13 bits = 8192 entries
- Ratio: 8192 / 500 = 16.38
- This is a reasonable ratio - solutions should exist

## Tips for Your Use Case

1. **Start small**: Test with fewer keys (e.g., 50) to verify the approach works

2. **Adjust K**: If no solution found, increase K (more hash table entries)

3. **Use Z3 for verification**: Even if you use random search, verify results with Z3

4. **Sparse vs Dense**:
   - Sparse magic (AND multiple randoms): Often better for bitboard-style keys
   - Dense magic (single random): Better for arbitrary keys

5. **Seed for reproducibility**: Save the random seed when you find a good magic

## Example Output

```
Found magic number after 123456 attempts!
Time: 2345ms
Magic: 0x0000100008004001
Popcount: 4 bits set

Verification:
✓ Magic is valid!

Hash distribution: 500 unique hashes for 500 keys
  No collisions - perfect hash!
```

## References

- [Multiplicative Hashing](https://en.wikipedia.org/wiki/Hash_function#Multiplicative_hashing)
- [Magic Bitboards in Chess Programming](https://www.chessprogramming.org/Magic_Bitboards)
- [Z3 Theorem Prover](https://github.com/Z3Prover/z3)
