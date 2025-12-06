#!/usr/bin/env python3
"""
Find magic numbers for perfect hashing using Z3 SMT solver.

This tool uses Z3 to exhaustively search for magic numbers that provide
perfect hashing for a given set of keys. A magic number 'M' is valid if
for all keys k1, k2 in the set:
    (k1 * M) >> shift != (k2 * M) >> shift  (when k1 != k2)

This ensures no collisions in the hash table.
"""

from z3 import *
import random
import sys

def find_magic_z3(keys, bits=13, timeout_ms=60000, max_solutions=5):
    """
    Find magic numbers using Z3 SMT solver.

    Args:
        keys: List of 64-bit keys to hash
        bits: Number of bits in the hash table (K in your code)
        timeout_ms: Timeout in milliseconds for Z3 solver
        max_solutions: Maximum number of solutions to find

    Returns:
        List of magic numbers that provide perfect hashing
    """
    print(f"Searching for magic numbers with {bits} bits ({2**bits} entries)")
    print(f"Keys to hash: {len(keys)}")
    print(f"Timeout: {timeout_ms}ms")
    print()

    # Create Z3 solver
    s = Solver()
    s.set("timeout", timeout_ms)

    # Magic number is a 64-bit bitvector
    magic = BitVec('magic', 64)

    # Shift amount
    shift = 64 - bits

    # Add constraints: for each pair of different keys, their hashes must differ
    print("Adding constraints...")
    constraint_count = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1 = BitVecVal(keys[i], 64)
            k2 = BitVecVal(keys[j], 64)

            # Hash values must be different
            hash1 = LShR(k1 * magic, shift)
            hash2 = LShR(k2 * magic, shift)

            s.add(hash1 != hash2)
            constraint_count += 1

    print(f"Added {constraint_count} constraints")
    print()

    # Find solutions
    solutions = []
    for solution_num in range(1, max_solutions + 1):
        print(f"Searching for solution #{solution_num}...")

        result = s.check()

        if result == sat:
            model = s.model()
            magic_val = model[magic].as_long()
            solutions.append(magic_val)

            print(f"✓ Found magic number: 0x{magic_val:016X}")
            print(f"  Binary: {bin(magic_val)}")
            print(f"  Popcount: {bin(magic_val).count('1')}")
            print()

            # Add constraint to exclude this solution and find another
            s.add(magic != magic_val)
        elif result == unsat:
            print("No more solutions exist")
            break
        else:
            print("Unknown result (timeout or error)")
            break

    return solutions


def verify_magic(magic, keys, bits=13):
    """
    Verify that a magic number provides perfect hashing for the given keys.

    Args:
        magic: The magic number to verify
        keys: List of keys
        bits: Number of bits in hash table

    Returns:
        True if magic number is valid, False otherwise
    """
    shift = 64 - bits
    seen = {}

    for k in keys:
        hash_val = (k * magic) >> shift
        if hash_val in seen and seen[hash_val] != k:
            print(f"Collision: key {k:016X} and {seen[hash_val]:016X} both hash to {hash_val}")
            return False
        seen[hash_val] = k

    return True


def generate_random_keys(n=500, seed=None):
    """Generate random 64-bit keys."""
    if seed is not None:
        random.seed(seed)
    return [random.getrandbits(64) for _ in range(n)]


def generate_sparse_keys(n=500, seed=None):
    """
    Generate sparse random keys (fewer bits set).
    This mimics the approach in your C++ code: d(mt) & d(mt) & d(mt)
    """
    if seed is not None:
        random.seed(seed)
    keys = []
    for _ in range(n):
        k = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)
        keys.append(k)
    return keys


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Find magic numbers for perfect hashing using Z3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find magic for 500 random keys, 13-bit hash
  python find_magic_z3.py --keys 500 --bits 13

  # Use sparse keys (fewer bits set)
  python find_magic_z3.py --keys 500 --bits 13 --sparse

  # Find up to 3 solutions with longer timeout
  python find_magic_z3.py --keys 500 --bits 13 --solutions 3 --timeout 120000

  # Verify a known magic number
  python find_magic_z3.py --verify 0x1234567890ABCDEF --keys 500 --bits 13
        """
    )

    parser.add_argument('--keys', type=int, default=500,
                       help='Number of random keys to generate (default: 500)')
    parser.add_argument('--bits', type=int, default=13,
                       help='Number of bits in hash table (default: 13)')
    parser.add_argument('--timeout', type=int, default=60000,
                       help='Timeout in milliseconds (default: 60000)')
    parser.add_argument('--solutions', type=int, default=5,
                       help='Maximum number of solutions to find (default: 5)')
    parser.add_argument('--sparse', action='store_true',
                       help='Generate sparse keys (fewer bits set)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verify a magic number (hex string, e.g., 0x1234567890ABCDEF)')

    args = parser.parse_args()

    # Generate keys
    print("=" * 70)
    print("MAGIC NUMBER FINDER USING Z3")
    print("=" * 70)
    print()

    if args.sparse:
        print("Generating sparse random keys...")
        keys = generate_sparse_keys(args.keys, args.seed)
    else:
        print("Generating random keys...")
        keys = generate_random_keys(args.keys, args.seed)

    print(f"Generated {len(keys)} keys")
    print(f"First key: 0x{keys[0]:016X}")
    print()

    # Verify mode
    if args.verify:
        magic = int(args.verify, 16)
        print(f"Verifying magic number: 0x{magic:016X}")
        print()

        if verify_magic(magic, keys, args.bits):
            print("✓ Magic number is VALID - no collisions!")
        else:
            print("✗ Magic number is INVALID - collisions detected")
        return

    # Search mode
    solutions = find_magic_z3(keys, args.bits, args.timeout, args.solutions)

    if solutions:
        print("=" * 70)
        print(f"Found {len(solutions)} magic number(s)")
        print("=" * 70)
        for i, magic in enumerate(solutions, 1):
            print(f"{i}. 0x{magic:016X}")
            print(f"   Binary: {bin(magic)}")
            print(f"   Popcount: {bin(magic).count('1')}")

            # Verify
            if verify_magic(magic, keys, args.bits):
                print("   ✓ Verified")
            else:
                print("   ✗ Verification FAILED")
            print()
    else:
        print("No magic numbers found within timeout")


if __name__ == '__main__':
    main()
