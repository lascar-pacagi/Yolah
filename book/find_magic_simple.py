#!/usr/bin/env python3
"""
Simple magic number finder using Z3 - minimal example.
"""

from z3 import *

# Example: Find magic for small set of keys
keys = [0x1234567890ABCDEF, 0xFEDCBA0987654321, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555]
bits = 8  # 8-bit hash (256 entries)
shift = 64 - bits

print(f"Finding magic for {len(keys)} keys with {bits}-bit hash")
print(f"Keys: {[hex(k) for k in keys]}")
print()

# Create solver
s = Solver()
magic = BitVec('magic', 64)

# Add constraints: all hashes must be different
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        k1 = BitVecVal(keys[i], 64)
        k2 = BitVecVal(keys[j], 64)

        hash1 = LShR(k1 * magic, shift)
        hash2 = LShR(k2 * magic, shift)

        s.add(hash1 != hash2)

print("Solving...")
if s.check() == sat:
    m = s.model()
    magic_val = m[magic].as_long()
    print(f"Found magic: 0x{magic_val:016X}")
    print(f"Binary: {bin(magic_val)}")

    # Verify
    print("\nVerification:")
    for k in keys:
        h = (k * magic_val) >> shift
        print(f"  {k:016X} -> hash {h:03X}")
else:
    print("No solution found")
