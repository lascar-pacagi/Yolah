# Z3 C++ API Tutorial

## Installation

### Ubuntu/Debian
```bash
sudo apt-get install libz3-dev
```

### From Source
```bash
git clone https://github.com/Z3Prover/z3.git
cd z3
python scripts/mk_make.py
cd build
make
sudo make install
```

## Compilation

To compile a program using Z3 C++ API:

```bash
g++ -std=c++17 find_magic_z3.cpp -o find_magic_z3 -lz3
```

Or with pkg-config:
```bash
g++ -std=c++17 find_magic_z3.cpp -o find_magic_z3 $(pkg-config --cflags --libs z3)
```

## Core Concepts

### 1. Context (`z3::context`)

The context is the main Z3 object that manages all other Z3 objects. You need one context per Z3 session.

```cpp
z3::context ctx;
```

### 2. Expressions (`z3::expr`)

Expressions represent formulas, variables, and values in Z3.

```cpp
z3::context ctx;

// Create integer variables
z3::expr x = ctx.int_const("x");
z3::expr y = ctx.int_const("y");

// Create boolean variables
z3::expr b = ctx.bool_const("b");

// Create bitvector variables (for bitwise operations)
z3::expr bv = ctx.bv_const("bv", 64);  // 64-bit bitvector
```

### 3. Solver (`z3::solver`)

The solver is used to check satisfiability of constraints.

```cpp
z3::context ctx;
z3::solver solver(ctx);

// Add constraints
z3::expr x = ctx.int_const("x");
solver.add(x > 0);
solver.add(x < 10);

// Check satisfiability
if (solver.check() == z3::sat) {
    std::cout << "Satisfiable!" << std::endl;
    z3::model m = solver.get_model();
    std::cout << "x = " << m.eval(x) << std::endl;
}
```

## Common Operations

### Integer Operations

```cpp
z3::context ctx;
z3::expr x = ctx.int_const("x");
z3::expr y = ctx.int_const("y");

// Arithmetic
z3::expr sum = x + y;
z3::expr diff = x - y;
z3::expr prod = x * y;
z3::expr div = x / y;
z3::expr mod = x % y;

// Comparisons
z3::expr eq = (x == y);
z3::expr neq = (x != y);
z3::expr lt = (x < y);
z3::expr le = (x <= y);
z3::expr gt = (x > y);
z3::expr ge = (x >= y);
```

### Bitvector Operations

```cpp
z3::context ctx;
z3::expr bv1 = ctx.bv_const("bv1", 64);
z3::expr bv2 = ctx.bv_const("bv2", 64);

// Create bitvector constants
z3::expr val = ctx.bv_val(42, 64);  // 42 as 64-bit bitvector
z3::expr val_hex = ctx.bv_val(0x123456789ABCDEF0ULL, 64);

// Arithmetic
z3::expr add = bv1 + bv2;
z3::expr sub = bv1 - bv2;
z3::expr mul = bv1 * bv2;

// Bitwise operations
z3::expr and_op = bv1 & bv2;
z3::expr or_op = bv1 | bv2;
z3::expr xor_op = bv1 ^ bv2;
z3::expr not_op = ~bv1;

// Shifts
z3::expr shl = z3::shl(bv1, bv2);      // Shift left
z3::expr lshr = z3::lshr(bv1, bv2);    // Logical shift right
z3::expr ashr = z3::ashr(bv1, bv2);    // Arithmetic shift right

// Extract bits
z3::expr low_byte = bv1.extract(7, 0);   // Extract bits 0-7
z3::expr high_byte = bv1.extract(63, 56); // Extract bits 56-63
```

### Boolean Operations

```cpp
z3::context ctx;
z3::expr p = ctx.bool_const("p");
z3::expr q = ctx.bool_const("q");

// Logical operations
z3::expr and_op = p && q;
z3::expr or_op = p || q;
z3::expr not_op = !p;
z3::expr implies = z3::implies(p, q);
z3::expr iff = p == q;  // Equivalence
```

## Working with Models

When a solver finds a satisfiable solution, you can extract values:

```cpp
z3::context ctx;
z3::solver solver(ctx);

z3::expr x = ctx.int_const("x");
z3::expr y = ctx.int_const("y");

solver.add(x + y == 10);
solver.add(x > y);

if (solver.check() == z3::sat) {
    z3::model model = solver.get_model();

    // Evaluate expressions in the model
    z3::expr x_val = model.eval(x);
    z3::expr y_val = model.eval(y);

    // Get integer values
    int x_int = x_val.get_numeral_int();
    int y_int = y_val.get_numeral_int();

    // Get 64-bit unsigned values (for bitvectors)
    uint64_t x_uint64 = x_val.get_numeral_uint64();

    std::cout << "x = " << x_int << ", y = " << y_int << std::endl;
}
```

## Example: Finding Magic Numbers

Magic numbers are used in chess programming for efficient move generation. Here's how to find them with Z3:

```cpp
#include <z3++.h>
#include <iostream>

int main() {
    z3::context ctx;
    z3::solver solver(ctx);

    // Create magic number variable (64-bit)
    z3::expr magic = ctx.bv_const("magic", 64);

    // We want to map different occupancy patterns to unique indices
    const int K = 6;  // We want 2^6 = 64 unique indices

    // Example occupancy patterns (simplified)
    std::vector<uint64_t> occupancies = {
        0x0000000000000001ULL,
        0x0000000000000002ULL,
        0x0000000000000004ULL,
        // ... more patterns
    };

    // For each pair of occupancies, their indices must be different
    for (size_t i = 0; i < occupancies.size(); i++) {
        for (size_t j = i + 1; j < occupancies.size(); j++) {
            // index = (magic * occupancy) >> (64 - K)
            z3::expr occ_i = ctx.bv_val(occupancies[i], 64);
            z3::expr occ_j = ctx.bv_val(occupancies[j], 64);

            z3::expr shift = ctx.bv_val(64 - K, 64);
            z3::expr index_i = z3::lshr(magic * occ_i, shift);
            z3::expr index_j = z3::lshr(magic * occ_j, shift);

            solver.add(index_i != index_j);
        }
    }

    if (solver.check() == z3::sat) {
        z3::model model = solver.get_model();
        uint64_t magic_val = model.eval(magic).get_numeral_uint64();
        std::cout << "Magic: 0x" << std::hex << magic_val << std::endl;
    }

    return 0;
}
```

## Tips and Best Practices

1. **Use bitvectors for bitwise operations**: If you need shifts, AND, OR, XOR operations, use bitvectors, not integers.

2. **Reuse context**: Create one context and reuse it. Don't create multiple contexts unless necessary.

3. **Check solver result**: Always check if `solver.check()` returns `sat`, `unsat`, or `unknown`.

4. **Simplify expressions**: Use `expr.simplify()` to simplify complex expressions.

5. **Timeout**: Set timeouts for long-running queries:
   ```cpp
   z3::params p(ctx);
   p.set(":timeout", 10000);  // 10 seconds
   solver.set(p);
   ```

6. **Debugging**: Print constraints to see what Z3 is solving:
   ```cpp
   std::cout << solver << std::endl;
   ```

## Common Gotchas

1. **Mixing contexts**: Don't mix expressions from different contexts.

2. **Type mismatches**: Make sure bitvector widths match when combining them.

3. **Unsigned operations**: Use `get_numeral_uint64()` for unsigned 64-bit values, not `get_numeral_int()`.

4. **Shift amounts**: Shift amounts must be bitvectors of the same width as the value being shifted.

## Resources

- Official Z3 C++ API: https://z3prover.github.io/api/html/namespacez3.html
- Z3 Guide: https://microsoft.github.io/z3guide/
- Z3 GitHub: https://github.com/Z3Prover/z3
