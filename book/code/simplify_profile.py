#!/usr/bin/env python3
"""
Simplify profile.txt for flame graph generation:
- Replace [unknown] and [[stack]] with main
- Remove Yolah:: prefix
- Filter out kernel and standard library functions
"""

import sys
import re

def should_keep_line(line):
    """Filter out kernel, interrupt, and standard library functions"""
    # List of patterns to exclude
    exclude_patterns = [
        r';asm_',
        r';__irq',
        r';sysvec_',
        r';__x64_',
        r';entry_',
        r';do_syscall_',
        r';syscall_',
        r';ret_from_',
        r';__kernel',
        r';native_',
        r';error_entry',
        r';sync_regs',
        r';std::_',
        r';__gnu_',
        r';_start;',
        r';__libc_',
        r';malloc',
        r';free;',
        r';\[kernel',
        r';\[vdso\]',
    ]

    for pattern in exclude_patterns:
        if re.search(pattern, line):
            return False
    return True

def simplify_stack(stack_part):
    """Simplify a single stack trace"""
    # Replace [unknown] and [[stack]] with main
    stack_part = stack_part.replace('[unknown]', 'main')
    stack_part = stack_part.replace('[[stack]]', 'main')

    # Remove Yolah:: prefix
    stack_part = stack_part.replace('Yolah::', '')

    # Split into function names
    functions = stack_part.split(';')

    # Remove consecutive duplicates
    simplified = []
    prev = None
    for func in functions:
        if func != prev:
            simplified.append(func)
            prev = func

    return ';'.join(simplified)

def main():
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'profile.txt')
    output_file = os.path.join(script_dir, 'profile_clean.txt')

    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.rstrip('\n')

                if not line:
                    continue

                # Skip lines that should be filtered
                if not should_keep_line(line):
                    continue

                # Split into stack trace and count
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    continue

                stack_part, count = parts

                # Simplify the stack trace
                simplified_stack = simplify_stack(stack_part)

                # Write to output
                f_out.write(f"{simplified_stack} {count}\n")

    print(f"Simplified profile written to {output_file}")

if __name__ == '__main__':
    main()
