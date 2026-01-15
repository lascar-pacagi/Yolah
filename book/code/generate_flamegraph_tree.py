#!/usr/bin/env python3
"""
Generate a tree-like flame graph from collapsed stack trace profile.
Shows percentages and sample counts in a hierarchical tree format.

Can also simplify raw profiles by:
- Replacing [unknown] and [[stack]] with main
- Removing namespace prefixes (e.g., Yolah::)
- Filtering out kernel/interrupt/standard library noise
"""

import sys
import re
from collections import defaultdict
from typing import Dict, List, Tuple


class FlameNode:
    """Represents a node in the flame graph tree"""
    def __init__(self, name: str):
        self.name = name
        self.self_samples = 0
        self.total_samples = 0
        self.children: Dict[str, FlameNode] = {}
        self.collapsed = False  # Whether this node represents collapsed children

    def add_stack(self, stack: List[str], samples: int):
        """Add a stack trace to this node"""
        self.total_samples += samples

        if not stack:
            self.self_samples += samples
            return

        # Get or create child node
        next_func = stack[0]
        if next_func not in self.children:
            self.children[next_func] = FlameNode(next_func)

        # Recursively add remaining stack
        self.children[next_func].add_stack(stack[1:], samples)

    def print_tree(self, total_samples: int, prefix: str = "", is_last: bool = True, depth: int = 0, max_depth: int = 10, min_percent: float = 0.5, max_name_len: int = 60):
        """Print the tree with percentages"""
        if depth > max_depth:
            return

        # Calculate percentage
        percentage = (self.total_samples / total_samples) * 100 if total_samples > 0 else 0

        # Skip nodes with very low percentage
        if percentage < min_percent and depth > 0:
            return

        # Format the node line
        if depth == 0:
            # Root node
            display_name = self.name
        else:
            # Child node with tree structure
            connector = "└── " if is_last else "├── "
            display_name = f"{prefix}{connector}{self.name}"

        # Truncate if too long
        if max_name_len > 0 and len(display_name) > max_name_len:
            # Calculate how much space we have for the actual name
            # Keep room for "..." at the end
            available = max_name_len - 3
            if depth == 0:
                truncated = display_name[:available] + "..."
            else:
                # For child nodes, preserve the prefix and connector
                prefix_len = len(prefix) + 4  # "└── " or "├── "
                name_space = available - prefix_len
                if name_space > 10:  # Only truncate if we have reasonable space
                    truncated = prefix + connector + self.name[:name_space] + "..."
                else:
                    truncated = display_name[:available] + "..."
            display_name = truncated

        # Add percentage only
        print(f"{display_name:<{max_name_len}} {percentage:6.2f}%")

        # Prepare prefix for children
        if depth == 0:
            child_prefix = ""
        else:
            extension = "    " if is_last else "│   "
            child_prefix = prefix + extension

        # Sort children by total samples (descending)
        sorted_children = sorted(
            self.children.items(),
            key=lambda x: x[1].total_samples,
            reverse=True
        )

        # Filter children that will actually be printed
        # (based on percentage threshold and depth)
        visible_children = []
        for name, child in sorted_children:
            child_percentage = (child.total_samples / total_samples) * 100 if total_samples > 0 else 0
            if child_percentage >= min_percent or depth + 1 == 0:
                if depth + 1 <= max_depth:
                    visible_children.append((name, child))

        # Print children
        for i, (name, child) in enumerate(visible_children):
            is_last_child = (i == len(visible_children) - 1)
            child.print_tree(total_samples, child_prefix, is_last_child, depth + 1, max_depth, min_percent, max_name_len)

    def collapse_chains(self, patterns: List[str], context_path: str = ""):
        """Collapse long call chains matching patterns into single nodes

        Args:
            patterns: List of patterns, either "pattern", "path:pattern", or regex patterns
                     Patterns starting with 're:' are treated as regular expressions
            context_path: Current path in the tree (for context-aware collapsing)
        """
        # Build current path
        current_path = f"{context_path}/{self.name}" if context_path else self.name

        # Recursively collapse children first
        for child in list(self.children.values()):
            child.collapse_chains(patterns, current_path)

        # Check if this node matches any pattern and should be collapsed
        for pattern_spec in patterns:
            # Parse pattern: either "pattern", "path:pattern", "re:regex", or "path:re:regex"
            if ':' in pattern_spec:
                parts = pattern_spec.split(':', 1)
                if parts[1].startswith('re:'):
                    # Format: "path:re:regex" or just "re:regex"
                    path_filter = parts[0] if parts[0] != 're' else None
                    regex_pattern = parts[1][3:]  # Remove 're:' prefix
                    is_regex = True
                else:
                    # Format: "path:pattern"
                    path_filter = parts[0]
                    pattern = parts[1]
                    is_regex = False

                # Check path filter
                if path_filter and path_filter not in current_path:
                    continue
            elif pattern_spec.startswith('re:'):
                # Format: "re:regex"
                regex_pattern = pattern_spec[3:]
                is_regex = True
                path_filter = None
            else:
                # Simple string pattern
                pattern = pattern_spec
                is_regex = False
                path_filter = None

            # Check if this node matches the pattern
            matches = False
            if is_regex:
                matches = re.search(regex_pattern, self.name) is not None
            else:
                matches = pattern in self.name

            if matches:
                # If this node has a single child, collapse them regardless of child's name
                # This allows collapsing chains like uniform_int -> uniform_int -> mersenne
                # or uniform_int -> mersenne -> ...
                if len(self.children) == 1:
                    child_name, child = next(iter(self.children.items()))
                    # Check if child also matches ANY of the patterns
                    child_matches = False
                    for p in patterns:
                        # Parse child pattern
                        if ':' in p:
                            p_parts = p.split(':', 1)
                            if p_parts[1].startswith('re:'):
                                child_path_filter = p_parts[0] if p_parts[0] != 're' else None
                                if child_path_filter and child_path_filter not in current_path:
                                    continue
                                child_regex = p_parts[1][3:]
                                if re.search(child_regex, child_name):
                                    child_matches = True
                                    break
                            else:
                                child_path_filter = p_parts[0]
                                child_pat = p_parts[1]
                                if child_path_filter not in current_path:
                                    continue
                                if child_pat in child_name:
                                    child_matches = True
                                    break
                        elif p.startswith('re:'):
                            child_regex = p[3:]
                            if re.search(child_regex, child_name):
                                child_matches = True
                                break
                        else:
                            if p in child_name:
                                child_matches = True
                                break

                    if child_matches:
                        # Collapse: merge child into this node
                        self.total_samples = child.total_samples
                        self.self_samples = child.self_samples
                        self.children = child.children
                        self.collapsed = True
                        # Continue collapsing recursively
                        self.collapse_chains(patterns, context_path)
                        return
                break

    @staticmethod
    def _format_samples(samples: int) -> str:
        """Format large sample counts in a readable way"""
        if samples >= 1_000_000_000:
            return f"{samples / 1_000_000_000:.1f}B"
        elif samples >= 1_000_000:
            return f"{samples / 1_000_000:.1f}M"
        elif samples >= 1_000:
            return f"{samples / 1_000:.1f}K"
        else:
            return str(samples)


def should_keep_line(line: str) -> bool:
    """Filter out kernel, interrupt, and standard library functions"""
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


def simplify_stack(stack_part: str, remove_prefix: str = "") -> str:
    """Simplify a stack trace string"""
    # Replace [unknown] and [[stack]] with main
    stack_part = stack_part.replace('[unknown]', 'main')
    stack_part = stack_part.replace('[[stack]]', 'main')

    # Remove specified prefix (e.g., "Yolah::")
    if remove_prefix:
        stack_part = stack_part.replace(remove_prefix, '')

    # Split into function names and remove consecutive duplicates
    functions = stack_part.split(';')
    simplified = []
    prev = None
    for func in functions:
        if func != prev:
            simplified.append(func)
            prev = func

    return ';'.join(simplified)


def simplify_profile(input_file: str, output_file: str, remove_prefix: str = ""):
    """Simplify a raw profile by filtering and cleaning function names"""
    print(f"Simplifying profile: {input_file} -> {output_file}")

    lines_in = 0
    lines_out = 0

    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                lines_in += 1
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
                simplified_stack = simplify_stack(stack_part, remove_prefix)

                # Write to output
                f_out.write(f"{simplified_stack} {count}\n")
                lines_out += 1

    print(f"Simplified: {lines_in} lines -> {lines_out} lines ({lines_in - lines_out} filtered)")


def parse_collapsed_stacks(filename: str, merge_orphans: bool = False) -> Tuple[FlameNode, int]:
    """Parse collapsed stack trace format and build flame tree

    Args:
        filename: Path to collapsed stack trace file
        merge_orphans: If True, merge stacks missing 'main' into the main branch
    """
    root = FlameNode("root")
    total_samples = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split stack trace and sample count
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue

            stack_str, count_str = parts
            try:
                samples = int(count_str)
            except ValueError:
                continue

            # Split stack into functions
            stack = stack_str.split(';')

            # Merge orphan stacks (those without 'main') into main if requested
            if merge_orphans and len(stack) >= 2:
                # Check if second element should be 'main' but isn't
                if stack[1] != 'main' and 'main' not in stack:
                    # Insert 'main' after the first element (typically 'chapter02')
                    stack.insert(1, 'main')

            # Add to tree
            root.add_stack(stack, samples)
            total_samples += samples

    return root, total_samples




def main():
    import os
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate a tree-like flame graph from collapsed stack trace profile',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s                                    # Use default files
  %(prog)s profile.txt                        # Specify input file
  %(prog)s profile.txt -o flamegraph.txt      # Specify input and output

  # Customize tree display
  %(prog)s profile.txt -d 15                  # Set max depth to 15
  %(prog)s profile.txt -m 1.0                 # Show only nodes >= 1.0%%
  %(prog)s profile.txt -w 60                  # Set max name width to 60 chars
  %(prog)s profile.txt -w 0                   # No limit on name width

  # Simplify raw profiles first
  %(prog)s --simplify profile.txt             # Simplify and generate tree
  %(prog)s --simplify profile.txt --prefix Yolah::  # Remove namespace prefix
  %(prog)s --simplify profile.txt --clean-output profile_clean.txt

  # Collapse redundant call chains
  %(prog)s profile.txt --collapse uniform_int_distribution  # Collapse all RNG chains
  %(prog)s profile.txt --collapse "play_random:uniform_int"  # Collapse only in play_random_games
  %(prog)s profile.txt --collapse "re:uniform_int|mersenne"  # Collapse using regex (matches either pattern)
  %(prog)s profile.txt --collapse "play_random:re:std::"  # Collapse std:: calls in play_random_games only
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        default='profile_clean.txt',
        help='Input file with collapsed stack traces (default: profile_clean.txt)'
    )

    parser.add_argument(
        '-o', '--output',
        default='flamegraph_tree.txt',
        help='Output file for flame graph tree (default: flamegraph_tree.txt)'
    )

    parser.add_argument(
        '-d', '--max-depth',
        type=int,
        default=10,
        help='Maximum tree depth to display (default: 10)'
    )

    parser.add_argument(
        '-m', '--min-percent',
        type=float,
        default=0.5,
        help='Minimum percentage to display (default: 0.5)'
    )

    parser.add_argument(
        '-w', '--max-width',
        type=int,
        default=50,
        help='Maximum width for function names, truncate with "..." if longer (default: 50, 0 = no limit)'
    )

    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify the profile first (remove kernel/std lib, clean names)'
    )

    parser.add_argument(
        '--prefix',
        default='Yolah::',
        help='Namespace prefix to remove during simplification (default: Yolah::)'
    )

    parser.add_argument(
        '--clean-output',
        help='Output file for simplified profile (default: <input>_clean.txt)'
    )

    parser.add_argument(
        '--merge-orphans',
        action='store_true',
        help='Merge stack traces missing "main" into the main branch (cleaner output)'
    )

    parser.add_argument(
        '--collapse',
        action='append',
        metavar='PATTERN',
        help='Collapse call chains containing PATTERN. Formats: "pattern", "path:pattern", "re:regex", "path:re:regex". Examples: "uniform_int", "play_random:uniform_int", "re:std::(uniform|mersenne)". Can be specified multiple times.'
    )

    args = parser.parse_args()

    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(args.input):
        input_file = os.path.join(script_dir, args.input)
    else:
        input_file = args.input

    if not os.path.isabs(args.output):
        output_file = os.path.join(script_dir, args.output)
    else:
        output_file = args.output

    # Handle simplification
    if args.simplify:
        # Determine cleaned profile output file
        if args.clean_output:
            if not os.path.isabs(args.clean_output):
                clean_file = os.path.join(script_dir, args.clean_output)
            else:
                clean_file = args.clean_output
        else:
            # Default: add _clean before extension
            base, ext = os.path.splitext(input_file)
            clean_file = f"{base}_clean{ext}"

        # Simplify the profile
        simplify_profile(input_file, clean_file, args.prefix)
        print()

        # Use the cleaned file as input for tree generation
        profile_file = clean_file
    else:
        profile_file = input_file

    print(f"Reading profile from: {profile_file}")

    # Parse the profile
    root, total_samples = parse_collapsed_stacks(profile_file, merge_orphans=args.merge_orphans)

    # Collapse chains if requested
    if args.collapse:
        print(f"Collapsing chains matching: {', '.join(args.collapse)}")
        root.collapse_chains(args.collapse)

    # Generate output
    with open(output_file, 'w') as f:
        # Redirect stdout to file
        old_stdout = sys.stdout
        sys.stdout = f

        print("Flame Graph Tree - Yolah Performance Profile")
        print("=" * 80)
        print()

        # Print the tree (skip the artificial root)
        for name, child in sorted(root.children.items(),
                                   key=lambda x: x[1].total_samples,
                                   reverse=True):
            child.print_tree(total_samples, "", True, 0,
                           max_depth=args.max_depth,
                           min_percent=args.min_percent,
                           max_name_len=args.max_width)
            print()

        print("=" * 80)
        print("Note: Percentages show portion of total CPU time")
        print("Tree shows call hierarchy from bottom (root) to top (leaves)")

        # Restore stdout
        sys.stdout = old_stdout

    print(f"Flame graph tree written to: {output_file}")

    # Also print to console
    print("\n" + "=" * 80)
    print("Preview:")
    print("=" * 80)
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines[:40]:  # Show first 40 lines
            print(line, end='')
    print("\n... (see full output in flamegraph_tree.txt)")


if __name__ == '__main__':
    main()
