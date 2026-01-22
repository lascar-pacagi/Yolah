#!/usr/bin/env python3
"""
Generate a LaTeX table from perf annotate output showing branch and cache hotspots.

Usage:
    # First, record and annotate:
    sudo perf record -e '{branches:pp,branch-misses:pp}' -g -F9999 -o perf_branch.data ./chapter02
    sudo perf annotate --stdio --show-total-period --group -i perf_branch.data > branch_raw.txt

    sudo perf record -e '{L1-dcache-loads:pp,L1-dcache-load-misses:pp}' -g -F9999 -o perf_cache.data ./chapter02
    sudo perf annotate --stdio --show-total-period --group -i perf_cache.data > cache_raw.txt

    # Then generate table:
    python3 generate_perf_table.py --branch branch_raw.txt --cache cache_raw.txt --function Yolah::moves -o table.tex
"""

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class LineStats:
    source_line: int
    source_code: str
    address: str
    branches: int = 0
    branch_misses: int = 0
    l1_loads: int = 0
    l1_misses: int = 0


def parse_perf_annotate(filename: str, function: Optional[str] = None) -> Tuple[Dict[str, LineStats], int, int]:
    """
    Parse perf annotate output with --show-total-period --group

    Returns:
        - Dict mapping address to LineStats
        - Total of first event (branches or L1 loads)
        - Total of second event (branch misses or L1 misses)
    """
    stats: Dict[str, LineStats] = {}
    total_event1 = 0
    total_event2 = 0

    in_target_function = function is None
    current_source_line = 0
    current_source_code = ""

    # Pattern for data lines: "  123456  789012 :   addr:   instruction"
    data_pattern = re.compile(r'^\s*(\d+)\s+(\d+)\s*:\s*([0-9a-f]+):\s*(.*)$')

    # Pattern for source lines: " : 123  source code"
    source_pattern = re.compile(r'^\s*:\s*(\d+)\s+(.*)$')

    # Pattern for function header
    func_pattern = re.compile(r'^Percent.*')
    func_name_pattern = re.compile(r"^.*<([^>]+)>.*$")

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()

            # Check for function boundary
            if function:
                if function in line and ('Percent' in line or '<' in line):
                    in_target_function = True
                elif in_target_function and line.startswith('Percent'):
                    # New function started
                    in_target_function = False

            if not in_target_function:
                continue

            # Try to match source line
            source_match = source_pattern.match(line)
            if source_match:
                current_source_line = int(source_match.group(1))
                current_source_code = source_match.group(2).strip()
                continue

            # Try to match data line
            data_match = data_pattern.match(line)
            if data_match:
                event1 = int(data_match.group(1))
                event2 = int(data_match.group(2))
                addr = data_match.group(3)

                total_event1 += event1
                total_event2 += event2

                if addr not in stats:
                    stats[addr] = LineStats(
                        source_line=current_source_line,
                        source_code=current_source_code,
                        address=addr
                    )

                # Store raw values (will be set as branches or cache depending on file)
                stats[addr].branches = event1
                stats[addr].branch_misses = event2
                stats[addr].source_line = current_source_line
                if current_source_code:
                    stats[addr].source_code = current_source_code

    return stats, total_event1, total_event2


def aggregate_by_source_line(stats: Dict[str, LineStats]) -> Dict[int, LineStats]:
    """Aggregate stats by source line number."""
    by_line: Dict[int, LineStats] = {}

    for addr, s in stats.items():
        line_num = s.source_line
        if line_num == 0:
            continue

        if line_num not in by_line:
            by_line[line_num] = LineStats(
                source_line=line_num,
                source_code=s.source_code,
                address=addr
            )

        by_line[line_num].branches += s.branches
        by_line[line_num].branch_misses += s.branch_misses
        by_line[line_num].l1_loads += s.l1_loads
        by_line[line_num].l1_misses += s.l1_misses

        # Keep longest source code
        if len(s.source_code) > len(by_line[line_num].source_code):
            by_line[line_num].source_code = s.source_code

    return by_line


def format_number(n: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)


def format_percent(value: float) -> str:
    """Format percentage."""
    if value == 0:
        return "—"
    elif value < 0.01:
        return "<0.01\\%"
    elif value < 1:
        return f"{value:.2f}\\%"
    else:
        return f"{value:.1f}\\%"


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def generate_latex_table(
    branch_stats: Dict[int, LineStats],
    cache_stats: Dict[int, LineStats],
    total_branches: int,
    total_branch_misses: int,
    total_l1_loads: int,
    total_l1_misses: int,
    min_threshold: float = 0.1
) -> str:
    """Generate LaTeX table from stats."""

    # Merge branch and cache stats
    all_lines = set(branch_stats.keys()) | set(cache_stats.keys())

    rows = []
    for line_num in sorted(all_lines):
        b_stat = branch_stats.get(line_num)
        c_stat = cache_stats.get(line_num)

        branches = b_stat.branches if b_stat else 0
        branch_misses = b_stat.branch_misses if b_stat else 0
        l1_loads = c_stat.branches if c_stat else 0  # stored in branches field
        l1_misses = c_stat.branch_misses if c_stat else 0  # stored in branch_misses field

        source_code = ""
        if b_stat and b_stat.source_code:
            source_code = b_stat.source_code
        elif c_stat and c_stat.source_code:
            source_code = c_stat.source_code

        # Calculate percentages
        pct_branches = (branches / total_branches * 100) if total_branches > 0 else 0
        pct_branch_misses = (branch_misses / total_branch_misses * 100) if total_branch_misses > 0 else 0
        branch_miss_rate = (branch_misses / branches * 100) if branches > 0 else 0

        pct_l1_loads = (l1_loads / total_l1_loads * 100) if total_l1_loads > 0 else 0
        pct_l1_misses = (l1_misses / total_l1_misses * 100) if total_l1_misses > 0 else 0
        l1_miss_rate = (l1_misses / l1_loads * 100) if l1_loads > 0 else 0

        # Filter out insignificant lines
        if (pct_branches < min_threshold and pct_branch_misses < min_threshold and
            pct_l1_loads < min_threshold and pct_l1_misses < min_threshold):
            continue

        rows.append({
            'line': line_num,
            'source': source_code,
            'branches': branches,
            'pct_branches': pct_branches,
            'branch_misses': branch_misses,
            'pct_branch_misses': pct_branch_misses,
            'branch_miss_rate': branch_miss_rate,
            'l1_loads': l1_loads,
            'pct_l1_loads': pct_l1_loads,
            'l1_misses': l1_misses,
            'pct_l1_misses': pct_l1_misses,
            'l1_miss_rate': l1_miss_rate,
        })

    # Sort by total impact (branch misses % + L1 misses %)
    rows.sort(key=lambda r: r['pct_branch_misses'] + r['pct_l1_misses'], reverse=True)

    # Generate LaTeX
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Branch and Cache Hotspots}")
    latex.append(r"\label{tab:perf-hotspots}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{r|l|rrr|rrr}")
    latex.append(r"\toprule")
    latex.append(r"Line & Source & \multicolumn{3}{c|}{Branches} & \multicolumn{3}{c}{L1 Cache} \\")
    latex.append(r" & & Count & \% Total & Miss\% & Count & \% Total & Miss\% \\")
    latex.append(r"\midrule")

    for row in rows:
        source = escape_latex(row['source'][:40])  # Truncate long lines
        if len(row['source']) > 40:
            source += "..."

        latex.append(
            f"{row['line']} & \\texttt{{{source}}} & "
            f"{format_number(row['branches'])} & {format_percent(row['pct_branches'])} & {format_percent(row['branch_miss_rate'])} & "
            f"{format_number(row['l1_loads'])} & {format_percent(row['pct_l1_loads'])} & {format_percent(row['l1_miss_rate'])} \\\\"
        )

    latex.append(r"\midrule")
    latex.append(
        f"\\textbf{{Total}} & & "
        f"\\textbf{{{format_number(total_branches)}}} & 100\\% & "
        f"\\textbf{{{format_percent(total_branch_misses / total_branches * 100 if total_branches > 0 else 0)}}} & "
        f"\\textbf{{{format_number(total_l1_loads)}}} & 100\\% & "
        f"\\textbf{{{format_percent(total_l1_misses / total_l1_loads * 100 if total_l1_loads > 0 else 0)}}} \\\\"
    )
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table from perf annotate output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Record branch data
    sudo perf record -e '{branches:pp,branch-misses:pp}' -g -F9999 -o perf_branch.data ./chapter02
    sudo perf annotate --stdio --show-total-period --group -i perf_branch.data > branch_raw.txt

    # Record cache data
    sudo perf record -e '{L1-dcache-loads:pp,L1-dcache-load-misses:pp}' -g -F9999 -o perf_cache.data ./chapter02
    sudo perf annotate --stdio --show-total-period --group -i perf_cache.data > cache_raw.txt

    # Generate table
    python3 generate_perf_table.py --branch branch_raw.txt --cache cache_raw.txt -o table.tex

    # For a specific function only
    python3 generate_perf_table.py --branch branch_raw.txt --cache cache_raw.txt --function "Yolah::moves" -o table.tex
        """
    )

    parser.add_argument('--branch', required=True, help='Branch perf annotate output file')
    parser.add_argument('--cache', required=True, help='L1 cache perf annotate output file')
    parser.add_argument('--function', '-f', help='Filter to specific function (optional)')
    parser.add_argument('--output', '-o', default='perf_table.tex', help='Output LaTeX file')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                        help='Minimum %% threshold to include a line (default: 0.1)')

    args = parser.parse_args()

    print(f"Parsing branch data from {args.branch}...")
    branch_raw, total_branches, total_branch_misses = parse_perf_annotate(args.branch, args.function)
    branch_stats = aggregate_by_source_line(branch_raw)
    print(f"  Found {len(branch_stats)} source lines")
    print(f"  Total branches: {format_number(total_branches)}")
    print(f"  Total branch misses: {format_number(total_branch_misses)}")

    print(f"Parsing cache data from {args.cache}...")
    cache_raw, total_l1_loads, total_l1_misses = parse_perf_annotate(args.cache, args.function)
    cache_stats = aggregate_by_source_line(cache_raw)
    print(f"  Found {len(cache_stats)} source lines")
    print(f"  Total L1 loads: {format_number(total_l1_loads)}")
    print(f"  Total L1 misses: {format_number(total_l1_misses)}")

    print(f"Generating LaTeX table...")
    latex = generate_latex_table(
        branch_stats, cache_stats,
        total_branches, total_branch_misses,
        total_l1_loads, total_l1_misses,
        args.threshold
    )

    with open(args.output, 'w') as f:
        f.write(latex)

    print(f"Table written to {args.output}")


if __name__ == '__main__':
    main()
