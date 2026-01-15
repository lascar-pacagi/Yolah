#!/bin/bash
# Comprehensive performance analysis for the moves() method
# Focuses on branch mispredictions and cache misses

BINARY=${1:-./chapter02}
OUTPUT_DIR="perf_analysis"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Performance Analysis for moves() method"
echo "=========================================="
echo ""

# 1. Overall statistics with branch and cache metrics
echo "1. Collecting overall performance statistics..."
perf stat -e cycles,instructions,branches,branch-misses \
  -e L1-dcache-loads,L1-dcache-load-misses \
  -e LLC-loads,LLC-load-misses \
  -e cache-references,cache-misses \
  -o "$OUTPUT_DIR/overall_stats.txt" \
  "$BINARY" 2>&1

echo "   ✓ Saved to $OUTPUT_DIR/overall_stats.txt"
echo ""

# 2. Record branch mispredictions
echo "2. Recording branch mispredictions..."
perf record -e branch-misses:pp \
  --call-graph dwarf \
  -o "$OUTPUT_DIR/branch_misses.data" \
  "$BINARY" 2>&1 | tail -n 3

echo "   ✓ Recorded to $OUTPUT_DIR/branch_misses.data"
echo ""

# 3. Record cache misses
echo "3. Recording cache misses..."
perf record -e cache-misses:pp,L1-dcache-load-misses:pp \
  --call-graph dwarf \
  -o "$OUTPUT_DIR/cache_misses.data" \
  "$BINARY" 2>&1 | tail -n 3

echo "   ✓ Recorded to $OUTPUT_DIR/cache_misses.data"
echo ""

# 4. Generate reports
echo "4. Generating reports..."

# Branch misses report
perf report -i "$OUTPUT_DIR/branch_misses.data" \
  --stdio --percent-limit=2.0 --sort=symbol --no-children \
  > "$OUTPUT_DIR/branch_misses_report.txt"

echo "   ✓ Branch misses: $OUTPUT_DIR/branch_misses_report.txt"

# Cache misses report
perf report -i "$OUTPUT_DIR/cache_misses.data" \
  --stdio --percent-limit=2.0 --sort=symbol --no-children \
  > "$OUTPUT_DIR/cache_misses_report.txt"

echo "   ✓ Cache misses: $OUTPUT_DIR/cache_misses_report.txt"

# 5. Annotate moves() function if it exists
echo ""
echo "5. Annotating moves() function..."

perf annotate -i "$OUTPUT_DIR/branch_misses.data" \
  --stdio moves \
  > "$OUTPUT_DIR/moves_branch_annotation.txt" 2>/dev/null

if [ -s "$OUTPUT_DIR/moves_branch_annotation.txt" ]; then
  echo "   ✓ Branch annotation: $OUTPUT_DIR/moves_branch_annotation.txt"
else
  echo "   ⚠ No branch data for moves() function"
fi

perf annotate -i "$OUTPUT_DIR/cache_misses.data" \
  --stdio moves \
  > "$OUTPUT_DIR/moves_cache_annotation.txt" 2>/dev/null

if [ -s "$OUTPUT_DIR/moves_cache_annotation.txt" ]; then
  echo "   ✓ Cache annotation: $OUTPUT_DIR/moves_cache_annotation.txt"
else
  echo "   ⚠ No cache data for moves() function"
fi

echo ""
echo "=========================================="
echo "Analysis complete! Summary:"
echo "=========================================="
echo ""

# Display summary
echo "=== Overall Statistics ==="
cat "$OUTPUT_DIR/overall_stats.txt" | grep -A 20 "Performance counter stats"

echo ""
echo "=== Top Functions with Branch Misses ==="
head -n 15 "$OUTPUT_DIR/branch_misses_report.txt" | grep -E "^\s+[0-9]"

echo ""
echo "=== Top Functions with Cache Misses ==="
head -n 15 "$OUTPUT_DIR/cache_misses_report.txt" | grep -E "^\s+[0-9]"

echo ""
echo "All detailed results are in: $OUTPUT_DIR/"
echo ""
echo "To view interactive reports:"
echo "  perf report -i $OUTPUT_DIR/branch_misses.data"
echo "  perf report -i $OUTPUT_DIR/cache_misses.data"
echo ""
echo "To annotate specific functions:"
echo "  perf annotate -i $OUTPUT_DIR/branch_misses.data"
