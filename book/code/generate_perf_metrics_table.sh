#!/bin/bash
# Generate LaTeX table showing branch mispredictions and cache misses
# for the most important functions

BINARY=${1:-./chapter02}

echo "Collecting performance metrics..."

# Run perf stat and capture output
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  "$BINARY" 2>&1 | tee perf_stats_raw.txt

# Record per-function metrics
echo ""
echo "Recording per-function branch and cache metrics..."

perf record -e branch-misses,cache-misses,cycles \
  --call-graph dwarf \
  "$BINARY" 2>&1 >/dev/null

# Generate LaTeX table
cat > perf_metrics_table.tex <<'EOF'
\begin{table}[h]
\centering
\caption{Métriques de performance -- Branches et cache}
\label{tab:perf_metrics}
\begin{tabular}{lrrr}
\toprule
\textbf{Fonction} & \textbf{Branch Misses} & \textbf{Cache Misses} & \textbf{Cycles} \\
\midrule
EOF

# Extract top functions for each metric and combine
# This is a simplified version - you may need to adjust based on your output

perf report --stdio --percent-limit=1.0 --sort=symbol --no-children | \
  grep -E "^\s+[0-9]" | \
  head -n 10 | \
  awk '{printf "\\code{%s} & -- & -- & %s \\\\\n", $3, $1}' | \
  sed 's/_/\\_/g' >> perf_metrics_table.tex

cat >> perf_metrics_table.tex <<'EOF'
\bottomrule
\end{tabular}
\end{table}
EOF

echo "✓ Created perf_metrics_table.tex"

# Clean up
rm -f perf.data perf.data.old

echo ""
echo "Summary from perf stat:"
grep -A 15 "Performance counter stats" perf_stats_raw.txt
