#!/bin/bash
# Generate a clean profile report suitable for book inclusion
#
# Usage: ./generate_profile_table.sh chapter02 profile.data

BINARY=${1:-chapter02}
PERF_DATA=${2:-perf.data}

echo "Generating clean profile report for ${BINARY}..."

# 1. Generate clean text report (top functions only)
perf report --stdio \
  -i "$PERF_DATA" \
  --percent-limit=2.0 \
  --no-children \
  --sort=symbol \
  --show-nr-samples \
  2>&1 > profile_clean.txt

if [ $? -ne 0 ]; then
  echo "✗ Error: Could not read $PERF_DATA"
  exit 1
fi

echo "✓ Created profile_clean.txt"

# 2. Generate CSV for easy table creation
perf report --stdio \
  -i "$PERF_DATA" \
  --percent-limit=2.0 \
  --no-children \
  --sort=symbol \
  --show-nr-samples | \
  grep -E "^\s+[0-9]" | \
  awk '{printf "%s,%s,%s\n", $1, $2, $3}' | \
  head -n 15 \
  > profile.csv

echo "✓ Created profile.csv"

# 3. Generate LaTeX table
echo "Generating LaTeX table..."

cat > profile_table.tex <<'EOF'
\begin{table}[h]
\centering
\caption{Profil d'exécution -- Fonctions les plus coûteuses}
\label{tab:profile_hotspots}
\begin{tabular}{lrr}
\toprule
\textbf{Fonction} & \textbf{Temps CPU} & \textbf{Échantillons} \\
\midrule
EOF

# Parse perf output and format for LaTeX
perf report --stdio \
  -i "$PERF_DATA" \
  --percent-limit=2.0 \
  --no-children \
  --sort=symbol \
  --show-nr-samples | \
  grep -E "^\s+[0-9]" | \
  grep -v "0x[0-9a-f]\{8,\}" | \
  awk '!seen[$3]++ {printf "\\code{%s} & %s & %s \\\\\n", $3, $1, $2}' | \
  head -n 10 | \
  sed 's/_/\\_/g' >> profile_table.tex

cat >> profile_table.tex <<'EOF'
\bottomrule
\end{tabular}
\end{table}
EOF

echo "✓ Created profile_table.tex"

# 4. Generate summary statistics
echo ""
echo "=== Profile Summary ==="
perf report --stdio -i "$PERF_DATA" --percent-limit=2.0 --no-children | head -n 20

echo ""
echo "Files created:"
echo "  - profile_clean.txt  (human-readable)"
echo "  - profile.csv        (for spreadsheets)"
echo "  - profile_table.tex  (ready for LaTeX)"
