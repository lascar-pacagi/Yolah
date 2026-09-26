#!/bin/bash
# Compile every TikZ figure to a 300 dpi PNG (used by both the HTML and the
# PDF rendering of ../alphazero_mcts.qmd).
set -e
cd "$(dirname "$0")"
for tex in fig-*.tex; do
    base="${tex%.tex}"
    [ -f "$base.png" ] && [ "$base.png" -nt "$tex" ] && continue
    echo "  $base"
    pdflatex -interaction=batchmode -halt-on-error "$tex" >/dev/null
    pdftocairo -png -r 300 -transp -singlefile "$base.pdf" "$base"
done
rm -f *.aux *.log
