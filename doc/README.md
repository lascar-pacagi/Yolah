# doc/

`alphazero_mcts.qmd` — the write-up of the network and the search: encoding,
architecture, training, export, the PUCT formulas, graph search, the root
refinements, the self-play learning loop (self-play driver, trainer, replay
window, symmetries, evaluation matches, cluster sizing, KataGo's auxiliary
heads), a code map, a parameter reference, the measurements, and an
appendix with the complete annotated source of every file involved.

## Rendering

Needs [Quarto](https://quarto.org) and a LaTeX install (TeX Live with
`scrartcl` and `fvextra`; the figures need `pdflatex`, `tikz`, `standalone` and
`pdftocairo`; the PDF listings need the `DejaVu Sans Mono` font for the
box-drawing and Greek characters in the sources).

```bash
figs/build.sh                  # TikZ sources -> 300 dpi PNGs (skips up-to-date ones)
./make_appendix.py             # sources -> appendix.qmd  (re-run after editing any of them)
quarto render alphazero_mcts.qmd
# -> alphazero_mcts.html (self-contained) and alphazero_mcts.pdf
```

## Files

| file | role |
|------|------|
| `alphazero_mcts.qmd` | the document |
| `learning.qmd` | the "Learning by self-play" chapter, included by the above |
| `solving.qmd` | the "Solving Yolah" chapter (retrograde analysis, exact endgames, solvability) |
| `appendix.qmd` | **generated** — the annotated source appendix, included by the above |
| `make_appendix.py` | generates it: the file list, the orientation notes and the symbol tables live here |
| `styles.css` | HTML-only layout: widens the code column for the appendix, holds prose to a readable measure |
| `figs/*.tex` | the TikZ diagram sources; `figs/preamble.tex` holds the shared styles |
| `figs/*.png` | committed, so the document renders without a LaTeX run |

The appendix is generated rather than pasted so the listings cannot drift from
the code. `make_appendix.py` prints a warning if a listed source has
uncommitted changes, which usually means the rendered output is ahead of what
was committed. Per-symbol prose belongs in `make_appendix.py`; line-by-line
commentary belongs in the sources themselves.
