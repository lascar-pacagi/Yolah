# doc/

`alphazero_mcts.qmd` — the write-up of the network and the search: encoding,
architecture, training, export, the PUCT formulas, graph search, the root
refinements, the self-play learning loop (self-play driver, trainer, replay
window, symmetries, evaluation matches, cluster sizing, KataGo's auxiliary
heads), a code map, a parameter reference, the measurements, and an
appendix with the complete annotated source of every file involved.

`minmax_nnue_dev_player.html` — the final minimax NNUE dev player
(`config/search_study/dev_final.cfg`), self-contained HTML with diagrams and
interactive widgets: the search (iterative deepening, aspiration windows,
negamax and its pruning, move ordering, the Yolah table, lazy SMP), the
evaluation cache and lazy accumulators, the NNUE network and its SIMD forward
pass, and its training (distillation, quantization-aware training, export).
Written by hand, not rendered by Quarto.

`alphazero_mcts_player.html` — the MCTS player with the value/policy ResNet,
in the same style: the network and its backends, one simulation (PUCT in
KataGo's form, the utility-stdev factor, first play urgency, backup), the
search graph (one node per position, Q from the node and N from the edge, the
node key, tree reuse), the batching of leaves and of threads, the network
cache, the root refinements (noise, forced playouts, policy target pruning,
LCB) and the measured strength.

`distillation_loop.html` — the distillation loop of `nnue/distill_loop.sh`, in
the same style: the games of the dev player (node limit, openings, the 40-byte
records, the tree nodes, the exact results), the training cache (the tree
nodes on the game's path, `zmask`, the window of rounds), the ResNet's
targets and training, the NNUE's teacher and distillation, the evaluation
matches and the SLURM jobs of a round.

Both reuse the CSS and the script framework of `minmax_nnue_dev_player.html`
(code blocks, contents, diagram helpers).

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
