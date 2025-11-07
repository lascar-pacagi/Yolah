# Yolah Book - Complete Setup Guide

## 📚 What You Now Have

I've created a complete professional LaTeX book template for documenting your Yolah project with the following files:

### Core Files

1. **`book_template.tex`** - Full LaTeX book template
   - Professional chapter and section formatting
   - Custom colored boxes for algorithms, results, and important concepts
   - Example chapters showing game engine, algorithms, and results
   - Code listing and figure examples
   - Built-in bibliography support

2. **`references.bib`** - Comprehensive bibliography database
   - 40+ citations covering game AI, deep learning, and optimization
   - Formatted for academic standards
   - Ready to extend with your own sources

3. **`Makefile`** - Automated build system
   - `make pdf` - Build complete PDF with bibliography
   - `make quick` - Fast build without bibliography
   - `make clean` - Remove build artifacts
   - `make view` - Open PDF in system viewer
   - `make wordcount` - Count words in document
   - `make watch` - Continuous build (requires `entr`)

4. **`generate_figures.py`** - Publication-quality figure generation
   - 6 different types of professional plots
   - 300 DPI resolution for printing
   - Color-blind friendly palette
   - Ready to customize with your data

### Documentation

5. **`BOOK_RECOMMENDATIONS.md`** - Comprehensive writing guide
   - LaTeX setup and compilation instructions
   - Figure design techniques (TikZ, PGFPlots, external tools)
   - Code formatting best practices
   - Document structure recommendations
   - Bibliography and styling tips

6. **`ADVANCED_LATEX_TIPS.md`** - Advanced techniques
   - Complex code formatting with highlighting
   - Publication-ready plots and subfigures
   - Mathematical typesetting
   - Professional table design
   - Custom environments and macros
   - Build optimization

7. **`LATEX_CHEATSHEET.md`** - Quick reference
   - Quick lookup for common LaTeX patterns
   - Code snippets for common tasks
   - Troubleshooting guide
   - File organization templates

---

## 🚀 Quick Start (5 minutes)

### 1. Install LaTeX and Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-full texlive-latex-extra biber python3-pygments
```

**macOS:**
```bash
brew install basictex
tlmgr install collection-fontsrecommended collection-latex
pip3 install pygments
```

**Windows (WSL):**
```bash
# Use Ubuntu instructions above in WSL terminal
```

### 2. Rename and Setup

```bash
cd /home/elucterio/Yolah
cp book_template.tex book.tex
mkdir -p figures
```

### 3. Generate Sample Figures

```bash
python3 generate_figures.py
```

Expected output:
```
✓ Saved: figures/performance_comparison.png
✓ Saved: figures/training_curves.png
✓ Saved: figures/search_tree_growth.png
✓ Saved: figures/nn_architecture.png
✓ Saved: figures/tournament_results.png
✓ Saved: figures/move_time_distribution.png
```

### 4. Build the Book

```bash
# Option 1: Using Make
make pdf

# Option 2: Manual
pdflatex --shell-escape -interaction=nonstopmode book.tex
biber book
pdflatex --shell-escape -interaction=nonstopmode book.tex
pdflatex --shell-escape -interaction=nonstopmode book.tex
```

### 5. View the Result

```bash
make view
# or: open book.pdf (macOS), xdg-open book.pdf (Linux)
```

---

## 📖 Writing Your Book

### Recommended Structure for Yolah

```
PART I: FOUNDATIONS
├── Chapter 1: Introduction
│   ├── Motivation for Yolah
│   ├── Game Rules (with figures)
│   └── Book Overview
├── Chapter 2: Game Engine
│   ├── Board Representation (bitboards)
│   ├── Move Generation (with code)
│   └── Performance Metrics

PART II: CLASSICAL AI
├── Chapter 3: Search Algorithms
│   ├── Minimax with pseudocode
│   ├── Alpha-Beta Pruning (with figures)
│   ├── Iterative Deepening
│   └── Transposition Tables
├── Chapter 4: Monte Carlo Tree Search
│   ├── MCTS Algorithm
│   ├── UCB1 Selection
│   └── Implementation Details

PART III: MODERN AI
├── Chapter 5: Neural Networks
│   ├── Architecture Design (with TikZ diagrams)
│   ├── Training Pipeline
│   └── Evaluation Functions
├── Chapter 6: AlphaZero Approach
│   ├── Self-Play Training
│   ├── Policy and Value Networks
│   ├── Results (with plots)
│   └── Analysis
├── Chapter 7: NNUE (Efficient Networks)
│   ├── Architecture for Speed
│   ├── Incremental Updates
│   └── Benchmarks

PART IV: IMPLEMENTATION
├── Chapter 8: System Architecture
│   ├── Software Design (flowcharts)
│   ├── Performance Optimization
│   └── Parallelization
├── Chapter 9: Training Infrastructure
│   ├── Data Generation
│   ├── Training Pipeline (code + diagrams)
│   └── Multi-GPU Training

PART V: EVALUATION
├── Chapter 10: Experimental Results
│   ├── Tournament System
│   ├── Statistical Analysis (tables + plots)
│   └── Ablation Studies
├── Chapter 11: Conclusions & Future Work

APPENDICES
├── Appendix A: Code Examples (full implementations)
├── Appendix B: Mathematical Proofs
└── Appendix C: Hyperparameter Tables
```

### Chapter Template

```latex
\chapter{Chapter Title}

\section{Introduction}
Brief introduction to the chapter.

\section{Main Topic}

\subsection{Subtopic}

\begin{importantbox}
Key insight or concept here.
\end{importantbox}

\subsubsection{Implementation}

Here's the algorithm:

\begin{listing}[H]
\begin{minted}[linenos,bgcolor=codecolor]{python}
# Your code here
\end{minted}
\caption{Description}
\label{code:example}
\end{listing}

Figure~\ref{fig:example} shows the result.

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/example.png}
\caption{Description}
\label{fig:example}
\end{figure}

\begin{resultbox}
Summary of findings
\end{resultbox}
```

---

## 🎨 Creating Beautiful Figures

### Method 1: TikZ (In LaTeX)
Best for: Diagrams, flowcharts, game states

```latex
\begin{figure}[H]
\centering
\begin{tikzpicture}
    % Your TikZ code
\end{tikzpicture}
\caption{Game board state}
\label{fig:board}
\end{figure}
```

### Method 2: PGFPlots (In LaTeX)
Best for: Performance graphs, training curves

```latex
\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[xlabel=Epoch, ylabel=Win Rate]
\addplot table {data/training.dat};
\end{axis}
\end{tikzpicture}
\caption{Training progress}
\label{fig:training}
\end{figure}
```

### Method 3: Python/Matplotlib (External)
Best for: Complex plots, real data

```bash
# Run generate_figures.py to create plots
python3 generate_figures.py

# Include in LaTeX
\includegraphics[width=0.8\textwidth]{figures/training_curves.png}
```

### Method 4: External Tools
- **Inkscape** - Vector graphics editor
- **Draw.io** - Flowchart and diagram maker
- **Graphviz** - Graph visualization
- **OmniGraffle** - Professional diagramming

---

## 💻 Professional Code Display

### Simple Code Snippet
```latex
\begin{minted}[linenos]{python}
def minimax(board, depth):
    if depth == 0:
        return evaluate(board)
    # ... rest of code
\end{minted}
```

### Code with Highlighting
```latex
\begin{minted}[
    linenos,
    highlightlines={3,5},
    bgcolor=codecolor
]{python}
def function():
    step1()
    critical_line()  # Highlighted
    step2()
    important()      # Highlighted
\end{minted}
```

### Side-by-Side Code Comparison
```latex
\begin{minipage}[t]{0.45\textwidth}
\begin{minted}[fontsize=\tiny]{python}
# Python version
\end{minted}
\end{minipage}
\hfill
\begin{minipage}[t]{0.45\textwidth}
\begin{minted}[fontsize=\tiny]{cpp}
// C++ version
\end{minted}
\end{minipage}
```

---

## 📊 Creating Tables

### Simple Performance Table
```latex
\begin{table}[H]
\centering
\caption{Algorithm Comparison}
\label{tab:algorithms}
\begin{tabular}{l|c|c|c}
\toprule
\textbf{Algorithm} & \textbf{Speed} & \textbf{Quality} & \textbf{Memory} \\
\midrule
Minimax & ✓ & ✓✓✓ & ✗✗ \\
MCTS & ✓✓ & ✓✓ & ✓ \\
AlphaZero & ✓✓ & ✓✓✓ & ✗ \\
NNUE & ✓✓✓ & ✓✓ & ✓ \\
\bottomrule
\end{tabularx}
\end{table}
```

### Results Table with Statistics
```latex
\begin{table}[H]
\centering
\caption{Tournament Results (1000 games)}
\label{tab:results}
\begin{tabularx}{\textwidth}{l|rrr}
\toprule
\textbf{Matchup} & \textbf{Wins} & \textbf{Losses} & \textbf{Win Rate} \\
\midrule
AlphaZero vs MCTS & 850 & 150 & 85.0\% \\
NNUE vs MCTS & 750 & 250 & 75.0\% \\
\bottomrule
\end{tabularx}
\end{table}
```

---

## 🔍 Useful LaTeX Patterns

### Highlighting a Definition
```latex
\begin{importantbox}
\textbf{Definition:} Monte Carlo Tree Search (MCTS) is a search algorithm
that builds a game tree incrementally using random simulations.
\end{importantbox}
```

### Showing an Algorithm
```latex
\begin{algorithmbox}
\begin{algorithmic}[1]
\Function{Minimax}{position, depth}
    \If{depth = 0}
        \Return $evaluate(position)$
    \EndIf
    \State $best \gets -\infty$
    \For{each move in legal\_moves}
        \State $value \gets -minimax(position, depth-1)$
        \State $best \gets \max(best, value)$
    \EndFor
    \Return best
\EndFunction
\end{algorithmic}
\end{algorithmbox}
```

### Highlighting Results
```latex
\begin{resultbox}
\textbf{Key Finding:} NNUE achieved 92\% win rate while being 100x faster
than AlphaZero, making it practical for real-time play.
\end{resultbox}
```

---

## 📝 Citation Examples

In `references.bib`:
```bibtex
@article{silver2016mastering,
    author = {Silver, David and others},
    title = {Mastering the game of Go},
    journal = {Nature},
    year = {2016}
}
```

In your document:
```latex
According to \cite{silver2016mastering}, deep learning combined
with tree search enables superhuman performance.

On page 42 \citep[p. 42]{silver2016mastering}, they discuss...

See \cref{tab:results} for comparison.
```

---

## 🛠️ Troubleshooting

### Issue: Code highlighting not working
**Solution:** Add `--shell-escape` flag
```bash
pdflatex --shell-escape book.tex
```

### Issue: Bibliography shows `[?]`
**Solution:** Run biber and recompile
```bash
biber book
pdflatex --shell-escape book.tex
```

### Issue: Figures not appearing
**Solution:** Check file paths and use:
```latex
\graphicspath{{figures/}}
```

### Issue: Compilation too slow
**Solution:** Use draft mode or cached compilation
```bash
pdflatex -draftmode book.tex  # Fast, no images
```

---

## 📈 Next Steps

1. **Customize the title page** - Update author, title, date
2. **Start writing chapters** - Begin with Chapter 1 or 2
3. **Add your code examples** - Copy from nnue_multigpu4.py and other files
4. **Generate your data** - Run generate_figures.py with real data
5. **Extend bibliography** - Add your sources to references.bib
6. **Build and review** - Compile frequently and proofread

---

## 📚 Recommended Reading Order

For writing effectively:

1. **LATEX_CHEATSHEET.md** - Quick reference while writing
2. **BOOK_RECOMMENDATIONS.md** - Deep dive into techniques
3. **ADVANCED_LATEX_TIPS.md** - Advanced styling and optimization
4. **book_template.tex** - Reference implementation

---

## 🎯 Pro Tips

✅ **Build frequently** - Catch errors early
✅ **Use labels** - Reference figures/tables by label, not number
✅ **Keep code snippets small** - Show only essential code
✅ **Use meaningful names** - `\label{fig:alphazero_training}` not `\label{fig:1}`
✅ **Version control** - Use git to track changes
✅ **Backup .bib file** - Your bibliography is valuable
✅ **Use Overleaf for sharing** - Upload and collaborate

---

## 📞 Getting Help

1. **LaTeX Errors:** Check `book.log` file
2. **Syntax:** Use `LATEX_CHEATSHEET.md`
3. **Advanced:** Check `ADVANCED_LATEX_TIPS.md`
4. **Online:**
   - TeX Stack Exchange: https://tex.stackexchange.com
   - Overleaf Documentation: https://overleaf.com/learn
   - TikZ Manual: https://pgf-tikz.github.io/

---

## 📋 Checklist Before Publishing

- [ ] All chapters written and reviewed
- [ ] All figures at 300+ DPI (for print)
- [ ] All code examples tested and working
- [ ] All references in bibliography (no [?])
- [ ] Table of contents updates
- [ ] Index created (optional)
- [ ] Proofread for typos
- [ ] Cross-references working
- [ ] Consistent formatting throughout
- [ ] PDF generates without warnings

---

**You're all set!** Happy writing! 📚✨

For questions or improvements, refer to the comprehensive documentation files included.
