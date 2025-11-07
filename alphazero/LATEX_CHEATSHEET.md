# LaTeX Quick Reference for Yolah Book

## Getting Started

### Basic File Structure
```latex
\documentclass[12pt,a4paper,twoside]{book}

% Preamble
\usepackage{...}
\title{...}
\author{...}

\begin{document}
\maketitle
\tableofcontents

\chapter{Introduction}
...

\appendix
\chapter{Code Examples}
...

\bibliography{references}
\end{document}
```

### Compilation Commands
```bash
# Standard
pdflatex --shell-escape book.tex

# With bibliography
pdflatex --shell-escape book.tex
biber book
pdflatex --shell-escape book.tex
pdflatex --shell-escape book.tex

# Using make
make pdf
```

---

## Document Structure

```latex
\part{Foundations}          % Part (Roman numerals)
\chapter{Title}             % Chapter (Large heading)
\section{Subsection}        % Section
\subsection{Item}           % Subsection
\subsubsection{Detail}      % Subsubsection
\paragraph{Point}           % Paragraph (no numbering)
\subparagraph{Detail}       % Subparagraph
```

---

## Code Examples

### Inline Code
```latex
The \inlinecode{minimax()} function is efficient.

Or use: \texttt{small code snippet}
```

### Code Block with Minted
```latex
\begin{minted}[linenos,bgcolor=codecolor]{python}
def minimax(board, depth):
    if depth == 0:
        return evaluate(board)
    return max(...) if maximizing else min(...)
\end{minted}
```

### Code from File
```latex
\inputminted[linenos,firstline=10,lastline=30]{cpp}{src/main.cpp}
```

### Highlighted Lines
```latex
\begin{minted}[highlightlines={3,5},linenos]{python}
# Line numbers start at 1
some_code()
highlighted_line()
more_code()
important_line()
\end{minted}
```

---

## Figures

### Basic Figure
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/algorithm.png}
\caption{Description of the figure}
\label{fig:algorithm}
\end{figure}

Reference: \ref{fig:algorithm} or \cref{fig:algorithm}
```

### Subfigures
```latex
\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{figures/fig1.png}
    \caption{First figure}
    \label{fig:1}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{figures/fig2.png}
    \caption{Second figure}
    \label{fig:2}
\end{subfigure}
\caption{Combined caption}
\label{fig:both}
\end{figure}
```

### Simple TikZ Diagram
```latex
\begin{tikzpicture}
\draw (0,0) -- (2,0) -- (2,2) -- (0,2) -- cycle;
\node at (1,1) {Box};
\end{tikzpicture}
```

### Plot with pgfplots
```latex
\begin{tikzpicture}
\begin{axis}[xlabel=X, ylabel=Y, title=Performance]
\addplot[color=blue] coordinates {
    (1,2) (2,4) (3,6) (4,8)
};
\end{axis}
\end{tikzpicture}
```

---

## Tables

### Simple Table
```latex
\begin{table}[H]
\centering
\caption{Results Comparison}
\label{tab:results}
\begin{tabular}{lcc}
\toprule
\textbf{Algorithm} & \textbf{Speed} & \textbf{Accuracy} \\
\midrule
Minimax & 500 & 95\% \\
MCTS & 300 & 85\% \\
NNUE & 2000 & 92\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Table with Colors
```latex
\begin{table}[H]
\centering
\begin{tabular}{l|cc}
\rowcolor{headercolor}
\textbf{Method} & \textbf{Result} & \textbf{Time} \\
\hline
Baseline & 50\% & 1000ms \\
\rowcolor{green!20}
Optimized & 85\% & 100ms \\
\end{tabular}
\end{table}
```

### Wide Table
```latex
\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{l|X|X|X}
\toprule
\textbf{Col 1} & \textbf{Col 2} & \textbf{Col 3} & \textbf{Col 4} \\
\midrule
Item & Description & Description & Description \\
\bottomrule
\end{tabularx}
\end{table}
```

---

## Math & Equations

### Inline Math
```latex
The equation $E = mc^2$ is famous.

Or use \(E = mc^2\) alternative syntax.
```

### Display Math
```latex
\[
\text{eval}(s) = \begin{cases}
    1 & \text{checkmate for me} \\
    -1 & \text{checkmate for opponent} \\
    f(s) & \text{otherwise}
\end{cases}
\]
```

### Numbered Equations
```latex
\begin{equation}
\label{eq:minimax}
\text{minimax}(n) = \begin{cases}
    \text{eval}(n) & \text{if } \text{leaf}(n) \\
    \max_{i} \text{minimax}(c_i) & \text{if max node} \\
    \min_{i} \text{minimax}(c_i) & \text{if min node}
\end{cases}
\end{equation}
```

### Aligned Equations
```latex
\begin{align}
P(a|s) &= \frac{\exp(f(s,a))}{\sum_b \exp(f(s,b))} \\
&= \text{softmax} \label{eq:policy}
\end{align}
```

---

## Text Formatting

```latex
\textbf{Bold text}
\textit{Italic text}
\texttt{Monospace text}
\textsc{Small caps}

\underline{Underlined}
\colorbox{yellow}{Highlighted}

\emph{Emphasis}  % Changes style based on context
\url{https://example.com}
```

---

## Lists

### Bullet Points
```latex
\begin{itemize}
    \item First item
    \item Second item
    \begin{itemize}
        \item Nested item
        \item Another nested
    \end{itemize}
    \item Third item
\end{itemize}
```

### Numbered List
```latex
\begin{enumerate}
    \item First
    \item Second
    \item Third
\end{enumerate}
```

### Description List
```latex
\begin{description}
    \item[Algorithm] A step-by-step procedure
    \item[Heuristic] A practical rule-of-thumb
    \item[Optimization] Improving efficiency
\end{description}
```

---

## Boxes and Highlights

### Simple Box
```latex
\boxed{Important Formula Here}
```

### Colored Box (requires tcolorbox)
```latex
\begin{tcolorbox}[colback=blue!10,colframe=blue!50!black]
    This is an important concept highlighted in a box.
\end{tcolorbox}
```

### Important/Algorithm/Result Boxes
```latex
\begin{importantbox}
    This is critical information!
\end{importantbox}

\begin{algorithmbox}
    \textbf{Key Algorithm:} Alpha-Beta Pruning
\end{algorithmbox}

\begin{resultbox}
    \textbf{Finding:} NNUE is 10x faster than AlphaZero
\end{resultbox}
```

---

## Bibliography and References

### In document
```latex
% Citation
\cite{silver2016mastering}
\citep[p. 42]{silver2016mastering}  % with page

% Reference
\label{sec:introduction}
See \ref{sec:introduction}
See \cref{sec:introduction}  % Clever ref
```

### In references.bib
```bibtex
@article{silver2016mastering,
    author = {Silver, David and Huang, Aja and ...},
    title = {Mastering the game of Go},
    journal = {Nature},
    year = {2016},
    volume = {529}
}

@book{knuth1973art,
    author = {Knuth, Donald E.},
    title = {The Art of Computer Programming},
    publisher = {Addison-Wesley},
    year = {1973}
}
```

---

## Useful Commands

### Spacing
```latex
\quad      % Medium space
\qquad     % Large space
\\         % Line break
\newline   % New line
\newpage   % New page
\pagebreak % Force page break
\vspace{1cm}  % Vertical space
\hspace{1cm}  % Horizontal space
```

### Page Style
```latex
\pagestyle{empty}       % No header/footer
\pagestyle{plain}       % Page number only
\pagestyle{headings}    % Section titles in header
\pagestyle{fancy}       % Custom (with fancyhdr)
```

### Special Characters
```latex
\%   \$   \&   \#   \_   \{   \}   \textasciitilde
```

### Comments
```latex
% Single line comment

\begin{comment}
Multi-line
comment
\end{comment}
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Code highlighting not working | Add `--shell-escape` to pdflatex |
| Bibliography empty | Run `biber book` then `pdflatex` twice |
| Figures not showing | Check file path, use `\graphicspath{{figures/}}` |
| Overfull hbox warning | Check line length, adjust margins or use `\raggedright` |
| References show `??` | Recompile twice |
| Section not in TOC | Use `\chapter{}` not `\textbf{}` |

---

## File Organization

```
book/
├── book.tex              # Main document
├── references.bib        # Bibliography
├── Makefile              # Build automation
├── figures/              # Generated images
│   ├── performance.png
│   ├── training_curves.png
│   └── architecture.png
├── chapters/             # Optional: separate files
│   ├── 01_introduction.tex
│   ├── 02_foundations.tex
│   └── 03_algorithms.tex
└── data/                 # Data for plots
    ├── alphazero.dat
    └── nnue.dat
```

### Split Document (optional)
```latex
% In main book.tex
\input{chapters/01_introduction.tex}
\input{chapters/02_foundations.tex}

% In chapters/01_introduction.tex
\chapter{Introduction}
...content...
```

---

## Productivity Tips

### Fast Compilation
```bash
# Draft mode (faster, no figures)
pdflatex -draftmode book.tex

# Quick rebuild
make quick

# Watch for changes
make watch
```

### Find Issues
```bash
# Check for undefined references
grep -i "undefined\|warning" book.log

# Word count
texcount book.tex

# Check bibliography
biber --output_all_macronames book
```

### Git-Friendly
```bash
# .gitignore for LaTeX
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
_minted-*
```

---

## Online Tools

- **Overleaf**: Cloud-based LaTeX editor (recommended for collaboration)
- **Detexify**: Draw symbol to find LaTeX command
- **LaTeX Tables Generator**: Create tables visually
- **Equation Editor**: Online equation builder
- **TikZ Editor**: Visual diagram creator

---

## Essential References

- **Official**: https://www.latex-project.org/
- **TikZ Manual**: https://pgf-tikz.github.io/pgf/pgfmanual.pdf
- **PGFPlots**: https://pgfplots.sourceforge.net/
- **Minted**: https://ctan.org/pkg/minted

---

**Happy writing!** 📚
