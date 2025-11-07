# Advanced LaTeX Tips for Technical Book Writing

## 1. Advanced Code Formatting

### Highlighting Specific Lines in Code

```latex
\begin{minted}[
    highlightlines={2,5,7-9},
    bgcolor=codecolor,
    linenos,
    fontsize=\small
]{python}
def minimax(board, depth, is_maximizing):
    if depth == 0:                          # Line 2 - Terminal condition
        return evaluate(board)

    if is_maximizing:                       # Line 5 - Maximizing player
        max_eval = float('-inf')
        for move in get_legal_moves():      # Lines 7-9 - Main loop
            eval_score = minimax(
                apply_move(board, move),
                depth - 1, False)
            max_eval = max(max_eval, eval_score)
        return max_eval
\end{minted}
```

### Side-by-Side Language Comparison

```latex
\begin{figure}[H]
\centering
\begin{minipage}[t]{0.48\textwidth}
\textbf{Python (slow)}
\begin{minted}[fontsize=\tiny, bgcolor=codecolor]{python}
# O(n²) - nested loop
for i in range(len(moves)):
    for j in range(len(moves)):
        if can_connect(moves[i], moves[j]):
            process(moves[i], moves[j])
\end{minted}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\textbf{C++ (fast)}
\begin{minted}[fontsize=\tiny, bgcolor=codecolor]{cpp}
// O(n log n) - optimized
sort(moves.begin(), moves.end());
for (auto& m : moves)
    if (can_fast_connect(m))
        process(m);
\end{minted}
\end{minipage}
\caption{Performance comparison: Language choice matters}
\end{figure}
```

### Including Entire Files

```latex
% Include specific range of lines
\inputminted[
    linenos,
    bgcolor=codecolor,
    firstline=50,
    lastline=100,
    fontsize=\small,
    frame=lines
]{python}{../nnue/nnue_multigpu4.py}
```

### Creating a Code Listing Environment

```latex
\newenvironment{codeblock}[1][python]{
    \minted{#1}
}{
    \endminted
}

% Usage:
\begin{codeblock}[cpp]
    // Your code here
\end{codeblock}
```

---

## 2. Advanced Figure Techniques

### Creating Publication-Ready Plots with pgfplots

```latex
\begin{tikzpicture}
\begin{axis}[
    xlabel=Epoch,
    ylabel=Win Rate (\%),
    title=AlphaZero Training Progress,
    legend pos=south right,
    grid=major,
    width=12cm,
    height=7cm,
    % Add error bars
    error bars/y dir=both,
    error bars/y explicit,
    % Smooth line
    smooth,
    % Add axis labels at specific positions
    xtick={0,10,20,30,40,50},
    ytick={0,25,50,75,100},
    % Set axis limits
    xmin=0, xmax=50,
    ymin=0, ymax=100,
    % Font settings
    tick label style={font=\small},
    label style={font=\small}
]

% Plot data with error bars
\addplot[
    color=blue,
    mark=o,
    line width=2pt,
    error bars/y explicit
] table[x=epoch, y=winrate, y error=stderr] {data/alphazero.dat};

\addplot[
    color=red,
    mark=square,
    line width=2pt
] table[x=epoch, y=winrate] {data/nnue.dat};

\legend{AlphaZero, NNUE}
\end{axis}
\end{tikzpicture}
```

### Subfigures with Shared Caption

```latex
\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/training_alphazero.png}
    \caption{AlphaZero training curve}
    \label{fig:train_alpha}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/training_nnue.png}
    \caption{NNUE training curve}
    \label{fig:train_nnue}
\end{subfigure}
\caption{Training progress comparison for different AI approaches}
\label{fig:training_both}
\end{figure}

% Reference all or specific subfigures:
In Figure~\ref{fig:training_both}, we show both approaches.
The AlphaZero results are in Figure~\ref{fig:train_alpha}.
```

### Wrapping Text Around Figures

```latex
\begin{wrapfigure}{r}{0.4\textwidth}
\centering
\includegraphics[width=0.38\textwidth]{figures/board_state.png}
\caption{Example board configuration}
\label{fig:board_example}
\end{wrapfigure}

Text wraps around the figure on the right. This is useful for
illustration-heavy chapters where space is at a premium. The text
will automatically reflow around the figure boundaries.
```

### TikZ Flowchart Example

```latex
\begin{tikzpicture}[node distance=2cm]
\tikzstyle{block} = [rectangle, draw, fill=blue!20, text width=3cm,
                     text centered, rounded corners, minimum height=1cm]
\tikzstyle{decision} = [diamond, draw, fill=yellow!20, text width=2cm,
                        text centered, aspect=2]
\tikzstyle{arrow} = [thick, ->, >=stealth]

% Nodes
\node (start) [block] {Start};
\node (input) [block, below of=start] {Read Position};
\node (check) [decision, below of=input] {Terminal?};
\node (eval) [block, below of=check, xshift=-3cm] {Evaluate};
\node (recurse) [block, below of=check, xshift=3cm] {Recurse};
\node (end) [block, below of=eval, xshift=3cm] {Return};

% Edges
\draw [arrow] (start) -- (input);
\draw [arrow] (input) -- (check);
\draw [arrow] (check) -- node[anchor=east] {yes} (eval);
\draw [arrow] (check) -- node[anchor=west] {no} (recurse);
\draw [arrow] (eval) -- (end);
\draw [arrow] (recurse) -- (end);
\end{tikzpicture}
```

---

## 3. Advanced Mathematical Typesetting

### Aligned Equations with Explanations

```latex
\begin{align}
P(a|s) &= \frac{\exp(v_{\theta}(s,a))}{\sum_b \exp(v_{\theta}(s,b))}
    \quad & \text{(softmax policy)} \\
&= \text{Neural network policy output} \quad & \text{(AlphaZero)} \\
V(s) &= \mathbb{E}[G_t | S_t = s] \quad & \text{(value function)} \\
&\approx f_{\phi}(s) \quad & \text{(value network approximation)}
\end{align}
```

### Complex Algorithm with Multiple Cases

```latex
\begin{equation}
\text{eval}(s) = \begin{cases}
    1 & \text{if } s \text{ is checkmate for me} \\
    -1 & \text{if } s \text{ is checkmate for opponent} \\
    f_{\theta}(s) & \text{otherwise (neural network)} \\
\end{cases}
\end{equation}
```

### Colored Box for Important Equations

```latex
\begin{center}
\fcolorbox{primaryblue}{lightblue}{
    \parbox{0.8\textwidth}{
        \centering
        \vspace{10pt}
        $\text{UCB}(n) = \frac{Q(n)}{N(n)} + C \sqrt{\frac{\ln N(\text{parent})}{N(n)}}$

        \vspace{5pt}
        \small \textit{UCB formula for MCTS node selection}
        \vspace{10pt}
    }
}
\end{center}
```

---

## 4. Professional Table Design

### Complex Table with Merged Cells

```latex
\begin{table}[H]
\centering
\caption{Comprehensive Algorithm Comparison}
\label{tab:comprehensive}
\begin{tabularx}{\textwidth}{l|cc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Algorithm}} &
\multicolumn{2}{c|}{\textbf{Performance}} &
\multicolumn{2}{c|}{\textbf{Resources}} &
\multicolumn{2}{c}{\textbf{Learning}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{Speed} & \textbf{Strength} &
  \textbf{Memory} & \textbf{Time} &
  \textbf{Data} & \textbf{Method} \\
\midrule
Minimax & \checkmark\checkmark & \checkmark\checkmark &
    $\times\times$ & \checkmark &
    None & Heuristic \\
MCTS & \checkmark & \checkmark\checkmark &
    \checkmark & \checkmark &
    Simulations & Self-play \\
AlphaZero & \checkmark & \checkmark\checkmark\checkmark &
    $\times$ & $\times$ &
    Games & Self-play \\
NNUE & \checkmark\checkmark\checkmark & \checkmark\checkmark &
    \checkmark & \checkmark\checkmark &
    Position evals & Supervised \\
\bottomrule
\end{tabularx}
\end{table}
```

### Table with Colored Rows

```latex
\definecolor{headerrow}{gray}{0.9}

\begin{table}[H]
\centering
\begin{tabular}{l|rr|rr}
\rowcolor{headerrow}
\textbf{Method} & \textbf{Epoch 1} & \textbf{Epoch 2} &
    \textbf{Epoch 3} & \textbf{Epoch 4} \\
\hline
Baseline & 45\% & 52\% & 58\% & 62\% \\
\rowcolor{yellow!20}
Baseline + LR & 48\% & 56\% & 65\% & 70\% \\
Baseline + Aug & 50\% & 58\% & 67\% & 73\% \\
\rowcolor{green!20}
Baseline + Both & 52\% & 62\% & 71\% & 78\% \\
\end{tabular}
\end{table}
```

---

## 5. Custom Environments and Macros

### Theorem and Lemma Environments

```latex
\usepackage{amsthm}

\theoremstyle{definition}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

% Usage:
\begin{theorem}
    Alpha-beta pruning reduces the effective branching factor
    from $b$ to $b^{1/2}$ in the best case.
\end{theorem}

\begin{proof}
    The proof follows from the minimax algorithm's tree structure...
\end{proof}
```

### Custom Highlighted Boxes

```latex
% In preamble:
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

\newtcolorbox{keyalgorithm}[1]{
    colback=green!5,
    colframe=green!50!black,
    colbacktitle=green!50!black,
    coltitle=white,
    title=\textbf{#1},
    fonttitle=\bfseries,
    left=10pt, right=10pt, top=10pt, bottom=10pt,
    arc=4pt,
    breakable
}

% Usage:
\begin{keyalgorithm}{Alpha-Beta Pruning}
\textbf{Key Insight:} If we've found a good move, we don't need to
    search all remaining branches if we know the opponent can force
    us to a worse position.
\end{keyalgorithm}
```

### Margin Notes for Additional Information

```latex
\usepackage{marginnote}

This is important text\marginnote{This is a margin note that appears
    on the side of the page}.

\reversemarginpar
This is left-side text\marginnote{This note appears on the left}.
```

---

## 6. Performance Optimization Tips

### Faster Compilation

```bash
# Use lualatex for faster compilation
lualatex --shell-escape -interaction=nonstopmode book.tex

# Or use xelatex for better font support
xelatex --shell-escape -interaction=nonstopmode book.tex

# Make figures conditional (don't compile if unchanged)
# Use \ifpdf for conditional compilation
```

### Reducing File Size

```latex
% Compress images before including
\usepackage{graphicx}
\pdfcompresslevel=9

% Use vector graphics when possible
% Use tikz or pdf/eps instead of png for diagrams

% Or convert in bash:
# for f in *.png; do
#   convert "$f" -quality 85 "compressed_$f"
# done
```

---

## 7. Advanced Referencing and Linking

### Smart References with Hyperref

```latex
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=primaryblue,
    citecolor=accentgreen,
    urlcolor=primaryblue,
    bookmarksnumbered=true,
    pdfencoding=unicode,
    pdftitle={Yolah: AI Implementation},
    pdfauthor={Your Name}
}

% Create custom reference commands
\newcommand{\figref}[1]{Figure~\ref{#1}}
\newcommand{\tabref}[1]{Table~\ref{#1}}
\newcommand{\secref}[1]{Section~\ref{#1}}
\newcommand{\chapref}[1]{Chapter~\ref{#1}}

% Usage:
\figref{fig:performance} shows the results.
\tabref{tab:comparison} summarizes the findings.
```

### Cleveref for Smarter References

```latex
\usepackage{cleveref}
\crefname{figure}{Figure}{Figures}
\crefname{table}{Table}{Tables}
\crefname{equation}{Eq.}{Eqs.}

% Automatic singular/plural
\cref{fig:a,fig:b,fig:c}  % Outputs: Figures 1, 2, and 3
\cref{fig:a}  % Outputs: Figure 1
```

---

## 8. Automation and Build Scripts

### Complete Build Script with Optimization

```bash
#!/bin/bash
# comprehensive_build.sh

MAIN_FILE="book"
BUILD_DIR="build"
FINAL_PDF="${MAIN_FILE}.pdf"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Yolah Book Build System ===${NC}"

# Clean
echo -e "${BLUE}[1/5] Cleaning...${NC}"
rm -rf $BUILD_DIR _minted-*
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot

# Generate figures if needed
if [ ! -d "figures" ] || [ -z "$(ls -A figures)" ]; then
    echo -e "${BLUE}[2/5] Generating figures...${NC}"
    python3 generate_figures.py
else
    echo -e "${BLUE}[2/5] Figures already exist, skipping generation${NC}"
fi

# First compilation
echo -e "${BLUE}[3/5] First LaTeX pass...${NC}"
pdflatex --shell-escape -interaction=nonstopmode $MAIN_FILE.tex > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ LaTeX compilation failed${NC}"
    exit 1
fi

# Bibliography
echo -e "${BLUE}[4/5] Processing bibliography...${NC}"
biber $MAIN_FILE > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Biber failed${NC}"
    exit 1
fi

# Final compilations
echo -e "${BLUE}[5/5] Final LaTeX passes...${NC}"
pdflatex --shell-escape -interaction=nonstopmode $MAIN_FILE.tex > /dev/null
pdflatex --shell-escape -interaction=nonstopmode $MAIN_FILE.tex > /dev/null

if [ -f "$FINAL_PDF" ]; then
    SIZE=$(ls -lh $FINAL_PDF | awk '{print $5}')
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}✓ Output: $FINAL_PDF ($SIZE)${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
```

---

## 9. Common Pitfalls and Solutions

| Problem | Solution |
|---------|----------|
| Text overflows margins | Use `\raggedright` or `hyphenpenalty=-1` |
| Figures float away | Use `[H]` placement or `float` package |
| Code highlighting fails | Add `--shell-escape` flag to pdflatex |
| Bibliography not updating | Run biber, then pdflatex twice |
| Widow/orphan lines | Use `\widowpenalty` and `\clubpenalty` |
| Slow compilation | Use `--interaction=batchmode` or cache figures |
| Broken cross-references | Recompile twice after adding new labels |

---

## 10. Quick Reference: Essential Packages

```latex
% Typography & Layout
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}

% Math & Science
\usepackage{amsmath,amssymb,mathtools}

% Graphics & Figures
\usepackage{graphicx,tikz,pgfplots}

% Code
\usepackage{minted}
\usepackage{listings}

% Tables
\usepackage{booktabs,array,tabularx,multirow}

% References
\usepackage{hyperref}
\usepackage{cleveref}

% Formatting
\usepackage{tcolorbox}
\usepackage{xcolor}

% Bibliography
\usepackage[backend=biber,style=alphabetic]{biblatex}

# Compile with:
pdflatex --shell-escape -interaction=nonstopmode book.tex
biber book
pdflatex --shell-escape -interaction=nonstopmode book.tex
```

---

Happy technical writing! 📚
