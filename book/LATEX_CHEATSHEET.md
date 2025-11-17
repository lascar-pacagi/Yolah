# LaTeX Cheatsheet for Technical Book Writing

Quick reference guide for common LaTeX commands and patterns used in technical book writing.

---

## Document Structure

```latex
\documentclass[12pt,a4paper]{book}
\begin{document}
% Content here
\end{document}
```

### Front Matter
```latex
\frontmatter              % Roman page numbers
\maketitle
\tableofcontents
\listoffigures
\listoftables
```

### Main Matter
```latex
\mainmatter               % Arabic page numbers, reset to 1
\chapter{Chapter Name}
\section{Section}
\subsection{Subsection}
\subsubsection{Subsubsection}
```

### Back Matter
```latex
\backmatter               % No chapter numbers
\appendix
\chapter{Appendix A}
\printbibliography
```

---

## Text Formatting

| Command | Effect |
|---------|--------|
| `\textbf{text}` | **Bold text** |
| `\textit{text}` | *Italic text* |
| `\texttt{text}` | `Monospace/code` |
| `\emph{text}` | Emphasis (usually italic) |
| `\underline{text}` | Underlined text |
| `\textsc{text}` | SMALL CAPS |

### Font Sizes
```latex
\tiny \scriptsize \footnotesize \small \normalsize
\large \Large \LARGE \huge \Huge
```

---

## Lists

### Itemized (Bullets)
```latex
\begin{itemize}
    \item First item
    \item Second item
    \item Third item
\end{itemize}
```

### Enumerated (Numbers)
```latex
\begin{enumerate}
    \item First item
    \item Second item
    \item Third item
\end{enumerate}
```

### Description Lists
```latex
\begin{description}
    \item[Term 1] Description of term 1
    \item[Term 2] Description of term 2
\end{description}
```

---

## Math Mode

### Inline Math
```latex
This is inline math: $E = mc^2$
```

### Display Math
```latex
\[ E = mc^2 \]
```

### Numbered Equations
```latex
\begin{equation}
    E = mc^2
    \label{eq:einstein}
\end{equation}
```

### Common Math Symbols
```latex
\alpha \beta \gamma \delta \epsilon
\sum \int \prod \lim
\leq \geq \neq \approx \equiv
\infty \partial \nabla
\frac{a}{b}
\sqrt{x}
x^2  x_i  x^{2y}  x_{i,j}
```

### Multi-line Equations
```latex
\begin{align}
    a &= b + c \\
    d &= e + f
\end{align}
```

---

## Figures

### Basic Figure
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/image.png}
    \caption{Description of the figure}
    \label{fig:example}
\end{figure}
```

### Position Parameters
- `h` - here (approximately)
- `t` - top of page
- `b` - bottom of page
- `p` - separate page
- `H` - exactly here (requires float package)

### Multiple Subfigures
```latex
\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{fig1.png}
        \caption{First subfigure}
        \label{fig:sub1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{fig2.png}
        \caption{Second subfigure}
        \label{fig:sub2}
    \end{subfigure}
    \caption{Overall caption}
    \label{fig:overall}
\end{figure}
```

---

## Tables

### Basic Table
```latex
\begin{table}[H]
    \centering
    \begin{tabular}{|l|c|r|}
        \hline
        Left & Center & Right \\
        \hline
        1 & 2 & 3 \\
        4 & 5 & 6 \\
        \hline
    \end{tabular}
    \caption{Table description}
    \label{tab:example}
\end{table}
```

### Column Alignment
- `l` - left aligned
- `c` - centered
- `r` - right aligned
- `|` - vertical line
- `p{width}` - paragraph column with specified width

### Professional Tables (booktabs)
```latex
\begin{table}[H]
    \centering
    \begin{tabular}{lcc}
        \toprule
        Algorithm & Time (ms) & Accuracy (\%) \\
        \midrule
        Minimax & 150 & 85.3 \\
        Alpha-Beta & 45 & 85.3 \\
        MCTS & 200 & 92.1 \\
        \bottomrule
    \end{tabular}
    \caption{Performance comparison}
    \label{tab:performance}
\end{table}
```

---

## Code Listings

### Using minted (Recommended)
```latex
\begin{listing}[H]
\begin{minted}[linenos, bgcolor=codebg]{python}
def minimax(position, depth, maximizing):
    if depth == 0:
        return evaluate(position)
    # ... rest of code
\end{minted}
\caption{Minimax algorithm implementation}
\label{code:minimax}
\end{listing}
```

### Inline Code
```latex
Use the \mintinline{python}{minimax()} function.
```

### Supported Languages
python, c, cpp, java, javascript, rust, go, bash, sql, latex, json, xml, html, css

---

## Algorithms

```latex
\begin{algorithmbox}
\caption{Minimax with Alpha-Beta Pruning}
\label{alg:alphabeta}
\begin{algorithmic}[1]
\Procedure{AlphaBeta}{$position, depth, \alpha, \beta, maximizing$}
    \If{$depth = 0$ \textbf{or} $position$ is terminal}
        \State \Return $evaluate(position)$
    \EndIf
    \If{$maximizing$}
        \State $value \gets -\infty$
        \For{each child of $position$}
            \State $value \gets \max(value, \textsc{AlphaBeta}(child, depth-1, \alpha, \beta, \textbf{false}))$
            \State $\alpha \gets \max(\alpha, value)$
            \If{$\beta \leq \alpha$}
                \State \textbf{break}
            \EndIf
        \EndFor
        \State \Return $value$
    \Else
        \State Similar logic for minimizing player
    \EndIf
\EndProcedure
\end{algorithmic}
\end{algorithmbox}
```

### Algorithm Commands
```latex
\Procedure{Name}{parameters}
\EndProcedure
\Function{Name}{parameters}
\EndFunction
\If{condition}
\ElsIf{condition}
\Else
\EndIf
\For{condition}
\EndFor
\While{condition}
\EndWhile
\State statement
\Return value
\Comment{comment text}
```

---

## Custom Boxes

### Important Box
```latex
\begin{importantbox}
This is an important concept that readers should remember.
\end{importantbox}
```

### Algorithm Box
```latex
\begin{algorithmbox}
Algorithm pseudocode goes here
\end{algorithmbox}
```

### Result Box
```latex
\begin{resultbox}
Key experimental results or findings.
\end{resultbox}
```

---

## Cross-References

```latex
% Define labels
\label{fig:example}
\label{tab:results}
\label{eq:formula}
\label{sec:introduction}
\label{code:implementation}

% Reference them
See Figure~\ref{fig:example}
As shown in Table~\ref{tab:results}
Equation~\ref{eq:formula} demonstrates
In Section~\ref{sec:introduction}
Listing~\ref{code:implementation} shows
```

### Page References
```latex
See page~\pageref{fig:example}
```

---

## Citations & Bibliography

### In-Text Citations
```latex
\cite{silver2016alphago}                    % [1]
\citep{silver2016alphago}                   % (Silver et al., 2016)
\citet{silver2016alphago}                   % Silver et al. (2016)
\cite{ref1,ref2,ref3}                       % Multiple citations
\citep[see][p. 42]{silver2016alphago}       % With page number
```

### Bibliography
```latex
\printbibliography
```

---

## Special Characters

```latex
\%  \$  \&  \#  \_  \{  \}
\textbackslash
\~{}  \^{}
```

### Quotes
```latex
`single quotes'
``double quotes''
```

### Dashes
```latex
- (hyphen)
-- (en-dash for ranges: 1--10)
--- (em-dash for punctuation)
```

---

## Spacing

### Horizontal Spacing
```latex
\hspace{1cm}
\hfill           % Fill horizontal space
\quad            % 1em space
\qquad           % 2em space
```

### Vertical Spacing
```latex
\vspace{1cm}
\vfill           % Fill vertical space
\smallskip
\medskip
\bigskip
```

### Line Breaks
```latex
\\               % New line
\\[5pt]          % New line with extra space
\newline
\newpage         % New page
\clearpage       % New page + flush floats
```

---

## Page Layout

### Margins
```latex
\usepackage[margin=1in]{geometry}
\usepackage[top=1in, bottom=1.5in, left=1.25in, right=1.25in]{geometry}
```

### Headers & Footers
```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Chapter \thechapter}
\fancyhead[R]{\leftmark}
\fancyfoot[C]{\thepage}
```

---

## Including Files

```latex
\input{chapters/chapter1.tex}      % Include file inline
\include{chapters/chapter1}        % Include with page break
```

---

## Hyperlinks

```latex
\usepackage{hyperref}

\href{https://example.com}{Link text}
\url{https://example.com}
\hyperref[label]{text}
```

---

## Colors

```latex
\usepackage{xcolor}

\textcolor{red}{Red text}
\colorbox{yellow}{Highlighted}
\definecolor{myblue}{RGB}{0,100,200}
\textcolor{myblue}{Custom color}
```

---

## TikZ Diagrams

### Basic Shape
```latex
\begin{tikzpicture}
    \draw (0,0) -- (2,0) -- (2,2) -- (0,2) -- cycle;
    \node at (1,1) {Text};
\end{tikzpicture}
```

### Common Commands
```latex
\draw (x1,y1) -- (x2,y2);           % Line
\draw (x,y) circle (radius);         % Circle
\draw (x,y) rectangle (x2,y2);       % Rectangle
\node at (x,y) {text};               % Text node
\fill[color] (x,y) circle (r);       % Filled circle
```

---

## Common Packages

### Must-Have Packages
```latex
\usepackage{amsmath, amssymb}      % Math symbols
\usepackage{graphicx}              % Images
\usepackage{hyperref}              % Hyperlinks
\usepackage{booktabs}              % Professional tables
\usepackage{minted}                % Code highlighting
\usepackage{algorithm}             % Algorithms
\usepackage{algpseudocode}         % Algorithm pseudocode
\usepackage{tikz}                  % Diagrams
\usepackage{geometry}              % Page layout
```

---

## Compilation

### Basic Compilation
```bash
pdflatex document.tex
```

### With Code Highlighting
```bash
pdflatex --shell-escape document.tex
```

### Full Cycle (with bibliography)
```bash
pdflatex --shell-escape document.tex
biber document
pdflatex --shell-escape document.tex
pdflatex --shell-escape document.tex
```

---

## Troubleshooting

### Common Fixes

| Problem | Solution |
|---------|----------|
| Undefined reference | Compile twice |
| Missing citations | Run biber, then pdflatex twice |
| Code not highlighted | Add `--shell-escape` |
| Image not found | Check path, use forward slashes |
| Math error | Check for missing `$` or `\[` |
| Package not found | Install via tlmgr or texlive-full |

### Quick Checks
```bash
# Check log for errors
grep -i error document.log

# Check for undefined references
grep "undefined" document.log

# Check for overfull boxes (bad formatting)
grep "Overfull" document.log
```

---

## Quick Tips

1. **Always compile twice** after adding references
2. **Use `\label{}` immediately** after `\caption{}`
3. **Don't hardcode figure numbers** - use `\ref{}`
4. **Keep lines short** in source (< 80 chars) for readability
5. **Use meaningful label names**: `fig:alphazero_architecture` not `fig:1`
6. **Comment your LaTeX**: Use `%` for notes
7. **Version control**: Use git, exclude build files
8. **Test early**: Don't wait until the end to compile

---

## Example Document Structure

```latex
\documentclass[12pt,a4paper]{book}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{minted}
\usepackage{booktabs}

% Document info
\title{My Book}
\author{Author Name}
\date{\today}

\begin{document}

\frontmatter
\maketitle
\tableofcontents

\mainmatter
\chapter{Introduction}
Content here...

\chapter{Main Content}
More content...

\backmatter
\printbibliography

\end{document}
```

---

**Pro Tip**: Keep this cheatsheet open while writing. The more you use these commands, the faster you'll memorize them!

For more advanced techniques, see `ADVANCED_LATEX_TIPS.md`.
