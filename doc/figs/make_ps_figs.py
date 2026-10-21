"""
make_ps_figs.py — the figures of the "recherche de programmes" chapter of
../nnue_roadmap.qmd: TikZ files (then ./build.sh) and one matplotlib PNG.

The board values are computed by nnue/program_search.py itself (the same
functions the search runs), on the positions of make_feature_figs.py.

    python3 make_ps_figs.py [RUN/all.tsv]     # all.tsv of a run → fig-ps-front.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nnue"))
import program_search as P  # noqa: E402
from make_feature_figs import (HIS, HOLE, MIDDLE, MINE, S, Fig, legend_line,  # noqa: E402
                               swatch)

PREAMBLE = "\\documentclass[border=6pt,tikz]{standalone}\n\\input{preamble.tex}\n\\begin{document}\n"


def env_of(pos):
    us, them, free = pos
    bb = lambda s: np.array([sum(1 << q for q in s)], dtype=np.uint64)   # noqa: E731
    M, H, F = bb(us), bb(them), bb(free)
    return {'M': M, 'H': H, 'F': F, 'X': ~(M | H | F)}


def squares(x):
    x = int(x[0])
    return {s for s in range(64) if x >> s & 1}


def write_tikz(name, body):
    with open(os.path.join(HERE, name + ".tex"), "w") as f:
        f.write(PREAMBLE + body + "\n\\end{document}\n")


# ── 1. the pipeline ──────────────────────────────────────────────────────────
def fig_pipeline():
    write_tikz("fig-ps-pipeline", r"""
\begin{tikzpicture}[node distance=4mm and 7mm]
\node[box] (games) {data/games\\[-1pt]\scriptsize 4,8 M parties};
\node[boxa,right=of games] (dump) {program\_search\_dump (C++)\\[-1pt]\scriptsize une position sur 20,\\[-1pt]\scriptsize valeur du réseau de base};
\node[box,right=of dump] (pos) {positions.bin\\[-1pt]\scriptsize 36 octets / position};
\node[boxb,below=9mm of dump,minimum width=36mm] (scorer) {le juge (1 M positions)\\[-1pt]\scriptsize gain : ce que le programme\\[-1pt]\scriptsize ajoute au réseau};
\node[boxb,right=of scorer,minimum width=34mm] (wit) {2 000 témoins\\[-1pt]\scriptsize doublons, nombre de tours,\\[-1pt]\scriptsize coût prédit};
\node[boxb,left=of scorer,minimum width=30mm] (harn) {20 000 positions\\[-1pt]\scriptsize chronométrage C++\\[-1pt]\scriptsize et vérification};
\node[box,below=9mm of scorer,minimum width=64mm,fill=white] (search) {la recherche : énumération par taille, puis évolution\\[-1pt]\scriptsize classement par Speed Prior ; une étape OOPS toutes les 4 h};
\node[plain,below=6mm of search] (out) {\ttfamily leaderboard.txt \quad library.txt \quad features.h \quad all.tsv};
\draw[ar] (games) -- (dump);
\draw[ar] (dump) -- (pos);
\draw[ar] (pos.south) |- ($(wit.north)+(0,0.35)$) -| (wit.north);
\draw[ar] (pos.south) |- ($(wit.north)+(0,0.35)$) -| (scorer.north);
\draw[ar] (pos.south) |- ($(wit.north)+(0,0.35)$) -| (harn.north);
\draw[ar] (scorer) -- (search);
\draw[ar] (wit.south) -- ($(search.north east)!0.2!(search.north west)$);
\draw[ar] (harn.south) -- ($(search.north west)!0.2!(search.north east)$);
\draw[ar] (search) -- (out);
\end{tikzpicture}""")


# ── 2. the four set operations on a board ────────────────────────────────────
def fig_ops():
    us, them, free = MIDDLE
    env = env_of(MIDDLE)
    fig = Fig()
    W = 8 * S + 0.9
    panels = [("N(M)", "les 8 voisines", "me"), ("S(M)", "un pas de roi sur une case libre", "me"),
              ("Q(M)", "un coup de dame", "me"), ("C(M)", "tout ce qu'un roi atteint", "me")]
    for i, (text, sub, col) in enumerate(panels):
        x = P.evaluate(P.parse(text), dict(env, cache=None))
        sq = squares(x)
        fig.board(us, them, free, ox=i * W, fills={s: col for s in sq},
                  title=f"\\texttt{{{text}}} : {len(sq)} cases",
                  extra=f"\\node[tiny] at ({4 * S:.3f},-0.55) {{{sub}}};")
        print(text, len(sq))
    fig.raw(4 * W - 0.4, legend_line(0, 3.6, [(MINE, "M : moi"), (HIS, "H : lui"), (HOLE, "X : trou"),
                                             (swatch("me"), "l'ensemble calculé")]))
    fig.write("fig-ps-ops")


# ── 3. the two loop templates ────────────────────────────────────────────────
def fig_loops():
    us, them, free = MIDDLE
    env = env_of(MIDDLE)
    fig = Fig()
    W = 8 * S + 1.2
    # RACE#[S(A) - B], replayed round by round to label each square
    g = P.parse("RACE#[S(A) - B]")[1]
    gs = P.swap(g)
    A, B, neu = env['M'].copy(), env['H'].copy(), np.zeros(1, np.uint64)
    when, fills = {}, {}
    for k in range(P.MAX_ROUNDS):
        loc = dict(env, A=A, B=B, cache=None)
        a = P.evaluate(g, loc) & ~A & ~neu
        b = P.evaluate(gs, loc) & ~B & ~neu
        both = a & b
        neu |= both
        a &= ~both
        b &= ~both
        for s, col in [(s, "me") for s in squares(a)] + [(s, "him") for s in squares(b)] + \
                [(s, "neutral") for s in squares(both)]:
            when[s], fills[s] = k + 1, col
        if not (a | b).any():
            break
        A |= a
        B |= b
    nm, nh = len(squares(A) & free), len(squares(B) & free)
    fig.board(us, them, free, fills=fills, labels={s: str(v) for s, v in when.items()},
              title=f"\\texttt{{RACE\\#[S(A) - B]}} : moi {nm}, lui {nh}",
              extra=f"\\node[tiny] at ({4 * S:.3f},-0.55) {{chiffre = tour où la case est prise}};")
    print("RACE#[S(A) - B]", nm, nh)
    # GROW#[S(A) - Q(H)]
    p = P.parse("GROW#[S(A) - Q(H)]")
    me, him, info = P.run_program(p, dict(env, cache={}, stats={}))
    qh = squares(P.evaluate(P.parse("Q(H)"), dict(env, cache=None)))
    mine = squares(info['set_me']) & free
    f2 = {s: "soft3" for s in qh}
    f2.update({s: "me" for s in mine})
    fig.board(us, them, free, ox=W, fills=f2, title=f"\\texttt{{GROW\\#[S(A) - Q(H)]}} : moi {int(me[0])}",
              extra=f"\\node[tiny] at ({4 * S:.3f},-0.55) {{pas de roi, sans entrer là où il va en un coup}};")
    print("GROW#[S(A) - Q(H)]", int(me[0]), int(him[0]))
    fig.raw(2 * W - 0.6, legend_line(0, 4.2, [(MINE, "moi"), (HIS, "lui"), (HOLE, "trou"),
                                             (swatch("me"), "A (à moi)"), (swatch("him"), "B (à lui)"),
                                             (swatch("neutral"), "neutre (pris au même tour)"),
                                             (swatch("soft3"), "Q(H) : interdit à A")]))
    fig.write("fig-ps-loops")


# ── 4. tree vs graph: the cost of L1 ─────────────────────────────────────────
def fig_graph():
    write_tikz("fig-ps-graph", r"""
\begin{tikzpicture}[x=1cm,y=1cm,
  op/.style={nd,minimum size=8mm},
  q/.style={nd,fill=soft2,minimum size=8mm},
  lf/.style={nd,fill=soft,minimum size=7mm}]
% leaves and hoisted part
\node[lf] (M) at (0,0) {M};
\node[lf] (H) at (6,0) {H};
\node[q] (QM) at (0,1.4) {Q};
\node[q] (QH) at (6,1.4) {Q};
\node[q] (QQM) at (0,2.8) {Q};
\node[q] (QQH) at (6,2.8) {Q};
\node[tiny,left=1mm of QQM] {Q(Q(M))};
\node[tiny,right=1mm of QQH] {Q(Q(H))};
\draw[ar] (M) -- (QM); \draw[ar] (QM) -- (QQM);
\draw[ar] (H) -- (QH); \draw[ar] (QH) -- (QQH);
\begin{scope}[on background layer]
\node[draw=accent2,dashed,rounded corners,fit=(M)(QQM)(H)(QQH),inner sep=6pt,
  label={[plain,text=accent2]below:hors de la boucle : 4 Q une seule fois, partagés par les deux camps}] {};
\end{scope}
% loop body
\node[lf] (A) at (1.2,4.6) {A};
\node[op] (SA) at (1.2,5.8) {S};
\node[op] (mA) at (1.9,7.0) {$-$};
\node[op] (oA) at (2.2,8.2) {$\cup$};
\node[lf] (B) at (4.8,4.6) {B};
\node[op] (SB) at (4.8,5.8) {S};
\node[op] (mB) at (4.1,7.0) {$-$};
\node[op] (oB) at (3.8,8.2) {$\cup$};
\draw[ar] (A) -- (SA); \draw[ar] (SA) -- (mA); \draw[ar] (mA) -- (oA);
\draw[ar] (B) -- (SB); \draw[ar] (SB) -- (mB); \draw[ar] (mB) -- (oB);
\draw[ar,accent2] (QQH) to[out=100,in=-10] (mA);
\draw[ar,accent2] (QQM) to[out=80,in=190] (mB);
\draw[ar,accent2] (QQM.north east) to[out=60,in=200] (oA);
\draw[ar,accent2] (QQH.north west) to[out=120,in=-20] (oB);
\node[box,fill=white] (race) at (3,9.5) {la course : $a$, $b$, pris ensemble $\to$ neutre ; $A \mathrel{|}= a$, $B \mathrel{|}= b$};
\draw[ar] (oA) -- (race); \draw[ar] (oB) -- (race);
\begin{scope}[on background layer]
\node[draw=accent,rounded corners,fill=soft!50,fit=(A)(B)(race),inner sep=6pt,
  label={[plain,text=accent,align=center]above:dans la boucle : répété à chaque tour (1,9 tours en moyenne)}] {};
\end{scope}
\node[tiny,text=ink,align=left,anchor=west] at (7.8,6.6) {\textbf{coût de \texttt{RACE\#[(S(A) - Q(Q(H))) | Q(Q(M))]}}\\[3pt]
arbre naïf : 8 Q à chaque tour\\$\approx 8 \times 8{,}8 \times 1{,}9 \approx 135$ ns\\[3pt]
graphe : 4 Q une fois (35 ns)\\+ 1,9 tours $\times$ (2 S + les $\&$, $|$, $-$)\\$\approx 46$ ns prédit, 41 ns mesuré};
\end{tikzpicture}""")


# ── 5. the OOPS stages ───────────────────────────────────────────────────────
def fig_oops():
    write_tikz("fig-ps-oops", r"""
\begin{tikzpicture}[node distance=6mm and 10mm]
\node[box,minimum width=46mm] (search) {la recherche\\[-1pt]\scriptsize énumération, évolution,\\[-1pt]\scriptsize mesure C++ toutes les 30 min};
\node[boxa,right=of search,minimum width=46mm] (pick) {toutes les 4 h : le meilleur\\[-1pt]\scriptsize selon le Speed Prior\\[-1pt]\scriptsize (si son score est $\ge 0{,}05$)};
\node[boxb,below=of pick,minimum width=46mm,align=left] (freeze) {\textbf{gel} de $L_k$ :\\[-1pt]\scriptsize $\bullet$ ses 2 colonnes rejoignent la base\\[-1pt]\scriptsize $\bullet$ son ensemble devient une feuille $L_k$\\[-1pt]\scriptsize $\bullet$ ses sous-calculs deviennent gratuits};
\node[box,below=of search,minimum width=46mm] (rescore) {renotation des 300 meilleurs\\[-1pt]\scriptsize et du front, contre la nouvelle base ;\\[-1pt]\scriptsize petits programmes autour de $L_k$};
\draw[ar] (search) -- (pick);
\draw[ar] (pick) -- (freeze);
\draw[ar] (freeze) -- (rescore);
\draw[ar] (rescore) -- (search);
\node[plain,below=4mm of $(rescore.south)!0.5!(freeze.south)$,text=muted] {\scriptsize écrit dans library.txt (le programme, son gain, son coût) et features.h (son C++)};
\end{tikzpicture}""")


# ── 6. the Pareto front of a run (matplotlib) ────────────────────────────────
def fig_front(tsv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [line.rstrip("\n").split("\t") for line in open(tsv)][1:]
    rows = [r for r in rows if r[0] == "0"]
    score = np.array([float(r[1]) for r in rows])
    ns = np.array([float(r[3]) for r in rows])
    prog = [r[-1] for r in rows]
    order = np.lexsort((-score, ns))
    front, best = [], -1e9
    for i in order:
        if score[i] > best + 1e-9:
            best = score[i]
            front.append(i)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=170)
    ax.scatter(ns, score, s=3, c="#9AA5B1", alpha=.35, lw=0, label=f"{len(rows)} programmes notés")
    ax.plot(ns[front], score[front], "-o", ms=3, c="#2F6FB5", lw=1.2, label="front de Pareto")
    xs = np.geomspace(1.5, 400, 100)
    for sp in (0.2, 0.3, 0.4):
        ax.plot(xs, sp + P.LAM * np.log2(xs), ":", c="#C1611F", lw=1)
        ax.text(1.6, sp + P.LAM * np.log2(1.6) + 0.01, f"sp = {sp}", color="#C1611F", fontsize=7)
    labels = {"|S(M)|", "|C(M)|", "RACE#[S(A)]", "|S(S(S(M)))|"}
    for i in [i for i, p in enumerate(prog) if p in labels] + [front[-1]]:
        ax.plot(ns[i], score[i], "o", ms=4, c="#1B2430")
        right = i != front[-1]
        ax.annotate(prog[i], (ns[i], score[i]), xytext=(6 if right else -6, -3), textcoords="offset points",
                    fontsize=7, ha="left" if right else "right", va="top")
    ax.set_xscale("log")
    ax.set_xlim(1.5, 400)
    ax.set_ylim(-0.02, max(score) + 0.06)
    ax.set_xlabel("coût prédit (ns par évaluation, les deux camps), échelle log")
    ax.set_ylabel("score (gain moyen, points de R²)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-ps-front.png"))
    print("front:", len(front), "points; best", prog[front[-1]], score[front[-1]], ns[front[-1]])


if __name__ == "__main__":
    fig_pipeline()
    fig_ops()
    fig_loops()
    fig_graph()
    fig_oops()
    if len(sys.argv) > 1:
        fig_front(sys.argv[1])
