"""
make_feature_figs.py — the board figures of the "caractéristiques candidates"
chapter of ../nnue_roadmap.qmd, as standalone TikZ files (then ./build.sh).

Every number drawn is computed by nnue/space_reference.py, the square-by-square
Python reference checked against space::compute_all (identical on 20,000
positions), so the figures show exactly what the network receives.

    python3 make_feature_figs.py          # writes fig-feat-*.tex, prints the values
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nnue"))
import space_reference as R  # noqa: E402

S = 0.62        # square size (cm)


def parse(rows):
    """8 strings, rank 8 first: M = my piece (side to move), H = his, x = hole, . = free."""
    us, them, free = set(), set(), set()
    for i, row in enumerate(rows):
        for c, ch in enumerate(row.split()):
            sq = 8 * (7 - i) + c
            {"M": us, "H": them, ".": free}.get(ch, set()).add(sq)
    assert len(us) == 4 and len(them) == 4, rows
    return us, them, free


def xy(sq):
    r, c = divmod(sq, 8)
    return (c + 0.5) * S, (r + 0.5) * S


class Fig:
    def __init__(self):
        self.parts = []

    def board(self, us, them, free, ox=0.0, fills=None, labels=None, title=None, coords=True, oy=0.0, extra=""):
        """fills: {sq: colour}, labels: {sq: text}; drawn under the pieces; extra: tikz drawn on top."""
        p = self.parts
        p.append(f"\\begin{{scope}}[shift={{({ox:.2f},{oy:.2f})}}]")
        for sq, col in (fills or {}).items():
            x, y = xy(sq)
            p.append(f"\\fill[{col}] ({x - S / 2:.3f},{y - S / 2:.3f}) rectangle +({S:.3f},{S:.3f});")
        for sq in range(64):
            if sq not in free and sq not in us and sq not in them:       # hole
                x, y = xy(sq)
                p.append(f"\\fill[hole] ({x - S / 2:.3f},{y - S / 2:.3f}) rectangle +({S:.3f},{S:.3f});")
        p.append(f"\\draw[muted!60,line width=.3pt,step={S}] (0,0) grid ({8 * S:.3f},{8 * S:.3f});")
        p.append(f"\\draw[ink,line width=.6pt] (0,0) rectangle ({8 * S:.3f},{8 * S:.3f});")
        for sq, text in (labels or {}).items():
            x, y = xy(sq)
            p.append(f"\\node[lab] at ({x:.3f},{y:.3f}) {{{text}}};")
        for sq in us:
            p.append(f"\\node[mine] at ({xy(sq)[0]:.3f},{xy(sq)[1]:.3f}) {{}};")
        for sq in them:
            p.append(f"\\node[his] at ({xy(sq)[0]:.3f},{xy(sq)[1]:.3f}) {{}};")
        if coords:
            for c in range(8):
                p.append(f"\\node[tiny] at ({(c + 0.5) * S:.3f},-0.22) {{{'abcdefgh'[c]}}};")
            for r in range(8):
                p.append(f"\\node[tiny] at (-0.2,{(r + 0.5) * S:.3f}) {{{r + 1}}};")
        if extra:
            p.append(extra)
        if title:
            p.append(f"\\node[plain,above] at ({4 * S:.3f},{8 * S + 0.05:.3f}) {{{title}}};")
        p.append("\\end{scope}")

    def raw(self, ox, tikz, oy=0.0):
        self.parts.append(f"\\begin{{scope}}[shift={{({ox:.2f},{oy:.2f})}}]{tikz}\\end{{scope}}")

    def write(self, name, legend=None):
        body = "\n".join(self.parts)
        leg = ""
        if legend:
            leg = "\n" + legend
        with open(os.path.join(HERE, name + ".tex"), "w") as f:
            f.write("\\documentclass[border=6pt,tikz]{standalone}\n\\input{preamble.tex}\n"
                    "\\tikzset{mine/.style={circle,fill=ink,minimum size=4.2mm,inner sep=0pt},\n"
                    "  his/.style={circle,draw=ink,line width=.8pt,fill=white,minimum size=4.2mm,inner sep=0pt},\n"
                    "  hole/.style={fill=muted!75},\n"
                    "  lab/.style={font=\\sffamily\\scriptsize,text=ink,inner sep=0pt}}\n"
                    "\\definecolor{me}{HTML}{CFE0F3}\\definecolor{him}{HTML}{F6D9C3}\\definecolor{neutral}{HTML}{E4E7EB}\n"
                    "\\begin{document}\n\\begin{tikzpicture}\n" + body + leg + "\n\\end{tikzpicture}\n\\end{document}\n")


def legend_line(x, y, items):
    """items: list of (tikz drawing at origin, text)."""
    out = []
    for i, (mark, text) in enumerate(items):
        out.append(f"\\begin{{scope}}[shift={{({x:.2f},{y - 0.5 * i:.2f})}}]{mark}"
                   f"\\node[plain,right] at (0.3,0) {{{text}}};\\end{{scope}}")
    return "\n".join(out)


MINE = "\\node[mine] at (0,0) {};"
HIS = "\\node[his] at (0,0) {};"
HOLE = "\\fill[hole] (-0.15,-0.15) rectangle (0.15,0.15);"


def swatch(col):
    return f"\\fill[{col}] (-0.15,-0.15) rectangle (0.15,0.15);\\draw[muted!60,line width=.3pt] (-0.15,-0.15) rectangle (0.15,0.15);"


def sq(name):
    return 8 * (int(name[1]) - 1) + "abcdefgh".index(name[0])


def name_of(s):
    return "abcdefgh"[s % 8] + str(s // 8 + 1)


# ── 1. blocks and the corrected flood fill ────────────────────────────────────
ENDGAME = parse([
    ". . . x . . . .",
    ". . . x . . . .",
    "x x x M x x x x",
    "x x x x x . H .",
    ". M . . x x . .",
    ". . x . . x H M",
    "H . . x . x x .",
    ". . M x . x H .",
])


def fig_corr():
    us, them, free = ENDGAME
    bl = R.blocks(free)
    bl.sort(key=lambda b: -len(b))
    colours = ["soft3", "me", "him", "neutral", "soft2"]
    fig = Fig()
    fills = {s: colours[i % len(colours)] for i, b in enumerate(bl) for s in b}
    labels = {min(b, key=lambda s: (-(s // 8), s % 8)): f"\\textbf{{{len(b)}}}" for b in bl}
    fig.board(us, them, free, fills=fills, labels=labels, title="blocs de cases libres (taille)")
    # options of my pieces
    opts = {p: [i for i, b in enumerate(bl) if R.around(p) & b] for p in sorted(us)}
    plain = len(set().union(*(b for b in bl if R.around_set(us) & b)))
    corr = R.corrected_flood_fill(us, free)
    fig.raw(0, legend_line(5.6, 4.6, [(MINE, "moi (au trait)"), (HIS, "lui"), (HOLE, "trou")]))
    fig.raw(0, f"\\node[plain,right,align=left] at (5.3,2.0) {{flood fill : {plain} cases\\\\"
               f"flood fill corrigé : \\textbf{{{corr}}} cases}};")
    fig.write("fig-feat-corr")
    print("corr:", {name_of(p): [len(bl[i]) for i in o] for p, o in opts.items()}, "plain", plain, "corrected", corr,
          "| his corrected", R.corrected_flood_fill(them, free))
    return plain, corr


# ── 2. queen distance, reached first, late zones ──────────────────────────────
MIDDLE = parse([
    ". . x . . . H .",
    ". x . . x . . .",
    "M . . x . . x .",
    ". . x . M . . .",
    ". x . . x . x H",
    ". . . H . x . .",
    "x . M . . . x .",
    ". . . x . M . H",
])


def fig_queen():
    us, them, free = MIDDLE
    lv = R.queen_levels(us, free)
    dist = {s: k + 1 for k, l in enumerate(lv) for s in l}
    shade = {1: "me!100", 2: "me!55", 3: "me!25"}
    fig = Fig()
    fig.board(us, them, free, fills={s: shade.get(d, "white") for s, d in dist.items()},
              labels={s: str(d) for s, d in dist.items()}, title="coups de dame pour moi")
    f16 = R.compute16(us, them, free)
    fig.raw(0, f"\\node[plain,right,align=left] at (5.3,3.0) {{Q1 = {f16[1]}\\\\Q2 = {f16[2]}\\\\Q3 = {f16[3]}\\\\"
               f"QW = {f16[5]}\\\\MOB = {f16[7]}}};")
    fig.write("fig-feat-queen")
    print("queen me:", f16[:8], "him:", f16[8:16])
    return f16


def fig_first():
    us, them, free = MIDDLE
    lm, lh = R.queen_levels(us, free), R.queen_levels(them, free)
    dm = {s: k for k, l in enumerate(lm) for s in l}
    dh = {s: k for k, l in enumerate(lh) for s in l}
    fills, labels = {}, {}
    for s in free:
        a, b = dm.get(s, 99), dh.get(s, 99)
        fills[s] = "me" if a < b else "him" if b < a else "neutral"
        labels[s] = f"{a + 1 if a < 99 else '-'}{{\\color{{muted}}/}}{b + 1 if b < 99 else '-'}"
    fig = Fig()
    fig.board(us, them, free, fills=fills, labels={s: f"\\tiny {t}" for s, t in labels.items()},
              title="qui arrive le premier (moi / lui)")
    # the 3 x 3 zones and the ones I reach late
    zones = R.zones9()
    late, ahead = [], []        # zones he reaches first (late for me), zones I reach first
    for i, z in enumerate(zones):
        zf = z & free
        if zf and R.first_level(lh, zf) < R.first_level(lm, zf):
            late.append(i)
        elif zf and R.first_level(lm, zf) < R.first_level(lh, zf):
            ahead.append(i)
    cuts = [0, 3, 5, 8]
    lines = "".join(f"\\draw[ink,line width=1.3pt,dashed] ({c * S:.3f},0) -- ({c * S:.3f},{8 * S:.3f});"
                    f"\\draw[ink,line width=1.3pt,dashed] (0,{c * S:.3f}) -- ({8 * S:.3f},{c * S:.3f});" for c in cuts[1:3])
    for i in late + ahead:
        zr, zc = divmod(i, 3)
        col = "accent2" if i in late else "accent"
        lines += (f"\\draw[{col},line width=1.6pt] ({cuts[zc] * S + .04:.3f},{cuts[zr] * S + .04:.3f}) rectangle "
                  f"({cuts[zc + 1] * S - .04:.3f},{cuts[zr + 1] * S - .04:.3f});")
    fig.raw(0, lines)
    f16 = R.compute16(us, them, free)
    fig.raw(0, legend_line(5.6, 4.6, [(swatch("me"), "j'arrive avant"), (swatch("him"), "il arrive avant"),
                                      (swatch("neutral"), "en même temps"),
                                      ("\\draw[accent,line width=1.6pt] (-0.15,-0.15) rectangle (0.15,0.15);",
                                       "zone où j'arrive avant"),
                                      ("\\draw[accent2,line width=1.6pt] (-0.15,-0.15) rectangle (0.15,0.15);",
                                       "zone où il arrive avant")]))
    fig.raw(0, f"\\node[plain,right,align=left] at (5.3,1.6) {{QFIRST : moi {f16[4]}, lui {f16[12]}\\\\"
               f"LATE9 : moi {f16[6]}, lui {f16[14]}}};")
    fig.write("fig-feat-first")
    print("first: QFIRST", f16[4], f16[12], "LATE9", f16[6], f16[14], "late zones", late, "ahead", ahead)


def fig_influence():
    us, them, free = MIDDLE
    # replay the spreading to label the step at which each square is taken
    own, front, neutral, prev, step, when = [set(us), set(them)], [set(us), set(them)], set(), None, 0, {}
    while prev != own:
        prev = [set(own[0]), set(own[1])]
        step += 1
        a, b = R.around_set(front[0]) & free - own[1], R.around_set(front[1]) & free - own[0]
        new_neutral = (R.around_set(neutral) & free) | (a & b)
        for s in new_neutral - neutral:
            when.setdefault(s, step)
        neutral |= new_neutral
        front = [a - neutral, b - neutral]
        for s in (front[0] | front[1]) - own[0] - own[1]:
            when.setdefault(s, step)
        own[0] |= front[0]
        own[1] |= front[1]
    fills = {s: "me" if s in own[0] else "him" if s in own[1] else "neutral" for s in free}
    fig = Fig()
    fig.board(us, them, free, fills=fills, labels={s: str(when[s]) for s in free if s in when},
              title="influence (pas de roi, simultanés)")
    i0, i1 = R.influence(us, them, free)
    fig.raw(0, legend_line(5.6, 4.6, [(swatch("me"), "à moi"), (swatch("him"), "à lui"), (swatch("neutral"), "neutre")]))
    fig.raw(0, f"\\node[plain,right,align=left] at (5.3,2.6) {{INFL : moi {i0}, lui {i1}\\\\"
               f"(chiffre = pas où\\\\la case est prise)}};")
    fig.write("fig-feat-infl")
    print("influence:", i0, i1)


# ── 3. connectivity per piece ─────────────────────────────────────────────────
LATE = parse([
    ". . x . . x . .",
    ". M x . . x H .",
    "x x x . . x x x",
    ". . . x . . . .",
    "H . x . . M . x",
    ". x . . x x . .",
    "x . H . . x M H",
    ". M . x . x . .",
])


def fig_connect():
    us, them, free = LATE
    bl = sorted(R.blocks(free), key=lambda b: -len(b))
    colours = ["soft3", "neutral", "soft2", "me", "him"]
    fills = {s: colours[i % len(colours)] for i, b in enumerate(bl) for s in b}
    rows = []
    for side, mine, opp in (("M", us, them), ("H", them, us)):
        desc = sorted(((R.connect_piece(p, opp, free), p) for p in mine), key=lambda t: (t[0][1], t[0][0], *t[0][2:]))
        for rank, (d, p) in enumerate(desc, 1):
            rows.append((side, rank, p, d))
    fig = Fig()
    fig.board(us, them, free, fills=fills, title="blocs et rang d'isolement")
    marks = "".join(f"\\node[font=\\sffamily\\bfseries\\tiny,text={'white' if s == 'M' else 'ink'}] at "
                    f"({xy(p)[0]:.3f},{xy(p)[1]:.3f}) {{{rank}}};" for s, rank, p, _ in rows)
    fig.raw(0, marks)
    # the king path of my most isolated piece that has a contact (if any)
    fig.raw(0, legend_line(5.6, 4.6, [(MINE, "moi : 1 = la plus isolée"), (HIS, "lui"), (HOLE, "trou")]))
    fig.write("fig-feat-connect")
    print("connect rows (side, rank, square, [block, 16*contact, 64/king, 64/go, 64/come]):")
    for s, rank, p, d in rows:
        print(f"  {s}{rank} {name_of(p)} {d}")
    return rows


# ── 4. encirclement ───────────────────────────────────────────────────────────
ENC = parse([
    ". . . . . . . .",
    ". . . . . . . .",
    ". . . M x x . .",
    ". M . . H x . .",
    ". . . x M H . .",
    ". . . . . . . .",
    ". . . . . . . H",
    "H M . . . . . .",
])


def fig_encircle():
    us, them, free = ENC
    p = sq("e5")
    e = R.encircle_piece(p, us, free)
    fig = Fig()
    nb = R.around(p)
    fills = {}
    for s in nb:
        fills[s] = "soft3" if s in free else fills.get(s, "white")
    att = {s for s in nb & free if any(s in R.queen_moves(q, free) for q in us)}
    fig.board(us, them, free, fills=fills, title="la pièce de lui en e5")
    # its lines: real (solid) and freed through my pieces (dashed, the squares taken)
    tik = ""
    x0, y0 = xy(p)
    for d in R.DIRS:
        real, freed = R.line(p, d, free), R.line(p, d, free | us)
        if real:
            x1, y1 = xy(real[-1])
            tik += f"\\draw[good,line width=1.1pt] ({x0:.3f},{y0:.3f}) -- ({x1:.3f},{y1:.3f});"
        if len(freed) > len(real):
            a = xy(real[-1]) if real else (x0, y0)
            x2, y2 = xy(freed[-1])
            tik += f"\\draw[accent2,line width=1.1pt,dashed] ({a[0]:.3f},{a[1]:.3f}) -- ({x2:.3f},{y2:.3f});"
            for s in freed[len(real):]:
                tik += f"\\node[circle,draw=accent2,line width=.8pt,minimum size=2.2mm,inner sep=0pt] at ({xy(s)[0]:.3f},{xy(s)[1]:.3f}) {{}};"
    for s in att:   # one of my pieces can plug it
        q = next(q for q in sorted(us) if s in R.queen_moves(q, free))
        tik += (f"\\draw[ar,accent,line width=1pt] ({xy(q)[0]:.3f},{xy(q)[1]:.3f}) -- "
                f"({xy(s)[0]:.3f},{xy(s)[1]:.3f});")
    fig.raw(0, tik)
    close = e["lib"] == 1 and e["att"] > 0
    fig.raw(0, legend_line(5.6, 4.6, [(swatch("soft3"), "liberté (voisine libre)"),
                                      ("\\draw[good,line width=1.1pt] (-0.2,0) -- (0.2,0);", "ses lignes"),
                                      ("\\draw[accent2,line width=1.1pt,dashed] (-0.2,0) -- (0.2,0);",
                                       "ce que mes pièces lui prennent"),
                                      ("\\draw[ar,accent,line width=1pt] (-0.2,0) -- (0.2,0);", "je peux boucher")]))
    fig.raw(0, f"\\node[plain,right,align=left] at (5.3,1.5) {{lib {e['lib']}, opp {e['opp']}, "
               f"wall {8 - e['lib'] - e['opp'] - len(R.around(p) & them)}, own {len(R.around(p) & them)}\\\\"
               f"short {e['short']}, gene {e['gene']}, contrib {e['contrib']}\\\\att {e['att']}, "
               f"fermable : {'oui' if close else 'non'}}};")
    fig.write("fig-feat-encircle")
    print("encircle e5:", e, "close", close, "att squares", [name_of(s) for s in att])


# ── 0. the occluded fill (Kogge-Stone) of slide<EAST> ─────────────────────────
def fig_occluded():
    """slide<EAST> step by step on a real board: gen (squares reached), pro (where a jump of the
    current length may land), the jumps of 1, 2 and 4, then the final step. Odd ranks are holes so
    that only the four interesting lines remain."""
    us = {sq(n) for n in ("a8", "a6", "c4", "e2")}
    them = {sq("f4")}
    free = {sq(n) for n in "b8 c8 d8 e8 f8 g8 h8 b6 c6 d6 e6 g6 h6 d4 e4 g4 h4 f2 g2 h2".split()}
    east = lambda s, k: {x + k for x in s if x % 8 + k < 8}      # shift<EAST> k times (no wrap)  # noqa: E731
    gen, pro = set(us), set(free)
    stages = [(set(gen), set(pro), [], "départ : gen = les pièces", "points : pro, arrivées des sauts de 1")]
    nxt = {1: "points : pro, arrivées des sauts de 2", 2: "points : pro, arrivées des sauts de 4", 4: ""}
    for k in (1, 2, 4):
        new = (pro & east(gen, k)) - gen
        gen |= new
        if k < 4:
            pro &= east(pro, k)
        stages.append((set(gen), set(pro) if k < 4 else set(), [(x - k, x) for x in sorted(new)],
                       f"tour {(1, 2, 4).index(k) + 1} : sauts de {k}", nxt[k]))
    result = east(gen, 1) & free
    expect = set().union(*(R.line(p, (0, 1), free) for p in us))
    assert result == expect, (sorted(result), sorted(expect))
    fig = Fig()
    W, H = 6.3, -6.9
    pos = [(0, 0), (W, 0), (2 * W, 0), (0, H), (W, H)]
    for (g, pr, arrows, title, sub), (ox, oy) in zip(stages, pos):
        extra = []
        for s in pr:
            x, y = xy(s)
            extra.append(f"\\fill[accent2] ({x + 0.17:.3f},{y + 0.17:.3f}) circle (0.065);")
        for a, b in arrows:
            (x1, y1), (x2, y2) = xy(a), xy(b)
            extra.append(f"\\draw[-{{Stealth[length=1.6mm]}},accent,line width=.9pt] ({x1:.3f},{y1 + 0.1:.3f}) "
                         f"to[bend left={40 if b - a == 1 else 22}] ({x2:.3f},{y2 + 0.1:.3f});")
        if sub:
            extra.append(f"\\node[tiny] at ({4 * S:.3f},-0.55) {{{sub}}};")
        # the title above the arrows of rank 8 (every panel at the same height)
        extra.append(f"\\node[plain,anchor=base] at ({4 * S:.3f},{8 * S + 0.55:.3f}) {{{title}}};")
        fig.board(us, them, free, ox=ox, oy=oy, fills={s: "me" for s in g}, extra="\n".join(extra))
    fig.board(us, them, free, ox=pos[4][0], oy=H, fills={s: "soft3" for s in result},
              extra=f"\\node[tiny] at ({4 * S:.3f},-0.55) {{les coups vers la droite de chaque pièce}};"
                    f"\\node[plain,anchor=base] at ({4 * S:.3f},{8 * S + 0.55:.3f}) {{résultat : un pas de plus, cases libres}};")
    leg = legend_line(0.0, 0, [(MINE, "moi"), (HIS, "lui"), (HOLE, "trou"),
                               (swatch("me"), "gen : cases déjà atteintes"),
                               ("\\fill[accent2] (0,0) circle (0.065);", "pro : arrivée permise au tour suivant"),
                               ("\\draw[-{Stealth[length=1.6mm]},accent,line width=.9pt] (-0.25,0) -- (0.2,0);",
                                "saut ajouté à ce tour"),
                               (swatch("soft3"), "résultat : les coups vers la droite")])
    fig.raw(2 * W + 0.2, leg, oy=H + 8 * S - 0.4)
    fig.write("fig-occluded")
    print("occluded fill: result", sorted(name_of(s) for s in result))


if __name__ == "__main__":
    fig_occluded()
    fig_corr()
    fig_queen()
    fig_first()
    fig_influence()
    fig_connect()
    fig_encircle()
    for name, pos in (("ENDGAME", ENDGAME), ("MIDDLE", MIDDLE), ("LATE", LATE), ("ENC", ENC)):
        print(name, "all 92:", R.compute_all(*pos))
