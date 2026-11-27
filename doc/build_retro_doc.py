#!/usr/bin/env python3
"""build_retro_doc.py: doc/retrograde.html, the retrograde analysis of private regions.

    python3 doc/build_retro_doc.py      (from the repository root)

The page reuses the CSS and the scripts of doc/az_gen_loop.html (as
build_ybwc_doc.py); its code excerpts are cut from the current sources
(player/solo_region.{h,cpp}, player/endgame_solver.cpp, test/solo_retro_main.cpp,
test/solo_retro_gpu.cu, player/alphazero_mcts.cpp, player/minmax_nnue_dev_player.cpp),
so re-running it after a change keeps them exact.
"""
import html

T = open('doc/az_gen_loop.html').read().split('\n')
head = '\n'.join(T[:142]).replace('<title>Yolah Generations Loop</title>', '<title>Yolah Retrograde Analysis</title>')
style_end = T[142]
tail_start = next(i for i, l in enumerate(T) if l.startswith('<svg width="0" height="0"'))
diag_start = next(i for i, l in enumerate(T) if l.startswith('/* 1.2 one generation */'))
rer = next(i for i, l in enumerate(T) if l.startswith('const rerender'))
tail_a = '\n'.join(T[tail_start:diag_start])
tail_b = '\n'.join(T[rer:])

solo_h = open('player/solo_region.h').read()
solo = open('player/solo_region.cpp').read()
solver = open('player/endgame_solver.cpp').read()
retro = open('test/solo_retro_main.cpp').read()
gpu = open('test/solo_retro_gpu.cu').read()
mcts = open('player/alphazero_mcts.cpp').read()
dev = open('player/minmax_nnue_dev_player.cpp').read()


def func(text, start):
    i = text.index(start)
    j = text.index('\n}\n', i)
    return text[i:j + 2]


def between(text, a, b):
    i = text.index(a)
    j = text.index(b, i)
    return text[i:j]


def code(path, title, body, lang='cpp'):
    return f'<script type="text/x-code" data-lang="{lang}" data-title="{html.escape(path)} · {html.escape(title)}">\n{body.rstrip()}\n</script>'


c_canon = func(solo, 'Key canonical(uint64_t free, uint64_t pieces) {')
c_clusters = func(solo, 'int clusters(uint64_t free, uint64_t pieces, Cluster out[64]) {')
c_value = func(solo, 'int Solver::cluster_value(uint64_t free, uint64_t pieces) {')
c_layers = between(retro, '    for (int n = 1; n <= MAX_FREE; n++) {', '    // Check: a sample against a fresh solver')
c_children = func(gpu, '__global__ void children_kernel(')
c_lookup = func(gpu, '__device__ int lookup(')
c_valuek = func(gpu, '__global__ void value_kernel(')
c_descent = between(gpu, '        const size_t chunk = size_t(1) << 20;', '        std::printf("layer %2d')
c_step = func(solver, 'bool EndgameSolver::regions_step(')
c_settled = func(mcts, 'bool Search::settled(const Yolah& state) const {')
c_harvest = func(dev, 'int MinMaxNNUE_DevPlayer::harvest(')

body = f'''<div class="page">
<nav class="toc" aria-label="Contents">
  <div class="toc-title">Contents</div>
  <ol id="toc-list"></ol>
</nav>
<main>
<header class="hero">
  <div class="eyebrow">Yolah · retrograde analysis · player/solo_region.{{h,cpp}} · test/solo_retro_main.cpp · test/solo_retro_gpu.cu</div>
  <h1>Yolah Retrograde Analysis</h1>
  <p class="lede">Chess endgame tables count the positions with few pieces. In Yolah eight pieces stay to the end and only the free space shrinks: tables of whole positions are out of reach. What is small is the <strong>private region</strong>, the free squares one side only can still reach. This page solves them exactly, builds their table backwards, layer by layer, on the CPU and on the GPU, and uses them in the exact solver, the minimax and the MCTS. The code, commented line by line in French: <code>doc/final_retro.{{h,cpp}}</code> and <code>doc/final_retro_gpu.cu</code>.</p>
  <div class="stat-row">
    <div class="stat"><b>97 %</b><span>of the private regions of real games have at most 6 squares</span></div>
    <div class="stat"><b>88.8 M</b><span>states in the table (≤ 9 squares, 1–2 pieces), built in 11 s on a GPU</span></div>
    <div class="stat"><b>×4.4</b><span>the exact solver on 8 threads, with the regions</span></div>
    <div class="stat"><b>0</b><span>mismatches in the checks</span></div>
  </div>
</header>
<details class="mobile-toc"><summary>Contents</summary><ol id="toc-list-m"></ol></details>

<section id="s1">
<h2><span class="num">1</span>Why regions</h2>
<div class="col">
<p>A table of every position with only 4 free squares would hold about 1.4 × 10<sup>16</sup> entries (<code>doc/solving.qmd</code>): the holes that bound the free space can be anywhere. But late positions fall apart. A queen move only crosses free squares, and two consecutive squares of a line are neighbours (8-connectivity): a piece only ever moves inside the <strong>regions</strong> of free squares it touches, and regions never merge, since squares only stop being free. A region touched by the pieces of one side only is <strong>private</strong> for good: an opponent piece can only move to free squares of its own regions, and a square next to ours would belong to ours.</p>
<p>Measured on 12 000 games of the dev player's data (<code>data/games</code>):</p>
</div>
<div class="tbl wide"><table>
<thead><tr><th class="num-r">Free squares</th><th class="num-r">Free space that is private</th><th class="num-r">Positions entirely private</th><th class="num-r">Largest private region (mean)</th></tr></thead>
<tbody>
<tr><td class="num-r">35</td><td class="num-r">1.1 %</td><td class="num-r">0 %</td><td class="num-r">0.4</td></tr>
<tr><td class="num-r">30</td><td class="num-r">5.0 %</td><td class="num-r">0 %</td><td class="num-r">1.3</td></tr>
<tr><td class="num-r">25</td><td class="num-r">17.9 %</td><td class="num-r">0.2 %</td><td class="num-r">3.1</td></tr>
<tr><td class="num-r">20</td><td class="num-r">44.2 %</td><td class="num-r">5.5 %</td><td class="num-r">4.4</td></tr>
<tr><td class="num-r">15</td><td class="num-r">70.3 %</td><td class="num-r">39.5 %</td><td class="num-r">4.4</td></tr>
<tr><td class="num-r">10</td><td class="num-r">77.0 %</td><td class="num-r">84.1 %</td><td class="num-r">3.4</td></tr>
</tbody></table></div>
<div class="col">
<p>The private regions are small: 42 % have 1 square, 88 % at most 4, 97 % at most 6, 99.8 % at most 10; 85 % are touched by one piece, 13 % by two.</p>
</div>
</section>

<section id="s2">
<h2><span class="num">2</span>Clusters and reserves</h2>
<div class="col">
<p>A piece touching two regions will go into one <em>or</em> the other: such regions are not independent. A <strong>cluster</strong> is a set of regions joined by the pieces that touch several of them, with those pieces (union-find over the regions). A private cluster is a one-player puzzle; its <strong>value</strong> is the largest number of moves its side can still make in it. It depends neither on the rest of the board nor on <em>when</em> the moves are made: a side whose private clusters are worth M owns M points and M tempi it can spend whenever it likes, a <strong>reserve</strong>.</p>
</div>
<div class="diagram wide" id="d-cluster"></div>
<div class="col">
<p>Two exact consequences, used by the players:</p>
<ul>
<li>a position where nothing is contested is settled: its remaining value is the reserve of the side to move minus the opponent's, without any search (checked on 11 585 all-private positions of real games against the exact endgame solver: 0 mismatch);</li>
<li>in a private cluster of the side to move, one move is enough to search: one that keeps the largest reserve. Two such moves lead to equivalent positions (only the reserve counts); the other moves of the cluster shrink it.</li>
</ul>
</div>
{code('player/solo_region.cpp', 'clusters', c_clusters)}
</section>

<section id="s3">
<h2><span class="num">3</span>The value of a cluster</h2>
<div class="col">
<p>value(cluster) = max over its moves of 1 + value(what is left), 0 without a move. After a move the cluster may split: the values of the parts add up. The search keeps its results in memory, keyed by the <strong>canonical form</strong>: queen lines depend neither on where a shape stands nor on the 8 symmetries of the square, so the key is the smallest of the 8 transformed shapes (built from three reflections: ranks, files, diagonal), translated to the corner. And it stops as soon as a sequence fills every square: nothing can do better (82 % of the table's states fill their region). Measured: 0.2 µs per position of a real game for all its clusters, with a direct-mapped cache of the clusters as they stand on the board in front of the canonical form.</p>
</div>
<div class="diagram wide" id="d-sym"></div>
{code('player/solo_region.cpp', 'canonical', c_canon)}
{code('player/solo_region.cpp', 'Solver::cluster_value', c_value)}
</section>

<section id="s4">
<h2><span class="num">4</span>The table, backwards (CPU)</h2>
<div class="col">
<p>Retrograde analysis computes from the end of the game backwards: in chess, from the mates. Here the states are ranked in <strong>layers</strong> by their number of free squares, and a move fills one square: the value of a state of layer n only needs states of lower layers. Computing layer 1, then 2, …, each layer finds everything it needs already solved. The shapes of layer n are grown from those of layer n − 1 (one more square next to the shape, at each placement near the corner where it fits on the board); the pieces stand on the squares around. The shape counts are those of the literature for 8-connected polyominoes (“polyplets”): 1, 2, 5, 22, 94, 524, 3 031, 18 770, 116 452 (with the board's limits) for 1 to 9 squares.</p>
</div>
{code('test/solo_retro_main.cpp', 'the layers', c_layers)}
<div class="tbl wide"><table>
<thead><tr><th class="num-r">Layer (free squares)</th><th class="num-r">Shapes</th><th class="num-r">States (1–2 pieces)</th><th class="num-r">Exceptions (value &lt; squares)</th></tr></thead>
<tbody>
<tr><td class="num-r">6</td><td class="num-r">524</td><td class="num-r">119 673</td><td class="num-r">7 884</td></tr>
<tr><td class="num-r">7</td><td class="num-r">3 031</td><td class="num-r">912 515</td><td class="num-r">89 226</td></tr>
<tr><td class="num-r">8</td><td class="num-r">18 770</td><td class="num-r">6 848 857</td><td class="num-r">951 214</td></tr>
<tr><td class="num-r">9</td><td class="num-r">116 452</td><td class="num-r">49 671 530</td><td class="num-r">9 286 045</td></tr>
</tbody></table></div>
<div class="col">
<p>Up to 9 squares: 57.6 M states in 3 minutes on one core (4.7 GB). Since most states fill their region, the table keeps only the exceptions (10.3 M, 175 MB in <code>data/solo_regions_9_2.bin</code>), the value of the others being their number of squares.</p>
</div>
</section>

<section id="s5">
<h2><span class="num">5</span>The same table on the GPU</h2>
<div class="col">
<p>A GPU runs the same small program (a <em>kernel</em>) on thousands of threads at once, each finding its own element from its block and thread numbers; it has its own memory, and the copies from and to it cost. A recursive search with a shared hash table does not fit; layers do. Three steps:</p>
<ol>
<li><strong>seeds</strong> (CPU): every single-region state, canonical;</li>
<li><strong>descent</strong>, layer N to 1: one thread per state plays every move and writes its children, normalised (the free squares no piece can reach and the pieces that cannot move removed: they change nothing) and canonical; sorted and deduplicated on the GPU (thrust), they join the layer of their number of squares. Every layer thus holds what the layers above will need;</li>
<li><strong>ascent</strong>, layer 1 to N: one thread per state tries every move, finds the child in its layer by binary search (the layers are sorted), and keeps max(1 + its value).</li>
</ol>
</div>
<div class="diagram wide" id="d-gpu"></div>
{code('test/solo_retro_gpu.cu', 'children_kernel', c_children, 'cpp')}
{code('test/solo_retro_gpu.cu', 'the descent: compact, sort, keep the next layer on the GPU', c_descent, 'cpp')}
{code('test/solo_retro_gpu.cu', 'lookup', c_lookup, 'cpp')}
{code('test/solo_retro_gpu.cu', 'value_kernel', c_valuek, 'cpp')}
<div class="tbl wide"><table>
<thead><tr><th>Table</th><th class="num-r">States</th><th class="num-r">GPU descent</th><th class="num-r">GPU values</th><th class="num-r">CPU</th></tr></thead>
<tbody>
<tr><td>≤ 8 squares, 1–2 pieces</td><td class="num-r">11.8 M</td><td class="num-r">1.4 s</td><td class="num-r">0.02 s</td><td class="num-r">18.6 s (6.8 M single-region states)</td></tr>
<tr><td>≤ 9 squares, 1–2 pieces</td><td class="num-r">88.8 M</td><td class="num-r">11 s</td><td class="num-r">0.14 s</td><td class="num-r">181 s (57.6 M)</td></tr>
<tr><td>≤ 10 squares, 1 piece</td><td class="num-r">30.2 M</td><td class="num-r">3.4 s</td><td class="num-r">0.05 s</td><td class="num-r">—</td></tr>
</tbody></table></div>
<div class="col">
<p>(RTX 3080; the CPU enumeration of the seeds, single-threaded, adds 20 s at 9 squares.) The first GPU version took 556 s at 9 squares, slower than the CPU: each chunk sorted 108 slots per state, almost all empty, and sent every chunk's children back to the CPU, duplicates between chunks included. Compacting before sorting (<code>remove_if</code>) and merging the next layer's children on the GPU (<code>merge</code> + <code>unique</code>) brought it to 11 s; computing the values themselves was never the problem (0.14 s for 88.8 M states). The next step up, 10 squares with 2 pieces (335 M placements), would need a parallel seed enumeration and care with the card's 10 GB.</p>
<p>Checks: the GPU's canonical form of every seed is the seed (the GPU and the CPU compute the same forms), no child is ever missing, and 2 000 states of each layer recomputed by the CPU solver give the same values.</p>
</div>
</section>

<section id="s6">
<h2><span class="num">6</span>In the players</h2>
<h3 id="s6-solver"><span class="num">6.1</span>The exact solver</h3>
<div class="col">
<p>Option <code>regions</code> of <code>EndgameSolver</code> (on in the app: “Vérifier”, the hints, the early end of the puzzles, “Solveur exact”): the leaf when nothing is contested, and one move per private cluster; a cheap test first (the squares each side can reach), the cluster solver of each thread kept between solves.</p>
</div>
{code('player/endgame_solver.cpp', 'regions_step', c_step)}
<div class="tbl wide"><table>
<thead><tr><th>100 positions of 22 to 26 free squares (real games)</th><th class="num-r">without</th><th class="num-r">with regions</th><th class="num-r">speed-up</th></tr></thead>
<tbody>
<tr><td>1 thread</td><td class="num-r">150 s, 2 599 M nodes</td><td class="num-r">61 s, 336 M nodes</td><td class="num-r">2.5</td></tr>
<tr><td>8 threads</td><td class="num-r">67 s, 10 115 M nodes</td><td class="num-r">15 s, 501 M nodes</td><td class="num-r">4.4</td></tr>
</tbody></table></div>
<h3 id="s6-dev"><span class="num">6.2</span>The minimax (dev player)</h3>
<div class="col">
<p>The dev player already had both ideas: <code>reserve</code> (one move per sealed pocket, kept: +10 Elo in U5) and <code>split solve</code> (the exact leaf, U4 neutral, −12 with reserve in U5). Its harvests were computed by <code>harvest_exact</code>, a new memo per call. Option <code>solo</code> computes them with the cluster solver of each thread, kept between moves. On the bench (209 positions, depth 11): with <code>reserve</code>, the same search (nodes ×1.000, same moves 209/209), time ×0.96; with <code>split solve</code>, time ×1.39 the old way (why it lost), ×0.98 with <code>solo</code>. Campaign U29 (<code>config/search_study/campaign_U29.txt</code>): <code>u29_solo</code> and <code>u29_solo_split</code> against dev_u9_spsa at 0.2 s and 1 s, 2000 games each.</p>
</div>
{code('player/minmax_nnue_dev_player.cpp', 'harvest', c_harvest)}
<h3 id="s6-mcts"><span class="num">6.3</span>The MCTS</h3>
<div class="col">
<p>Option <code>regions</code> of the AlphaZero search: a leaf where nothing is contested is a finished game for the search (never expanded), worth the exact final difference with the same score utility as the real ends; the root keeps the usual test, so a move is always chosen. Match: <code>nnue/az_regions_match.sh</code> (the app's AlphaZero network with the option against itself without, 2000 games over 5 GPUs, both with the app's score utility).</p>
</div>
{code('player/alphazero_mcts.cpp', 'settled', c_settled)}
</section>

<section id="s7">
<h2><span class="num">7</span>Checks and files</h2>
<div class="col">
<ul>
<li><code>test/solo_region_check.cpp</code>: the cluster values against the exact endgame solver on the all-private positions of real games (7 744 / 2 000 games, 0 mismatch).</li>
<li><code>test/solo_retro_main.cpp</code>: the CPU table, a sample against a fresh solver (5 000 states, 0 wrong).</li>
<li><code>test/solo_retro_gpu.sh</code>: the GPU table, its canonical forms against the CPU's, its values against the CPU solver (18 000 states at 9 squares, 0 wrong).</li>
<li><code>test/final_retro_check.sh</code>: the commented copy <code>doc/final_retro.cpp</code> against the real code (55 278 positions: same splits, canonical forms and values; 11 585 all-private positions against the exact solver; its table of 1.05 M states against the real solver: 0 mismatch).</li>
<li><code>test/final_retro_gpu_check.sh</code>: the commented copy <code>doc/final_retro_gpu.cu</code> gives the same output as <code>test/solo_retro_gpu.cu</code>.</li>
<li>The exact solver with <code>regions</code>: 2 983 positions of 9 to 12 free squares against a plain negamax, 500 positions of 22 to 26 against the solver without it (0 difference).</li>
</ul>
</div>
</section>
<footer class="col" style="margin-top:56px; padding-top:14px; border-top:1px solid var(--rule); font-family:var(--font-display); font-size:13px; color:var(--ink-3)">
Repository state: November 2026, the retrograde analysis project. Measures on an i7-12700K and an RTX 3080 (WSL2).
</footer>
</main>
</div>'''

diagrams = r'''
/* 2 a cluster */
(function () {
  let s = "";
  const q = 30, x0 = 20, y0 = 30;
  // a 6 x 4 board: F free, H hole, N black piece, W white piece
  const rows = ["FFHHFF", "FNHHFF", "HHHHWF", "FFHFFF"];
  rows.forEach((r, j) => r.split("").forEach((c, i) => {
    const k = c === "F" ? "" : c === "H" ? "k-dim" : c === "N" ? "k-b" : "k-w";
    s += `<rect class="dg-box ${k}" x="${x0 + i * q}" y="${y0 + j * q}" width="${q - 2}" height="${q - 2}" rx="4"/>`;
    if (c === "N" || c === "W") s += T(x0 + i * q + q / 2 - 1, y0 + j * q + q / 2 + 4, c === "N" ? "B" : "W", "dg-t", "middle");
  }));
  s += T(x0 + 200, y0 + 12, "left region: touched by the black piece only: PRIVATE", "dg-lbl");
  s += T(x0 + 200, y0 + 30, "its value: the most moves black can still make there (3: it fills it)", "dg-s");
  s += T(x0 + 200, y0 + 56, "right region: touched by the white piece only: private to white", "dg-lbl");
  s += T(x0 + 200, y0 + 92, "nothing contested: remaining = reserve(side to move) − reserve(opponent)", "dg-ret");
  svg("d-cluster", 720, 160, s, "private regions");
})();

/* 3 the 8 symmetries */
(function () {
  let s = "";
  const shape = [[0,0],[1,0],[2,0],[2,1]];
  const tf = [p => p, p => [2 - p[0], p[1]], p => [p[0], 2 - p[1]], p => [2 - p[0], 2 - p[1]],
              p => [p[1], p[0]], p => [2 - p[1], p[0]], p => [p[1], 2 - p[0]], p => [2 - p[1], 2 - p[0]]];
  tf.forEach((f, k) => {
    const pts = shape.map(f);
    const mx = Math.min(...pts.map(p => p[0])), my = Math.min(...pts.map(p => p[1]));
    const ox = 14 + k * 86, oy = 20;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += `<rect class="dg-box k-dim" x="${ox + i * 22}" y="${oy + j * 22}" width="20" height="20" rx="3"/>`;
    pts.forEach(p => { s += `<rect class="dg-box ${k === 0 ? "k-acc" : ""}" x="${ox + (p[0] - mx) * 22}" y="${oy + (2 - (p[1] - my)) * 22}" width="20" height="20" rx="3"/>`; });
  });
  s += T(14, 112, "the same shape under the 8 symmetries, each translated to the corner: the smallest bitboard is its key", "dg-lbl");
  svg("d-sym", 720, 125, s, "the 8 symmetries");
})();

/* 5 GPU */
(function () {
  let s = "";
  s += B(10, 20, 170, 50, "seeds (CPU)", ["single-region states"], "");
  s += A(180, 45, 220, 45);
  s += B(220, 10, 220, 70, "descent N → 1 (GPU)", ["1 thread per state: children,", "normalised, canonical; sort, unique"], "k-w");
  s += A(440, 45, 480, 45);
  s += B(480, 10, 230, 70, "ascent 1 → N (GPU)", ["1 thread per state: max over moves", "of 1 + value(child) (binary search)"], "k-acc");
  s += T(220, 104, "layer n only needs the layers below: a move fills one square", "dg-lbl");
  svg("d-gpu", 720, 115, s, "the GPU table");
})();
'''

out = head + '\n' + style_end + '\n' + body + '\n' + tail_a + '\n' + diagrams + '\n' + tail_b + '\n'
open('doc/retrograde.html', 'w').write(out)
print(len(out))
