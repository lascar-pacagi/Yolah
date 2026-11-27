#!/usr/bin/env python3
"""build_ybwc_doc.py: doc/endgame_ybwc.html, the parallel endgame solver.

    python3 doc/build_ybwc_doc.py      (from the repository root)

The page reuses the CSS and the scripts of doc/az_gen_loop.html; its code
excerpts are cut from the current sources (player/endgame_solver.{h,cpp},
app/yolah_app.cpp), so re-running it after a change keeps them exact.
"""
import re, html
T = open('doc/az_gen_loop.html').read().split('\n')
head = '\n'.join(T[:142]).replace('<title>Yolah Generations Loop</title>', '<title>Yolah Parallel Solver</title>')
style_end = T[142]  # </style>
tail_start = next(i for i, l in enumerate(T) if l.startswith('<svg width="0" height="0"'))
diag_start = next(i for i, l in enumerate(T) if l.startswith('/* 1.2 one generation */'))
rer = next(i for i, l in enumerate(T) if l.startswith('const rerender'))
tail_a = '\n'.join(T[tail_start:diag_start])
tail_b = '\n'.join(T[rer:])

src = open('player/endgame_solver.cpp').read()
hdr = open('player/endgame_solver.h').read()
app = open('app/yolah_app.cpp').read()
def func(text, start):
    i = text.index(start)
    j = text.index('\n}\n', i)
    return text[i:j + 2]
def between(text, a, b):
    i = text.index(a); j = text.index(b, i)
    return text[i:j]
def code(path, title, body, lang='cpp'):
    return f'<script type="text/x-code" data-lang="{lang}" data-title="{html.escape(path)} · {html.escape(title)}">\n{body.rstrip()}\n</script>'

c_split_cond = between(src, '    for (size_t i = 0; i < moves.size(); i++) {\n        // Young brothers wait', '        const Move m = moves[i];\n        const int v = move_value(w, yolah, hash, m, alpha, beta);\n        if (stopped(w)) return 0;')
c_split = func(src, 'void EndgameSolver::split(')
c_work = func(src, 'void EndgameSolver::work_on(')
c_help = func(src, 'bool EndgameSolver::help(')
c_loop = func(src, 'void EndgameSolver::helper_loop(')
c_store = func(src, 'void EndgameSolver::store(')
c_sp = between(hdr, '    struct SplitPoint {', '    Options options;')
c_entry = between(hdr, '    struct Entry {', '    // The bounds of the position `hash`')
c_probe = between(hdr, '    bool probe(uint64_t hash', '    // A node whose remaining moves')
c_stopped = between(hdr, '    bool stopped(const Worker& w) const {', '    int  search(Worker& w')
c_gain = between(hdr, '    int move_gain(Move m) const {', '    // The search must stop')
c_targets = between(src, '    if (options.goal != EndgameGoal::Points) {', '    } else if (!pass_rule_now) {')
c_keys = between(app, '        if (pos.current_player() == human && feasible) {', '            if (feasible) {\n                std::string list;')

body = f'''<div class="page">
<nav class="toc" aria-label="Contents">
  <div class="toc-title">Contents</div>
  <ol id="toc-list"></ol>
</nav>
<main>
<header class="hero">
  <div class="eyebrow">Yolah · exact endgame solver · player/endgame_solver.{{h,cpp}} · the editor's “Vérifier”</div>
  <h1>Yolah Parallel Endgame Solver</h1>
  <p class="lede">How the exact solver behind the puzzle editor's “Vérifier” shares its search between the cores: the “young brothers wait” scheme of the parallel chess and Othello solvers, a transposition table shared without a lock, the target goals of the puzzles, and the key moves proved in parallel. The sequential solver it builds on is explained in <code>doc/final_solver.{{h,cpp}}</code>; this one, line by line in French, in <code>doc/final_ybwc.{{h,cpp}}</code>.</p>
  <div class="stat-row">
    <div class="stat"><b>×4 – ×5.6</b><span>“Vérifier” on puzzles of 26 to 28 free squares</span></div>
    <div class="stat"><b>8</b><span>threads at most for the main proof</span></div>
    <div class="stat"><b>12</b><span>free squares at least for a node to be shared</span></div>
    <div class="stat"><b>0</b><span>mismatches in the checks (values, moves, nodes)</span></div>
  </div>
</header>
<details class="mobile-toc"><summary>Contents</summary><ol id="toc-list-m"></ol></details>

<section id="s1">
<h2><span class="num">1</span>What the solver proves</h2>
<div class="col">
<p>Every move fills a free square and frees none (its origin becomes a hole), so a position with k free squares ends in at most k moves: when k is small enough, the game can be searched to its very end, and the result is a proof. The solver computes the <strong>remaining</strong> value of a position, what the side to move will still gain minus what the opponent will, with perfect play. It depends on the board and the side to move only, not on the scores so far, so the transposition table can share it between positions reached with different scores.</p>
<p>The search is a fail-soft alpha-beta. For a puzzle, only the sign of the final difference matters (win, draw, loss), which a narrow window around 0 decides far faster than the exact value; “win by N” is checked as “win” with N − 1 more points for the opponent. All of this is the sequential solver of <code>doc/final_solver.h</code>, unchanged. What follows is what makes it parallel.</p>
</div>
<div class="tbl wide"><table>
<thead><tr><th>Where</th><th>What</th></tr></thead>
<tbody>
<tr><td><code>player/endgame_solver.{{h,cpp}}</code></td><td><code>EndgameSolver</code>: the solver, sequential with <code>threads = 1</code> (node for node the same search as before), parallel above</td></tr>
<tr><td><code>app/yolah_app.cpp</code></td><td><code>PuzzleCheck</code>, <code>puzzle_check_start</code>, <code>puzzle_lead</code>: the verdict in a thread of its own, 10 minutes at most, “Arrêter”; the key moves in parallel</td></tr>
<tr><td><code>player/solver_player.{{h,cpp}}</code></td><td>“Solveur exact”, the puzzles' opponent (<code>"nb threads": 8</code>)</td></tr>
<tr><td><code>doc/final_ybwc.{{h,cpp}}</code></td><td>the commented copy (French), checked against the real one by <code>test/final_ybwc_check.sh</code></td></tr>
<tr><td><code>player/endgame_solver_{{lazy,pro}}.{{h,cpp}}</code></td><td>the variants measured and not kept (section 7)</td></tr>
</tbody></table></div>
</section>

<section id="s2">
<h2><span class="num">2</span>Why the moves cannot simply be shared out</h2>
<div class="col">
<p>In an alpha-beta search, the first move of a node narrows the window of the next ones, and often it is enough on its own: the node is cut and the other moves are never searched. A good move order makes this the common case. Searching all the moves at once throws that away: the threads search moves that the sequential search would have skipped. Measured on 12 positions of 24 to 26 free squares: sharing out the root's moves between 8 threads is <strong>twice slower</strong> than one thread (38.1 s against 18.6 s, 2 061 M nodes against 203 M).</p>
</div>
<h3 id="s2-ybwc"><span class="num">2.1</span>Young brothers wait</h3>
<div class="col">
<p>The rule of the parallel chess and Othello solvers (YBWC): the <strong>eldest brother</strong>, the first move of a node in the move order, is always searched alone, by the thread that is at the node. Only if it has not cut, the <strong>young brothers</strong>, the other moves, may be shared: the node becomes a <em>split point</em>, and the idle threads come and take its moves one by one. A node is shared only when it has at least 12 free squares (below, its subtree costs less than organising the sharing) and when some thread is idle.</p>
</div>
{code('player/endgame_solver.cpp', 'search: the move loop', c_split_cond)}
<div class="diagram wide" id="d-split"></div>
<div class="col">
<p>Each thread searches its move on its own copy of the position and reports the value under the split point's lock: the best value, alpha raised, and, when a value reaches beta, the cutoff. A value obtained with an older, lower alpha is still correct: below that alpha it is an upper bound that changes nothing, above it is exact.</p>
</div>
{code('player/endgame_solver.h', 'SplitPoint and Worker', c_sp)}
{code('player/endgame_solver.cpp', 'work_on', c_work)}
</section>

<section id="s3">
<h2><span class="num">3</span>Cutoffs and the chain of split points</h2>
<div class="col">
<p>When one move of a split point reaches beta, the others are useless. The split point raises its <code>cutoff</code> flag, and every thread working <em>below</em> that node, however deep, must stop. A thread may work under several split points at once (it helps at one, then splits a node of its own subtree), so each split point knows its parent, and <code>stopped()</code>, called at every node, walks the chain.</p>
</div>
<div class="diagram wide" id="d-chain"></div>
{code('player/endgame_solver.h', 'stopped', c_stopped)}
<div class="col">
<p>A thread stopped this way returns a meaningless value. The search tests <code>stopped()</code> after every move: such a value is never compared, and the node is never stored in the table.</p>
</div>
</section>

<section id="s4">
<h2><span class="num">4</span>The master, its helpers, and the end of a split point</h2>
<div class="col">
<p>The split point lives on the stack of the thread that created it, its <em>master</em>: the node cannot be left while a thread still searches one of its moves (<code>workers</code> &gt; 0). The master takes moves like the others; when none is left, it does not wait idle: it helps at the split points opened <em>below</em> its own, by its helpers, the only ones whose end it waits for.</p>
<p>Two orders matter. A helper joins under the global lock and increments <code>workers</code> before releasing it: the master cannot leave without it. And the master <strong>closes</strong> the split point (no more moves handed out) before it waits: <code>work_on</code> also returns when the search above is stopped while moves remain, and a thread joining at that moment would work on a split point about to disappear. A helper's last access is the decrement itself, without the split point's lock: once the count is 0 the master may return.</p>
</div>
{code('player/endgame_solver.cpp', 'split', c_split)}
{code('player/endgame_solver.cpp', 'help', c_help)}
{code('player/endgame_solver.cpp', 'helper_loop', c_loop)}
</section>

<section id="s5">
<h2><span class="num">5</span>The table shared without a lock</h2>
<div class="col">
<p>All the threads read and write the same table at every node; a lock would cost too much. An entry is two 64-bit words written one after the other: the data (bounds, move, a “valid” bit) and a check word, the key XOR the data. A reader accepts an entry only if check XOR data gives back the key. When two threads write the same slot at once, a reader may get the data of one and the check of the other: the test fails and the entry is ignored. A torn entry can never pass for a true bound, so the proofs stay exact (the technique of Hyatt and Mann). The valid bit tells an empty slot (two zeros) from an entry: without it, a position whose key is 0 would read the bounds [0, 0].</p>
</div>
<div class="diagram wide" id="d-entry"></div>
{code('player/endgame_solver.h', 'Entry, pack', c_entry)}
{code('player/endgame_solver.h', 'probe', c_probe)}
{code('player/endgame_solver.cpp', 'store', c_store)}
</section>

<section id="s6">
<h2><span class="num">6</span>Target goals and the verdict</h2>
<div class="col">
<p>A puzzle can mark target squares; a target is taken by the piece that lands on it, once. Besides the points, two goals use them: take more targets than the opponent (<em>Collect</em>), or be the first to land on one (<em>First</em>: that move wins and ends the game). For the solver only the gain of a move changes. The remaining value still depends on the board alone (the targets are fixed for the solver's life), so the table works as it is. Two bounds are new: when no target is in the free regions next to either side's pieces, nothing more can be gained (exactly 0); with Collect, a side blocked for good leaves the other at most the targets of its regions.</p>
</div>
{code('player/endgame_solver.h', 'move_gain', c_gain)}
{code('player/endgame_solver.cpp', 'search: the bounds of the target goals', c_targets)}
<h3 id="s6-keys"><span class="num">6.1</span>The key moves in parallel</h3>
<div class="col">
<p>When the puzzle is feasible and the player moves first, the editor counts the <strong>key moves</strong>, the moves after which the goal is still reached: one proof per move, all independent. They are spread over the cores, each thread with its own single-threaded solver and table, taking the next move until none is left. When the puzzle is impossible there is no key move, and none is searched.</p>
</div>
{code('app/yolah_app.cpp', 'puzzle_check_start: the key moves', c_keys)}
</section>

<section id="s7">
<h2><span class="num">7</span>Measurements</h2>
<h3 id="s7-variants"><span class="num">7.1</span>Four ways to use the cores</h3>
<div class="col">
<p>12 random positions of 24 to 26 free squares, win/draw/loss, one bench run (i7-12700K, 20 hardware threads). All give the same values.</p>
</div>
<div class="tbl wide"><table>
<thead><tr><th>Solver</th><th class="num-r">Time</th><th class="num-r">Nodes</th><th class="num-r">Speed-up</th></tr></thead>
<tbody>
<tr><td>sequential</td><td class="num-r">18.6 s</td><td class="num-r">203 M</td><td class="num-r">1</td></tr>
<tr><td>root moves shared out, 8 threads (<code>EndgameSolverLazy</code>, <code>root_split</code>)</td><td class="num-r">38.1 s</td><td class="num-r">2 061 M</td><td class="num-r">0.5</td></tr>
<tr><td>Lazy SMP, 4 / 8 threads (<code>EndgameSolverLazy</code>): every thread searches the whole root, only the table is shared</td><td class="num-r">12.0 / 14.7 s</td><td class="num-r">616 / 1 375 M</td><td class="num-r">1.6 / 1.3</td></tr>
<tr><td><strong>YBWC, 4 / 8 threads (<code>EndgameSolver</code>)</strong></td><td class="num-r"><strong>5.9 / 5.3 s</strong></td><td class="num-r">286 / 440 M</td><td class="num-r"><strong>3.2 / 3.5</strong></td></tr>
<tr><td>YBWC + ETC + ordering by short searches, 1 / 4 / 8 (<code>EndgameSolverPro</code>)</td><td class="num-r">17.7 / 7.5 / 7.0 s</td><td class="num-r">208 / 313 / 502 M</td><td class="num-r">1.05 / 2.5 / 2.7</td></tr>
</tbody></table></div>
<div class="col">
<p>Lazy SMP is the simple scheme of the dev minimax player; here its threads search the same subtrees, and more threads hardly help. ETC (a child the table already proves bad enough for the opponent cuts the node before any search) and the ordering by short searches (each move first gets a small node budget; a move proven to reach beta cuts, the moves proven to fail go last) are techniques of the Othello solver Edax: 5 % faster alone, slower with threads, since the short searches are wasted work too.</p>
</div>
<h3 id="s7-threads"><span class="num">7.2</span>Threads, and the false sharing</h3>
<div class="col">
<p>The extra nodes of the parallel search are the moves already started when another one wins: 203 M at one thread, 286 M at 4, 440 M at 8, 676 M at 16. Beyond 8 threads this wasted work grows faster than the cores it uses (16 threads: 7.5 s against 4.9 s at 8, in an earlier run), so the editor stops at 8.</p>
<p>The first version reached only 1.9 times at 8 threads. A core reads memory by lines of 64 bytes: the threads' node counters, written at every node, were neighbours in memory, and the cutoff flag, read at every node by every thread, shared its line with the split point's lock. Each write evicted the line from all the other cores' caches. Giving each its own line (<code>alignas(64)</code>) brought 3.3.</p>
</div>
<h3 id="s7-puzzles"><span class="num">7.3</span>“Vérifier” on puzzles</h3>
<div class="tbl wide"><table>
<thead><tr><th>Puzzle (random positions)</th><th class="num-r">All cores</th><th class="num-r">One core</th><th class="num-r">Speed-up</th></tr></thead>
<tbody>
<tr><td>28 free squares, impossible</td><td class="num-r">2.85 s</td><td class="num-r">15.87 s</td><td class="num-r">5.6</td></tr>
<tr><td>28 free squares, impossible (draw)</td><td class="num-r">1.43 s</td><td class="num-r">7.19 s</td><td class="num-r">5.0</td></tr>
<tr><td>28 free squares, feasible, 8 key moves out of 19</td><td class="num-r">3.19 s</td><td class="num-r">14.58 s</td><td class="num-r">4.6</td></tr>
<tr><td>28 free squares, feasible, 19 key moves out of 23</td><td class="num-r">11.69 s</td><td class="num-r">58.07 s</td><td class="num-r">5.0</td></tr>
<tr><td>26 free squares, feasible, 14 key moves out of 18</td><td class="num-r">1.1 – 1.4 s</td><td class="num-r">5.4 – 6.1 s</td><td class="num-r">≈ 4.5</td></tr>
</tbody></table></div>
</section>

<section id="s9">
<h2><span class="num">8</span>Private regions as exact reserves</h2>
<div class="col">
<p>Option <code>regions</code> (Points goal; on in the app). A region of free squares only one side can reach is private for good, and its value, the largest number of moves that side can still make in it, is known exactly by the cluster solver of <code>player/solo_region.{{h,cpp}}</code> (regions linked by a piece touching several, solved with memory, canonical up to translation and the 8 symmetries; built by retrograde analysis for every region of up to 9 squares with 1 or 2 pieces by <code>test/solo_retro_main.cpp</code>). Two exact shortcuts follow: a position with no contested region left is a leaf (remaining = reserve of the side to move − the opponent's), and in each private cluster of the side to move one move is searched, one that keeps the largest reserve: the other moves lead to equivalent positions or to smaller reserves. A cheap test first (the squares each side can reach: no private region, nothing to do), and a direct-mapped cache of the clusters as they stand.</p>
</div>
<div class="tbl wide"><table>
<thead><tr><th>100 positions of 22 to 26 free squares from real games</th><th class="num-r">regions off</th><th class="num-r">on</th><th class="num-r">speed-up</th></tr></thead>
<tbody>
<tr><td>1 thread</td><td class="num-r">150 s, 2 599 M nodes</td><td class="num-r">61 s, 336 M nodes</td><td class="num-r">2.5</td></tr>
<tr><td>8 threads</td><td class="num-r">67 s, 10 115 M nodes</td><td class="num-r">15 s, 501 M nodes</td><td class="num-r">4.4</td></tr>
</tbody></table></div>
<div class="col"><p>Checked: 2 983 positions of 9 to 12 free squares against a plain negamax (0 wrong), 500 positions of 22 to 26 free squares against the solver without the option (0 sign difference); the cluster values against the exact solver on 7 744 all-private positions (0 wrong).</p></div>
</section>

<section id="s8">
<h2><span class="num">9</span>Checks</h2>
<div class="col">
<ul>
<li><code>test/final_solver_check.sh</code>: the solver with one thread, against its commented sequential copy and a plain negamax: same values, same moves, <strong>same node counts</strong> (1 000 positions, no mismatch).</li>
<li><code>test/final_ybwc_check.sh</code>: the commented parallel copy, with one thread against <code>EndgameSolver</code> (same nodes), with 8 threads against a plain negamax (values, signs, and the move reaches the value), its verdict and key moves against the negamax, the three goals; and 1 thread against 8 on bigger positions (1 000 + 50 positions, no mismatch).</li>
<li>During the development: 8 threads against 1 on positions of 16 to 22 free squares, the three goals (300 checks), under AddressSanitizer; 30 solves stopped in the middle (as “Arrêter” does), then the same solver reused, under AddressSanitizer. The two bugs found that way: a copied <code>Yolah::MoveList</code> keeps its end pointer into the original list (a split point now copies its moves into a plain array), and a thread could join a split point while its master was leaving it (it is now closed first).</li>
<li>An outside review (Gemini) found five more defects, all fixed and checked: with a node budget and several threads, a helper reaching its budget dropped its move without a word and its master could conclude without it (now one shared flag stops all: 300 limited solves at 4 threads, 54 incomplete, 0 wrong); with the target goals the value so far defaulted to the score difference instead of 0 (the moves were right, the values shown by “Solveur exact” were not); the pass rule gave a false bound with the scores of a puzzle (it now turns itself off unless the scores are those of a normal game: 1 433 positions, 0 wrong); <code>EndgameSolverPro</code> crashed without a table and ran its short searches without a budget when <code>probe_nodes</code> was 0. A sixth point, a helper falling asleep just after a split, cost up to 200 µs of delay: the split points are now counted (<code>split_gen</code>) and a helper does not sleep when one appeared.</li>
</ul>
</div>
</section>
<footer class="col" style="margin-top:56px; padding-top:14px; border-top:1px solid var(--rule); font-family:var(--font-display); font-size:13px; color:var(--ink-3)">
Repository state: November 2026, the parallel solver of the puzzle editor. Measures on an i7-12700K (WSL2).
</footer>
</main>
</div>'''

diagrams = r'''
/* 2.1 a split point */
(function () {
  let s = "";
  s += B(10, 20, 170, 56, "node N", ["4 moves: a b c d"], "k-acc");
  s += A(180, 48, 230, 48, "", "T0 searches a alone", 186, 40);
  s += B(370, 20, 200, 56, "a has not cut", ["N becomes a split point"], "");
  s += A(470, 76, 470, 106);
  s += B(250, 106, 440, 60, "split point of N (on T0's stack)", ["next →  b   c   d      alpha, best, cutoff (under its lock)"], "k-b");
  [["T0 (master)", 290], ["T1 (was idle)", 440], ["T2 (was idle)", 590]].forEach(([t, x], i) => {
    s += A(x, 166, x, 206);
    s += B(x - 70, 206, 140, 52, t, ["searches " + "bcd"[i] + " on its copy"], i ? "k-w" : "k-acc");
  });
  s += T(250, 290, "each reports its value: best, alpha raised; ≥ beta → cutoff, the others stop", "dg-lbl");
  svg("d-split", 720, 300, s, "a split point");
})();

/* 3 the chain */
(function () {
  let s = "";
  const xs = [10, 150, 330, 510];
  s += B(xs[0], 30, 110, 44, "root", "", "");
  s += B(xs[1], 30, 150, 44, "split point N", "cutoff?", "k-b");
  s += B(xs[2], 30, 150, 44, "split point M", "cutoff?", "k-b");
  s += B(xs[3], 30, 160, 44, "current node", "of thread T", "k-acc");
  s += A(120, 52, 150, 52); s += A(300, 52, 330, 52); s += A(480, 52, 510, 52);
  s += P("M590,74 L590,104 L405,104 L405,75", "acc");
  s += P("M405,104 L225,104 L225,75", "acc");
  s += T(240, 124, "stopped(): from T's split point up through its parents, at every node", "dg-lbl");
  svg("d-chain", 700, 140, s, "the chain of split points");
})();

/* 5 an entry */
(function () {
  let s = "";
  const fields = [["valid", 1, "k-acc"], ["…", 3, ""], ["move", 4, "k-w"], ["upper", 2, "k-b"], ["lower", 2, "k-b"]];
  let x = 20; const u = 46;
  s += T(20, 22, "data (64 bits)", "dg-lbl");
  fields.forEach(([n, w, k]) => { s += B(x, 30, w * u - 4, 40, n, "", k); x += w * u; });
  s += T(20, 100, "check (64 bits) = key ⊕ data", "dg-lbl");
  s += B(20, 108, 12 * u - 4, 40, "key XOR data", "", "k-dim");
  s += T(20, 176, "read: accept only if valid and check ⊕ data = key — a torn entry (halves of two writes) fails the test", "dg-lbl");
  svg("d-entry", 600, 190, s, "a table entry");
})();
'''
out = head + '\n' + style_end + '\n' + body + '\n' + tail_a + '\n' + diagrams + '\n' + tail_b + '\n'
open('doc/endgame_ybwc.html', 'w').write(out)
print(len(out))
