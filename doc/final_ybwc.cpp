// =============================================================================
//  final_ybwc.cpp — le solveur parallèle de l'éditeur de puzzles, commenté
// =============================================================================
//  Lire d'abord doc/final_ybwc.h (et, pour les bases, doc/final_solver.h).
//  Plan de ce fichier :
//    1. reach        les cases libres qu'un camp peut encore atteindre
//    2. la table     construction, lecture (probe), écriture (store)
//    3. gain, ordre  ce que rapporte un coup ; l'ordre des coups
//    4. search       l'alpha-bêta fail-soft, avec le partage
//    5. le partage   split, work_on, help, helper_loop
//    6. solve        la racine : les threads, la valeur finale
//    7. verify       le verdict d'un puzzle, les coups clés en parallèle
// =============================================================================

#include "final_ybwc.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <thread>

namespace final_ybwc {

// ── 1. Les cases libres atteignables ─────────────────────────────────────────
// Les cases libres que `pieces` pourront atteindre un jour, en glissant sur des
// cases libres : une « inondation » 8-connexe à partir des pièces (expliquée en
// détail dans doc/final_solver.cpp). grow(b) = b et ses 8 voisines, sans
// passer d'un bord du plateau à l'autre.
static uint64_t reach(uint64_t pieces, uint64_t free) {
    auto grow = [](uint64_t b) {
        constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
        return row | (row << 8) | (row >> 8);
    };
    uint64_t r = grow(pieces) & free, prev = 0;
    while (r != prev) {
        prev = r;
        r = grow(r) & free;
    }
    return r;
}

// ── 2. La table partagée ─────────────────────────────────────────────────────
// make_unique<Entry[]> : un tableau d'entrées toutes à zéro (case vide : pas
// de bit valide). Pas de std::vector : un atomique ne se copie pas.
Solver::Solver(Options opt) : options(opt) {
    if (options.tt_bits > 0) {
        table_size = size_t(1) << options.tt_bits;
        table = std::make_unique<Entry[]>(table_size);
    }
}

// Les bornes de la position `hash`, si la table les a. La case est choisie
// par les bits bas de la clé (table_size est une puissance de 2). On lit les
// deux mots ; l'entrée n'est acceptée que si elle est valide ET que check ⊕
// data redonne la clé (sinon : autre position, ou entrée déchirée par deux
// écritures simultanées). memory_order_relaxed : aucun ordre n'est nécessaire,
// la vérification suffit à rejeter un mélange.
bool Solver::probe(uint64_t hash, int& lower, int& upper, Move& move) const {
    const Entry& e = table[hash & (table_size - 1)];
    const uint64_t d = e.data.load(std::memory_order_relaxed), c = e.check.load(std::memory_order_relaxed);
    if (!(d & VALID) || (c ^ d) != hash) return false;
    lower = int8_t(d & 0xFF);          // l'octet bas, relu comme un entier signé
    upper = int8_t(d >> 8 & 0xFF);
    move = Move(uint16_t(d >> 16));
    return true;
}

// Range ce qu'une recherche dans ]alpha, beta[ a appris : une valeur v ≤ alpha
// est une borne supérieure, v ≥ beta une borne inférieure, entre les deux une
// valeur exacte (les deux bornes). Les bornes déjà connues de la même position
// sont gardées (fusion : l'entrée ne fait que se resserrer) ; une autre
// position dans la case est remplacée. −127 et 127 : « rien de prouvé ».
// Deux threads peuvent écrire la même case en même temps : l'entrée garde
// l'une des deux écritures, ou est déchirée et sera ignorée. Toute borne lue
// reste vraie ; au pire une fusion est perdue.
void Solver::store(uint64_t hash, int alpha, int beta, int value, Move move) {
    if (table_size == 0) return;
    int lower = -127, upper = 127;
    Move old = Move::none();
    probe(hash, lower, upper, old);   // autre position ou case vide : les valeurs par défaut restent
    if (value > alpha) lower = std::max(lower, value);
    if (value < beta)  upper = std::min(upper, value);
    if (move == Move::none()) move = old;
    Entry& e = table[hash & (table_size - 1)];
    const uint64_t d = pack(lower, upper, move);
    e.data.store(d, std::memory_order_relaxed);
    e.check.store(hash ^ d, std::memory_order_relaxed);
}

// ── 3. Ce que rapporte un coup ; l'ordre des coups ───────────────────────────
// Points : 1 par coup, 0 pour un passe. Collect : 1 si le coup arrive sur une
// cible. First : FIRST_WIN sur une cible (la partie est gagnée là, sans
// chercher plus loin), 0 sinon.
int Solver::gain(Move m) const {
    if (options.goal == Goal::Points) return m != Move::none();
    const bool hit = m != Move::none() && (options.targets >> m.to_sq() & 1);
    if (options.goal == Goal::Collect) return hit;
    return hit ? FIRST_WIN : 0;
}

// Le coup de la table d'abord (le meilleur d'une recherche précédente de cette
// position), puis « le plus rapide d'abord » : les coups qui laissent le moins
// de réponses à l'adversaire (souvent forts, et leur sous-arbre est petit, donc
// peu cher à réfuter quand ils ne le sont pas). Tri par insertion : quelques
// dizaines de coups au plus. Expliqué en détail dans doc/final_solver.cpp.
void Solver::order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const {
    const int n = int(moves.size());
    int first = 0;
    if (tt_move != Move::none()) {
        for (int i = 0; i < n; i++) {
            if (moves[i] == tt_move) {
                std::swap(moves[0], moves[i]);
                first = 1;
                break;
            }
        }
    }
    if (n - first > 1) {
        int replies[Yolah::MAX_NB_MOVES];
        for (int i = first; i < n; i++) {
            yolah.play(moves[i]);
            const auto [b0, b1, b2, b3] = yolah.moves_bb(yolah.current_player());
            replies[i] = std::popcount(b0) + std::popcount(b1) + std::popcount(b2) + std::popcount(b3);
            yolah.undo(moves[i]);
        }
        for (int i = first + 1; i < n; i++) {
            const Move m = moves[i];
            const int r = replies[i];
            int j = i - 1;
            while (j >= first && replies[j] > r) {
                moves[j + 1] = moves[j];
                replies[j + 1] = replies[j];
                j--;
            }
            moves[j + 1] = m;
            replies[j + 1] = r;
        }
    }
}

// ── 4. La recherche ──────────────────────────────────────────────────────────
// Faut-il s'arrêter ? Le drapeau `stop` (le bouton « Arrêter », le budget de
// l'app), ou une coupure à l'un des points de partage sous lesquels ce thread
// travaille : on remonte la chaîne (w.sp, puis son parent, …). La chaîne est
// courte (quelques points de partage) et chaque drapeau est sur sa propre
// ligne de cache : ce test, fait à chaque nœud, reste bon marché.
bool Solver::stopped(const Worker& w) const {
    if (stop && stop->load(std::memory_order_relaxed)) return true;
    for (const SplitPoint* p = w.sp; p; p = p->parent)
        if (p->cutoff.load(std::memory_order_relaxed)) return true;
    return false;
}

// L'alpha-bêta fail-soft sur le restant (voir doc/final_solver.h pour chaque
// étape commune ; ici, ce qui diffère est signalé).
int Solver::search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta) {
    ++w.nodes;                                   // le compteur DE CE THREAD (pas de partage)
    if (stopped(w)) return 0;                    // valeur sans signification : l'appelant ne l'utilise pas
    if (yolah.game_over()) return 0;             // plus rien à gagner
    const int nb_free = std::popcount(yolah.free_squares());
    const bool brute = nb_free <= BRUTE_FORCE_FREE || table_size == 0;
    // La table : des bornes déjà prouvées pour cette position (par n'importe quel thread).
    Move tt_move = Move::none();
    if (!brute) {
        int lower, upper;
        Move m;
        if (probe(hash, lower, upper, m)) {
            if (lower >= beta)  return lower;    // déjà prouvé ≥ beta : coupure
            if (upper <= alpha) return upper;    // déjà prouvé ≤ alpha : inutile
            if (lower == upper) return lower;    // valeur exacte connue
            alpha = std::max(alpha, lower);      // les deux bornes sont vraies :
            beta  = std::min(beta, upper);       // la fenêtre peut se resserrer
            tt_move = m;
        }
    }
    const int alpha_orig = alpha;                // pour savoir, à la fin, quel type de borne ranger
    Yolah::MoveList moves;
    yolah.moves(moves);                          // un passe (Move::none()) seul si le camp est bloqué
    const uint64_t free = yolah.free_squares();
    const uint8_t us = yolah.current_player();
    if (options.goal != Goal::Points) {
        // Cibles (section 3) : seules les cases libres voisines des pièces d'un camp
        // peuvent encore recevoir un de ses coups.
        const uint64_t r_us = reach(yolah.bitboard(us), free), r_them = reach(yolah.bitboard(Yolah::other_player(us)), free);
        if (((r_us | r_them) & options.targets) == 0) return 0;   // plus aucune cible prenable : 0 exactement
        if (options.goal == Goal::Collect) {
            if (moves[0] == Move::none()) {          // nous sommes bloqués pour toujours : restant ∈ [−cibles(eux), 0]
                if (0 <= alpha) return 0;
                const int lower = -std::popcount(r_them & options.targets);
                if (lower >= beta) return lower;
            } else if (r_them == 0) {                // eux bloqués pour toujours : restant ∈ [0, cibles(nous)]
                if (0 >= beta) return 0;
                const int upper = std::popcount(r_us & options.targets);
                if (upper <= alpha) return upper;
            }
        }
    } else {
        // Points : les bornes du joueur seul (doc/final_solver.cpp, étape 4).
        if (moves[0] == Move::none()) {              // bloqués : restant ∈ [−reach(eux), −1]
            if (-1 <= alpha) return -1;
            const int lower = -std::popcount(reach(yolah.bitboard(Yolah::other_player(us)), free));
            if (lower >= beta) return lower;
        } else if (reach(yolah.bitboard(Yolah::other_player(us)), free) == 0) {   // eux bloqués : ∈ [1, reach(nous)]
            if (1 >= beta) return 1;
            const int upper = std::popcount(reach(yolah.bitboard(us), free));
            if (upper <= alpha) return upper;
        }
    }
    if (!brute) order_moves(yolah, tt_move, moves);
    int best = -127;
    Move best_move = Move::none();
    for (size_t i = 0; i < moves.size(); i++) {
        // NOUVEAU : le partage. L'aîné (i = 0) a été cherché seul et n'a pas coupé
        // (sinon on serait sorti de la boucle) ; si le nœud est assez gros et qu'un
        // thread est inactif, les cadets (à partir de i = 1) sont partagés. split()
        // les cherche tous (avec les aides) et met à jour alpha, best, best_move :
        // la boucle est finie.
        if (i == 1 && options.threads > 1 && nb_free >= SPLIT_MIN_FREE && idle.load(std::memory_order_relaxed) > 0) {
            split(w, yolah, hash, moves, 1, alpha, beta, best, best_move);
            if (stopped(w)) return 0;            // arrêt venu d'au-dessus : rien à ranger
            break;
        }
        const Move m = moves[i];
        const int v = move_value(w, yolah, hash, m, alpha, beta);
        if (stopped(w)) return 0;
        if (v > best) {
            best = v;
            best_move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;            // coupure
            }
        }
    }
    if (!brute) store(hash, alpha_orig, beta, best, best_move);
    return best;
}

// La valeur du coup m au nœud (yolah, hash) pour le camp au trait :
// gain(m) − restant(après m), dans la fenêtre ]alpha, beta[. v est dans
// ]alpha, beta[ ⇔ le restant de l'enfant est dans ]gain − beta, gain − alpha[,
// d'où la fenêtre de l'appel récursif. La clé de l'enfant se calcule à partir
// de celle du parent (zobrist::update : quelques ou-exclusifs).
int Solver::move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta) {
    const int g = gain(m);
    if (g == FIRST_WIN) return 1;                // First : sur une cible, la partie est gagnée
    const uint64_t child = zobrist::update(hash, yolah.current_player(), m);
    yolah.play(m);
    const int v = g - search(w, yolah, child, g - beta, g - alpha);
    yolah.undo(m);
    return v;
}

// ── 5. Le partage ────────────────────────────────────────────────────────────
// split : le nœud devient un point de partage. Le point de partage vit sur la
// pile du maître (cette fonction) : il ne doit pas être quitté tant qu'un
// thread y travaille.
void Solver::split(Worker& w, Yolah& yolah, uint64_t hash, const Yolah::MoveList& moves, size_t first,
                   int& alpha, int beta, int& best, Move& best_move) {
    SplitPoint sp;
    sp.parent = w.sp;                            // la chaîne que stopped() remontera
    sp.pos = yolah;                              // chaque thread en fera sa copie
    sp.hash = hash;
    sp.nb_moves = moves.size();
    std::copy(moves.begin(), moves.end(), sp.moves);   // pas de copie de MoveList (pointeur de fin)
    sp.next = first;                             // les cadets seulement
    sp.alpha = alpha;
    sp.beta = beta;
    sp.best = best;                              // la valeur de l'aîné
    sp.best_move = best_move;
    sp.workers = 1;                              // le maître lui-même
    {
        std::lock_guard lk(sp_mutex);            // visible des aides à partir d'ici
        active.push_back(&sp);
        nb_active.store(int(active.size()), std::memory_order_relaxed);
        split_gen++;                             // pour les aides qui allaient s'endormir
    }
    sp_cv.notify_all();                          // réveille les aides inactifs
    w.sp = &sp;
    work_on(w, sp);                              // le maître prend des coups comme les autres
    {
        // Fermé : plus rien n'est distribué. work_on rend aussi la main quand la
        // recherche au-dessus est arrêtée alors qu'il reste des coups : aucun thread
        // ne doit rejoindre ce point de partage pendant qu'on le quitte.
        std::lock_guard lk(sp.m);
        sp.next = sp.nb_moves;
        sp.workers--;
    }
    // Les aides encore sur un coup : le maître aide en dessous (seulement s'il
    // existe d'autres points de partage, pour ne pas prendre le verrou pour rien),
    // sinon il cède son tour au système (yield) et revérifie.
    while (sp.workers.load(std::memory_order_acquire) > 0)
        if (nb_active.load(std::memory_order_relaxed) <= 1 || !help(w, &sp)) std::this_thread::yield();
    {
        std::lock_guard lk(sp_mutex);
        active.erase(std::find(active.begin(), active.end(), &sp));
        nb_active.store(int(active.size()), std::memory_order_relaxed);
    }
    w.sp = sp.parent;
    alpha = sp.alpha;                            // ce que les coups partagés ont donné
    best = sp.best;
    best_move = sp.best_move;
}

// Prend les coups du point de partage un par un, jusqu'à ce qu'il n'y en ait
// plus (ou qu'il soit coupé). Le coup et l'alpha courant sont pris sous le
// verrou ; la recherche se fait sans (sur une copie de la position) ; le
// résultat est rapporté sous le verrou.
void Solver::work_on(Worker& w, SplitPoint& sp) {
    for (;;) {
        size_t i;
        int a;
        {
            std::lock_guard lk(sp.m);
            if (!sp.open()) return;
            i = sp.next++;
            a = sp.alpha;                        // l'alpha le plus récent : la meilleure fenêtre connue
        }
        Yolah y = sp.pos;
        const int v = move_value(w, y, sp.hash, sp.moves[i], a, sp.beta);
        if (stopped(w)) return;                  // arrêt, ou coupure ici ou au-dessus : valeur non voulue
        std::lock_guard lk(sp.m);
        // Une valeur obtenue avec un alpha périmé (plus bas que l'actuel) reste
        // correcte : ≤ a ≤ alpha actuel, c'est une borne supérieure inutile ; au-dessus,
        // elle est exacte. Rien de faux ne peut donc entrer.
        if (v > sp.best) {
            sp.best = v;
            sp.best_move = sp.moves[i];
            if (v > sp.alpha) {
                sp.alpha = v;
                if (v >= sp.beta) sp.cutoff.store(true, std::memory_order_relaxed);   // les autres s'arrêtent
            }
        }
    }
}

// Rejoint un point de partage ouvert : le premier de la liste (le moins
// profond : le plus de travail), et, pour un maître qui attend (`under`), un
// point de partage SOUS le sien seulement. On rejoint en tenant sp_mutex : un
// point de partage ne quitte `active` (et la pile de son maître) qu'une fois
// fermé et quitté par tous, et un point fermé ne se rouvre jamais. Le compteur
// workers est incrémenté avant de lâcher le verrou : le maître ne peut plus
// partir sans nous.
bool Solver::help(Worker& w, const SplitPoint* under) {
    SplitPoint* sp = nullptr;
    {
        std::lock_guard lk(sp_mutex);
        for (SplitPoint* c : active) {
            if (under) {                         // descendant de `under` ?
                const SplitPoint* p = c->parent;
                while (p && p != under) p = p->parent;
                if (!p) continue;
            }
            std::lock_guard lk2(c->m);
            if (c->open()) { c->workers++; sp = c; break; }
        }
    }
    if (!sp) return false;
    SplitPoint* saved = w.sp;
    w.sp = sp;                                   // la chaîne passe par ce point de partage
    work_on(w, *sp);
    // Le dernier accès : dès que le compteur tombe à 0, le maître peut repartir et
    // le point de partage disparaît. Donc pas de verrou de sp->m autour.
    sp->workers.fetch_sub(1, std::memory_order_release);
    w.sp = saved;
    return true;
}

// La boucle d'un thread aide : tant que la recherche dure, rejoindre un point
// de partage ; sinon se déclarer inactif (ce qui autorise les autres à
// partager) et dormir au plus 200 µs, ou jusqu'à ce qu'un split le réveille.
// split_gen évite un réveil perdu : un split fait entre help() et la mise en
// sommeil change split_gen, et le thread ne s'endort pas.
void Solver::helper_loop(Worker& w) {
    while (solving.load(std::memory_order_acquire)) {
        uint64_t gen;
        {
            std::lock_guard lk(sp_mutex);
            gen = split_gen;
        }
        if (help(w, nullptr)) continue;
        idle.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock lk(sp_mutex);
            sp_cv.wait_for(lk, std::chrono::microseconds(200), [&] { return split_gen != gen || !solving.load(std::memory_order_relaxed); });
        }
        idle.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ── 6. La racine ─────────────────────────────────────────────────────────────
// La même boucle qu'un nœud, en gardant le meilleur coup. La valeur finale =
// la valeur actuelle + le restant. Les aides naissent ici et meurent à la fin
// de la recherche.
Solver::Result Solver::solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop_flag, const int* current_value) {
    stop = stop_flag;
    Yolah yolah = position;
    Result res;
    // La valeur déjà acquise : donnée par l'appelant, sinon l'écart des scores
    // (Points). Avec les cibles, les scores comptent des coups, pas des cibles : 0.
    const int current = current_value ? *current_value : options.goal == Goal::Points ? yolah.score(yolah.current_player()) : 0;
    if (yolah.game_over()) {
        res.value = current;
        res.complete = true;
        return res;
    }
    // La fenêtre sur le restant. Gagné/nul/perdu : finale ∈ ]−1, 1[ ⇔ restant ∈
    // ]−1 − actuel, 1 − actuel[ : il suffit de savoir < 0, 0 ou > 0.
    int alpha = wld_only ? -1 - current : -127;
    const int beta = wld_only ? 1 - current : 127;
    const uint64_t hash = zobrist::hash(yolah);
    Yolah::MoveList moves;
    yolah.moves(moves);
    Move tt_move = Move::none();
    if (table_size) {
        int lower, upper;
        Move m;
        if (probe(hash, lower, upper, m)) tt_move = m;
    }
    order_moves(yolah, tt_move, moves);
    // Les aides : threads − 1 threads qui attendent des points de partage.
    std::vector<Worker> workers(std::max(1, options.threads));
    std::vector<std::thread> helpers;
    solving.store(true, std::memory_order_release);
    for (size_t k = 1; k < workers.size(); k++) helpers.emplace_back([this, &workers, k] { helper_loop(workers[k]); });
    Worker& w = workers[0];
    int best = -127;
    bool complete = true;
    for (size_t i = 0; i < moves.size(); i++) {
        if (i == 1 && options.threads > 1) {     // la racine aussi : l'aîné seul, puis le partage
            split(w, yolah, hash, moves, 1, alpha, beta, best, res.move);
            if (stopped(w)) complete = false;
            break;
        }
        const Move m = moves[i];
        const int v = move_value(w, yolah, hash, m, alpha, beta);
        if (stopped(w)) { complete = false; break; }
        if (v > best) {
            best = v;
            res.move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;            // gagné/nul/perdu : une victoire suffit
            }
        }
    }
    {
        std::lock_guard lk(sp_mutex);            // sous le verrou : un aide sur le point de dormir le voit
        solving.store(false, std::memory_order_release);   // les aides sortent de leur boucle
    }
    sp_cv.notify_all();
    for (auto& h : helpers) h.join();
    for (const Worker& x : workers) res.nodes += x.nodes;
    if (!complete) {
        res.move = Move::none();
        return res;                              // complete = false
    }
    res.value = current + best;
    res.complete = true;
    return res;
}

// ── 7. Le verdict d'un puzzle ────────────────────────────────────────────────
// Le signe de l'objectif depuis y, pour le joueur (> 0 atteint, 0 nul, < 0
// manqué), avec le meilleur jeu des deux côtés. taken : les cibles déjà prises
// par chaque camp (Collect). « Au moins N de plus » = « plus » quand
// l'adversaire a N − 1 de plus : gagné/nul/perdu suffit.
static int lead(Solver& solver, const Yolah& y0, const Puzzle& p, std::array<int, 2> taken, const std::atomic_bool* stop, bool& ok) {
    const int stm = y0.current_player();
    const int sign = stm == p.human ? 1 : -1;    // les valeurs du solveur sont pour le camp au trait
    Solver::Result r;
    if (p.goal == Goal::Points) {
        // Les N − 1 points de plus vont dans les scores mêmes de la position.
        Yolah y = y0;
        auto [b, w] = y.score();
        (p.human == Yolah::BLACK ? w : b) += uint16_t(p.margin - 1);
        y.set_state(y.bitboard(Yolah::BLACK), y.bitboard(Yolah::WHITE), y.empty_bitboard(), b, w, y.nb_plies());
        r = solver.solve(y, true, stop);
    } else {
        // Cibles : la valeur acquise est passée au solveur (les scores de la
        // position comptent des points, pas des cibles).
        const int current = p.goal == Goal::Collect ? sign * (taken[p.human] - taken[1 - p.human] - (p.margin - 1)) : 0;
        r = solver.solve(y0, true, stop, &current);
    }
    ok = r.complete;
    return sign * r.value;
}

Check verify(const Puzzle& p, int threads, const std::atomic_bool* stop) {
    Check c;
    if (std::popcount(p.start.free_squares()) > MAX_FREE) return c;   // Inconnu sans essayer
    Options o;
    o.goal = p.goal;
    o.targets = p.targets;
    o.tt_bits = 23;
    o.threads = threads;
    Solver main_solver(o);                       // la preuve principale, partagée entre threads
    bool ok = false;
    c.lead = lead(main_solver, p.start, p, {0, 0}, stop, ok);
    if (!ok) return c;
    const bool feasible = c.lead > 0;
    c.verdict = feasible ? Verdict::Feasible : Verdict::Impossible;
    if (!feasible || p.start.current_player() != p.human) return c;
    // Les coups clés : une preuve par coup, indépendantes. Chaque thread a son
    // solveur (un thread, sa table de 2^21 entrées) et prend le coup suivant
    // (`next`, atomique) jusqu'à épuisement.
    Yolah::MoveList moves;
    Yolah y = p.start;
    y.moves(moves);
    const size_t n = moves.size();
    std::vector<int> key(n, 0);                  // 1 clé, 0 non, −1 pas de preuve (arrêt)
    std::atomic<size_t> next{0};
    auto prove = [&] {
        Options ko = o;
        ko.tt_bits = 21;
        ko.threads = 1;
        Solver s(ko);
        for (size_t i; (i = next.fetch_add(1)) < n;) {
            const Move m = moves[i];
            const bool hit = m != Move::none() && (p.targets >> m.to_sq() & 1) && p.goal != Goal::Points;
            if (hit && p.goal == Goal::First) { key[i] = 1; continue; }   // sur une cible : gagné aussitôt
            Yolah ch = y;
            ch.play(m);
            std::array<int, 2> taken{0, 0};
            if (hit) taken[p.human] = 1;         // Collect : ce coup prend une cible
            bool ok2 = false;
            const int d = lead(s, ch, p, taken, stop, ok2);
            key[i] = !ok2 ? -1 : d > 0;
        }
    };
    const int nb = std::clamp(threads, 1, int(n));
    std::vector<std::thread> helpers;
    for (int k = 1; k < nb; k++) helpers.emplace_back(prove);
    prove();
    for (auto& h : helpers) h.join();
    for (size_t i = 0; i < n; i++) {
        if (key[i] < 0) return c;                // arrêté pendant le comptage : faisable, coups clés inconnus
        if (key[i] > 0) c.keys.push_back(moves[i]);
    }
    c.keys_counted = true;
    c.nb_moves = int(n);
    return c;
}

}  // namespace final_ybwc
