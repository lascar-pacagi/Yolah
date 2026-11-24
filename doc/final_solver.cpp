// =============================================================================
//  final_solver.cpp — le solveur exact de l'éditeur de puzzles, commenté
// =============================================================================
//  Lire d'abord doc/final_solver.h : les valeurs (le score « restant »), la
//  fenêtre gagné/nul/perdu, pourquoi pas de règle du passe, la vérification.
//  Plan de ce fichier :
//    1. reach        les cases libres qu'un camp peut encore atteindre
//    2. Solver       construction, table (store), ordre des coups
//    3. search       l'alpha-bêta fail-soft sur le restant
//    4. solve        la racine : écart final, meilleur coup
//    5. verify       le verdict d'un puzzle
// =============================================================================

#include "final_solver.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>

namespace final_solver {

// ── 1. Les cases libres atteignables ─────────────────────────────────────────
// Les cases libres que `pieces` peuvent encore atteindre un jour, en passant
// par des cases libres. Un coup glisse en ligne droite sur des cases libres :
// c'est une suite de pas d'une case vers une voisine libre (8 directions). Une
// « inondation » 8-connexe à partir des pièces donne donc exactement les cases
// où elles pourront aller, en un coup ou en plusieurs.
//
// grow(b) = b et ses 8 voisines : d'abord la ligne (b, b décalé d'une colonne
// à l'est et à l'ouest, masqué pour ne pas passer de la colonne h à la
// colonne a de la rangée suivante), puis cette ligne décalée d'une rangée vers
// le haut et vers le bas (<< 8 et >> 8 ; ce qui sort du plateau disparaît).
// On répète grow, restreint aux cases libres, jusqu'à ce que plus rien ne
// change. Les pièces elles-mêmes ne sont pas des cases libres : elles ne sont
// pas dans le résultat.
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

// ── 2. Construction, table, ordre des coups ──────────────────────────────────
Solver::Solver(int tt_bits) {
    if (tt_bits > 0) table.resize(size_t(1) << tt_bits);
}

void Solver::clear() {
    std::fill(table.begin(), table.end(), Entry{});
    nb_nodes = 0;
}

// Range ce qu'une recherche dans la fenêtre ]alpha, beta[ a appris. Une valeur
// fail-soft v est une borne supérieure si v ≤ alpha, une borne inférieure si
// v ≥ beta, exacte entre les deux (alors les deux bornes sont posées). Les
// bornes de la MÊME position se cumulent (toutes restent vraies : on garde la
// plus serrée), donc l'entrée ne fait que se préciser au fil des recherches.
// Une autre position dans la même case de la table est remplacée (la plus
// récente est la plus utile : c'est celle qu'on est en train de chercher).
void Solver::store(uint64_t hash, int alpha, int beta, int value, Move move) {
    if (table.empty()) return;
    Entry& e = table[hash & (table.size() - 1)];     // la taille est une puissance de 2
    if (e.key != hash) e = Entry{hash};
    if (value > alpha) e.lower = int8_t(std::max<int>(e.lower, value));
    if (value < beta)  e.upper = int8_t(std::min<int>(e.upper, value));
    if (move != Move::none()) e.move = move;
}

// Met le coup de la table en tête, puis trie les autres par le nombre de
// réponses qu'ils laissent à l'adversaire, du plus petit au plus grand
// (« le plus rapide d'abord », comme dans les solveurs de fin de partie
// d'Othello). Pourquoi ça marche dans une recherche exacte : un coup qui
// laisse peu de réponses est souvent fort (l'adversaire est à l'étroit,
// peut-être bientôt bloqué), et le sous-arbre qui le suit est petit : même
// s'il n'est pas le meilleur, le réfuter coûte peu. Le meilleur coup en tête
// fait couper l'alpha-bêta au plus tôt.
// Compter les réponses coûte une génération de coups par coup : rentable avec
// beaucoup de cases libres, pas tout près de la fin (BRUTE_FORCE_FREE).
void Solver::order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const {
    const int n = int(moves.size());
    int first = 0;
    if (tt_move != Move::none()) {
        for (int i = 0; i < n; i++) {
            if (moves[i] == tt_move) {
                std::swap(moves[0], moves[i]);
                first = 1;                            // le coup de table reste en tête, hors du tri
                break;
            }
        }
    }
    if (n - first > 1) {
        int replies[Yolah::MAX_NB_MOVES];
        for (int i = first; i < n; i++) {
            yolah.play(moves[i]);
            // moves_bb : les destinations possibles de chacune des 4 pièces du
            // camp au trait (l'adversaire, après le coup) ; leur total = son
            // nombre de coups
            const auto [b0, b1, b2, b3] = yolah.moves_bb(yolah.current_player());
            replies[i] = std::popcount(b0) + std::popcount(b1) + std::popcount(b2) + std::popcount(b3);
            yolah.undo(moves[i]);
        }
        // tri par insertion (quelques dizaines de coups au plus) ; stable :
        // à égalité, l'ordre du générateur est gardé
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

// ── 3. La recherche ──────────────────────────────────────────────────────────
// Rend le RESTANT de la position (camp au trait), fail-soft dans ]alpha, beta[
// (voir le .h). `hash` est la clé Zobrist de la position, mise à jour coup par
// coup par l'appelant (zobrist::update) plutôt que recalculée.
int Solver::search(Yolah& yolah, uint64_t hash, int alpha, int beta) {
    ++nb_nodes;
    // Arrêt demandé : on remonte n'importe quelle valeur ; solve() le voit et
    // déclare le résultat incomplet, donc cette valeur n'est jamais utilisée.
    if (stopped()) return 0;
    // Partie finie : plus aucun coup de part et d'autre, restant = 0.
    if (yolah.game_over()) return 0;
    const int nb_free = std::popcount(yolah.free_squares());
    const bool brute = nb_free <= BRUTE_FORCE_FREE || table.empty();

    // Étape 1 — la table : des bornes prouvées par des recherches précédentes
    // de cette même position (atteinte par un autre ordre de coups, ou lors
    // d'une recherche précédente : coups clés, autre objectif).
    Move tt_move = Move::none();
    if (!brute) {
        const Entry& e = table[hash & (table.size() - 1)];
        if (e.key == hash) {
            if (e.lower >= beta)  return e.lower;    // déjà prouvé ≥ beta : coupure
            if (e.upper <= alpha) return e.upper;    // déjà prouvé ≤ alpha : rien à gagner ici
            if (e.lower == e.upper) return e.lower;  // valeur exacte connue
            // Sinon les deux bornes restent vraies : la fenêtre peut se
            // resserrer (la recherche coupera plus tôt), et le coup de la
            // table sera essayé en premier.
            alpha = std::max<int>(alpha, e.lower);
            beta  = std::min<int>(beta, e.upper);
            tt_move = e.move;
        }
    }
    // La fenêtre telle qu'à l'entrée (après la table) : c'est par rapport à
    // elle que store() saura si le résultat est une borne ou une valeur exacte.
    const int alpha_orig = alpha;

    // Étape 2 — les coups. Un camp bloqué a un seul « coup » : le passe,
    // Move::none(), en moves[0].
    Yolah::MoveList moves;
    yolah.moves(moves);

    // Étape 3 — les bornes du joueur seul. Si un camp est bloqué, il l'est pour
    // toujours (propriété (2) du .h) et l'autre joue seul jusqu'à la fin ;
    // chacun de ses coups remplit une case libre (propriété (1)), donc il
    // jouera au plus un coup par case libre qu'il peut atteindre. D'où des
    // bornes EXACTES sur le restant, quels que soient les scores :
    //   • le camp au trait doit passer : l'adversaire jouera au moins un coup
    //     (la partie n'est pas finie, donc lui peut bouger) et au plus
    //     reach(adversaire) coups ; nous, aucun :
    //         restant ∈ [−reach(adversaire), −1] ;
    //   • l'adversaire est bloqué (aucune case libre voisine de ses pièces,
    //     donc rien d'atteignable) : nous jouerons au moins un coup (nous
    //     avons des coups : sinon la partie serait finie) et au plus reach(nous) :
    //         restant ∈ [1, reach(nous)].
    // Si une borne suffit à sortir de la fenêtre, on la rend tout de suite.
    // Exemple : le camp au trait est bloqué mais mène de 6 points (un bonus de
    // puzzle) ; les pièces adverses ne peuvent plus atteindre que 4 cases
    // libres : elles joueront au plus 4 coups, restant ≥ −4, écart final ≥ +2.
    // Gagné, sans chercher dans quel ordre elles joueront.
    // Sans ces bornes, le solveur essaierait tous les ordres de coups du joueur
    // seul pour en trouver le nombre exact : sur la « dernière pièce » (doc de
    // l'app) avec 6 points de bonus, 4,6 s au lieu de 0,27 s.
    {
        const uint64_t free = yolah.free_squares();
        const uint8_t us = yolah.current_player();
        if (moves[0] == Move::none()) {
            if (-1 <= alpha) return -1;
            const int lower = -std::popcount(reach(yolah.bitboard(Yolah::other_player(us)), free));
            if (lower >= beta) return lower;
        } else if (reach(yolah.bitboard(Yolah::other_player(us)), free) == 0) {
            if (1 >= beta) return 1;
            const int upper = std::popcount(reach(yolah.bitboard(us), free));
            if (upper <= alpha) return upper;
        }
    }

    // Étape 4 — le négamax sur les coups, les meilleurs d'abord.
    if (!brute) order_moves(yolah, tt_move, moves);
    const uint8_t player = yolah.current_player();
    int best = -127;                                  // « pire que tout » (le restant est ≥ −64)
    Move best_move = Move::none();
    for (Move m : moves) {
        // un coup rapporte un point à son camp, un passe aucun
        const int gain = m != Move::none();
        const uint64_t child = zobrist::update(hash, player, m);
        yolah.play(m);
        // v = gain − restant(enfant). La fenêtre de l'enfant se déduit de la
        // nôtre : v ∈ ]alpha, beta[ ⇔ restant(enfant) ∈ ]gain − beta, gain − alpha[.
        const int v = gain - search(yolah, child, gain - beta, gain - alpha);
        yolah.undo(m);
        if (stopped()) return 0;                      // résultat inutilisable : on remonte
        if (v > best) {
            best = v;
            best_move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;                 // coupure : l'adversaire évitera cette position
            }
        }
    }
    if (!brute) store(hash, alpha_orig, beta, best, best_move);
    return best;
}

// ── 4. La racine ─────────────────────────────────────────────────────────────
// La même boucle qu'à l'étape 4, en gardant le meilleur coup, et avec la
// conversion entre l'écart FINAL (ce que veut l'appelant) et le RESTANT (ce que
// calcule la recherche) : final = écart actuel + restant.
Solver::Result Solver::solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop_flag) {
    stop = stop_flag;
    const uint64_t nodes_before = nb_nodes;
    Yolah yolah = position;                           // une copie : la recherche joue et déjoue dessus
    Result res;
    const int current = yolah.score(yolah.current_player());   // écart actuel, vu du camp au trait
    if (yolah.game_over()) {
        res.value = current;
        res.complete = true;
        return res;
    }
    // La fenêtre sur le restant. Gagné/nul/perdu : on veut savoir si
    // final ≥ 1, = 0 ou ≤ −1, donc final ∈ ]−1, 1[ ⇔ restant ∈ ]−1 − actuel,
    // 1 − actuel[. Exact : toute la plage possible (]−127, 127[).
    int alpha = wld_only ? -1 - current : -127;
    const int beta = wld_only ? 1 - current : 127;
    const uint64_t hash = zobrist::hash(yolah);
    Yolah::MoveList moves;
    yolah.moves(moves);
    Move tt_move = Move::none();
    if (!table.empty()) {
        const Entry& e = table[hash & (table.size() - 1)];
        if (e.key == hash) tt_move = e.move;          // à la racine, seulement le coup (pas de coupure)
    }
    order_moves(yolah, tt_move, moves);
    const uint8_t player = yolah.current_player();
    int best = -127;
    for (Move m : moves) {
        const int gain = m != Move::none();
        yolah.play(m);
        const int v = gain - search(yolah, zobrist::update(hash, player, m), gain - beta, gain - alpha);
        yolah.undo(m);
        if (stopped()) {
            res.nodes = nb_nodes - nodes_before;
            return res;                               // complete = false
        }
        if (v > best) {
            best = v;
            res.move = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) break;                 // gagné/nul/perdu : une victoire suffit
            }
        }
    }
    res.value = current + best;
    res.complete = true;
    res.nodes = nb_nodes - nodes_before;
    return res;
}

// ── 5. Le verdict d'un puzzle ────────────────────────────────────────────────
// Dans l'app (puzzle_check_start), ce code tourne dans un thread à part, et un
// second thread (« chien de garde ») met `stop` à vrai au bout de 10 s ; le
// bouton « Arrêter » fait de même. Le verdict est affiché sous le plateau.
Check verify(const Yolah& start, uint8_t human, int margin, const std::atomic_bool* stop) {
    Check c;
    c.free_squares = std::popcount(start.free_squares());
    // Trop de cases libres : la preuve prendrait des heures (le nombre de
    // positions croît très vite avec les cases libres). On ne tente rien.
    if (c.free_squares > MAX_FREE) {
        c.too_big = true;
        return c;                                     // Inconnu
    }
    Solver solver(22);                                // une table pour toutes les recherches du verdict
    // Le signe de l'écart final du JOUEUR avec le meilleur jeu des deux camps,
    // après avoir donné margin − 1 points de plus à l'adversaire :
    // > 0 ⇔ le joueur gagne de margin ou plus. (Sur le vrai plateau, les
    // scores sont seulement décalés : la table reste valable d'une recherche à
    // l'autre, puisqu'elle ne contient que des restants.)
    auto lead = [&](const Yolah& y0, bool& ok) {
        Yolah y = y0;
        auto [b, w] = y.score();
        (human == Yolah::BLACK ? w : b) += uint16_t(margin - 1);
        y.set_state(y.bitboard(Yolah::BLACK), y.bitboard(Yolah::WHITE), y.empty_bitboard(), b, w, y.nb_plies());
        auto r = solver.solve(y, true, stop);
        ok = r.complete;
        // value est vu du camp au trait : on le ramène au joueur
        return y.current_player() == human ? r.value : -r.value;
    };
    bool ok = false;
    const int d = lead(start, ok);
    if (!ok) return c;                                // Inconnu : arrêté avant la preuve
    c.lead = d;
    c.verdict = d > 0 ? Verdict::Feasible : Verdict::Impossible;
    // Les coups clés, quand le joueur a le trait : chaque coup, puis la même
    // preuve depuis la position suivante (l'adversaire au trait).
    if (start.current_player() == human) {
        Yolah y = start;
        Yolah::MoveList moves;
        y.moves(moves);
        std::vector<Move> keys;
        for (Move m : moves) {
            Yolah child = y;
            child.play(m);
            bool ok2 = false;
            const int dc = lead(child, ok2);
            if (!ok2) return c;                       // arrêté : verdict sans coups clés
            if (dc > 0) keys.push_back(m);
        }
        c.keys = keys;
        c.nb_moves = int(moves.size());
        c.keys_counted = true;
    }
    return c;
}

}  // namespace final_solver
