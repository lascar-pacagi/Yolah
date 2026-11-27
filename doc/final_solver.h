#ifndef FINAL_SOLVER_H
#define FINAL_SOLVER_H

// =============================================================================
//  FINAL_SOLVER — Version épurée et commentée du solveur de l'éditeur de puzzles
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (et son compagnon doc/final_solver.cpp) contient le code qui
//  répond au bouton « Vérifier » de l'éditeur de puzzles de l'application :
//  un solveur EXACT de fin de partie (il cherche jusqu'à la toute fin du jeu,
//  sans aucune évaluation approchée), et la fonction qui en tire le verdict
//  d'un puzzle : « Faisable », « Impossible » ou « Inconnu », avec les coups
//  clés.
//
//  Le vrai code est réparti dans deux fichiers :
//    player/endgame_solver.{h,cpp}   le solveur (classe EndgameSolver), partagé
//                                    avec l'auto-apprentissage AlphaZero, le
//                                    joueur hybride (feuilles prouvées) et le
//                                    joueur « Solveur exact » des puzzles
//    app/yolah_app.cpp               PuzzleCheck et puzzle_check_start : le
//                                    verdict, calculé dans un thread à part,
//                                    avec un budget de temps (section
//                                    « Puzzle verification »)
//  Le solveur y a des options que l'éditeur n'utilise pas : la « règle du
//  passe » (fausse pour les puzzles, voir plus bas), trois ordres de coups
//  au choix, une taille de table, une limite de nœuds. Ici il ne reste que
//  la configuration de l'éditeur pour l'objectif « points » : pas de règle
//  du passe, ordre « le plus rapide d'abord », force brute à ≤ 6 cases
//  libres, et UN seul thread.
//  Ce que cette copie ne reprend pas (voir player/endgame_solver.h) :
//    • les objectifs à cibles (EndgameGoal : récolter des cibles, atteindre
//      une cible en premier) : seul le gain d'un coup change ;
//    • la recherche sur plusieurs threads (EndgameSolver::split, « young
//      brothers wait ») : l'éditeur prouve le verdict sur 8 threads au plus
//      (≈ 5 fois plus vite), avec une table partagée sans verrou (clé XOR
//      donnée), et compte les coups clés en parallèle, un solveur par
//      thread. Avec un seul thread, le solveur du joueur visite toujours les
//      mêmes nœuds que celui-ci.
//  Le thread et le budget de temps restent dans l'app (ce sont de
//  l'interface) ; ici, l'arrêt se demande par un drapeau `stop`.
//
//  « Même code » veut dire : avec les mêmes options, ce solveur visite
//  EXACTEMENT les mêmes nœuds que EndgameSolver, dans le même ordre, et rend
//  les mêmes valeurs et les mêmes coups (voir « VÉRIFICATION » à la fin de ce
//  commentaire).
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF
//  ---------------------------------------------------------------------------
//  (Les règles en détail sont au début de doc/final_dev.h.) Plateau 8 × 8,
//  4 pièces par camp. Une pièce se déplace comme une dame d'échecs vers une
//  case libre, sans rien sauter ; sa case de départ devient un TROU définitif.
//  Chaque coup rapporte 1 point à son camp. Un camp bloqué passe
//  (`Move::none()`) ; la partie finit quand les deux camps sont bloqués, et
//  gagne celui qui a le plus de points. Une case = un entier 0..63 (a1 = 0,
//  h8 = 63) ; un ensemble de cases = un `uint64_t` (« bitboard »).
//
//  Deux propriétés font tout le solveur :
//    (1) chaque coup remplit une case libre (la destination) et n'en libère
//        aucune (le départ devient un trou) : le nombre de cases libres baisse
//        de 1 à chaque coup. Avec k cases libres, la partie finit en au plus k
//        coups (plus des passes). Une recherche jusqu'à la fin est donc
//        possible quand k est petit, et elle est alors une PREUVE ;
//    (2) un camp bloqué le reste pour toujours : une case libre ne revient
//        jamais. Après un passe, l'adversaire joue SEUL jusqu'à la fin.
//
//  ---------------------------------------------------------------------------
//  LES VALEURS : LE SCORE « RESTANT »
//  ---------------------------------------------------------------------------
//  Le score d'un camp est le nombre de coups qu'il a joués (plus son bonus de
//  départ, dans un puzzle). Ce qui intéresse l'appelant, c'est l'écart FINAL
//  (camp au trait − adversaire) avec le meilleur jeu des deux côtés. Mais la
//  recherche calcule seulement la partie RESTANTE de cet écart :
//
//      restant(position) = nombre de coups que le camp au trait jouera encore
//                          − nombre de coups que l'adversaire jouera encore
//      final = écart actuel + restant
//
//      restant(partie finie) = 0
//      restant(position) = max sur les coups m de  1 − restant(après m)   (un coup : +1 point)
//                                              ou    − restant(après m)   (un passe : 0 point)
//
//  (Le signe moins : après le coup, c'est l'adversaire qui est au trait, et sa
//  valeur est vue de son côté. C'est le « négamax ».)
//
//  Pourquoi le restant et pas l'écart final : le restant ne dépend que du
//  plateau et du camp au trait, pas des scores déjà acquis. Deux positions au
//  même plateau mais aux scores différents (atteintes par des ordres de coups
//  différents, ou parce que le puzzle donne un bonus) ont le même restant :
//  la table de transposition (indexée par la clé Zobrist, qui ne contient pas
//  les scores) peut le partager. Les scores n'interviennent qu'à la racine,
//  pour placer la fenêtre de recherche (voir solve()).
//
//  Exemple : Noir au trait, 3 points contre 1 (écart actuel +2). Si, avec le
//  meilleur jeu, Noir joue encore 4 coups et Blanc 5, restant = −1 et
//  final = +2 − 1 = +1 : Noir gagne d'un point.
//
//  ---------------------------------------------------------------------------
//  LA RECHERCHE : ALPHA-BÊTA « FAIL-SOFT »
//  ---------------------------------------------------------------------------
//  search(position, alpha, beta) ne calcule pas toujours la valeur exacte : il
//  suffit de savoir où elle se place par rapport à la fenêtre ]alpha, beta[.
//    • résultat v ≤ alpha : la vraie valeur est ≤ v (borne supérieure) ; le
//      parent a déjà mieux ailleurs, inutile d'en savoir plus ;
//    • résultat v ≥ beta  : la vraie valeur est ≥ v (borne inférieure) ;
//      l'adversaire, au-dessus, ne laissera pas faire : coupure ;
//    • alpha < v < beta   : v est exacte.
//  « Fail-soft » : en dehors de la fenêtre, on rend la meilleure borne trouvée
//  (et pas simplement alpha ou beta), ce qui donne des bornes plus serrées à
//  la table.
//
//  Gagné / nul / perdu (`wld_only`) : on ne veut souvent que le SIGNE de
//  l'écart final. La fenêtre à la racine est alors ]−1, +1[ sur l'écart final
//  (donc ]−1 − actuel, 1 − actuel[ sur le restant) : la recherche s'arrête dès
//  qu'elle sait « ≥ +1 » (gagné) ou « ≤ −1 » (perdu). C'est beaucoup moins
//  cher que l'écart exact : sur la « dernière pièce » (21 cases libres,
//  objectif « gagner de 3 »), le calcul exact n'avait pas fini en 10 s, la
//  version gagné/perdu répond en 0,07 s (voir verify()).
//
//  ---------------------------------------------------------------------------
//  LA RÈGLE DU PASSE, ET POURQUOI L'ÉDITEUR NE L'UTILISE PAS
//  ---------------------------------------------------------------------------
//  Le vrai solveur a une option `pass_rule` (vraie par défaut) : « un joueur
//  qui doit passer alors que la partie n'est pas finie a perdu ». Elle est
//  vraie en partie NORMALE : si Blanc doit passer, Noir a déjà un coup de plus
//  et Blanc n'en jouera plus ; si Noir doit passer, les scores sont égaux et
//  Blanc jouera encore au moins un coup. Mais elle suppose ces scores-là.
//  Dans un puzzle, c'est faux : avec 5 points de bonus, un joueur bloqué peut
//  très bien gagner ; et un puzzle où Blanc commence à 0-0 n'a pas non plus
//  les scores d'une partie normale. L'éditeur la désactive donc, et elle n'est
//  pas dans ce fichier.
//
//  Sans elle, un camp bloqué posait un problème de vitesse : l'adversaire joue
//  seul jusqu'à la fin, et le solveur essayait TOUS les ordres possibles de
//  ses coups pour en trouver le nombre maximal (sur la « dernière pièce »
//  avec un bonus de 6 : 4,6 s). D'où les « bornes du joueur seul »
//  (search(), étape 4), exactes quels que soient les scores, qui ramènent ce
//  cas à 0,27 s.
//
//  ---------------------------------------------------------------------------
//  LE VERDICT D'UN PUZZLE (verify)
//  ---------------------------------------------------------------------------
//  Un puzzle donne une position (avec les bonus dans les scores), le camp du
//  joueur, et un objectif « gagner de N ou plus » (N = 1 : simplement gagner).
//    • « Gagner de N » se vérifie comme « gagner » après avoir donné N − 1
//      points de plus à l'adversaire : écart ≥ N ⇔ écart − (N − 1) ≥ 1. On
//      reste ainsi en gagné/nul/perdu (rapide).
//    • Faisable : le joueur gagne contre TOUTE défense (preuve).
//      Impossible : la meilleure défense l'en empêche (preuve).
//      Inconnu : trop de cases libres (> MAX_FREE = 36 ici : on n'essaie même
//      pas), ou arrêt avant la fin (bouton « Arrêter », budget de l'app).
//      L'app a depuis relevé ces limites : 44 cases libres et 10 minutes, la
//      preuve sur plusieurs threads (voir doc/final_ybwc.h).
//    • Quand le joueur a le trait, on compte ses COUPS CLÉS : les coups après
//      lesquels l'objectif reste atteint. Un bon puzzle n'en a souvent qu'un.
//      (Chaque coup demande une recherche de plus ; la table, partagée,
//      rend les suivantes moins chères.)
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_solver_check.cpp (construit par test/final_solver_check.sh) tire
//  des positions au hasard (6 à 13 cases libres, scores de 0 à 6 de chaque
//  côté, trait au hasard) et compare :
//    • ce solveur à EndgameSolver sans règle du passe : mêmes valeurs, mêmes
//      coups, MÊME NOMBRE DE NŒUDS (preuve que le code est le même) ;
//    • les deux à un négamax brut jusqu'à la fin de la partie (sans table,
//      sans ordre, sans bornes) : mêmes valeurs exactes et mêmes signes ;
//    • verify() à ce que dit le négamax brut pour des objectifs N = 1 à 3.
//  Toute modification de ce code doit être revérifiée ainsi.
// =============================================================================

#include "game.h"
#include <atomic>
#include <cstdint>
#include <vector>

namespace final_solver {

class Solver {
public:
    struct Result {
        Move move = Move::none();   // le meilleur coup à la racine (passe si bloqué)
        int  value = 0;             // écart FINAL (camp au trait − adversaire) ; en gagné/nul/perdu,
                                    // seul son signe est exact
        bool complete = false;      // faux : interrompu par `stop`, `move` et `value` ne valent rien
        uint64_t nodes = 0;         // nœuds visités par cette recherche
    };

    // tt_bits : la table a 2^tt_bits entrées de 16 octets (22 : 64 Mo, ce
    // qu'utilise l'éditeur).
    explicit Solver(int tt_bits = 22);

    // Meilleur coup et écart final de `position`. wld_only : seulement
    // gagné / nul / perdu (le signe de `value` est exact, pas sa grandeur).
    // stop : drapeau mis à vrai par un autre thread pour arrêter (peut être nul).
    Result solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop = nullptr);

    // Vide la table (les bornes prouvées sont oubliées) et remet le compteur à 0.
    void clear();
    uint64_t nodes() const { return nb_nodes; }

private:
    // Une entrée de la table : des BORNES prouvées sur le restant de la
    // position de clé `key`, et le meilleur coup trouvé. lower = −127 et
    // upper = 127 veulent dire « rien de prouvé » (le restant est toujours
    // dans [−64, 64] : au plus un coup par case). int8_t : 16 octets en tout.
    struct Entry {
        uint64_t key = 0;
        int8_t   lower = -127;      // le restant est ≥ lower
        int8_t   upper = 127;       //             et ≤ upper
        Move     move = Move::none();
    };

    // En dessous de ce nombre de cases libres : ni table ni tri des coups. Si
    // près de la fin, le sous-arbre est si petit qu'il coûte moins cher de le
    // chercher que de consulter la table (accès mémoire aléatoire) ou de
    // compter les réponses de chaque coup (une génération de coups par coup).
    static constexpr int BRUTE_FORCE_FREE = 6;

    std::vector<Entry> table;
    uint64_t nb_nodes = 0;
    const std::atomic_bool* stop = nullptr;

    bool stopped() const { return stop && stop->load(std::memory_order_relaxed); }
    int  search(Yolah& yolah, uint64_t hash, int alpha, int beta);
    void order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const;
    void store(uint64_t hash, int alpha, int beta, int value, Move move);
};

// ── Le verdict d'un puzzle ───────────────────────────────────────────────────
enum class Verdict { Feasible, Impossible, Unknown };

struct Check {
    Verdict verdict = Verdict::Unknown;
    int  free_squares = 0;          // cases libres de la position de départ
    bool too_big = false;           // Inconnu sans essayer : free_squares > MAX_FREE
    int  lead = 0;                  // signe du résultat (objectif décalé) : > 0 atteint,
                                    // 0 nul, < 0 perdu (Faisable / Impossible seulement)
    bool keys_counted = false;      // le joueur a le trait ET les coups clés ont été comptés
    std::vector<Move> keys;         // les coups clés (si keys_counted)
    int  nb_moves = 0;              // ses coups légaux (si keys_counted)
};

// Au-delà de ce nombre de cases libres, l'éditeur ne tente pas de preuve.
constexpr int MAX_FREE = 36;

// Le verdict du puzzle : `start` (bonus compris dans les scores), le camp du
// joueur `human`, l'objectif « gagner de `margin` ou plus » (1 = gagner).
// `stop` arrête la recherche (verdict Inconnu si c'est avant la preuve
// principale ; Faisable/Impossible sans coups clés si c'est pendant leur
// comptage).
Check verify(const Yolah& start, uint8_t human, int margin, const std::atomic_bool* stop = nullptr);

}  // namespace final_solver

#endif
