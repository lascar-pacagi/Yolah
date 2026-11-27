#ifndef FINAL_YBWC_H
#define FINAL_YBWC_H

// =============================================================================
//  FINAL_YBWC — Version épurée et commentée du solveur PARALLÈLE de l'éditeur
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Le bouton « Vérifier » de l'éditeur de puzzles prouve qu'un puzzle est
//  faisable ou non en cherchant la partie jusqu'à la toute fin. Depuis
//  novembre 2026, cette preuve tourne sur plusieurs cœurs (8 au plus) : environ
//  5 fois plus vite sur des puzzles de 26 à 28 cases libres (par exemple
//  15,9 s → 2,9 s). Ce fichier (avec doc/final_ybwc.cpp) contient ce solveur,
//  commenté ligne par ligne.
//
//  À LIRE AVANT : doc/final_solver.h. Il explique, sur la version à un seul
//  thread (plus simple), tout ce qui ne change pas ici : le score « restant »,
//  l'alpha-bêta fail-soft, la fenêtre gagné/nul/perdu, pourquoi pas de règle
//  du passe, les bornes du joueur seul, le verdict d'un puzzle. Ce fichier-ci
//  ne réexplique que l'essentiel et se concentre sur ce qui est NOUVEAU :
//    1. la recherche PARTAGÉE entre threads (« young brothers wait », YBWC) ;
//    2. la table de transposition partagée SANS VERROU ;
//    3. les objectifs à CIBLES des puzzles (récolter des cibles, atteindre une
//       cible en premier) ;
//    4. le verdict, avec les coups clés prouvés EN PARALLÈLE.
//
//  Le vrai code est dans :
//    player/endgame_solver.{h,cpp}   le solveur (classe EndgameSolver), partagé
//                                    avec l'auto-apprentissage AlphaZero, le
//                                    joueur hybride et le joueur « Solveur
//                                    exact » des puzzles
//    app/yolah_app.cpp               PuzzleCheck, puzzle_check_start,
//                                    puzzle_lead : le verdict dans un thread à
//                                    part, avec un budget de 10 minutes
//  Ici ne reste que la configuration de l'éditeur : pas de règle du passe,
//  ordre « le plus rapide d'abord », force brute à ≤ 6 cases libres, partage à
//  partir de 12 cases libres. Le thread de l'app, son chronomètre et le bouton
//  « Arrêter » restent dans l'app ; ici, l'arrêt se demande par le drapeau
//  `stop`.
//
//  « Même code » veut dire : avec UN thread, ce solveur visite EXACTEMENT les
//  mêmes nœuds que EndgameSolver, dans le même ordre ; avec plusieurs, il rend
//  les mêmes valeurs (voir VÉRIFICATION à la fin de ce commentaire).
//
//  ---------------------------------------------------------------------------
//  RAPPEL EN TROIS LIGNES
//  ---------------------------------------------------------------------------
//  Chaque coup remplit une case libre et n'en libère aucune (le départ devient
//  un trou) : avec k cases libres, la partie finit en au plus k coups. La
//  recherche calcule le RESTANT (ce que le camp au trait gagnera encore, moins
//  ce que gagnera l'adversaire), qui ne dépend que du plateau : la table peut
//  le partager entre positions aux scores différents.
//
//      restant(position) = max sur les coups m de  gain(m) − restant(après m)
//      restant(partie finie) = 0
//
//  gain(m) vaut 1 pour un coup et 0 pour un passe dans le jeu normal ; il
//  change avec les objectifs à cibles (section 3).
//
//  ---------------------------------------------------------------------------
//  1. LA RECHERCHE PARTAGÉE : « YOUNG BROTHERS WAIT » (YBWC)
//  ---------------------------------------------------------------------------
//  Pourquoi on ne peut pas simplement donner un coup à chaque thread : dans
//  un alpha-bêta, le premier coup d'un nœud sert à resserrer la fenêtre des
//  suivants, et souvent il suffit à couper (les autres ne sont alors jamais
//  cherchés). Chercher tous les coups en même temps, c'est chercher beaucoup
//  de coups pour rien. Mesuré : partager seulement les coups de la RACINE
//  entre 8 threads est 2 fois PLUS LENT que la recherche à un thread.
//
//  La règle YBWC (utilisée par les solveurs d'échecs et d'Othello) :
//    • le FRÈRE AÎNÉ (le premier coup d'un nœud, le meilleur selon l'ordre)
//      est toujours cherché seul, par le thread qui est sur ce nœud ;
//    • s'il n'a pas coupé, les FRÈRES CADETS (les coups suivants) peuvent
//      être partagés : le nœud devient un « point de partage » (SplitPoint),
//      et les threads inactifs viennent y prendre des coups, un par un.
//  On ne partage que les gros nœuds (au moins 12 cases libres : en dessous,
//  le sous-arbre coûte moins cher que l'organisation du partage), et
//  seulement quand un thread est inactif.
//
//  Exemple, 3 threads, un nœud N à 4 coups a, b, c, d (dans l'ordre) :
//
//      thread T0 sur N :  cherche a seul (l'aîné)  →  pas de coupure
//                         N devient un point de partage, coups b, c, d
//
//                    point de partage de N
//                  ┌────────┬────────┬────────┐
//                  │  b     │  c     │  d     │   « next » : le prochain coup
//                  └────────┴────────┴────────┘     à distribuer
//                     ▲         ▲         ▲
//                    T0        T1        T2        (T1, T2 étaient inactifs)
//
//      T0 (le MAÎTRE de N) prend b, T1 prend c, T2 prend d. Chacun cherche
//      sur SA copie de la position. Quand un thread finit, il rapporte sa
//      valeur au point de partage (sous verrou) : meilleure valeur, alpha
//      relevé, et si la valeur atteint beta → COUPURE.
//
//  La coupure : si T1 trouve que c atteint beta, b et d ne servent plus à
//  rien. Le point de partage lève alors son drapeau `cutoff`, et tout thread
//  qui travaille SOUS ce nœud (même très profond) s'arrête : à chaque nœud,
//  stopped() remonte la chaîne des points de partage du thread et regarde
//  leurs drapeaux.
//
//      racine ── … ── N (partage, cutoff ?) ── … ── M (partage, cutoff ?) ── nœud courant
//                     ▲                             ▲
//                     └──── stopped() regarde toute la chaîne ────┘
//
//  Un thread qui s'arrête ainsi rend une valeur sans signification : elle
//  n'est ni rangée dans la table, ni utilisée (on teste stopped() après
//  chaque coup).
//
//  Le maître attend ses aides. Le nœud N (et son point de partage, qui vit
//  sur la PILE du maître) ne peut être quitté que quand plus personne ne
//  cherche un de ses coups (compteur `workers` à 0). Mais le maître n'attend
//  pas les bras croisés : il aide aux points de partage ouverts SOUS le sien
//  (par ses aides, dans leurs sous-arbres), les seuls dont la fin l'intéresse.
//
//  Un piège évité : quand la recherche au-dessus est arrêtée (coupure plus
//  haut, ou drapeau `stop`), le maître sort de sa boucle alors qu'il reste des
//  coups à distribuer. Il FERME donc le point de partage (next = nombre de
//  coups) avant d'attendre : sinon un thread inactif pourrait le rejoindre au
//  moment où le maître le quitte, et travailler sur une mémoire déjà libérée.
//
//  Ce que coûte le parallélisme : des nœuds en plus. Quand un coup gagne,
//  les threads ont déjà commencé les coups suivants, en vain. Mesuré
//  (12 positions de 24 à 26 cases libres) : 203 M nœuds à 1 thread, 286 M à
//  4, 440 M à 8. Au-delà de 8 threads, ce travail perdu grandit plus vite
//  que les cœurs ajoutés : l'éditeur s'arrête à 8.
//
//  Comparaisons faites (même banc) : séquentiel 18,6 s ; partage de la
//  racine seulement, 8 threads 38,1 s ; « Lazy SMP » (chaque thread cherche
//  toute la racine, seule la table est partagée) 12,0 s à 4 threads, 14,7 s
//  à 8 ; YBWC 5,9 s à 4, 5,3 s à 8 ; YBWC + coupures par la table (ETC) + tri
//  par recherches courtes (techniques d'Edax) 7,5 s à 4, 7,0 s à 8. Les
//  variantes perdantes sont gardées dans player/endgame_solver_{lazy,pro}.
//
//  ---------------------------------------------------------------------------
//  2. LA TABLE PARTAGÉE SANS VERROU
//  ---------------------------------------------------------------------------
//  Tous les threads lisent et écrivent la même table de transposition, à
//  chaque nœud : un verrou coûterait trop cher. Une entrée fait deux mots de
//  64 bits, écrits l'un après l'autre :
//
//      data  = [ 1 bit « valide » | … | coup (16 bits) | upper (8) | lower (8) ]
//      check = clé ⊕ data         (⊕ : ou exclusif)
//
//  Lecture : on lit data et check, puis on vérifie check ⊕ data == clé. Si
//  deux threads ont écrit la même case en même temps, on peut lire le data de
//  l'un et le check de l'autre : la vérification échoue presque sûrement, et
//  l'entrée est simplement ignorée. Une entrée déchirée ne peut donc PAS
//  passer pour une vraie borne : les preuves restent exactes. (Technique de
//  Hyatt et Mann, utilisée par les programmes d'échecs.) Au pire, une borne
//  utile est perdue, ce qui ne coûte que du temps.
//
//  Le bit « valide » distingue une case vide (data = 0, check = 0) d'une vraie
//  entrée : sans lui, une position de clé 0 lirait des bornes [0, 0].
//
//  ---------------------------------------------------------------------------
//  3. LES OBJECTIFS À CIBLES
//  ---------------------------------------------------------------------------
//  Un puzzle peut marquer des cases « cibles ». Une cible est prise par le
//  pion qui s'y pose, une seule fois (la case est ensuite occupée, puis trou).
//    • Points  (le jeu normal) : gain(coup) = 1, gain(passe) = 0 ;
//    • Collect (« plus de cibles ») : gain = 1 si le coup arrive sur une
//      cible, 0 sinon ; la valeur est l'écart de cibles prises ;
//    • First   (« première cible ») : le premier coup qui arrive sur une cible
//      GAGNE (+1 pour son camp) et la partie s'arrête là ; personne : 0 (nul).
//  Le restant ne dépend toujours que du plateau et du camp au trait (les
//  cibles sont fixées pour toute la vie du solveur) : la table marche telle
//  quelle. Deux bornes changent :
//    • aucune cible atteignable par aucun camp (dans les zones libres voisines
//      de leurs pièces) : plus rien à gagner, restant = 0 exactement ;
//    • Collect, un camp bloqué pour toujours : l'autre prend au plus les
//      cibles de sa zone, peut-être aucune.
//
//  ---------------------------------------------------------------------------
//  4. LE VERDICT ET LES COUPS CLÉS EN PARALLÈLE
//  ---------------------------------------------------------------------------
//  verify() fait comme dans doc/final_solver.h : « gagner de N » se prouve
//  comme « gagner » quand l'adversaire a N − 1 points (ou cibles) de plus.
//  La preuve principale utilise le solveur parallèle. Si le puzzle est
//  faisable et que le joueur a le trait, on cherche ensuite ses COUPS CLÉS
//  (ceux après lesquels l'objectif reste atteint) : une preuve par coup,
//  toutes indépendantes. Elles sont réparties sur les threads, chacun avec SON
//  solveur à un thread et sa table, qui prend le coup suivant de la liste
//  jusqu'à épuisement. (Si le puzzle est impossible, il n'y a aucun coup clé :
//  on ne les cherche pas.)
//
//  ---------------------------------------------------------------------------
//  DEUX DÉTAILS QUI ONT COMPTÉ
//  ---------------------------------------------------------------------------
//    • Le faux partage de cache. Un cœur lit la mémoire par lignes de 64
//      octets. Les compteurs de nœuds des threads, écrits à chaque nœud,
//      étaient voisins en mémoire, et le drapeau `cutoff`, lu à chaque nœud
//      par tous, partageait sa ligne avec le verrou du point de partage : à
//      chaque écriture, la ligne était retirée des caches de tous les autres
//      cœurs. Chacun a maintenant sa ligne (alignas(64)) : l'accélération à
//      8 threads est passée de 1,9 à 3,3.
//    • La copie d'une Yolah::MoveList garde son pointeur de fin vers la liste
//      ORIGINALE (sur la pile du maître). Le point de partage copie donc ses
//      coups dans un simple tableau.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_ybwc_check.cpp (construit par test/final_ybwc_check.sh) tire
//  des positions au hasard et compare :
//    • ce solveur à UN thread à EndgameSolver (même configuration, les trois
//      objectifs) : mêmes valeurs, mêmes coups, MÊME NOMBRE DE NŒUDS ;
//    • ce solveur à 8 threads à un négamax brut jusqu'à la fin (sans table,
//      sans ordre, sans bornes) : mêmes valeurs exactes, mêmes signes en
//      gagné/nul/perdu, et un coup qui atteint bien la valeur annoncée ;
//    • verify() au négamax brut (verdict et nombre de coups clés).
//  Toute modification de ce code doit être revérifiée ainsi.
// =============================================================================

#include "game.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace final_ybwc {

// Sur quoi se compte la partie (section 3).
enum class Goal { Points, Collect, First };

struct Options {
    Goal goal = Goal::Points;
    uint64_t targets = 0;   // les cases cibles (Collect, First), a1 = bit 0
    int tt_bits = 22;       // la table a 2^tt_bits entrées de 16 octets
    int threads = 1;        // 1 : la recherche séquentielle, nœud pour nœud
};

class Solver {
public:
    struct Result {
        Move move = Move::none();   // le meilleur coup à la racine (passe si bloqué)
        int  value = 0;             // valeur FINALE pour le camp au trait (en gagné/nul/perdu, seul
                                    // son signe est exact)
        bool complete = false;      // faux : interrompu par `stop`, `move` et `value` ne valent rien
        uint64_t nodes = 0;         // nœuds visités par cette recherche (tous threads)
    };

    explicit Solver(Options options = Options());

    // Meilleur coup et valeur finale de `position`. wld_only : seulement
    // gagné / nul / perdu. stop : drapeau mis à vrai par un autre thread pour
    // arrêter (peut être nul). current : la valeur déjà acquise par le camp au
    // trait, quand ce n'est pas l'écart des scores de la position (Collect : les
    // cibles déjà prises, camp au trait − adversaire ; First : 0) ; nul : l'écart
    // des scores.
    Result solve(const Yolah& position, bool wld_only, const std::atomic_bool* stop = nullptr, const int* current = nullptr);

private:
    // ── La table partagée (section 2) ──
    struct Entry {
        std::atomic<uint64_t> check{0}, data{0};
    };
    static constexpr uint64_t VALID = uint64_t(1) << 63;
    // data = valide | coup << 16 | upper << 8 | lower (les bornes sur 8 bits signés).
    static uint64_t pack(int lower, int upper, Move m) {
        return VALID | uint64_t(uint8_t(int8_t(lower))) | uint64_t(uint8_t(int8_t(upper))) << 8 | uint64_t(m.raw()) << 16;
    }
    bool probe(uint64_t hash, int& lower, int& upper, Move& move) const;
    void store(uint64_t hash, int alpha, int beta, int value, Move move);

    // ── Le partage (section 1) ──
    // Un nœud dont les coups cadets sont distribués. Protégé par `m`, sauf
    // `workers` et `cutoff` (atomiques).
    struct SplitPoint {
        SplitPoint* parent = nullptr;   // le point de partage pour lequel travaillait le maître
        Yolah pos;                      // la position du nœud, et sa clé
        uint64_t hash = 0;
        Move moves[Yolah::MAX_NB_MOVES];   // un tableau simple : voir « deux détails »
        size_t nb_moves = 0;
        size_t next = 0;                // le prochain coup à distribuer
        int alpha = 0, beta = 0, best = -127;
        Move best_move = Move::none();
        std::atomic<int> workers{0};    // threads sur un de ses coups (le maître compris)
        std::mutex m;
        // Lu à chaque nœud par tous les threads en dessous : sur sa propre ligne de cache.
        alignas(64) std::atomic<bool> cutoff{false};
        char pad[63];
        // Reste-t-il des coups à distribuer ? (sous m)
        bool open() const { return !cutoff.load(std::memory_order_relaxed) && next < nb_moves; }
    };
    // Un thread de la recherche : son compteur de nœuds (une ligne de cache à
    // lui : il l'écrit à chaque nœud) et le point de partage pour lequel il
    // travaille.
    struct alignas(64) Worker {
        uint64_t nodes = 0;
        SplitPoint* sp = nullptr;
    };

    // En dessous de ce nombre de cases libres : ni table ni tri (voir doc/final_solver.h).
    static constexpr int BRUTE_FORCE_FREE = 6;
    // Un nœud n'est partagé qu'à partir de ce nombre de cases libres.
    static constexpr int SPLIT_MIN_FREE = 12;
    // gain() d'un coup qui arrive sur une cible avec l'objectif First : la partie est gagnée.
    static constexpr int FIRST_WIN = 1000;

    Options options;
    std::unique_ptr<Entry[]> table;
    size_t table_size = 0;
    const std::atomic_bool* stop = nullptr;
    std::mutex sp_mutex;                 // protège `active`
    std::vector<SplitPoint*> active;     // les points de partage ouverts aux aides
    std::atomic<int> nb_active{0};       // active.size(), lisible sans verrou
    std::condition_variable sp_cv;       // réveille les aides inactifs
    std::atomic<int> idle{0};            // nombre de threads inactifs
    std::atomic<bool> solving{false};    // une recherche est en cours (les aides tournent)
    uint64_t split_gen = 0;              // nombre de splits faits (sous sp_mutex) : réveille les aides

    int  gain(Move m) const;
    bool stopped(const Worker& w) const;
    int  search(Worker& w, Yolah& yolah, uint64_t hash, int alpha, int beta);
    int  move_value(Worker& w, Yolah& yolah, uint64_t hash, Move m, int alpha, int beta);
    void order_moves(Yolah& yolah, Move tt_move, Yolah::MoveList& moves) const;
    void split(Worker& w, Yolah& yolah, uint64_t hash, const Yolah::MoveList& moves, size_t first,
               int& alpha, int beta, int& best, Move& best_move);
    void work_on(Worker& w, SplitPoint& sp);
    bool help(Worker& w, const SplitPoint* under);
    void helper_loop(Worker& w);
};

// ── Le verdict d'un puzzle (section 4) ──────────────────────────────────────
enum class Verdict { Feasible, Impossible, Unknown };

struct Puzzle {
    Yolah start;                    // la position (bonus compris dans les scores)
    uint8_t human = Yolah::BLACK;   // le camp du joueur
    int margin = 1;                 // « au moins `margin` de plus » (points, ou cibles)
    Goal goal = Goal::Points;
    uint64_t targets = 0;
};

struct Check {
    Verdict verdict = Verdict::Unknown;
    int  lead = 0;                  // > 0 objectif atteint, 0 nul, < 0 perdu (si prouvé)
    bool keys_counted = false;      // faisable, le joueur a le trait, coups clés tous prouvés
    std::vector<Move> keys;         // les coups clés (si keys_counted)
    int  nb_moves = 0;              // ses coups légaux (si keys_counted)
};

constexpr int MAX_FREE = 44;        // au-delà, l'éditeur ne tente pas de preuve

// Le verdict de `p`, la preuve principale sur `threads` threads, les coups clés
// répartis sur autant de threads. `stop` arrête tout (Inconnu avant la preuve
// principale ; faisable sans coups clés pendant leur comptage).
Check verify(const Puzzle& p, int threads, const std::atomic_bool* stop = nullptr);

}  // namespace final_ybwc

#endif
