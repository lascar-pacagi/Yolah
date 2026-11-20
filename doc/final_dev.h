#ifndef FINAL_DEV_H
#define FINAL_DEV_H

// =============================================================================
//  FINAL_DEV (dev_u9_spsa) — Version épurée et commentée de MinMaxNNUE_DevPlayer
// =============================================================================
//
//  Ce fichier (et son compagnon doc/final_dev.cpp) présente le code exact du
//  moteur « Minimax NNUE dev » (configuration `dev_u9_spsa`) utilisé dans
//  l'application Yolah, débarrassé de la centaine d'options expérimentales
//  testées et écartées au fil des campagnes A à U (plus de 250 variantes et
//  des centaines de milliers de parties sur cluster).
//
//  Objectif de ce document-code :
//    Permettre une lecture linéaire, claire et approfondie du moteur tel qu'il
//    joue réellement, avec toutes ses constantes explicites et des commentaires
//    détaillant le « pourquoi » et le « comment » de chaque mécanisme.
//
//  ---------------------------------------------------------------------------
//  Vue d'ensemble des briques actives dans `dev_u9_spsa` (et gains Elo mesurés) :
//  ---------------------------------------------------------------------------
//
//  1. APPROFONDISSEMENT ITÉRATIF, PVS & FENÊTRES D'ASPIRATION ADAPTATIVES (B, A2)
//     • Recherche à variation principale (PVS) en fail-soft : premier coup cherché
//       à fenêtre pleine (alpha, beta), tous les suivants à fenêtre nulle
//       (alpha, alpha + 1), re-cherchés à fenêtre pleine uniquement s'ils
//       battent alpha.
//     • Fenêtres d'aspiration adaptatives à la racine (à partir de depth = 4) :
//       demi-largeur δ = max(345, 0.92 × moyenne_glissante(|v_d − v_{d−1}|)).
//       En cas d'échec bas/haut, la fenêtre est recentrée sur la valeur fail-soft
//       et élargie d'un facteur 1.34 (−22 % de nœuds à profondeur 12, mesuré
//       avec 300, 1.25 et 1.5 ; valeurs actuelles : SPSA, voir plus bas).
//     • Tri des coups à la racine (« root_fix ») : meilleur coup en tête, les
//       suivants triés par la taille (en nœuds) du sous-arbre qu'il a fallu
//       explorer pour les réfuter, cumulée sur les tentatives de la profondeur.
//
//  2. TABLE DE TRANSPOSITION DÉDIÉE À YOLAH (`SearchTable`, H)
//     • Chaque entrée stocke simultanément les DEUX bornes [lower, upper] de la
//       position (au lieu d'écraser une borne inférieure par une borne supérieure).
//     • Tirant parti de l'absence de cycles à Yolah (chaque coup détruit une case
//       pour toujours), le remplacement évince en priorité les positions qui ont
//       PLUS de cases libres que la racine actuelle (devenues inatteignables).
//     • Aux nœuds PV, la table ne coupe que sur une valeur exacte (lower == upper)
//       ou une fin de partie prouvée (|borne| > WIN), jamais sur une simple borne
//       issue d'un nœud non-PV où l'élagage agressif s'est appliqué.
//     • Prefetch matériel (`__builtin_prefetch`) dès qu'un coup est choisi, avant
//       même de jouer le coup et de mettre à jour l'accumulateur.
//
//  3. ÉVALUATION NNUE PERSPECTIVE, ACCUMULATEURS PARESSEUX & CORRECTION (D, S, U6)
//     • Réseau NNUE vu du camp au trait (`NNUE_Persp`), avec première couche
//       factorisée par phase (4 jeux de poids fc1 + cache de rafraîchissement) et
//       8 têtes de phase selon le nombre de cases libres.
//     • Produit creux (`sparse_h1`, +61 Elo à 0.2 s, +35 Elo à 1 s) : dans la
//       couche 1024 → 64, seuls les groupes de 4 entrées non nulles après ReLU
//       sont multipliés (AVX2/VNNI).
//     • Accumulateurs paresseux (`lazy_accumulator`) : on ne met PAS à jour le
//       réseau à chaque coup joué/déjoué dans l'arbre. On empile simplement les
//       coups joués, et lorsqu'une feuille (ou un élagage statique) a réellement
//       besoin de l'évaluation, on remonte au dernier ancêtre valide et on
//       descend en une passe fusionnée par pli (aucun `undo` nécessaire).
//     • Cache d'évaluations sans verrou (`eval_cache`, 2^20 entrées de 64 bits
//       atomiques) : évite de recalculer le réseau quand la même feuille est
//       atteinte par transposition de coups.
//     • Historique de correction (`corr_hist = 2`, +10.6 Elo) : table indexée par
//       un hachage des 8 pièces (noir + blanc) et le camp au trait, qui apprend
//       l'écart (valeur_recherche − éval_statique) et corrige l'évaluation du
//       réseau aussi bien pour l'élagage interne qu'aux feuilles (depth == 0).
//
//  4. ÉLAGAGE AVANT LA BOUCLE DE COUPS (aux nœuds non-PV uniquement)
//     • Reverse Futility Pruning (RFP, F1, +76 Elo) à depth ≤ 5 :
//       si `static_eval − 2609 × depth ≥ beta`, on coupe immédiatement.
//     • Multi-ProbCut bidirectionnel (MPC, O, +57 Elo à 1 s) à depth ≥ 4 :
//       recherche la position courante à profondeur réduite `d' = max(1, 0.46 × d)` ;
//       si elle dépasse `beta + 4054 + 644 × (d − d')` (quand `static_eval ≥ beta`)
//       ou tombe sous `alpha − margin` (quand `static_eval ≤ alpha`), on coupe.
//     • Proxy Cut (N, +47 Elo à 0.2 s, +59 Elo à 1 s) à depth ≥ 4 :
//       à Yolah, le *null move pruning* classique (passer son tour) échoue (−81 Elo)
//       car le jeu est dominé par le zugzwang (jouer détruit une case de son propre
//       espace). Le Proxy Cut le remplace par un VRAI coup ordinaire « témoin »
//       (le meilleur coup ordinaire au tri), cherché à profondeur réduite
//       `d − 1 − (3 + d / 4)` contre `beta`. Si même ce coup ordinaire tient
//       `beta`, alors le meilleur coup du nœud tiendra `beta` a fortiori (même en
//       zugzwang : `valeur(meilleur) ≥ valeur(témoin)` est toujours vrai).
//
//  5. GÉNÉRATION PAR ÉTAPES & ORDONNANCEMENT DES COUPS (C, J, K, U4, U7)
//     • Génération par étapes (`staged`) : s'il existe un coup de table légal
//       (`tt_move`), on le teste en étape 1 AVANT de générer et noter les autres
//       coups (−4 % de nœuds, −6 % de temps).
//     • Hiérarchie de tri des coups en étape 2 (`score_moves`) :
//         1. Coup de la table de transposition (`SCORE_TT = 2^30`)
//         2. Contre-coup (`SCORE_COUNTER = 2^29 − 2`) : le dernier coup ayant
//            réfuté le coup que l'adversaire vient de jouer (`played[ply − 1]`).
//         3. Coups ordinaires, triés par la somme :
//              score(m) = history[us][from][to]
//                       + cont1[us][prev.to][m.to]         (U4 : +32.6 Elo)
//                       + (intercepte(prev.to, m.to) ? 2000 : 0) (U7 : +17.0 Elo)
//                       + (est_killer(m) ? 4667 : 0)       (U7 : +17.7 Elo)
//                       + 1038 × (couverture(après m) − couverture(avant))
//                                                      (U19-U22, voir ci-dessous)
//            La COUVERTURE de notre groupe est le nombre de cases libres que
//            nos pièces atteignent collectivement en un coup (union de leurs
//            rayons de dame, sans tenir compte de l'adversaire). Un coup qui
//            l'agrandit ouvre de l'espace à notre groupe ; un coup qui la
//            réduit l'enferme. Ce terme REMPLACE l'influence Voronoï de dev_u8
//            (`512 × influence_fast(m)` à depth ≥ 4, J : +36 Elo) : contre
//            dev_u8, +17.9 Elo [+7.6, +28.2] à 0.2 s/coup (2000 parties) et
//            +6.9 Elo [+0.9, +13.0] à 1 s/coup (4000 parties). Il est appliqué
//            à TOUTES les profondeurs : bien moins cher que l'inondation
//            Voronoï (8 calculs de rayons par coup), il paie même près des
//            feuilles (seuil 2 ou 3 : ≈ 0 Elo ; ajouté EN PLUS de l'influence : ≈ 0).
//            Noter que dans `dev_u9_spsa` (comme dans `dev_u8`), les coups tueurs (`killer1`, `killer2`) ne
//            sont PAS placés brutalement avant tous les coups ordinaires : ils
//            reçoivent un bonus de +4667 sur leur score d'historique, ce qui
//            permet à un coup ayant un fort historique de continuation (`cont1`)
//            et d'interception de passer devant un killer hors contexte.
//
//  6. ÉLAGAGE PAR COUP, POCHES SCELLÉES & RÉDUCTIONS LMR (E, F3, U5, U7)
//     • Late Move Pruning (LMP, F3, +35 Elo) à depth ≤ 4 (non-PV) :
//       au-delà de `2 + depth²` coups essayés (et si un coup non perdant a déjà
//       été trouvé), on coupe la fin de la liste.
//     • Poches scellées comme réserves de tempos (`reserve`, U5, +10.3 Elo, EXACT) :
//       si un groupe de nos pièces est enfermé dans une composante connexe privée
//       de ≤ 20 cases libres inaccessible à l'adversaire, l'ordre dans lequel on
//       remplit cette poche n'a aucun impact sur le reste du plateau. On résout
//       exactement la récolte maximale `h` de la poche (`harvest_exact`), et on
//       ne garde dans la boucle qu'UN SEUL coup qui préserve `h − 1` coups futurs.
//     • Late Move Reductions (LMR, E + U7) pour `depth ≥ 3` et `i ≥ 1` :
//       réduction de base `r = 1.1158 + ln(depth) × ln(i + 1) / 1.4152`, modulée par :
//         − 1058/1024 pli aux nœuds PV
//         − 829/1024 pli pour le contre-coup (`SCORE_COUNTER`)
//         − history[us][from][to] / 6945 plis (historique principal seul)
//         − 1 pli si le coup intercepte le dernier coup adverse (`intercept_lmr`)
//         − 1 pli si ce couple (prev.to → m.to) a souvent contredit sa réduction
//           par le passé (`lmrh[us][prev.to][m.to] ≥ 4453`, U7 +12.3 Elo).
//
//  7. PARALLÉLISME LAZY SMP (L)
//     • `nb_threads` travailleurs lancent l'approfondissement itératif sur la
//       même racine en partageant sans verrou `SearchTable` et `eval_cache`, mais
//       avec chacun son `Worker` privé (historiques, killers, fenêtres).
//
//  8. RÉGLAGE DES CONSTANTES PAR SPSA (U23)
//     • Les 29 constantes numériques de la recherche (fenêtres d'aspiration,
//       LMR, RFP, MPC, Proxy Cut, LMP, bonus d'ordonnancement, historiques,
//       correction) ont été réglées ENSEMBLE par SPSA, comme le fait Fishtest
//       pour Stockfish (nnue/spsa.py) : à chaque lot, toutes sont décalées d'un
//       ±c aléatoire, θ+ joue θ− sur des ouvertures équilibrées, et chaque
//       constante se déplace vers le côté qui a gagné. 50 004 paires de
//       parties (100 008 parties) à 0.2 s/coup.
//     • Gain contre dev_u9 (mêmes heuristiques, constantes réglées à la main) :
//       +14.4 Elo [+4.2, +24.7] à 0.2 s et +18.3 Elo [+9.2, +27.3] à 1 s
//       (2000 parties chacun) : le réglage tient à une cadence 5 fois plus
//       longue que celle où il a été fait. Contre dev_u8 (avant la couverture),
//       à 1 s : +19.3 Elo [+10.5, +28.1] (2000 parties).
//     • Les changements les plus nets : LMP plus agressif (2 + d² coups au lieu
//       de 4 + d², jusqu'à depth 4), Proxy Cut plus réduit (R = 3 + d/4 au lieu
//       de 2 + d/4), MPC dès depth 4 avec une marge plus large, RFP jusqu'à
//       depth 5 avec une marge plus faible, fenêtres d'aspiration plus serrées.
//       Le poids de la couverture (1038) reste au réglage trouvé à la main.
//       Les anciennes valeurs sont rappelées entre crochets dans les constantes.
// =============================================================================

#include "player.h"
#include "heuristic.h"
#include "search_table.h"
#include "nnue_persp.h"
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

class FinalDevPlayer : public Player {
public:
    struct Result {
        Move     move     = Move::none();
        int16_t  value    = 0;
        uint8_t  depth    = 0;     // dernière profondeur entièrement terminée
        uint64_t nb_nodes = 0;
        uint64_t nb_hits  = 0;
        double   seconds  = 0;
    };

    // Constructeur : charge le réseau NNUE perspective, alloue la table de
    // transposition Yolah, le cache d'évaluations et la table LMR logarithmique.
    FinalDevPlayer(uint64_t microseconds,
                   size_t tt_size_mb,
                   const std::string& nnue_persp_filename,
                   bool verbose = false,
                   size_t nb_threads = 4);

    Move play(Yolah yolah) override;
    Result search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds);
    void clear_table(bool keep_ordering = false);
    std::string info() override;
    json config() override;
    std::vector<Move> principal_variation(Yolah yolah, int max_len = 16) const;

private:
    // ── Échelle des valeurs de recherche ─────────────────────────────────────
    // Les calculs internes se font en `int` (pas de débordement lors des
    // décalages de fenêtres) et sont stockés en `int16_t` dans la table.
    //   • [-WIN + 1, WIN - 1] : évaluations du réseau NNUE (±30 000 = tanh ±1)
    //   • ±(WIN + écart)      : fin de partie exacte (une vraie victoire d'1 point
    //                           vaut WIN + 1 > toute évaluation heuristique).
    static constexpr int INFINITE = 32000;
    static constexpr int WIN      = heuristic::MAX_VALUE;   // 30 000
    static constexpr int EVAL_NONE = -32768;

    // ── Constantes des heuristiques de `dev_u9_spsa` ─────────────────────────
    // Les valeurs viennent du SPSA de dev_u9 (U23 : 29 paramètres, 100 008 parties
    // à 0,2 s ; +14,4 Elo à 0,2 s, +18,3 à 1 s contre dev_u9). Entre crochets :
    // la valeur avant le SPSA.
    // 1. Fenêtres d'aspiration adaptatives (B, A2)
    static constexpr int    ASPIRATION_MIN_DELTA   = 345;     // [300]
    static constexpr double ASPIRATION_K           = 0.9187;  // [1.25]
    static constexpr double ASPIRATION_GROW        = 1.3372;  // [1.5]
    static constexpr int    ASPIRATION_STATS_DEPTH = 4;

    // 2. Late Move Reductions logarithmiques (E, U7)
    static constexpr double LMR_BASE      = 1.1158;  // [1.25]
    static constexpr double LMR_DIVISOR   = 1.4152;  // [1.5]
    static constexpr int    LMRH_THRESHOLD = 4453;  // seuil d'historique d'erreurs LMR [4000]
    static constexpr int    LMR_PV         = 1058;  // −1058/1024 pli en nœud PV [1024]
    static constexpr int    LMR_SPECIAL    = 829;   // −829/1024 pli pour le contre-coup [1024]
    static constexpr int    LMR_HIST_DIV   = 6945;  // − hist / 6945 plis [8192]

    // 3. Reverse Futility Pruning (F1)
    static constexpr int RFP_DEPTH  = 5;              // [4]
    static constexpr int RFP_MARGIN = 2609;           // par pli de profondeur [3000]

    // 4. Multi-ProbCut bidirectionnel (O)
    static constexpr int MPC_DEPTH          = 4;     // [5]
    static constexpr int MPC_RATIO          = 46;    // d' = max(1, depth * 46 / 100) [40]
    static constexpr int MPC_MARGIN         = 4054;  // [4000]
    static constexpr int MPC_MARGIN_PER_PLY = 644;   // [500]

    // 5. Proxy Cut (N)
    static constexpr int PROXY_DEPTH     = 4;
    static constexpr int PROXY_REDUCTION = 4;        // R = 4 + depth / 4 - 1 [3]

    // 6. Late Move Pruning (F3)
    static constexpr int LMP_DEPTH = 4;                // [3]
    static constexpr int LMP_MOVES = 2;              // garde 2 + depth² coups [4]

    // 7. Poches scellées comme réserves de tempos (U5)
    static constexpr int RESERVE_MAX_SQUARES = 20;   // taille max d'une poche privée
    static constexpr int HARVEST_BUDGET      = 200000;

    // 8. Ordonnancement des coups et tables d'historique (C, J, U4, U7)
    static constexpr int HISTORY_MAX      = 16384;
    static constexpr int KILLER_BONUS     = 4667;    // killer_mode = 1 : bonus sur hist(m) [4000]
    static constexpr int HIST_BONUS_A     = 16;      // bonus d'historique à la coupure :
    static constexpr int HIST_BONUS_B     = 34;      //   min(A·d² + B·d, CAP) [16, 32, 2000]
    static constexpr int HIST_BONUS_CAP   = 2166;
    static constexpr int INTERCEPT_BONUS  = 2000;    // coup sur rayon ou voisin 3×3 de prev.to
    static constexpr int COVER_DEPTH      = 1;       // couverture à toutes les profondeurs (U20)
    static constexpr int COVER_WEIGHT     = 1038;    // poids de (couverture après − avant) (U20-U22) [1024]

    // Scores spéciaux de tri (au-dessus de tout score d'historique)
    static constexpr int SCORE_TT           = 1 << 30;
    static constexpr int SCORE_KILLER1      = 1 << 29;
    static constexpr int SCORE_COUNTER      = SCORE_KILLER1 - 2;
    static constexpr int SCORE_ARTICULATION = 1 << 28; // plafond sous lequel vivent les coups ordinaires

    // 9. Historique de correction de l'évaluation statique (U4/U6)
    static constexpr int CORR_TABLE_SIZE = 16384;
    static constexpr int CORR_SCALE      = 8;        // bonus = (best - static_eval) * depth / 8
    static constexpr int CORR_MAX        = 7917;     // [8000]

    // 10. Cache d'évaluations des feuilles (D)
    static constexpr int EVAL_CACHE_BITS = 20;       // 2^20 = 1 048 576 entrées (8 Mo)

    // ── État partagé entre les threads ───────────────────────────────────────
    const uint64_t thinking_time;
    const size_t   tt_size_mb;
    const std::string weights_filename;
    const bool     verbose;
    const size_t   nb_threads;
    std::atomic_bool stop = false;

    // Table de transposition dédiée à Yolah (64 octets par cluster, 2 bornes)
    std::unique_ptr<SearchTable> yolah_table;

    // Réseau NNUE perspective (sparse_h1 activé)
    std::unique_ptr<NNUE_Persp> persp;
    using Accumulator = NNUE_Persp::Accumulator;

    // Table des réductions LMR précalculées en 1/1024 de pli : lmr_table[depth][move_nb]
    int lmr_table[64][Yolah::MAX_NB_MOVES]{};

    // Cache d'évaluations atomique (1 mot de 64 bits par entrée : clé 47 bits | valeur 16 bits | bit valide)
    std::vector<std::atomic<uint64_t>> eval_cache;

    // ── Mémoire persistante propre à chaque thread (Worker) ──────────────────
    // Conservée d'un coup de la partie au suivant (avec vieillissement ÷ 2 de
    // l'historique principal à chaque nouveau coup). Ne pas partager ces tables
    // évite tout verrou et fait diverger utilement les arbres de Lazy SMP.
    struct Worker {
        // Historique principal : history[joueur][ origine ][ destination ] ∈ [-16384, 16384]
        int16_t history[2][SQUARE_NB][SQUARE_NB]{};

        // Contre-coup : countermoves[ origine_adverse ][ dest_adverse ] = meilleur coup de réfutation
        Move countermoves[SQUARE_NB][SQUARE_NB]{};

        // U4 (7) : Historique de continuation à 1 pli (+32.6 Elo, le gain majeur de U1–U10) :
        //   cont1[joueur][ destination_adverse_précédente ][ notre_destination ] ∈ [-16384, 16384]
        //
        // Pourquoi indexer par (prev.to, m.to) et non par (prev.from, prev.to, m.from, m.to) ?
        //   • Une table complète 4D (2 × 64^4 = 32 Mo) serait beaucoup trop creuse : au cours
        //     d'une recherche d'une seconde, la plupart des quadruplets ne seraient vus qu'une
        //     seule fois.
        //   • À Yolah, l'urgence tactique créée par le dernier coup adverse dépend avant tout de
        //     la case où sa pièce VIENT D'ATTERRIR (`prev.to`), et la réfutation dépend de la
        //     case clé que nous allons OCCUPER (`m.to`), quelle que soit celle de nos 4 pièces
        //     qui s'y rend.
        //   • Avec une table 64 × 64 par camp (8 Ko par camp, qui tient entièrement dans le
        //     cache L1), l'apprentissage converge très vite dès les premières profondeurs et se
        //     généralise immédiatement à toutes nos pièces pouvant atteindre `m.to`.
        int16_t cont1[2][SQUARE_NB][SQUARE_NB]{};

        // U7 (C5) : Historique contextuel des réductions LMR erronées (+12.3 Elo) :
        //   lmrh[joueur][ dest_adverse_précédente ][ notre_destination ]
        // Même indexation (prev.to, m.to) que `cont1` : mémorise les réponses `m.to` à `prev.to`
        // qui ont été réduites par LMR (`r_used > 0`) mais qui ont finalement battu `alpha`.
        // Dès que `lmrh >= 4000`, ce couple reçoit 1 pli de réduction LMR en moins.
        int16_t lmrh[2][SQUARE_NB][SQUARE_NB]{};

        // U4/U6 (6) : Historique de correction de l'évaluation statique :
        // corr[joueur][ hachage_14_bits_des_pièces ] ∈ [-8000, 8000]
        int16_t corr[2][CORR_TABLE_SIZE]{};

        // Fenêtres d'aspiration adaptatives : somme pondérée (décroissance 0.9)
        // des écarts |v_d − v_{d−1}| observés par ce thread.
        double asp_sum = 0;
        double asp_n   = 0;
    };
    std::vector<std::unique_ptr<Worker>> workers;

    // ── État d'un coup à la racine ───────────────────────────────────────────
    struct RootMove {
        Move     move;
        uint64_t nodes = 0;   // nœuds explorés sous ce coup (cumulés sur la profondeur courante)
        int      value = 0;
    };

    // ── État local d'une recherche pour un thread (`Search`) ─────────────────
    struct Search {
        int      id       = 0;         // 0 = thread principal, 1..k = threads assistants (Lazy SMP)
        Worker*  w        = nullptr;
        uint8_t  depth    = 0;
        int16_t  value    = 0;
        Move     move     = Move::none();
        size_t   nb_nodes = 0;
        size_t   nb_hits  = 0;
        uint64_t nb_evals = 0, nb_eval_hits = 0;

        // Deux coups tueurs par pli de l'arbre
        Move killer1[Yolah::MAX_NB_PLIES]{};
        Move killer2[Yolah::MAX_NB_PLIES]{};

        // Pile des coups joués le long de la branche courante :
        // played[p] est le coup joué au pli p (sert au contre-coup, à cont1, à
        // l'interception, à lmrh et à la mise à jour paresseuse de l'accumulateur).
        Move played[Yolah::MAX_NB_PLIES + 1]{};

        // Évaluation statique corrigée à chaque pli (EVAL_NONE si non calculée)
        int eval_stack[Yolah::MAX_NB_PLIES + 2]{};

        // Pile d'accumulateurs NNUE paresseux : accs[p] est valide ssi acc_ok[p] == true.
        Accumulator accs[Yolah::MAX_NB_PLIES + 1];
        bool        acc_ok[Yolah::MAX_NB_PLIES + 1]{};
        NNUE_Persp::RefreshCache refresh;

        // Liste ordonnée des coups à la racine
        std::vector<RootMove> root_moves;
    };

    // ── Accès à la table de transposition ────────────────────────────────────
    struct TTView {
        bool found = false;
        Move move  = Move::none();
        int  depth = 0;
        int  lower = -32767;
        int  upper =  32767;
    };
    TTView tt_probe(uint64_t hash) const;
    void   tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta,
                    int value, Move move, bool pv = false);
    Move   tt_move(uint64_t hash) const;

    // ── Méthodes internes du moteur ──────────────────────────────────────────
    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    void iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth);
    int  root_search(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth, Move& res);
    int  negamax(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth);
    int  search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i,
                     int reduction, int alpha, int beta, int depth);
    int  late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const;

    void score_moves(const Yolah& yolah, const Search& s, Move tt_move,
                     const Yolah::MoveList& moves, int* scores, int depth) const;
    static uint64_t coverage(uint64_t mine, uint64_t occupied);
    static bool is_legal(const Yolah& yolah, Move m);
    static Move pick_move(Yolah::MoveList& moves, int* scores, size_t i, size_t n);
    void update_history(Search& s, uint8_t player, Move m, int bonus);

    int  evaluate(const Yolah& yolah, Search& s, uint64_t hash);
    int  raw_network_value(const Yolah& yolah, Search& s);
    void print_pv(Yolah yolah, uint64_t hash, int8_t depth);
};

#endif // FINAL_DEV_H
