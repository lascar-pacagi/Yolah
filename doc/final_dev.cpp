// =============================================================================
//  FINAL_DEV (dev_u9_spsa) — Implémentation épurée et commentée du joueur Minimax NNUE
// =============================================================================
//
//  Ce fichier rassemble uniquement le code effectivement exécuté par la
//  configuration `dev_u9_spsa` (utilisée dans l'application Yolah pour « Minimax
//  NNUE dev » et pour la barre d'analyse). Lire d'abord le grand commentaire de
//  doc/final_dev.h : règles du jeu, glossaire (fenêtre, fail-soft, nœud PV…),
//  échelle des valeurs, vue d'ensemble et constantes.
//
//  Plan de lecture du fichier (l'ordre suit celui d'une recherche) :
//    1. Constructeur, lancement de la recherche et parallélisme Lazy SMP
//    2. Approfondissement itératif & fenêtres d'aspiration adaptatives
//    3. Recherche à la racine (`root_search`) & tri par taille de sous-arbre
//    4. Recherche d'un coup (`search_move` : LMR + PVS) & formule LMR
//    5. Fonctions géométriques auxiliaires (poches scellées,
//       récolte exacte `harvest_exact`, clé de correction `corr_key`)
//    6. Nœuds internes (`negamax`) :
//         a) Fin de partie
//         b) Table de transposition (`SearchTable`)
//         c) Feuilles
//         d) Évaluation statique corrigée, RFP, MPC
//         e) Génération par étapes (étape 1 : `tt_move` avant génération)
//         f) Proxy Cut (substitut au null-move par coup ordinaire témoin)
//         g) Boucle des coups : étape 2, Late Move Pruning (LMP),
//            poches scellées (`reserve`), LMR contextuel (interception, `lmrh`)
//         h) Coupure bêta : mise à jour de la table, de la correction,
//            des killers, de l'historique, de la continuation (`cont1`)
//            et du contre-coup
//    7. Ordonnancement des coups (`score_moves`) & couverture (`coverage`)
//    8. Évaluation NNUE paresseuse & cache d'évaluations
//    9. Accès à la table de transposition (`tt_probe`, `tt_store`) & variation principale
//
//  Fonctions du reste du projet utilisées ici :
//    • `attacks_bb(sq, occupied)` (game/magic.h) : les cases atteintes par une
//      dame posée en `sq`, rayons arrêtés par la première case de `occupied`
//      (cette case bloquante est INCLUSE : on la retire ensuite avec
//      `& free_squares()` ou `& ~occupied`). Une seule lecture de table
//      (« magic bitboards »).
//    • `square_bb(s)` = 1 << s ; `pop_lsb(b)` retire et renvoie la plus petite
//      case de `b` ; `around(b)` (misc/misc.h) = les 8 cases voisines (roi) de
//      l'unique case de `b`.
//    • `zobrist::hash(yolah)` : hachage 64 bits de la position ;
//      `zobrist::update(h, player, m)` : le hachage après le coup `m` de `player`,
//      calculé par différence (sans jouer le coup).
//    • `yolah.play(m)` / `yolah.undo(m)` : jouer / déjouer (y compris la passe
//      Move::none()) ; `yolah.moves(list)` : coups légaux du camp au trait (la
//      liste contient la seule passe Move::none() s'il n'en a aucun).
// =============================================================================

#include "final_dev.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>

using std::cout;

// =============================================================================
//  1. CONSTRUCTEUR, GESTION DU TEMPS & PARALLÉLISME LAZY SMP
// =============================================================================

FinalDevPlayer::FinalDevPlayer(uint64_t microseconds,
                               size_t tt_size_mb,
                               const std::string& nnue_persp_filename,
                               bool verbose,
                               size_t nb_threads)
    : thinking_time(microseconds),
      tt_size_mb(tt_size_mb),
      weights_filename(nnue_persp_filename),
      verbose(verbose),
      nb_threads(std::max<size_t>(1, nb_threads)) {

    // 1. Un `Worker` privé par thread (historiques, killers, fenêtres d'aspiration).
    //    Il vit aussi longtemps que le joueur : ce qui est appris pendant un coup
    //    sert aux coups suivants de la partie.
    for (size_t i = 0; i < this->nb_threads; i++) {
        workers.push_back(std::make_unique<Worker>());
    }

    // 2. Chargement du réseau NNUE vu du camp au trait (perspective + 8 têtes de phase)
    //    et activation du produit matriciel creux (sparse_h1 : +61 Elo à 0.2 s).
    //    `sparse` ne change aucune valeur, seulement la vitesse.
    persp = std::make_unique<NNUE_Persp>(weights_filename);
    persp->sparse = true;

    // 3. Allocation de la table de transposition dédiée à Yolah (SearchTable)
    //    et du cache d'évaluations des feuilles (2^20 entrées de 8 octets = 8 Mo,
    //    initialisées à 0 = « vide », puisque le bit 0 d'une entrée occupée vaut 1).
    yolah_table = std::make_unique<SearchTable>(tt_size_mb);
    eval_cache  = std::vector<std::atomic<uint64_t>>(size_t(1) << EVAL_CACHE_BITS);

    // 4. Précalcul de la table logarithmique des Late Move Reductions (en 1/1024 de pli) :
    //        r(d, n) = 1.1158 + ln(d) × ln(n) / 1.4152      (d = profondeur, n = rang + 1)
    //    Cette table ne donne que la réduction DE BASE. Au moment de chercher le
    //    coup, `late_move_reduction_of` (section 4) lui retire trois ajustements
    //    qui ne sont pas des nombres entiers de plis :
    //        − 1058/1024 ≈ 1,03 pli  en nœud PV,
    //        −  829/1024 ≈ 0,81 pli  pour le contre-coup,
    //        − hist / 6945 plis       (hist = −2500 → +0,36 pli ; hist = 5000 → −0,72 pli),
    //    puis seulement tronque le total en plis entiers.
    //
    //    Pourquoi « en 1/1024 de pli » : il faut garder les fractions jusqu'à la
    //    fin, car ce sont elles qui, en s'additionnant, font passer ou non un
    //    pli entier. On les garde en virgule fixe : r est stocké comme un entier
    //    valant r × 1024 (4,69 pli → 4803). Tous les calculs se font alors en
    //    entiers (pas de flottant dans la boucle des coups), avec une précision
    //    de 1/1024 ≈ 0,001 pli, et la conversion finale en plis est une simple
    //    division entière par 1024 (une puissance de 2 : un décalage de 10 bits).
    //    Exemple où cela compte : d = 10, 9e coup (n = 9), nœud non-PV, pas le
    //    contre-coup, hist = −2500 (un coup qui a plutôt échoué ailleurs : on le
    //    réduit un peu PLUS) :
    //      1. La table : 1.1158 + ln(10) × ln(9) / 1.4152 = 4.6905 plis,
    //         stocké int(1024 × 4.6905) = int(4803.1) = 4803.
    //      2. L'historique, ligne `r -= hist * 1024 / LMR_HIST_DIV` :
    //         hist × 1024 = −2500 × 1024 = −2 560 000 ;
    //         −2 560 000 / 6945 = −368.6, tronqué vers zéro (division entière) = −368 ;
    //         r −= −368, donc r = 4803 + 368 = 5171.
    //         Ces 368 valent 368 / 1024 = 0.36 pli (= 2500 / 6945).
    //      3. En plis entiers : 5171 / 1024 = 5.05, division entière = 5 plis
    //         (borné dans [0, d − 2 = 8] : inchangé). Le coup est cherché à
    //         10 − 1 − 5 = 4 plis au lieu de 9.
    //    Si la table était déjà en plis entiers : la base 4.69 deviendrait 4 (même
    //    troncature), l'historique 0.36 deviendrait 0, total 4 plis. Or la vraie
    //    somme est 4.69 + 0.36 = 5.05 > 5 : les deux fractions 0.69 et 0.36, chacune
    //    trop petite pour faire un pli, en font un ensemble. Arrondir chaque
    //    terme séparément les perdrait ; la virgule fixe les additionne d'abord.
    //
    //    Exemples SANS AJUSTEMENT (c.-à-d. la valeur brute de la table, celle d'un
    //    nœud non-PV, coup qui n'est pas le contre-coup, hist = 0) :
    //        d = 10, n = 9 (9e coup) → 1.1158 + 2.303 × 2.197 / 1.4152 = 4.69
    //                                 → stocké 4803 → 4 plis ;
    //        d = 3,  n = 2 (2e coup) → 1.1158 + 1.099 × 0.693 / 1.4152 = 1.65
    //                                 → stocké 1693 → 1 pli.
    //    Le logarithme fait croître la réduction avec la profondeur ET le rang,
    //    mais de moins en moins vite.
    for (int d = 1; d < 64; d++) {
        for (int n = 1; n < Yolah::MAX_NB_MOVES; n++) {
            lmr_table[d][n] = int(1024 * (LMR_BASE + std::log(d) * std::log(n) / LMR_DIVISOR));
        }
    }
}

// Recherche le meilleur coup de `yolah` : jusqu'à la profondeur `max_depth`, ou
// jusqu'à ce que `microseconds` soient écoulées (0 = pas d'horloge). Avec
// plusieurs threads, tous cherchent la même position en même temps (Lazy SMP).
FinalDevPlayer::Result FinalDevPlayer::search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds) {
    const auto start = std::chrono::steady_clock::now();
    stop = false;

    // ── Horloge d'arrêt ──────────────────────────────────────────────────────
    // Un thread léger (`std::jthread`) dort pendant `microseconds` puis passe
    // le booléen atomique `stop` à `true`. Tous les threads de recherche lisent
    // ce booléen à chaque nœud (`stopped()`) et remontent aussitôt en renvoyant
    // des valeurs sans signification, que les appelants ignorent.
    // Si la recherche se termine avant (profondeur max atteinte), on réveille
    // l'horloge plus bas (`request_stop`) : `wait_for` rend alors `true` et
    // l'horloge s'arrête sans toucher à `stop`. Le prédicat `[] { return false; }`
    // veut dire « aucune autre raison de se réveiller » : seuls le délai ou la
    // demande d'arrêt mettent fin à l'attente.
    std::mutex m;
    std::condition_variable_any cv;
    std::jthread clock;
    if (microseconds > 0) {
        clock = std::jthread([&](std::stop_token st) {
            std::unique_lock lock(m);
            if (!cv.wait_for(lock, st, std::chrono::microseconds(microseconds), [] { return false; })) {
                stop = true;   // délai écoulé : on arrête la recherche
            }
        });
    }

    // Indique à la table de transposition le nombre de cases libres à la racine :
    // toute entrée de la table ayant PLUS de cases libres que la racine actuelle
    // appartient au passé de la partie et ne pourra plus jamais être atteinte ;
    // elle sera remplacée en priorité.
    yolah_table->new_search(std::popcount(yolah.free_squares()));

    // ── Vieillissement de l'historique principal ─────────────────────────────
    // Ce qui a été appris aux coups précédents reste pertinent (les positions
    // sont proches), mais on divise par 2 l'historique principal pour que les
    // nouvelles coupures pèsent davantage que les anciennes.
    // Les tables contextuelles `cont1` et `lmrh`, la correction `corr` et les
    // contre-coups ne sont PAS vieillis (vieillir `cont1` a été mesuré neutre).
    for (auto& w : workers) {
        // history[joueur][origine][destination] : 3 boucles imbriquées.
        for (auto& of_player : w->history)          // of_player = history[joueur]             : [64][64]
            for (auto& of_from : of_player)         // of_from   = history[joueur][origine]    : [64]
                for (int16_t& h : of_from) h /= 2;  // h         = history[joueur][origine][destination]
    }

    // ── Parallélisme Lazy SMP (L) ────────────────────────────────────────────
    // Les `nb_threads` threads lancent le MÊME approfondissement itératif sur
    // la même position racine :
    //
    //     thread 0 (principal) : prof. 1, 2, 3, 4, 5, 6, 7, 8 …
    //     thread 1             : prof. 1,    3,    5,    7 …   (saute certaines
    //     thread 2             : prof. 1,       4, 5,       8 … profondeurs,
    //                                 ╲      │      ╱             section 2)
    //                        SearchTable & eval_cache partagés
    //
    // Ils partagent sans verrou la table de transposition et le cache d'éval,
    // mais chacun possède son propre `Worker` (historique, killers, ordre des
    // coups à la racine) : leurs arbres divergent naturellement et explorent
    // en parallèle différentes branches. Ce qu'un thread prouve et range dans
    // la table évite le même travail aux autres. « Lazy » : aucun partage
    // explicite du travail, aucune synchronisation, seulement la table commune.
    std::vector<std::unique_ptr<Search>> searches;
    for (size_t i = 0; i < nb_threads; i++) {
        auto s = std::make_unique<Search>();   // sur le tas : la structure est grosse (accumulateurs)
        s->id = int(i);
        s->w  = workers[i].get();

        // Seul l'accumulateur NNUE de la racine est calculé de zéro ; tous les
        // accumulateurs des nœuds descendants seront calculés paresseusement
        // par différence depuis leur parent (section 8).
        persp->init(yolah, s->accs[yolah.nb_plies()].acc);
        s->acc_ok[yolah.nb_plies()] = true;
        searches.push_back(std::move(s));
    }

    {
        std::vector<std::jthread> helpers;
        for (size_t i = 1; i < nb_threads; i++) {
            helpers.emplace_back([&, i] { iterative_deepening(yolah, *searches[i], max_depth); });
        }
        // Le thread courant mène la recherche principale (thread 0) ; dès qu'il
        // a fini (temps écoulé ou profondeur max), il stoppe les threads
        // assistants. En sortant de ce bloc, le destructeur de `helpers` attend
        // (join) qu'ils aient tous rendu la main.
        iterative_deepening(yolah, *searches[0], max_depth);
        stop = true;
    }

    // Si l'horloge dort encore (recherche finie avant le délai), on la réveille et on l'attend.
    if (clock.joinable()) {
        clock.request_stop();
        clock.join();
    }

    // On retient le coup du thread qui a terminé la profondeur la plus élevée
    // (le thread 0 en cas d'égalité, car il est examiné en premier et l'on
    // n'en change que pour une profondeur STRICTEMENT plus grande).
    const Search* best = searches[0].get();
    Result r;
    for (const auto& s : searches) {
        if (s->depth > best->depth && s->move != Move::none()) best = s.get();
        r.nb_nodes += s->nb_nodes;
        r.nb_hits  += s->nb_hits;
    }
    r.move    = best->move;
    r.value   = best->value;
    r.depth   = best->depth;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

// Interface `Player` : recherche sans limite de profondeur (63 = le maximum
// géré) pendant `thinking_time` microsecondes, puis renvoie le coup.
Move FinalDevPlayer::play(Yolah yolah) {
    Result r = search(yolah, 63, thinking_time);
    if (verbose) {
        cout << "##########\n"
             << "depth  : " << int(r.depth) << '\n'
             << "value  : " << r.value << '\n'
             << "# nodes: " << r.nb_nodes << '\n'
             << "# hits : " << r.nb_hits << '\n'
             << "tt load: " << yolah_table->load() << '\n';
        print_pv(yolah, zobrist::hash(yolah), r.depth);
        cout << '\n';
    }
    return r.move;
}

// Oublie tout ce qui dépend des positions déjà vues : la table de transposition,
// le cache d'évaluations et (sauf `keep_ordering`) la mémoire des threads
// (historiques, contre-coups, continuation, lmrh, correction, statistiques
// d'aspiration). Après `clear_table()`, une recherche est reproductible.
//
// Qui l'appelle, et avec quoi :
//   • `clear_table()` (keep_ordering = false, le défaut) : tous les appels réels —
//     entre deux parties (test/selfplay_label_main.cpp avant chaque partie,
//     test/opening_book_main.cpp), et avant chaque position du banc d'essai
//     (test/search_bench_main.cpp `bench`, `cutstats`) pour que deux joueurs
//     soient comparés à partir du même état vide.
//   • `clear_table(true)` : UN seul usage, de mesure, `search_bench bench --warm 1`
//     (via le joueur d'origine MinMaxNNUE_DevPlayer, dont cette classe est la
//     copie commentée) : on vide la table et le cache mais on garde les
//     historiques appris sur les positions PRÉCÉDENTES du banc, pour estimer
//     la recherche « en cours de partie », où ces tables sont déjà chaudes.
//   • Pendant une partie, personne ne l'appelle : d'un coup à l'autre la
//     table de transposition est gardée (seule son ancienneté change, voir
//     `new_search`) et l'historique principal est divisé par 2 (vieillissement
//     au début de `search`). Ni le jeu de l'application ni les matchs de
//     tournoi n'utilisent `keep_ordering`.
void FinalDevPlayer::clear_table(bool keep_ordering) {
    yolah_table->clear();
    if (!keep_ordering) {
        for (auto& w : workers) *w = Worker{};
    }
    for (auto& e : eval_cache) e.store(0, std::memory_order_relaxed);
}


// =============================================================================
//  2. APPROFONDISSEMENT ITÉRATIF & FENÊTRES D'ASPIRATION ADAPTATIVES (B, A2)
// =============================================================================
//
//  Approfondissement itératif : on cherche à profondeur 1, 2, 3, … Chaque
//  profondeur terminée donne un coup et une valeur ; si l'horloge interrompt
//  une profondeur, on garde le résultat de la précédente (ou mieux, voir
//  `stopped()` plus bas).
//
//  Pourquoi des fenêtres d'aspiration adaptatives ?
//    À partir de la profondeur 4, au lieu de chercher dans (−∞, +∞), on cherche
//    dans une fenêtre étroite (v − δ, v + δ) centrée sur la valeur `v` de la
//    profondeur précédente. Une fenêtre étroite provoque beaucoup plus de
//    coupures alpha-bêta. Si la vraie valeur sort de la fenêtre, on le saura
//    (échec bas ou haut) et on recommence avec une fenêtre élargie.
//
//    Or, avec une demi-largeur fixe δ = 300, la moitié des itérations échouaient
//    (la variation médiane de la valeur d'une profondeur à la suivante vaut
//    ~630, le 90e centile ~4600). Dans `dev_u9_spsa`, δ s'adapte à la volatilité
//    réelle de la partie :
//
//        δ = max(345, 0.9187 × moyenne_glissante(|v_d − v_{d−1}|))
//
//    La moyenne glissante est celle du `Worker` (décroissance 0.9 par
//    profondeur terminée, gardée d'un coup de la partie au suivant).
//
//    Lorsqu'une recherche échoue bas (v' ≤ α) ou haut (v' ≥ β) :
//      • On recentre le bord dépassé directement autour de la valeur fail-soft `v'`
//        renvoyée par `root_search` (α = v' − δ en fail-low, β = v' + δ en fail-high).
//      • En cas de fail-low, on abaisse aussi β au milieu de l'ancienne fenêtre,
//        (α + β) / 2, calculé avec l'ancien α avant de le déplacer. L'échec
//        bas dit que la valeur est ≤ α : garder l'ancien β laisserait au-dessus
//        de α une zone presque sûrement vide, et une fenêtre plus large coupe
//        moins. On ne descend pas β jusqu'à α, car la recherche est instable :
//        relancée avec une autre fenêtre et une table qui a appris entre-temps,
//        elle peut revenir au-dessus de l'ancien α. Avec β = α, cela ferait un
//        échec haut et une troisième recherche (un « rebond »). Le milieu laisse
//        une marge. Mesuré (campagne A2) : garder β (`aspiration keep beta`)
//        divise les rebonds par 4 sur le bench (201 → 51), mais perd 2 à 7 Elo :
//        la fenêtre plus étroite fait gagner plus de temps que les rebonds n'en
//        coûtent.
//      • On multiplie δ par ASPIRATION_GROW = 1.3372 avant de relancer.
//
//    Exemple : valeur précédente 1500, δ = 900 → fenêtre (600, 2400).
//      1. root_search renvoie 600 (≤ α : en fail-soft, la valeur renvoyée
//         colle souvent à la borne) : échec bas → β = (600 + 2400) / 2 = 1500,
//         α = 600 − 900 = −300, δ = int(900 × 1.3372) = 1203.
//      2. Fenêtre (−300, 1500) : elle renvoie −300, nouvel échec bas →
//         β = (−300 + 1500) / 2 = 600, α = −300 − 1203 = −1503, δ = 1608.
//      3. Fenêtre (−1503, 600) : elle renvoie −820, à l'intérieur : la
//         profondeur est terminée avec la valeur −820, et la moyenne glissante
//         apprend l'écart |−820 − 1500| = 2320.
// =============================================================================

void FinalDevPlayer::iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth) {
    const uint64_t hash = zobrist::hash(yolah);
    Move    res   = Move::none();   // meilleur coup retenu jusqu'ici
    uint8_t depth = 0;              // dernière profondeur terminée
    int     value = 0;              // sa valeur

    // Initialisation de la liste des coups à la racine, triée une première fois
    // comme un nœud interne (coup de la table, contre-coup, historique, couverture).
    // 63 = « profondeur très grande » pour `score_moves` (tous les termes actifs).
    // Ensuite, c'est `root_search` qui réordonne cette liste après chaque recherche.
    {
        Yolah::MoveList moves;
        yolah.moves(moves);
        int scores[Yolah::MAX_NB_MOVES];
        score_moves(yolah, s, tt_move(hash), moves, scores, 63);
        s.root_moves.clear();
        for (size_t i = 0; i < moves.size(); i++) {
            s.root_moves.push_back({pick_move(moves, scores, i, moves.size())});
        }
    }

    // ── Motif de saut de profondeurs des threads secondaires (Lazy SMP, L) ───
    // Problème : en Lazy SMP, des threads qui cherchent tous la profondeur 12 au
    // même moment visitent largement les mêmes nœuds près de la racine (travail
    // dupliqué). Solution reprise de Stockfish (avant 2018) : chaque thread
    // secondaire SAUTE certaines profondeurs, selon son propre motif.
    //
    // Comment chaque thread choisit les profondeurs qu'il saute :
    //   1. Son numéro `s.id` (0 = principal, 1, 2, …) donne un indice k = s.id % 20
    //      (au-delà de 20 threads, les motifs recommencent).
    //   2. k donne deux nombres, lus dans les tables SKIP_SIZE et SKIP_PHASE
    //      ci-dessous : une taille de bloc S et un décalage P.
    //   3. Il découpe les profondeurs en blocs de S profondeurs consécutives et
    //      cherche un bloc sur deux : il cherche le premier, saute le suivant,
    //      cherche le suivant, etc. Le décalage P fait commencer ce découpage
    //      plus tôt, pour que deux threads de même taille S n'aient pas les
    //      mêmes blocs.
    //   En formule : la profondeur d est sautée quand q = (d + P) / S (division
    //   entière) est impair. q est le numéro du bloc de d ; blocs pairs cherchés,
    //   blocs impairs sautés.
    //
    // Exemple de calcul, thread 3 (k = 3 : S = 2, P = 1) :
    //   d = 1 : q = (1 + 1) / 2 = 1, impair → sautée
    //   d = 2 : q = (2 + 1) / 2 = 1, impair → sautée
    //   d = 3 : q = (3 + 1) / 2 = 2, pair   → cherchée
    //   d = 4 : q = (4 + 1) / 2 = 2, pair   → cherchée
    //   d = 5 : q = (5 + 1) / 2 = 3, impair → sautée     … d'où 3, 4, 7, 8, 11, 12…
    //
    // Le tableau complet des 10 premiers threads (X = cherchée, . = sautée ;
    // sans la règle « jamais la dernière profondeur », voir plus bas) :
    //     d =                     1  2  3  4  5  6  7  8  9 10 11 12 13 14
    //     thread 0 (principal)    X  X  X  X  X  X  X  X  X  X  X  X  X  X
    //     thread 1 (S 1, P 1)     X  .  X  .  X  .  X  .  X  .  X  .  X  .
    //     thread 2 (S 2, P 0)     X  .  .  X  X  .  .  X  X  .  .  X  X  .
    //     thread 3 (S 2, P 1)     .  .  X  X  .  .  X  X  .  .  X  X  .  .
    //     thread 4 (S 2, P 2)     .  X  X  .  .  X  X  .  .  X  X  .  .  X
    //     thread 5 (S 2, P 3)     X  X  .  .  X  X  .  .  X  X  .  .  X  X
    //     thread 6 (S 3, P 0)     X  X  .  .  .  X  X  X  .  .  .  X  X  X
    //     thread 7 (S 3, P 1)     X  .  .  .  X  X  X  .  .  .  X  X  X  .
    //     thread 8 (S 3, P 2)     .  .  .  X  X  X  .  .  .  X  X  X  .  .
    //     thread 9 (S 3, P 3)     .  .  X  X  X  .  .  .  X  X  X  .  .  .
    //   Threads 10 et 11 : S 3, P 4 et 5 ; threads 12 à 19 : S 4, P 0 à 7 ;
    //   thread 20 (k = 0 : S 1, P 0) : les profondeurs paires 2, 4, 6… ; puis
    //   thread 21 = thread 1, etc. Chaque colonne a des X et des . : à chaque
    //   profondeur, une partie des threads y travaille et les autres sont déjà
    //   plus loin.
    // Un thread qui saute la moitié des profondeurs arrive plus tôt aux
    // suivantes : pendant que le thread principal cherche d, certains sont déjà
    // à d + 1 ou d + 2 et remplissent la table de résultats plus profonds, que
    // le principal trouve en y arrivant (coups de table, coupures directes).
    //
    // Trois conditions pour sauter :
    //   • `s.id > 0` : le thread principal (id 0) ne saute jamais ; c'est lui qui
    //     fixe le rythme et la fin de la recherche.
    //   • le test de parité ci-dessus ;
    //   • `d < max_depth` : on ne saute jamais la dernière profondeur demandée
    //     (dans une recherche à profondeur fixe, chacun finit à `max_depth`).
    // (Le vrai joueur a une option "smp skip" pour désactiver ces sauts. Avec un
    //  seul thread, ils n'existent pas.) Campagne U24 (2026-11-21, 2000 parties
    //  par match, sauts désactivés contre sauts actifs) : 4 threads −1.2 à 0.2 s
    //  et −2.4 à 1 s ; 8 threads +6.3 à 0.2 s et −3.5 à 1 s, tous les intervalles
    //  à 95 % contenant 0. Les sauts ne rapportent donc rien de mesurable à 4 ou
    //  8 threads ; ils sont gardés car le joueur est réglé et mesuré avec eux, et
    //  le léger avantage à 1 s va plutôt dans leur sens.
    static constexpr int SKIP_SIZE[]  = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    static constexpr int SKIP_PHASE[] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
    const int k = s.id % 20;

    for (int d = 1; d <= max_depth && d < 64; d++) {
        if (s.id > 0 && ((d + SKIP_PHASE[k]) / SKIP_SIZE[k]) % 2 == 1 && d < max_depth) {
            continue;
        }

        // Calcul de la demi-largeur initiale δ de la fenêtre d'aspiration :
        // au moins 345, ou 0.9187 × la variation moyenne déjà observée.
        int delta = ASPIRATION_MIN_DELTA;
        if (s.w->asp_n > 0) {
            delta = std::max(delta, int(ASPIRATION_K * s.w->asp_sum / s.w->asp_n));
        }

        int alpha = -INFINITE, beta = INFINITE;
        // Remise à zéro du compteur de nœuds par coup racine au début de la profondeur `d`
        // (`root_fix` : les nœuds seront cumulés sur toutes les tentatives de la profondeur `d`).
        for (RootMove& rm : s.root_moves) rm.nodes = 0;

        // Fenêtre étroite seulement à partir de la profondeur 4 (avant, la
        // valeur bouge trop et la recherche est de toute façon instantanée),
        // et pas si la valeur précédente est une fin de partie prouvée (|v| ≥ WIN :
        // une fenêtre de ±δ autour d'un score exact n'a pas de sens).
        if (d >= 4 && std::abs(value) < WIN) {
            alpha = std::max(value - delta, -INFINITE);
            beta  = std::min(value + delta,  INFINITE);
        }

        // Boucle de résolution de la fenêtre d'aspiration à la profondeur `d` :
        // on recommence jusqu'à obtenir une valeur à l'intérieur de la fenêtre.
        for (;;) {
            Move m;
            const int v = root_search(yolah, s, hash, alpha, beta, d, m);

            if (stopped()) {
                // L'horloge a interrompu la recherche. `v` n'a pas de sens, mais si
                // un coup `m` a eu le temps d'être COMPLÈTEMENT exploré et de battre
                // alpha, il est meilleur que l'ancien meilleur coup (cherché en
                // premier, il a été battu) : on le garde.
                if (m != Move::none()) res = m;
                goto done;
            }

            // Pourquoi `alpha > -INFINITE` : un échec bas n'a de sens que si la
            // fenêtre a un bord bas à élargir. Si alpha vaut déjà −∞ (fenêtre
            // pleine : profondeur < 4, fin de partie prouvée, ou alpha déjà
            // descendu jusqu'à −∞ par les élargissements), « v ≤ alpha » ne peut
            // arriver qu'avec v = −32000, et relancer donnerait la MÊME fenêtre
            // (max(v − δ, −∞) = −∞) : la même recherche, le même résultat, une
            // boucle sans fin. Avec alpha = −∞, la valeur v est exacte (rien
            // n'est plus bas que −∞) : on la garde. Les vraies valeurs restent
            // dans [−30000, 30000] et `root_search` part de best = −32000 : v =
            // −32000 n'arriverait que si aucun coup n'avait été évalué. C'est
            // une garde de sécurité ; même rôle pour `beta < INFINITE` plus bas.
            if (v <= alpha && alpha > -INFINITE) {
                // Échec bas (fail-low) : tous les coups ont échoué sous alpha ;
                // la vraie valeur est ≤ v. Voir l'explication en tête de section.
                beta  = (alpha + beta) / 2;
                alpha = std::max(v - delta, -INFINITE);
                delta = int(delta * ASPIRATION_GROW);
                continue;
            }
            if (v >= beta && beta < INFINITE) {
                // Échec haut (fail-high) : `m` a atteint beta ; la vraie valeur est
                // ≥ v. Le coup `m` est meilleur que tout ce qu'on connaissait : on
                // le retient tout de suite (utile si l'horloge coupe la relance),
                // et on relève beta. alpha ne bouge pas.
                if (m != Move::none()) res = m;
                beta  = std::min(v + delta, INFINITE);
                delta = int(delta * ASPIRATION_GROW);
                continue;
            }

            // La valeur `v` est tombée à l'intérieur de (alpha, beta) : valeur exacte !
            res = m;
            // Apprentissage de la variation moyenne (pour les δ suivants), à partir
            // de la profondeur 4, s'il y a une valeur précédente, et hors fins de
            // partie prouvées (un saut vers ±30 000 fausserait la moyenne).
            if (d >= ASPIRATION_STATS_DEPTH && depth > 0 && std::abs(v) < WIN && std::abs(value) < WIN) {
                s.w->asp_sum = 0.9 * s.w->asp_sum + std::abs(v - value);
                s.w->asp_n   = 0.9 * s.w->asp_n   + 1.0;
            }
            depth = uint8_t(d);
            value = v;
            break;
        }
    }

done:
    // Résultat de ce thread (lu par `search` pour choisir le coup final).
    s.depth = depth;
    s.value = int16_t(value);
    s.move  = res;
}


// =============================================================================
//  3. RECHERCHE À LA RACINE (`root_search`) & TRI DES COUPS RACINE (C, A2)
// =============================================================================
//
//  Particularités de la racine par rapport à un nœud interne (`negamax`) :
//    • On ne coupe jamais par la table de transposition ni par élagage avant-coup :
//      on veut un coup légal à jouer et sa valeur.
//    • Pas de LMP, de poches, ni de réductions contextuelles : seule la
//      réduction LMR de base s'applique (la racine est un nœud PV).
//    • Seul un coup dont l'exploration s'est terminée AVANT le déclenchement de
//      `stopped()` peut devenir `res` (sinon on renvoie aussitôt).
//    • `res` ne devient un coup que s'il BAT alpha : en cas d'échec bas (aucun
//      coup au-dessus de alpha), `res` reste Move::none().
//    • Après la recherche, on réordonne `s.root_moves` pour la tentative ou la
//      profondeur suivante (`root_fix = true`) :
//        - En cas de fail-low (`res == Move::none()`), aucun coup n'a battu alpha :
//          on ne touche PAS à l'ordre (l'ancien meilleur coup reste en tête).
//        - En cas de fail-high (`best >= beta`), les coups après la coupure n'ont
//          pas été cherchés : on fait simplement passer le nouveau meilleur coup
//          `res` en première position (`std::rotate`, les autres gardent leur ordre).
//        - En cas de recherche complète (`alpha < best < beta`), `res` passe en
//          tête et tous les autres coups sont triés par `rm.nodes` décroissant :
//          un coup secondaire qui a demandé beaucoup de nœuds pour être réfuté
//          est le principal challenger à tester tôt à la profondeur d + 1.
//          (Correction « root fix » de la campagne A2 : +9 Elo à 0.2 s.)
//
//          Pourquoi les nœuds plutôt que la valeur : après le premier coup, les
//          autres sont cherchés en fenêtre nulle (alpha, alpha + 1) (PVS,
//          section 4). La recherche est fail-soft : un coup qui ne bat pas le
//          meilleur renvoie une valeur v ≤ alpha qui n'est pas forcément alpha,
//          et c'est un MAJORANT (la vraie valeur est ≤ v). Il y a donc bien une
//          information : un coup renvoyant alpha − 2000 est sûrement mauvais.
//          Mais ce majorant est flou près de alpha : il vient des coupures chez
//          l'adversaire, qui s'arrêtent dès qu'une réponse suffit, pas à la
//          meilleure ; et le coup a souvent été cherché réduit (LMR), donc
//          moins profond. Deux coups renvoyant alpha − 3 et alpha − 40 ne
//          sont pas classés de façon fiable.
//          Le nombre de nœuds mesure autre chose : la difficulté de la réfutation.
//            • un coup nettement mauvais est réfuté vite : la première réponse
//              de l'adversaire essayée (coup de table, killer) suffit à couper
//              dans presque chaque nœud, et la recherche réduite (LMR, étape 1)
//              échoue déjà sous alpha, sans relance ;
//            • un coup presque aussi bon que le meilleur est réfuté difficilement :
//              la coupure n'arrive qu'après plusieurs réponses essayées, la
//              recherche réduite peut dépasser alpha et forcer une relance à
//              pleine profondeur (étape 2), et ses sous-arbres sont plus larges.
//          Mesuré (campagne R1, 2026-11-05, journal de la racine à profondeur 12,
//          où se place le coup qui deviendra le meilleur à la profondeur
//          suivante parmi les autres) :
//            • classé par nœuds : dans les 3 premiers 56 à 61 % des fois, rang moyen 8.5 à 10.1 ;
//            • classé par valeur fail-soft : 48 à 51 %, rang moyen 6.1 à 6.2
//              (moins souvent en tête, mais moins de ratés lointains) ;
//            • classé par la somme des deux rangs : 60 à 61 %, rang moyen 5.0 à 5.6.
//          La somme était donc le meilleur classement sur le papier (option
//          "root order gap" du vrai joueur) ; en parties elle a perdu :
//          −8.7 Elo [−19.1, +1.7] à 0.2 s (1000 parties), campagne abandonnée.
//          Le tri par nœuds seul est resté.
//          Beaucoup de nœuds = valeur proche de alpha = le coup qui a le plus de
//          chances de devenir meilleur une fois cherché un pli plus loin.
//
//          Pourquoi le tester tôt : (1) s'il devient le meilleur, alpha monte
//          tout de suite et tous les coups suivants sont coupés plus vite ;
//          (2) la réduction LMR croît avec le rang (ln(i + 1), section 4) : à
//          la profondeur 10 en nœud PV, rangé 15e (i = 14) il serait réduit de
//          1.1158 + ln 10 × ln 15 / 1.4152 − 1.03 = 4.49 → 4 plis ; rangé 2e
//          (i = 1) de 2.24 − 1.03 = 1.21 → 1 pli. Réduit de 4 plis, il pourrait
//          échouer sous alpha à tort ; réduit d'1, sa vraie valeur a une chance
//          d'apparaître.
//          Ce tri par nœuds existait avant A2 ; le « root fix » d'A2 (+9 Elo) y a
//          ajouté les trois règles ci-dessus : nœuds cumulés sur toutes les
//          tentatives de la profondeur (une relance après échec d'aspiration ne
//          remet pas les compteurs à zéro), ordre inchangé après un fail-low
//          (avant, un coup réfuté vite pouvait passer en tête), simple rotation
//          après un fail-high (les coups non cherchés ont des compteurs
//          incomplets).
// =============================================================================

int FinalDevPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash,
                                int alpha, int beta, int depth, Move& res) {
    res = Move::none();
    const int alpha_orig = alpha;   // la fenêtre d'entrée, pour ranger le résultat dans la table
    int best = -INFINITE;
    const uint8_t player = yolah.current_player();

    for (size_t i = 0; i < s.root_moves.size(); i++) {
        RootMove& rm = s.root_moves[i];
        const uint64_t nodes_before = s.nb_nodes;

        // La racine est un nœud PV (`pv_node = true`), sans coup spécial (`special = false`) ;
        // la réduction LMR dépend du rang `i` et de l'historique principal du coup.
        // (i = 0, le meilleur coup de la profondeur précédente, n'est jamais réduit.)
        const int r = late_move_reduction_of(
            depth, i, /*pv_node=*/true, /*special=*/false,
            s.w->history[player][rm.move.from_sq()][rm.move.to_sq()]
        );

        // PVS + LMR (section 4). `alpha` monte au fil de la boucle : les coups
        // suivants doivent battre le meilleur trouvé.
        const int v = search_move(yolah, s, hash, rm.move, i, r, alpha, beta, depth);
        rm.value  = v;
        rm.nodes += s.nb_nodes - nodes_before;   // cumulé sur les tentatives de la profondeur

        if (stopped()) return best;   // valeur sans signification ; `res` = dernier coup complet ayant battu alpha

        if (v > best) {
            best = v;
            if (v > alpha) {
                res   = rm.move;
                alpha = v;
                if (v >= beta) {
                    break;   // fail-high de la fenêtre d'aspiration : inutile de chercher les autres coups
                }
            }
        }
    }

    // Réordonnancement des coups de la racine (`root_fix = true`)
    if (res == Move::none()) {
        // Fail-low : on conserve l'ordre actuel (l'ancien meilleur reste 1er).
    } else if (best >= beta) {
        // Fail-high : seul le coup qui a coupé remonte en 1re place.
        auto it = std::find_if(s.root_moves.begin(), s.root_moves.end(),
                               [&](const RootMove& rm) { return rm.move == res; });
        std::rotate(s.root_moves.begin(), it, it + 1);
    } else {
        // Fenêtre résolue : `res` en 1er, les autres par nombre de nœuds décroissant.
        // (`stable_sort` : à nombre de nœuds égal, l'ordre précédent est conservé.)
        std::stable_sort(s.root_moves.begin(), s.root_moves.end(),
                         [&](const RootMove& a, const RootMove& b) {
                             if ((a.move == res) != (b.move == res)) return a.move == res;
                             return a.nodes > b.nodes;
                         });
    }

    // Le résultat de la racine va dans la table comme celui de tout nœud
    // (marqué PV) : c'est là que `principal_variation` et `tt_move(hash)`
    // retrouveront le meilleur coup.
    tt_store(hash, yolah, depth, alpha_orig, beta, best, res, /*pv=*/true);
    return best;
}


// =============================================================================
//  4. FORMULE LMR & RECHERCHE D'UN COUP (`search_move` : LMR + PVS)
// =============================================================================

// Calcule la réduction de profondeur de base (en plis) du i-ème coup (i ≥ 0) :
//   • Pas de réduction à depth < 3 ni sur le tout premier coup (i == 0).
//   • Formule de base (en 1/1024 de pli, lue dans `lmr_table`) :
//         r = LMR_BASE + ln(depth) × ln(i + 1) / LMR_DIVISOR
//         avec LMR_BASE = 1.1158 et LMR_DIVISOR = 1.4152
//   • Ajustements :
//         − 1058/1024 pli  si c'est un nœud PV (on y veut la valeur exacte :
//                          moins de risque de rater le bon coup)
//         − 829/1024 pli   si c'est le contre-coup (`special == true`)
//         − hist / 6945 plis (hist ∈ [−16384, 16384] : entre −2,4 et +2,4 plis ;
//                          un coup qui coupe souvent ailleurs est moins réduit,
//                          un coup souvent inutile l'est davantage)
//   • Le total est tronqué en plis entiers (division entière par 1024, donc
//     vers zéro) et borné dans [0, depth − 2] pour que la recherche réduite
//     garde au moins 1 pli (depth − 1 − r ≥ 1).
//   Exemple : depth 10, i = 8, nœud non-PV, hist = 0 → 4803/1024 → 4 plis :
//   le coup est d'abord cherché à profondeur 10 − 1 − 4 = 5 au lieu de 9.
int FinalDevPlayer::late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const {
    if (depth < 3 || i == 0) return 0;
    int r = lmr_table[std::min(depth, 63)][std::min<size_t>(i + 1, Yolah::MAX_NB_MOVES - 1)];
    if (pv_node) r -= LMR_PV;
    if (special) r -= LMR_SPECIAL;
    r -= hist * 1024 / LMR_HIST_DIV;
    return std::clamp(r / 1024, 0, depth - 2);
}

// Exécute le coup `m` (rang `i` dans le nœud) et renvoie la valeur du sous-arbre
// enfant, du point de vue du nœud courant, selon le schéma PVS + LMR en 3 étapes :
//
//   Étape 1 (si `reduction > 0`) :
//     Recherche à profondeur réduite `depth − 1 − reduction` avec fenêtre nulle
//     `(-(alpha + 1), -alpha)`. Si `v <= alpha`, le coup est réputé mauvais et
//     on s'arrête là (économie massive sur les coups tardifs).
//
//   Étape 2 (si l'étape 1 a battu `alpha`, ou s'il n'y avait pas de réduction) :
//     • coup de rang i > 0 : recherche à pleine profondeur `depth − 1` mais
//       toujours avec fenêtre nulle `(-(alpha + 1), -alpha)` pour vérifier à
//       moindre coût si le coup bat vraiment `alpha` (PVS) ;
//     • premier coup (i == 0) : directement l'étape 3.
//
//   Étape 3 (si `i == 0`, ou si l'étape 2 a donné `alpha < v < beta`) :
//     Recherche à pleine profondeur `depth − 1` et fenêtre pleine `(-beta, -alpha)`
//     pour obtenir la valeur exacte. À un nœud non-PV (beta = alpha + 1), la
//     condition `alpha < v < beta` est impossible pour un entier : il n'y a
//     jamais d'étape 3 pour i > 0, la fenêtre nulle suffit.
//
//     Pourquoi l'étape 2 peut-elle donner `alpha < v < beta` alors qu'elle
//     cherche en fenêtre nulle (alpha, alpha + 1) ? La fenêtre nulle ne dit
//     pas « v vaut alpha ou alpha + 1 ». Elle dit seulement de quel côté
//     d'alpha se trouve la vraie valeur :
//       • v <= alpha : le coup ne bat pas alpha (v est un majorant) ;
//       • v >= alpha + 1 : le coup bat alpha, et v n'est qu'un MINORANT.
//     La recherche est en fail-soft : dans le second cas, v n'est pas
//     forcément alpha + 1. C'est la meilleure valeur trouvée avant la coupure,
//     qui peut être plus haute.
//     À un nœud PV, beta peut être bien au-dessus d'alpha + 1, donc un
//     minorant tel que alpha < v < beta est possible, et c'est même le cas
//     courant. On sait alors seulement que le coup vaut au moins v. On ne sait
//     pas s'il atteint beta, ni de combien il dépasse alpha. Il faut donc
//     relancer en fenêtre pleine pour avoir sa valeur exacte (et sa PV).
//     Exemple : nœud PV de fenêtre (10, 50). L'étape 2 cherche en (10, 11),
//     l'enfant coupe et renvoie v = 23. La vraie valeur vaut 23 ou plus :
//     peut-être 23, 40 ou 60. L'étape 3 cherche en (10, 50) et tranche.
//     Si au contraire v >= beta (ici v >= 50), le minorant suffit : le coup
//     vaut au moins beta, le nœud coupe, et la valeur exacte est inutile.
//     D'où la condition `v > alpha && v < beta`.
//
// Les signes : la valeur de l'enfant est vue par l'adversaire, d'où
// `v = -negamax(enfant, -beta, -alpha)` ; la fenêtre (alpha, alpha+1) du nœud
// devient (-(alpha+1), -alpha) pour l'enfant.
// `v` est initialisé à alpha + 1 pour que, sans réduction, l'étape 2 ait lieu.
int FinalDevPlayer::search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i,
                                int reduction, int alpha, int beta, int depth) {
    const uint8_t  player = yolah.current_player();
    const uint64_t child  = zobrist::update(hash, player, m);   // hachage de l'enfant, sans jouer le coup
    const uint16_t ply    = yolah.nb_plies();

    // Préchargement matériel du cluster de la table de transposition de l'enfant :
    // l'enfant le consultera dès son entrée dans `negamax`.
    yolah_table->prefetch(child);

    // Deux piles indexées par le pli, une case par niveau de la branche courante
    // (racine → nœud courant). `ply` est le pli du nœud courant, avant de jouer `m`.
    // L'enfant est donc au pli `ply + 1`.
    //
    // • `s.played[ply] = m` : le coup qui mène du pli `ply` au pli `ply + 1`.
    //   L'enfant et ses descendants le lisent en `played[ply_enfant − 1]` pour
    //   le contre-coup, cont1, l'interception et lmrh. L'accumulateur paresseux
    //   le lit aussi (point suivant).
    //
    // • `s.acc_ok[ply + 1] = false` : la case `accs[ply + 1]` contient encore
    //   l'accumulateur d'un FRÈRE déjà exploré. Avant `m`, le nœud a peut-être
    //   joué m1, et l'évaluation d'une feuille sous m1 a rempli accs[ply + 1]
    //   avec la position « nœud + m1 » et l'a marquée valide. Pour l'enfant
    //   « nœud + m », ce contenu est faux. On le marque donc invalide.
    //   On ne le recalcule PAS maintenant (« paresseux ») : un calcul NNUE coûte
    //   cher, et souvent aucune feuille ne sera évaluée sous cet enfant (coupure
    //   par la table de transposition, fin de partie, valeur trouvée dans le
    //   cache d'évaluation).
    //   Si une évaluation a lieu plus bas, au pli P, `raw_network_value` remonte
    //   jusqu'au dernier pli k <= P dont `acc_ok[k]` est vrai. Au pire c'est la
    //   racine, calculée de zéro dans `search`. Puis elle redescend de k à P :
    //   accs[k + 1] = accs[k] + l'effet du coup played[k], et ainsi de suite.
    //   Chaque case recalculée redevient valide et sert aux autres feuilles du
    //   même sous-arbre (section 8).
    //   Pourquoi une case marquée vraie au-dessus de P est forcément correcte :
    //   chaque descente invalide la case de l'enfant AVANT d'y entrer, et une
    //   case n'est remise à vrai que par un calcul fait pour la branche courante.
    //
    //   Exemple : racine au pli 20 (accs[20] valide). Le nœud du pli 22 joue m :
    //   played[22] = m, acc_ok[23] = false. Sous l'enfant, une feuille au pli 25
    //   est évaluée. Si acc_ok[21] et acc_ok[22] sont vrais, la remontée
    //   s'arrête à 22, et il y a 3 mises à jour : 23 (coup played[22]),
    //   24 (played[23]) et 25 (played[24]). Le frère suivant m' remettra
    //   acc_ok[23] à faux, mais accs[22] restera valide et resservira.
    s.played[ply]     = m;
    s.acc_ok[ply + 1] = false;

    yolah.play(m);
    int v = alpha + 1;

    // 1. Recherche réduite (LMR) en fenêtre nulle
    if (reduction > 0) {
        const int d = depth - 1 - reduction;
        v = -negamax(yolah, s, child, -(alpha + 1), -alpha, d);
    }

    // 2. Si la recherche réduite promet > alpha (ou s'il n'y avait pas de réduction) :
    if (v > alpha && !stopped()) {
        if (i > 0) {
            // Recherche à pleine profondeur en fenêtre nulle (PVS)
            //
            // Même à un nœud PV, seul le PREMIER coup (i == 0, branche else
            // plus bas) est cherché en fenêtre pleine. C'est le principe de la
            // PVS (Principal Variation Search) : le tri des coups met en tête le
            // coup de la table, qui est presque toujours le meilleur. Les autres
            // coups n'ont alors besoin que d'une réponse oui/non : « bat-il alpha ? ».
            // La fenêtre nulle (alpha, alpha + 1) répond à cette question beaucoup
            // moins cher : chaque coupure y arrive plus tôt, et les nœuds en
            // dessous sont tous non-PV, donc réduits et élagués plus fort.
            // Dans la grande majorité des cas la réponse est non (v <= alpha) et le
            // coup est écarté sans qu'on connaisse sa valeur exacte : inutile.
            // Rarement la réponse est oui : le tri s'est trompé. Il faut alors la
            // valeur exacte de ce nouveau meilleur coup, d'où la relance en
            // fenêtre pleine juste après (seulement si v < beta : si v >= beta,
            // le nœud coupe de toute façon).
            // Le nœud reste un nœud PV, dont la fenêtre (alpha, beta) est pleine.
            // Ce qui est nul, c'est la fenêtre donnée à l'enfant pour ce test.
            // Le surcoût d'un oui (deux recherches au lieu d'une) est largement
            // compensé par l'économie faite sur tous les non.
            v = -negamax(yolah, s, child, -(alpha + 1), -alpha, depth - 1);
            // Aux nœuds PV (beta - alpha > 1), si le coup bat alpha sans dépasser beta,
            // on relance en fenêtre pleine pour connaître sa valeur exacte.
            if (v > alpha && v < beta && !stopped()) {
                v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
            }
        } else {
            // Tout premier coup (i == 0) : directement en fenêtre pleine
            v = -negamax(yolah, s, child, -beta, -alpha, depth - 1);
        }
    }

    yolah.undo(m);
    return v;
}


// =============================================================================
//  5. FONCTIONS GÉOMÉTRIQUES AUXILIAIRES (POCHES SCELLÉES & CORRECTION)
// =============================================================================

namespace {

    // Région de cases libres (`free`) accessible depuis un ensemble de pièces
    // (`pieces`), par propagation de proche en proche (8-connexité, comme un
    // roi d'échecs) jusqu'à ce que plus rien ne change.
    // Pourquoi la 8-connexité suffit-elle pour des pièces qui bougent comme des
    // dames ? Un déplacement de dame passe par des cases successives qui se
    // touchent (par un côté ou un coin), toutes libres : la case d'arrivée est
    // donc dans la même région 8-connexe que la case de départ. Et inversement,
    // toute case de la région est atteignable en une suite de coups. La région
    // est donc exactement l'ensemble des cases où ces pièces pourront un jour
    // aller (en ignorant ce que l'adversaire bouchera entre-temps).
    // Le résultat n'inclut pas les cases des pièces elles-mêmes (occupées).
    uint64_t region_reach(uint64_t pieces, uint64_t free) {
        // grow(b) = b et ses 8 voisines : d'abord la ligne (b, b décalé d'une
        // colonne à l'est et à l'ouest, masqué pour ne pas passer de la colonne h
        // à la colonne a), puis cette ligne décalée d'une rangée vers le haut et le bas.
        auto grow = [](uint64_t b) {
            constexpr uint64_t NOT_A = 0xfefefefefefefefeULL;   // toutes les cases sauf la colonne a
            constexpr uint64_t NOT_H = 0x7f7f7f7f7f7f7f7fULL;   // toutes les cases sauf la colonne h
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

    // ── Récolte exacte d'une poche privée (`harvest_exact`, U4/U5) ───────────
    // Calcule par recherche exhaustive mémorisée (parcours en profondeur + table
    // de hachage des états déjà vus) le nombre MAXIMUM de coups que les pièces
    // `pieces` peuvent jouer à l'intérieur de la région de cases libres
    // `region` (sans jamais en sortir), si on les laisse jouer seules.
    // Chaque coup consomme une case de la région (la destination devient
    // occupée ; le départ devient un trou, il n'était déjà pas dans la région).
    // Renvoie −1 si le budget de nœuds `budget` est dépassé (poche trop compliquée :
    // l'appelant l'ignore alors simplement, ce qui est toujours sûr).
    int harvest_exact_rec(uint64_t pieces, uint64_t region,
                          std::unordered_map<uint64_t, int>& memo, int64_t& budget) {
        if (--budget < 0) return -1;
        // Clé de mémorisation : mélange des deux bitboards (des collisions sont
        // possibles mais astronomiquement rares sur quelques milliers d'états).
        const uint64_t key = (pieces * 0x9E3779B97F4A7C15ULL)
                           ^ (region + 0x632BE59BD9B4E019ULL)
                           ^ (region >> 17);
        if (auto it = memo.find(key); it != memo.end()) return it->second;

        int best = 0;
        const int max_possible = std::popcount(region);   // on ne peut pas faire plus de coups qu'il n'y a de cases
        for (uint64_t pb = pieces; pb;) {
            const Square from = pop_lsb(pb);
            // Destinations : les cases de la région atteintes en ligne droite,
            // tout ce qui n'est pas dans la région bloquant les rayons.
            for (uint64_t t = attacks_bb(from, ~region) & region; t;) {
                const Square to = pop_lsb(t);
                if (best >= max_possible) break;   // impossible de faire mieux que de tout remplir
                const int v = harvest_exact_rec(
                    pieces ^ square_bb(from) ^ square_bb(to),   // la pièce passe de `from` à `to`
                    region & ~square_bb(to),                    // `to` n'est plus libre
                    memo, budget
                );
                if (v < 0) return -1;
                best = std::max(best, 1 + v);
            }
        }
        memo.emplace(key, best);
        return best;
    }

    int harvest_exact(uint64_t pieces, uint64_t region, int budget) {
        std::unordered_map<uint64_t, int> memo;
        int64_t b = budget;
        return harvest_exact_rec(pieces, region, memo, b);
    }

    // ── Clé de l'historique de correction (`corr_key`, U4/U6) ────────────────
    // Hachage rapide sur 14 bits (0..16383) de la position exacte des 4 pièces
    // noires et des 4 pièces blanches (multiplications par de grandes constantes
    // impaires qui mélangent les bits, puis repli des bits hauts sur les bas).
    // Deux positions ayant les mêmes pièces au même endroit (mais éventuellement
    // des trous différents) partagent le même terme correctif
    // `corr[joueur][corr_key]` : l'erreur du réseau dépend surtout de la
    // configuration des pièces, et une clé qui ignore les trous revient assez
    // souvent pour que la table apprenne vite. Des positions sans rapport
    // peuvent aussi tomber sur la même clé ; la correction est alors un peu
    // bruitée, ce qui a été mesuré acceptable (+10.6 Elo au total).
    inline int corr_key(const Yolah& yolah) {
        uint64_t x = yolah.bitboard(Yolah::BLACK) * 0x9E3779B97F4A7C15ULL
                   ^ yolah.bitboard(Yolah::WHITE) * 0xC2B2AE3D27D4EB4FULL;
        x ^= x >> 29;
        return int(x & (16384 - 1));
    }

} // namespace


// =============================================================================
//  6. NŒUDS INTERNES (`negamax`)
// =============================================================================
//
//  Recherche alpha-bêta en convention Negamax fail-soft (voir le glossaire de
//  final_dev.h) :
//    • Si la valeur retournée `v` vérifie `alpha < v < beta`, c'est la valeur exacte.
//    • Si `v <= alpha`, tous les coups ont échoué bas : `v` est une borne supérieure.
//    • Si `v >= beta`, un coup a coupé : `v` est une borne inférieure.
//  `depth` est le nombre de plis restant à chercher ; `hash` le hachage Zobrist
//  de `yolah` (passé en paramètre pour ne pas le recalculer).
//  Si l'horloge s'est déclenchée (`stopped()`), la valeur renvoyée n'a aucun
//  sens : chaque appelant teste `stopped()` avant d'en tirer quoi que ce soit.
//
//  Ordre des étapes d'un nœud :
//    a) fin de partie → score exact
//    b) table de transposition → coupure si une borne connue suffit
//    c) depth ≤ 0 → évaluation du réseau (feuille)
//    d) éval. statique corrigée, puis RFP et MPC (non-PV seulement)
//    e) étape 1 de la génération : le coup de la table seul
//    f) Proxy Cut (non-PV seulement)
//    g) boucle des coups (LMP, poches, LMR)
//    h) coupure bêta (apprentissage), ou fin du nœud (rangement dans la table)
// =============================================================================

int FinalDevPlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth) {
    ++s.nb_nodes;
    if (stopped()) return 0;

    const uint16_t ply_now = yolah.nb_plies();   // pli absolu de ce nœud dans la partie
    s.eval_stack[ply_now] = EVAL_NONE;           // pas (encore) d'évaluation statique ici

    // ── 6.a) Fin de partie exacte ────────────────────────────────────────────
    // Lorsque les deux joueurs n'ont plus aucun coup légal, le score exact
    // (différence de points pour le camp au trait) est décalé au-delà de ±WIN
    // pour qu'une victoire certaine de +1 point (+30001) batte toujours
    // n'importe quelle évaluation heuristique du réseau (< 30000). Une victoire
    // par plus de points vaut plus : le moteur cherche aussi à maximiser l'écart.
    if (yolah.game_over()) {
        const int score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? WIN : -WIN);
    }

    // ── 6.b) Consultation de la table de transposition (`SearchTable`) ───────
    // Comme chaque coup de Yolah détruit définitivement une case, il n'y a
    // jamais de répétition de position ni de dépendance au chemin : une entrée
    // de profondeur `tt.depth >= depth` est toujours utilisable.
    //
    // Règle de coupure (`tt_cut_pv = false, tt_cut_pv_exact = true`) :
    //   • Aux nœuds non-PV (`beta - alpha == 1`), toute borne suffisante coupe.
    //   • Aux nœuds PV (`beta - alpha > 1`), on n'accepte de couper que si l'entrée
    //     est EXACTE (`tt.lower == tt.upper`) ou prouve une fin de partie
    //     (`lower > WIN` : gain prouvé ; `upper < -WIN` : perte prouvée).
    //     Une simple borne vient souvent d'une recherche à fenêtre nulle très
    //     élaguée : s'y fier en PV dégraderait la variation principale.
    // Une borne « suffit » quand elle tranche la question posée par la fenêtre :
    //   lower ≥ beta  → le nœud vaut au moins beta : coupure (on renvoie lower) ;
    //   upper ≤ alpha → le nœud vaut au plus alpha : échec bas (on renvoie upper) ;
    //   lower == upper → valeur exacte connue.
    // Même sans coupure, `tt_move` (le meilleur coup connu) servira à ordonner.
    const TTView tt = tt_probe(hash);
    const Move tt_move = tt.move;
    if (tt.found) {
        s.nb_hits++;
        if (tt.depth >= depth && (beta - alpha == 1 || tt.lower == tt.upper || tt.lower > WIN || tt.upper < -WIN)) {
            if (tt.lower >= beta)  return tt.lower;
            if (tt.upper <= alpha) return tt.upper;
            if (tt.lower == tt.upper) return tt.lower;
        }
    }

    // ── 6.c) Feuilles de l'arbre (`depth <= 0`) ──────────────────────────────
    // À l'horizon, on évalue la position avec le réseau NNUE perspective et on
    // lui ajoute la correction apprise `corr[joueur][corr_key(yolah)]` (`corr_hist = 2` :
    // la correction s'applique aussi aux feuilles, pas seulement à l'élagage).
    // Le résultat reste dans ]−WIN, WIN[ : une évaluation ne doit jamais passer
    // pour une fin de partie prouvée.
    // On ne range JAMAIS les feuilles (depth 0) dans la table de transposition :
    // elles sont très nombreuses et en évinceraient des nœuds internes plus
    // précieux (le cache d'évaluations, section 8, évite déjà de recalculer le réseau).
    // Pas de recherche de quiescence : à Yolah aucun coup n'est « violent »
    // (pas de prise), donc rien ne rend l'évaluation statique particulièrement
    // trompeuse à un pli plutôt qu'à un autre.
    if (depth <= 0) {
        int leaf = evaluate(yolah, s, hash);
        leaf = std::clamp(leaf + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        return leaf;
    }

    // ── 6.d) Évaluation statique corrigée & Élagage avant-coup (non-PV) ──────
    // Tous les élagages avant-coup (RFP, MPC, Proxy Cut) ne s'appliquent qu'aux
    // nœuds non-PV (`beta == alpha + 1`) et hors scores de fin de partie (`|beta| < WIN` :
    // quand la question porte sur un résultat prouvé, une estimation ne suffit pas).
    const bool pv_node  = beta - alpha > 1;
    const bool prunable = !pv_node && std::abs(beta) < WIN;

    // Évaluation statique = valeur du réseau pour CE nœud (avant toute recherche
    // en dessous), corrigée par `corr`. Avec les profondeurs de RFP (≤ 5), du
    // Proxy Cut et du MPC (≥ 4), toutes les profondeurs sont couvertes : on la
    // calcule donc à tout nœud élaguable. Mémorisée dans `eval_stack` pour
    // `corr_update` (fin du nœud). Aux nœuds PV, elle n'est pas calculée
    // (et la correction n'y apprend donc rien).
    int static_eval = 0;
    if (prunable) {
        static_eval = evaluate(yolah, s, hash);
        static_eval = std::clamp(static_eval + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        s.eval_stack[ply_now] = static_eval;
    }

    // ── F1. Reverse Futility Pruning (RFP, +76 Elo) ──────────────────────────
    // Aussi appelé « static null move ». Aux faibles profondeurs (depth ≤ 5),
    // si l'évaluation statique dépasse déjà `beta` d'une marge de sécurité de
    // `2609 × depth`, une recherche peu profonde ne la ramènerait presque
    // jamais sous beta : on renvoie l'évaluation statique (borne inférieure
    // fail-soft) sans générer les coups. 2609 ≈ un dixième de victoire sûre
    // par pli ; la marge grandit avec la profondeur parce que plus on cherche
    // loin, plus la valeur peut s'éloigner de l'estimation.
    if (prunable && depth <= RFP_DEPTH && static_eval - RFP_MARGIN * depth >= beta) {
        return static_eval;
    }

    // ── O. Multi-ProbCut bidirectionnel (MPC, +57 Elo à 1 s) ─────────────────
    // Idée (Buro, Othello) : le résultat d'une recherche profonde se prédit
    // bien par une recherche peu profonde de la même position. À `depth >= 4`,
    // on cherche donc la position COURANTE (pas un enfant) à une profondeur
    // réduite `d2 = max(1, ⌊depth × 46 / 100⌋)`, avec une fenêtre nulle placée
    // loin au-delà de la fenêtre du nœud :
    //   • Coupe haute : si `static_eval >= beta` et que la recherche à profondeur `d2`
    //     atteint `beta + 4054 + 644 × (depth − d2)`, la recherche complète
    //     atteindrait presque sûrement beta : on coupe en renvoyant `beta`.
    //   • Coupe basse : si `static_eval <= alpha` et que la recherche à profondeur `d2`
    //     reste sous `alpha − (4054 + 644 × (depth − d2))`, on échoue bas en renvoyant `alpha`.
    // La marge grandit avec l'écart de profondeur (plus on extrapole loin,
    // moins la prédiction est sûre). On ne tente chaque test que si
    // l'évaluation statique est déjà du bon côté (sinon il échouerait presque
    // toujours et coûterait pour rien), et seulement si la borne reste
    // une évaluation (|borne| < WIN). On renvoie beta ou alpha, pas `v` : la
    // valeur peu profonde n'est qu'un indice, pas une borne prouvée.
    // Exemple : depth 10 → d2 = 4, marge = 4054 + 644 × 6 = 7918 : avec beta =
    // 1000, la recherche à profondeur 4 doit atteindre 8918 pour couper.
    if (prunable && depth >= MPC_DEPTH) {
        const int d2     = std::max(1, depth * MPC_RATIO / 100);
        const int margin = MPC_MARGIN + MPC_MARGIN_PER_PLY * (depth - d2);

        if (static_eval >= beta && beta + margin < WIN) {
            const int bound = beta + margin;
            // fenêtre nulle (bound − 1, bound) : « la valeur atteint-elle bound ? »
            const int v = negamax(yolah, s, hash, bound - 1, bound, d2);
            if (stopped()) return 0;
            if (v >= bound) return beta;
        }
        if (static_eval <= alpha && alpha - margin > -WIN) {
            const int bound = alpha - margin;
            // fenêtre nulle (bound, bound + 1) : « la valeur reste-t-elle ≤ bound ? »
            const int v = negamax(yolah, s, hash, bound, bound + 1, d2);
            if (stopped()) return 0;
            if (v <= bound) return alpha;
        }
    }

    // ── 6.e) Préparation de la génération par étapes (`staged = true`, K) ────
    // Si la table de transposition propose un coup `tt_move` légal ICI, on le
    // cherche seul d'abord (étape 1), sans générer ni noter les autres coups :
    // très souvent il suffit à couper, et la génération est économisée.
    // `is_legal` est indispensable : l'entrée peut venir d'une autre position
    // (collision de hachage, ou entrée « déchirée » par une écriture
    // simultanée d'un autre thread, puisque la table n'a pas de verrou).
    // Si le Proxy Cut doit s'exécuter ci-dessous, il générera et triera les
    // coups lui-même pour trouver son témoin ordinaire.
    //
    // Représentation commune aux étapes 1 et 2 dans la boucle 6.g :
    //   • `generated` faux : la liste n'existe pas encore ; le coup d'indice 0
    //     est `tt_move`, et `n = 1` (on suppose au moins ce coup) ;
    //   • `generated` vrai : `moves[0..n)` et `scores[0..n)` sont remplis.
    Yolah::MoveList moves;
    const bool tt_first = tt_move != Move::none() && is_legal(yolah, tt_move);
    bool generated = false;
    int  scores[Yolah::MAX_NB_MOVES];
    size_t n = 1;

    if (!tt_first) {
        // Pas de coup de table utilisable : on génère et note tout de suite.
        yolah.moves(moves);
        generated = true;
        score_moves(yolah, s, tt_move, moves, scores, depth);
        n = moves.size();
    }

    // ── 6.f) Proxy Cut (N, +47 Elo à 0.2 s, +59 Elo à 1 s) ──────────────────
    // Pourquoi le Proxy Cut au lieu du Null Move Pruning ?
    //   Le null move des moteurs d'échecs demande « même si je passe mon tour,
    //   suis-je encore ≥ beta ? » ; il suppose que passer est pire que le
    //   meilleur coup. À Yolah c'est faux très souvent (zugzwang : bouger détruit
    //   une case de son propre territoire, passer serait un cadeau). Le null move
    //   croit donc à tort que la position est bonne et coupe à tort (−81 Elo).
    //   À l'inverse, tester un VRAI coup légal `m` est toujours sain vis-à-vis
    //   du zugzwang :
    //       valeur(meilleur_coup) ≥ valeur(m)
    //   Donc si même un coup ordinaire `m` (ni le coup de table, ni le contre-coup),
    //   cherché à profondeur réduite `d = depth − 1 − (3 + depth / 4)`, atteint
    //   `beta`, le meilleur coup du nœud atteindra `beta` a fortiori !
    //   La seule approximation est la profondeur réduite (comme pour le null move).
    // Conditions : nœud non-PV, depth ≥ 4, et évaluation statique déjà ≥ beta
    // (sinon le test échouerait presque toujours).
    if (prunable && depth >= PROXY_DEPTH && static_eval >= beta) {
        if (!generated) {
            yolah.moves(moves);
            generated = true;
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
        }
        // Le témoin est le coup ordinaire (`scores[j] < SCORE_COUNTER` : ni le
        // coup de table, ni le contre-coup ; les killers en font partie) ayant
        // le meilleur score de tri : le plus susceptible de tenir beta.
        int best_ordinary = -1;
        for (size_t j = 0; j < n; j++) {
            if (scores[j] < SCORE_COUNTER && (best_ordinary < 0 || scores[j] > scores[best_ordinary])) {
                best_ordinary = int(j);
            }
        }
        if (best_ordinary >= 0) {
            const Move wit = moves[best_ordinary];
            const int  R   = PROXY_REDUCTION + depth / 4 - 1;   // 3 + depth/4 : depth 8 → R = 5
            const int  d   = depth - 1 - R;                     //                 → d = 2
            const uint8_t  player = yolah.current_player();
            const uint64_t child  = zobrist::update(hash, player, wit);

            // Jouer le témoin comme dans `search_move` (pile des coups, accumulateur paresseux).
            s.played[ply_now]     = wit;
            s.acc_ok[ply_now + 1] = false;
            yolah.play(wit);
            // Fenêtre nulle (beta − 1, beta) vue de l'enfant : « le témoin atteint-il beta ? »
            const int v = -negamax(yolah, s, child, -beta, -beta + 1, d);
            yolah.undo(wit);

            if (stopped()) return 0;
            // Coupure. Si la recherche réduite prétend à une victoire prouvée
            // (v ≥ WIN), on ne la croit pas (la profondeur réduite a pu manquer
            // une défense) : on renvoie seulement beta. Rien n'est rangé dans
            // la table pour ce nœud.
            if (v >= beta) return v >= WIN ? beta : v;
        }
    }

    // ── Mise à jour de l'historique de correction (`corr_update`, U4/U6) ─────
    // Appelée à la fin du nœud (coupure ou fin de boucle) avec la valeur `bestv`
    // trouvée par la recherche. L'écart `bestv − static_eval` dit de combien le
    // réseau s'est trompé sur ce nœud ; on en déplace la correction de la
    // configuration de pièces correspondante.
    const int alpha_orig = alpha;   // fenêtre d'entrée (alpha montera dans la boucle)
    int  best      = -INFINITE;     // meilleure valeur trouvée (fail-soft)
    Move best_move = Move::none();  // coup correspondant, s'il a battu alpha

    auto corr_update = [&](int bestv) {
        // Rien à apprendre : pas d'évaluation statique ici (nœud PV), résultat
        // prouvé (l'écart avec une évaluation n'a pas de sens), ou recherche interrompue.
        if (s.eval_stack[ply_now] == EVAL_NONE || std::abs(bestv) >= WIN || stopped()) return;
        // `bestv` n'est souvent qu'une borne. Une borne inférieure (coupure bêta)
        // située SOUS static_eval, ou une borne supérieure (échec bas) située
        // AU-DESSUS de static_eval, ne contredit pas static_eval : elle ne dit
        // pas dans quel sens le réseau s'est trompé. Seule une valeur exacte, ou
        // une borne qui contredit l'évaluation, apprend quelque chose.
        if (bestv >= beta       && bestv <= static_eval) return;
        if (bestv <= alpha_orig && bestv >= static_eval) return;

        // Pas d'apprentissage proportionnel à l'erreur et à la profondeur (une
        // recherche profonde est plus fiable), divisé par 8 et borné à ±2000 ;
        // puis mise à jour par « gravité » (comme l'historique, section 7),
        // qui garde la correction dans [−7917, 7917].
        const int b = std::clamp((bestv - static_eval) * depth / CORR_SCALE, -2000, 2000);
        int16_t& e  = s.w->corr[yolah.current_player()][corr_key(yolah)];
        e = int16_t(std::clamp(e + b - e * std::abs(b) / CORR_MAX, -CORR_MAX, CORR_MAX));
    };

    // ── U5 (1) : Détection paresseuse des poches scellées (`reserve = 1`) ────
    // Une « poche scellée » est une région de cases libres `comp` (de taille
    // ≤ 20) que NOUS sommes seuls à pouvoir atteindre, et telle que nos pièces
    // `pieces` qui la touchent n'ont accès à rien d'autre (`sealed == true`).
    //
    // Pourquoi c'est EXACT :
    //   • `comp` est une région 8-connexe entière de cases libres que
    //     l'adversaire ne peut pas atteindre (`region_reach`, section 5) :
    //     aucune pièce adverse n'y entrera jamais.
    //   • Pour entrer dans `comp`, une pièce doit passer par une case voisine de
    //     `comp` ; une case voisine libre ferait partie de `comp`. Donc seules nos
    //     pièces VOISINES de `comp` peuvent y aller, et ce sont exactement `pieces`.
    //   • `sealed` exige que ces pièces n'aient aucune autre case libre voisine :
    //     elles ne peuvent aller que dans `comp`.
    //   La poche et ses pièces forment donc un petit jeu à part, sans effet sur
    //   le reste du plateau : la seule chose qui compte est le nombre de coups
    //   qu'on pourra encore y jouer, une réserve de tempos (on joue ces coups
    //   quand on veut, par exemple pour ne pas avoir à abîmer son espace ailleurs).
    //   Le maximum est `h = harvest_exact(pieces, comp)`. Un coup de ces pièces
    //   laisse au mieux `h − 1` coups dans la poche ; tous ceux qui en laissent
    //   exactement `h − 1` sont équivalents (même reste du plateau, même
    //   réserve), et ceux qui en laissent moins sont moins bons. Il suffit donc
    //   d'essayer UN SEUL coup `keep` qui laisse `h − 1`, et d'ignorer tous les
    //   autres coups de `pieces` : on ne perd rien d'exact.
    // Mesuré : +10.3 Elo à 1 s (surtout en fin de partie, où les poches abondent).
    uint64_t reserve_pieces[4]{};   // jusqu'à 4 poches (une par pièce au plus)
    Move     reserve_keep[4]{};     // le seul coup gardé pour chacune
    int      n_reserves   = 0;
    bool     reserve_done = false;  // calcul fait une seule fois par nœud, et seulement si besoin

    auto compute_reserves = [&]() {
        const uint8_t  us       = yolah.current_player();
        const uint64_t free     = yolah.free_squares();
        const uint64_t mine     = yolah.bitboard(us);
        const uint64_t occupied = yolah.occupied_squares();
        // Cases que nous pouvons atteindre un jour et l'adversaire jamais.
        uint64_t priv = region_reach(mine, free) & ~region_reach(yolah.bitboard(Yolah::other_player(us)), free);

        // Découpage de `priv` en régions 8-connexes, une par une.
        while (priv && n_reserves < 4) {
            const uint64_t seed = priv & -priv;                    // la plus petite case restante
            const uint64_t comp = region_reach(seed, priv) | seed; // sa région entière dans `priv`
            priv &= ~comp;
            if (std::popcount(comp) > RESERVE_MAX_SQUARES) continue;  // trop grande pour un calcul exact

            // Nos pièces voisines de la poche ; la poche n'est scellée que si
            // aucune d'elles ne touche aussi une case libre hors de la poche.
            uint64_t pieces = 0;
            bool sealed = true;
            for (uint64_t b = mine; b;) {
                const uint64_t q  = square_bb(pop_lsb(b));
                const uint64_t nb = around(q) & free;
                if (nb & comp) {
                    pieces |= q;
                    if (nb & ~comp) sealed = false;   // la pièce touche aussi une autre région
                }
            }
            if (!sealed || !pieces) continue;

            const int h = harvest_exact(pieces, comp, HARVEST_BUDGET);
            if (h <= 0) continue;   // budget dépassé (−1), ou poche où l'on ne peut rien jouer

            // Trouve le premier coup dans la poche qui conserve une récolte optimale (h - 1)
            Move keep = Move::none();
            for (uint64_t b = pieces; b && keep == Move::none();) {
                const Square from = pop_lsb(b);
                for (uint64_t t = attacks_bb(from, occupied) & comp; t;) {
                    const Square to = pop_lsb(t);
                    if (harvest_exact(pieces ^ square_bb(from) ^ square_bb(to),
                                      comp & ~square_bb(to), HARVEST_BUDGET) == h - 1) {
                        keep = Move(from, to);
                        break;
                    }
                }
            }
            if (keep == Move::none()) continue;   // (budget dépassé dans les sous-calculs)
            reserve_pieces[n_reserves] = pieces;
            reserve_keep[n_reserves++] = keep;
        }
    };

    // ── 6.g) Boucle principale sur les coups du nœud ─────────────────────────
    // Tant que la liste n'est pas générée (étape 1), la condition `!generated`
    // fait entrer dans la boucle pour i = 0 (le coup de table) puis pour i = 1,
    // où l'étape 2 génère la liste.
    for (size_t i = 0; i < n || !generated; i++) {
        // Étape 2 de la génération par étapes : le coup de la table (`i == 0`)
        // n'a pas suffi à couper. On génère maintenant tous les coups, on place
        // `tt_move` en `moves[0]` (déjà exploré : la sélection reprendra à
        // l'indice 1 et ne le reverra pas) et on note tous les coups.
        if (!generated && i == 1) {
            yolah.moves(moves);
            generated = true;
            for (size_t j = 0; j < moves.size(); j++) {
                if (moves[j] == tt_move) {
                    std::swap(moves[0], moves[j]);
                    break;
                }
            }
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
            if (i >= n) break;   // le coup de table était le seul coup
        }

        // F3. Late Move Pruning (LMP, +35 Elo) :
        // Aux nœuds non-PV peu profonds (depth ≤ 4), dès qu'au moins un coup
        // non perdant (`best > -WIN`) a été trouvé et qu'on a déjà testé
        // `2 + depth²` coups (soit 3, 6, 11 ou 18 coups aux profondeurs 1 à 4),
        // on élague le reste de la liste : ce sont les coups les moins bien
        // classés, et près des feuilles les chercher coûte plus qu'ils ne
        // rapportent. (« non perdant » : on ne s'arrête jamais en ne connaissant
        // que des coups prouvés perdants, il pourrait rester un sauvetage.)
        // Le compteur `i` inclut les coups écartés par les poches ci-dessous.
        if (prunable && depth <= LMP_DEPTH && i >= size_t(LMP_MOVES + depth * depth) && best > -WIN) {
            break;
        }

        // Sélection paresseuse du meilleur coup restant en position `i`
        // (évite un tri complet O(n log n) alors que la majorité des nœuds
        // coupent dès le 1er ou le 2e coup). Après l'appel, `moves[i]` et
        // `scores[i]` sont ceux de `m`.
        const Move m = generated ? pick_move(moves, scores, i, n) : tt_move;

        // U5 (1) : Filtrage exact des poches scellées (`reserve = 1`) :
        // un coup d'une pièce d'une poche scellée, autre que le coup gardé
        // `keep`, est sauté (voir la justification plus haut). Le calcul des
        // poches n'est fait qu'au premier coup de la liste générée, donc pas du
        // tout si le coup de table a coupé.
        if (generated && m != Move::none()) {
            if (!reserve_done) {
                compute_reserves();
                reserve_done = true;
            }
            bool drop = false;
            for (int k = 0; k < n_reserves; k++) {
                if ((reserve_pieces[k] & square_bb(m.from_sq())) && m != reserve_keep[k]) {
                    drop = true;
                }
            }
            if (drop) continue;
        }

        // ── Calcul de la réduction LMR du coup `m` (E + U7) ──────────────────
        const uint8_t player = yolah.current_player();
        // • Pas de réduction à l'étape 1 (coup de table, `!generated`).
        // • « PV » se lit sur la fenêtre COURANTE : alpha a pu monter dans la boucle.
        // • `special` : coup noté au moins SCORE_COUNTER, c'est-à-dire le
        //   contre-coup (le coup de table, à l'indice 0, n'est jamais réduit).
        // • L'historique passé à la formule est l'historique PRINCIPAL seul (pas
        //   la continuation ni la couverture : les mêler à la réduction a été
        //   mesuré moins bon) ; le contre-coup, noté au-dessus de
        //   SCORE_ARTICULATION, compte pour 0.
        int r = !generated ? 0
              : late_move_reduction_of(
                    depth, i,
                    /*pv_node=*/beta - alpha > 1,
                    /*special=*/scores[i] >= SCORE_COUNTER,
                    scores[i] >= SCORE_ARTICULATION ? 0 : s.w->history[player][m.from_sq()][m.to_sq()]
                );

        // Réductions LMR contextuelles (U7 : `intercept_lmr` et `lmrh`) :
        // Si le coup est réduit (`r > 0`) et répond au dernier coup adverse `q1`
        // (connu : pas le premier pli de la partie, et pas une passe ni le coup
        // inconnu d'avant la racine) :
        //   1. Interception : si `m` atterrit sur une ligne de la pièce adverse qui
        //      vient de jouer (ses déplacements de dame depuis `q1.to_sq()`, à
        //      travers les cases libres) ou sur une case voisine d'elle, on réduit
        //      d'un pli de moins (c'est la réponse à une menace : bloquer la dame
        //      qui vient d'avancer, ou contester l'espace qu'elle vient de prendre).
        //   2. `lmrh` : si la paire `(q1.to_sq(), m.to_sq())` a souvent mis en défaut
        //      la réduction LMR par le passé (`lmrh >= 4453`), on réduit aussi
        //      d'un pli de moins : la recherche réduite serait de toute façon refaite.
        if (r > 0 && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {
            const Move q1 = s.played[ply_now - 1];
            const uint64_t intercept_zone = attacks_bb(q1.to_sq(), yolah.occupied_squares())
                                          | around(square_bb(q1.to_sq()));
            if (intercept_zone & square_bb(m.to_sq())) {
                r = std::max(0, r - 1);
            }
            if (s.w->lmrh[player][q1.to_sq()][m.to_sq()] >= LMRH_THRESHOLD) {
                r = std::max(0, r - 1);
            }
        }

        // Toujours au moins 1 pli de recherche réduite (déjà assuré par la
        // formule ; répété ici par sécurité, comme dans le joueur).
        const int r_used = std::min(r, std::max(0, depth - 2));
        const int v = search_move(yolah, s, hash, m, i, r_used, alpha, beta, depth);

        if (stopped()) return 0;

        // U7 (C5) : Apprentissage des réductions erronées (`lmrh`) :
        // Si le coup a été réduit (`r_used > 0`) mais qu'il a finalement battu
        // `alpha` après re-recherche à pleine profondeur, la réduction était
        // trop optimiste (elle a coûté une recherche de plus) : on augmente
        // `lmrh[player][prev.to][m.to]` d'un bonus croissant avec la profondeur,
        // par gravité (h + b − h·b/16384 : reste sous 16384). Ce bonus garde les
        // valeurs d'origine 16 d² + 32 d plafonnées à 2000 : le SPSA n'a réglé
        // que celui de l'historique principal (HIST_BONUS_*).
        // (Strictement, lmrh apprend où une réduction a COÛTÉ une re-recherche,
        // pas où elle a fait rater un bon coup : un coup réduit qui échoue bas
        // n'est jamais re-cherché, la recherche ne peut pas voir cette erreur.)
        if (r_used > 0 && v > alpha && ply_now > 0 && s.played[ply_now - 1] != Move::none()) {
            int16_t& h = s.w->lmrh[player][s.played[ply_now - 1].to_sq()][m.to_sq()];
            const int b = std::min(16 * depth * depth + 32 * depth, 2000);
            h = int16_t(h + b - h * b / HISTORY_MAX);
        }

        // ── 6.h) Mise à jour des bornes et coupure bêta ──────────────────────
        if (v > best) {
            best = v;
            if (v > alpha) {
                best_move = m;
                if (v >= beta) {
                    // Coupure bêta : l'adversaire n'entrera pas dans cette position
                    // (il a mieux ailleurs). On arrête le nœud, et la recherche
                    // apprend de cette coupure à cinq endroits.

                    // 1. Enregistrement dans la table de transposition (borne
                    //    inférieure v, coup m) et dans la correction `corr`
                    tt_store(hash, yolah, depth, alpha_orig, beta, v, m, /*pv=*/false);
                    corr_update(v);

                    // 2. Mise à jour des deux coups tueurs (`killer1`, `killer2`) du pli :
                    //    m devient le premier, l'ancien premier passe second (sauf si
                    //    m était déjà le premier : rien ne change).
                    if (s.killer1[ply_now] != m) {
                        s.killer2[ply_now] = s.killer1[ply_now];
                        s.killer1[ply_now] = m;
                    }

                    // 3. Mise à jour par « gravité » de l'historique principal (`history`)
                    //    et de l'historique de continuation à 1 pli (`cont1`) :
                    //    • Le coup `m` qui a provoqué la coupure bêta reçoit `+bonus`.
                    //    • Tous les coups `moves[0..i-1]` placés avant lui reçoivent
                    //      un malus `-bonus` : ils ont été essayés sans couper (ou sautés
                    //      par le filtre des poches, qui les laisse aussi à ces indices).
                    //    Le bonus grandit avec la profondeur (une coupure profonde en dit
                    //    plus long) : min(16 d² + 34 d, 2166), soit 50 à d = 1, 2166 dès d = 11.
                    //    (Si la liste n'est pas générée, i = 0 : aucun malus.)
                    const int bonus = std::min(HIST_BONUS_A * depth * depth + HIST_BONUS_B * depth, HIST_BONUS_CAP);
                    update_history(s, player, m, bonus);
                    for (size_t k = 0; k < i; k++) {
                        update_history(s, player, moves[k], -bonus);
                    }

                    // `cont1` : même bonus et mêmes malus, dans le contexte de la case
                    // d'arrivée du dernier coup adverse `p1` (s'il est connu).
                    const Move p1 = ply_now > 0 ? s.played[ply_now - 1] : Move::none();
                    if (p1 != Move::none()) {
                        auto upd_cont1 = [&](Move mv, int b) {
                            if (mv == Move::none()) return;
                            int16_t& h = s.w->cont1[player][p1.to_sq()][mv.to_sq()];
                            h = int16_t(h + b - h * std::abs(b) / HISTORY_MAX);
                        };
                        upd_cont1(m, bonus);
                        for (size_t k = 0; k < i; k++) {
                            upd_cont1(moves[k], -bonus);
                        }
                    }
                    // 4. Enregistrement du contre-coup (`countermoves`) : m a réfuté p1.
                    //    Y compris quand p1 est une passe adverse ou le coup inconnu
                    //    d'avant la racine (tous deux Move::none() : case [0][0] de la
                    //    table), exactement comme le vrai joueur.
                    if (ply_now > 0) s.w->countermoves[p1.from_sq()][p1.to_sq()] = m;

                    return v;   // fail-soft : v peut dépasser beta
                }
                alpha = v;   // nouveau meilleur coup : les suivants doivent faire mieux
            }
        }
    }

    // Fin du nœud sans coupure bêta : valeur exacte (si `best > alpha_orig`)
    // ou borne supérieure fail-soft (si tous les coups ont échoué sous `alpha_orig` ;
    // `best_move` vaut alors Move::none(), et la table garde l'ancien meilleur coup
    // de cette position s'il y en avait un). La table déduit la nature du
    // résultat de la fenêtre (alpha_orig, beta).
    tt_store(hash, yolah, depth, alpha_orig, beta, best, best_move, /*pv=*/pv_node);
    corr_update(best);
    return best;
}


// =============================================================================
//  7. ORDONNANCEMENT DES COUPS (`score_moves`) & HISTORIQUE DE CONTINUATION
// =============================================================================
//
//  Pourquoi l'ordre des coups compte tant : l'alpha-bêta ne coupe que si un
//  bon coup est essayé tôt. Avec le meilleur coup en premier, un nœud coupe
//  après un seul coup et tous les autres sont épargnés ; avec un mauvais
//  ordre, l'arbre explose. Les réductions LMR et le LMP supposent aussi que
//  les coups tardifs sont les moins bons.
//
//  À Yolah, il n'y a ni prises ni promotions : tous les coups sont « calmes ».
//  Dans le moteur de base (`dev_final`), le tri des coups ordinaires reposait
//  sur `history[player][m.from][m.to]`, qui mesure si un coup `(from → to)` est
//  bon EN MOYENNE dans tout l'arbre, indépendamment de ce que l'adversaire
//  vient de jouer.
//
//  Or, à Yolah, la valeur d'un coup est essentiellement RÉACTIVE : si
//  l'adversaire vient d'amener une pièce en `p1.to`, la bonne réponse consiste
//  souvent à lui fermer l'accès à une zone ou à contester son voisinage autour
//  de `p1.to`. C'est le rôle des trois mécanismes contextuels indexés par `p1.to`
//  (1 à 3), auxquels s'ajoute la couverture (4) :
//
//    1. L'historique de continuation à 1 pli (`cont1[player][p1.to][m.to]`, +32.6 Elo) :
//       • Indexé par (dernière destination adverse `p1.to`, notre destination `m.to`).
//       • Lorsqu'un coup `m` coupe bêta après le coup adverse `p1`, `cont1[us][p1.to][m.to]`
//         reçoit le même bonus de gravité (`+bonus`) que `history`, et tous les coups
//         essayés avant `m` sans couper reçoivent `-bonus` (section 6.h).
//       • Ignorer les cases de départ (`p1.from` et `m.from`) réduit la table à
//         64 × 64 entrées par joueur : si la case `c5` est la bonne case de blocage
//         quand l'adversaire joue en `d5`, cet apprentissage profite immédiatement
//         à TOUTES nos pièces capables d'aller en `c5`, quelle que soit la case
//         d'où venait la pièce adverse !
//       • Contrairement à `history`, `cont1` n'est PAS divisé par 2 entre deux coups
//         de la partie (`cont_age = false`), et il intervient uniquement dans le tri
//         des coups (`score_moves`), pas dans le terme `-hist / 6945` de LMR
//         (`cont_lmr = false`).
//
//    2. Le bonus d'interception (`INTERCEPT_BONUS = +2000`, +17.0 Elo avec la
//       réduction diminuée de la section 6.g) :
//       • Priorité géométrique a priori (dès le début de la recherche, avant même que
//         `cont1` ait appris) pour tout coup dont la destination `m.to` se trouve
//         sur l'un des rayons de dame partant de `p1.to` (à travers les cases
//         libres) ou sur l'une des 8 cases voisines de `p1.to`.
//
//    3. Les coups tueurs en bonus (`KILLER_BONUS = +4667` après SPSA, `killer_mode = 1`, +17.7 Elo) :
//       • Au lieu de trier `killer1` et `killer2` avec un score fixe géant devant tous
//         les coups ordinaires, on leur donne `hist(m) + 4667` (où `hist(m)` inclut
//         `history + cont1 + intercept`). Ainsi, un coup ordinaire doté d'une forte
//         continuation (`cont1`) et d'une interception peut dépasser un killer issu
//         d'une autre branche où l'adversaire avait joué ailleurs.
//
//    4. La couverture de notre groupe (`COVER_WEIGHT = 1038` après SPSA, à toutes les profondeurs,
//       U19-U22 : +17.9 Elo à 0.2 s, +6.9 Elo à 1 s contre dev_u8) :
//       • `coverage(mine, occupied)` = les cases LIBRES atteintes en un coup par
//         au moins une de nos pièces (union des déplacements de dame `attacks_bb`),
//         chaque case comptée une fois.
//       • Chaque coup ordinaire reçoit `1038 × (|C_après| − |C_avant|)` : après m,
//         la pièce part de `from` (qui devient un trou, donc reste occupée) et
//         arrive en `to` (qui n'est plus libre). Les pièces adverses ne comptent
//         que comme obstacles : ce n'est pas un territoire disputé, c'est
//         l'espace que notre groupe garde ouvert. Une case de couverture vaut
//         1038 points d'historique.
//       • Ce terme remplace l'influence de dev_u8 (`512 × influence` à depth ≥ 4,
//         J : +36 Elo contre l'ordre par l'historique seul) : la couverture se
//         calcule avec 4 consultations de la table d'attaques par coup (une par
//         pièce) au lieu d'une inondation pas à pas du plateau, et s'applique
//         donc à toutes les profondeurs.
//
//  Résumé du score d'un coup (le plus grand est essayé d'abord) :
//      coup de table        : SCORE_TT = 2^30
//      contre-coup          : SCORE_COUNTER = 2^29 − 2   (s'il n'est pas aussi un killer)
//      killer               : min(hist(m) + 4667, 2^28 − 1) + couverture
//      autre coup           : hist(m) + couverture
//  avec hist(m) = history + cont1 + (interception ? 2000 : 0).
// =============================================================================

void FinalDevPlayer::score_moves(const Yolah& yolah, const Search& s, Move tt_move,
                                 const Yolah::MoveList& moves, int* scores, int depth) const {
    const bool     cover     = depth >= COVER_DEPTH;   // toujours vrai ici (depth ≥ 1)
    const uint16_t ply       = yolah.nb_plies();
    const uint8_t  player    = yolah.current_player();
    // Dernier coup adverse (Move::none() au premier pli, après une passe, ou
    // à la racine, dont le coup précédent n'est pas connu de la recherche).
    const Move     p1        = ply > 0 ? s.played[ply - 1] : Move::none();
    // Contre-coup lu aussi après une passe adverse (ou à la racine) : la case
    // [0][0] de la table sert alors, comme dans le joueur.
    const Move     counter   = ply > 0 ? s.w->countermoves[p1.from_sq()][p1.to_sq()] : Move::none();
    // Couverture de notre groupe AVANT le coup (une fois par nœud)
    const uint64_t mine0     = yolah.bitboard(player);
    const uint64_t occ0      = ~yolah.free_squares();   // pièces des deux camps + trous
    const int      c0        = cover ? std::popcount(coverage(mine0, occ0)) : 0;

    // Score d'historique composite (historique principal + continuation 1 pli + interception).
    // Sans dernier coup adverse connu, seul l'historique principal compte.
    auto hist = [&](Move m) {
        int h = s.w->history[player][m.from_sq()][m.to_sq()];
        if (p1 != Move::none() && m != Move::none()) {
            h += s.w->cont1[player][p1.to_sq()][m.to_sq()];
            const uint64_t intercept_mask = attacks_bb(p1.to_sq(), yolah.occupied_squares())
                                          | around(square_bb(p1.to_sq()));
            if (intercept_mask & square_bb(m.to_sq())) {
                h += INTERCEPT_BONUS;
            }
        }
        return h;
    };

    for (size_t i = 0; i < moves.size(); i++) {
        const Move m   = moves[i];
        const bool kil = (m == s.killer1[ply] || m == s.killer2[ply]);

        if (m == tt_move) {
            scores[i] = SCORE_TT;
        } else if (kil) {
            // `killer_mode = 1` : les killers restent dans la catégorie des coups
            // ordinaires (< SCORE_ARTICULATION) avec un bonus de +4667 (KILLER_BONUS) sur `hist(m)`
            // (testé AVANT le contre-coup : un coup à la fois tueur et contre-coup
            // est traité en tueur, comme dans le joueur).
            scores[i] = std::min(hist(m) + KILLER_BONUS, SCORE_ARTICULATION - 1);
        } else if (m == counter && counter != Move::none()) {
            scores[i] = SCORE_COUNTER;
        } else {
            scores[i] = hist(m);
        }

        // Terme de couverture (coups ordinaires seulement) : variation du nombre
        // de cases libres que notre groupe atteint, si l'on joue m.
        // Après m : nos pièces sont (mine0 sans from) + to ; les cases occupées
        // sont occ0 + to (from reste occupée : c'est désormais un trou).
        if (scores[i] < SCORE_ARTICULATION && cover) {
            const uint64_t from = square_bb(m.from_sq()), to = square_bb(m.to_sq());
            const uint64_t c1   = coverage((mine0 & ~from) | to, occ0 | to);
            scores[i] += COVER_WEIGHT * (std::popcount(c1) - c0);
        }
    }
}

// ── Couverture d'un groupe de pièces (`coverage`) ───────────────────────────
// Les cases libres qu'au moins une des pièces `mine` atteint en un coup
// (déplacement de dame, arrêté par toute case occupée : pièce ou trou).
// `attacks_bb` inclut la case bloquante de chaque rayon ; `& ~occupied` la retire.
uint64_t FinalDevPlayer::coverage(uint64_t mine, uint64_t occupied) {
    uint64_t c = 0;
    for (uint64_t b = mine; b;) c |= attacks_bb(pop_lsb(b), occupied);
    return c & ~occupied;
}

// Le coup `m` est-il légal dans `yolah` ? (Sert pour les coups lus dans la
// table de transposition, qui peuvent venir d'une autre position.)
// Légal = la case de départ porte une pièce du camp au trait, et la case
// d'arrivée est libre et atteinte en ligne droite sans obstacle.
// La passe (Move::none()) n'est jamais considérée comme un coup de table légal.
bool FinalDevPlayer::is_legal(const Yolah& yolah, Move m) {
    if (m == Move::none()) return false;
    const uint64_t from = uint64_t(1) << m.from_sq();
    const uint64_t to   = uint64_t(1) << m.to_sq();
    if (!(yolah.bitboard(yolah.current_player()) & from)) return false;
    return (attacks_bb(m.from_sq(), yolah.occupied_squares()) & yolah.free_squares() & to) != 0;
}

// Sélection : amène en position `i` le coup de meilleur score parmi
// moves[i..n) (échange de `moves` et de `scores` ensemble) et le renvoie.
// En cas d'égalité, le premier rencontré gagne (ordre de génération).
Move FinalDevPlayer::pick_move(Yolah::MoveList& moves, int* scores, size_t i, size_t n) {
    size_t best = i;
    for (size_t j = i + 1; j < n; j++) {
        if (scores[j] > scores[best]) best = j;
    }
    std::swap(moves[i], moves[best]);
    std::swap(scores[i], scores[best]);
    return moves[i];
}

// Mise à jour par « gravité » :
//     h ← h + bonus − h × |bonus| / HISTORY_MAX
// Plus `h` s'approche de ±16384, plus un bonus de même signe est amorti et
// plus un malus de signe opposé le ramène vite : `h` reste naturellement
// borné dans [-16384, +16384] sans saturation brutale, et les coupures
// récentes pèsent plus que les anciennes.
// Exemple : h = 8192, bonus = 2000 → h = 8192 + 2000 − 1000 = 9192 ;
//           h = 8192, malus = −2000 → h = 8192 − 2000 − 1000 = 5192.
void FinalDevPlayer::update_history(Search& s, uint8_t player, Move m, int bonus) {
    int16_t& h = s.w->history[player][m.from_sq()][m.to_sq()];
    h = int16_t(h + bonus - h * std::abs(bonus) / HISTORY_MAX);
}


// =============================================================================
//  8. ÉVALUATION NNUE PARESSEUSE (`lazy_accumulator`) & CACHE D'ÉVALUATIONS (D)
// =============================================================================
//
//  Fonctionnement du cache d'évaluations (`eval_cache`) :
//    • 2^20 emplacements indexés par les 20 bits de poids faible du hash Zobrist.
//    • Chaque entrée tient dans UN SEUL mot atomique de 64 bits :
//          bits 63..17 : les 47 bits de poids fort du hash Zobrist (clé)
//          bits 16..1  : la valeur `int16_t` du réseau (non corrigée)
//          bit  0      : 1 si l'emplacement est occupé
//    • Lecture/écriture atomique `relaxed` : même en multithread, une entrée ne
//      peut jamais être « déchirée » (clé d'une position avec valeur d'une autre),
//      puisque clé et valeur sont écrites ensemble en une seule opération.
//    • Une nouvelle valeur écrase toujours l'ancienne (pas de politique de
//      remplacement). Deux positions différentes ne sont confondues que si
//      elles ont les mêmes 20 bits bas ET les mêmes 47 bits hauts.
//    • Le cache garde la valeur BRUTE du réseau ; la correction `corr`, qui
//      change pendant la recherche, est ajoutée par l'appelant (section 6).
//
//  Fonctionnement des accumulateurs paresseux (`lazy_accumulator`) :
//    • L'accumulateur d'une position est la sortie de la première couche du
//      réseau (avant activation), mise à jour par différence d'un coup à l'autre.
//    • Lorsqu'on descend dans l'arbre (`search_move`, Proxy Cut), on se contente
//      de noter `s.played[ply] = m` et `s.acc_ok[ply + 1] = false`.
//    • Si le sous-arbre coupe par la table de transposition, par une fin de
//      partie, ou si la valeur est dans le cache, aucune opération NNUE n'aura
//      été payée !
//    • Si `raw_network_value` est appelée au pli `ply`, on remonte au dernier
//      ancêtre `k <= ply` dont `s.acc_ok[k]` est vrai (au pire la racine, calculée
//      dans `search`), puis on redescend de `k` à `ply` en appelant
//      `persp->update(...)` une fois par pli (chaque accumulateur intermédiaire
//      devient valide au passage et servira aux nœuds frères).
//    • Pas besoin de « déjouer » : l'accumulateur du parent n'est jamais modifié,
//      chaque pli a le sien.
// =============================================================================

// Valeur du réseau (non corrigée) pour le camp au trait, en passant par le cache.
int FinalDevPlayer::evaluate(const Yolah& yolah, Search& s, uint64_t hash) {
    s.nb_evals++;
    constexpr uint64_t KEY_MASK = ~uint64_t(0) << 17;   // bits 63..17
    const uint64_t key = hash & KEY_MASK;

    std::atomic<uint64_t>& slot = eval_cache[hash & (eval_cache.size() - 1)];
    const uint64_t word = slot.load(std::memory_order_relaxed);
    if ((word & 1) && (word & KEY_MASK) == key) {
        s.nb_eval_hits++;
        // bits 16..1 → uint16_t → int16_t : retrouve la valeur signée.
        return int16_t(uint16_t(word >> 1));
    }

    const int v = raw_network_value(yolah, s);
    // Valeur signée sur 16 bits, décalée d'un bit, + bit « occupé ».
    slot.store(key | (uint64_t(uint16_t(int16_t(v))) << 1) | 1, std::memory_order_relaxed);
    return v;
}

// Calcule réellement le réseau pour la position du pli courant.
int FinalDevPlayer::raw_network_value(const Yolah& yolah, Search& s) {
    const int ply = yolah.nb_plies();
    int k = ply;
    while (!s.acc_ok[k]) k--;   // remonte jusqu'au dernier accumulateur déjà calculé

    // Redescend pli par pli : accs[k + 1] = accs[k] + le coup played[k], joué
    // par le camp `k & 1` (Noir aux plis pairs, Blanc aux impairs). Une passe
    // (Move::none()) ne change aucune case.
    for (; k < ply; k++) {
        persp->update(s.accs[k].acc, s.accs[k + 1].acc, uint8_t(k & 1), s.played[k], &s.refresh);
        s.acc_ok[k + 1] = true;
    }
    // Le réseau rend une valeur dans [−1, 1] pour le camp au trait ; × 30 000
    // et tronquée vers zéro pour l'échelle de la recherche.
    return int(persp->value(s.accs[ply].acc, yolah) * WIN);
}


// =============================================================================
//  9. ACCÈS À LA TABLE DE TRANSPOSITION (`SearchTable`) & VARIATION PRINCIPALE
// =============================================================================
//
//  Rappel du fonctionnement de `SearchTable` (player/search_table.{h,cpp}) :
//    • Clusters de 4 entrées de 16 octets (64 octets = une ligne de cache),
//      choisis par les bits bas du hachage ; l'entrée est reconnue grâce aux
//      bits 63..16 du hachage.
//    • `store` à profondeur PLUS grande que l'entrée : nouvelles bornes
//      (le meilleur coup connu est gardé si le nouveau résultat n'en a pas) ;
//      à profondeur ÉGALE : les bornes sont fusionnées (l'intervalle
//      [lower, upper] ne fait que rétrécir ; si elles se croisent, signe d'une
//      incohérence due aux élagages, le nouveau résultat remplace l'ancien) ;
//      à profondeur PLUS FAIBLE : les bornes ne sont pas touchées.
//    • Remplacement quand le cluster est plein : d'abord les positions plus
//      « anciennes » que la racine (plus de cases libres), puis la moins profonde.
//    • Pas de verrou : une entrée peut être lue pendant qu'un autre thread
//      l'écrit. D'où la vérification `is_legal` des coups lus.
// =============================================================================

FinalDevPlayer::TTView FinalDevPlayer::tt_probe(uint64_t hash) const {
    TTView view;
    if (const SearchTable::Entry* e = yolah_table->probe(hash)) {
        view = {true, e->move, e->depth, e->lower, e->upper};
    }
    return view;
}

// `fragile = false` : option expérimentale du vrai joueur, inactive dans dev_u9_spsa.
// Le nombre de cases libres de la position sert au remplacement (voir plus haut).
void FinalDevPlayer::tt_store(uint64_t hash, const Yolah& yolah, int depth,
                              int alpha, int beta, int value, Move move, bool pv) {
    yolah_table->store(hash, std::popcount(yolah.free_squares()), depth, alpha, beta, value, move, pv, /*fragile=*/false);
}

Move FinalDevPlayer::tt_move(uint64_t hash) const {
    return yolah_table->get_move(hash);
}

// Affiche (mode verbeux) la variation principale : la suite des meilleurs coups
// de la table, en vérifiant la légalité de chacun, sur au plus `depth` coups.
void FinalDevPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth <= 0) return;
    const Move m = tt_move(hash);
    if (m == Move::none() || !is_legal(yolah, m)) return;
    const uint8_t player = yolah.current_player();
    cout << m << ' ';
    yolah.play(m);
    print_pv(yolah, zobrist::update(hash, player, m), depth - 1);
}

// Même chose, renvoyée comme une liste (le vrai joueur a la même fonction,
// dont l'application se sert pour afficher la variation prévue).
std::vector<Move> FinalDevPlayer::principal_variation(Yolah yolah, int max_len) const {
    std::vector<Move> pv;
    uint64_t hash = zobrist::hash(yolah);
    while (int(pv.size()) < max_len && !yolah.game_over()) {
        const Move m = tt_move(hash);
        if (m == Move::none() || !is_legal(yolah, m)) break;
        pv.push_back(m);
        hash = zobrist::update(hash, yolah.current_player(), m);
        yolah.play(m);
    }
    return pv;
}

std::string FinalDevPlayer::info() {
    return "final dev player (dev_u9_spsa, " + std::to_string(nb_threads) + " threads)";
}

// Configuration minimale (les constantes de recherche sont figées dans final_dev.h).
json FinalDevPlayer::config() {
    json j;
    j["name"]         = "FinalDevPlayer (dev_u9_spsa)";
    j["microseconds"] = thinking_time;
    j["tt size"]      = tt_size_mb;
    j["nb threads"]   = nb_threads;
    j["weights"]      = weights_filename;
    return j;
}
