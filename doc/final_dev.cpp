// =============================================================================
//  FINAL_DEV (dev_u8) — Implémentation épurée et commentée du joueur Minimax NNUE
// =============================================================================
//
//  Ce fichier rassemble uniquement le code effectivement exécuté par la
//  configuration `dev_u8` (utilisée dans l'application Yolah pour « Minimax
//  NNUE dev » et pour la barre d'analyse).
//
//  Plan de lecture du fichier :
//    1. Constructeur, lancement de la recherche et parallélisme Lazy SMP
//    2. Approfondissement itératif & fenêtres d'aspiration adaptatives
//    3. Recherche à la racine (`root_search`) & tri par taille de sous-arbre
//    4. Recherche d'un coup (`search_move` : LMR + PVS) & formule LMR
//    5. Fonctions géométriques auxiliaires (influence rapide, poches scellées,
//       récolte exacte `harvest_exact`, clé de correction `corr_key`)
//    6. Nœuds internes (`negamax`) :
//         a) Fin de partie & Table de transposition (`SearchTable`)
//         b) Évaluation statique corrigée (`evaluate` + `corr`)
//         c) Reverse Futility Pruning (RFP)
//         d) Multi-ProbCut bidirectionnel (MPC)
//         e) Génération par étapes (Stage 1 : `tt_move` avant génération)
//         f) Proxy Cut (substitut au null-move par coup ordinaire témoin)
//         g) Boucle des coups : Stage 2, Late Move Pruning (LMP),
//            Poches scellées (`reserve`), LMR contextuel (`intercept_lmr`, `lmrh`)
//         h) Coupure bêta : mise à jour de la table, de la correction,
//            des killers, de l'historique, de la continuation (`cont1`)
//            et du contre-coup
//    7. Ordonnancement des coups (`score_moves`) & Territoire (`territory_after`)
//    8. Évaluation NNUE paresseuse (`lazy_accumulator`) & Cache d'évaluations
//    9. Accès à la table de transposition Yolah (`tt_probe`, `tt_store`)
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

    // 1. Un `Worker` privé par thread (historiques, killers, fenêtres d'aspiration)
    for (size_t i = 0; i < this->nb_threads; i++) {
        workers.push_back(std::make_unique<Worker>());
    }

    // 2. Chargement du réseau NNUE vu du camp au trait (perspective + 8 têtes de phase)
    //    et activation du produit matriciel creux (sparse_h1 : +61 Elo à 0.2 s).
    persp = std::make_unique<NNUE_Persp>(weights_filename);
    persp->sparse = true;

    // 3. Allocation de la table de transposition dédiée à Yolah (SearchTable)
    //    et du cache d'évaluations des feuilles (2^20 entrées de 8 octets = 8 Mo).
    yolah_table = std::make_unique<SearchTable>(tt_size_mb);
    eval_cache  = std::vector<std::atomic<uint64_t>>(size_t(1) << EVAL_CACHE_BITS);

    // 4. Précalcul de la table logarithmique des Late Move Reductions (en 1/1024 de pli) :
    //        r(d, n) = 1.25 + ln(d) × ln(n) / 1.5
    //    Stocker r en millièmes de pli (× 1024) permet d'appliquer des ajustements
    //    continus (comme −hist / 8192) avant d'arrondir au nombre entier de plis.
    for (int d = 1; d < 64; d++) {
        for (int n = 1; n < Yolah::MAX_NB_MOVES; n++) {
            lmr_table[d][n] = int(1024 * (LMR_BASE + std::log(d) * std::log(n) / LMR_DIVISOR));
        }
    }
}

FinalDevPlayer::Result FinalDevPlayer::search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds) {
    const auto start = std::chrono::steady_clock::now();
    stop = false;

    // ── Horloge d'arrêt ──────────────────────────────────────────────────────
    // Un thread léger (`std::jthread`) dort pendant `microseconds` puis passe
    // le booléen atomique `stop` à `true`. Si la recherche se termine avant
    // (profondeur max atteinte), le destructeur du jthread le réveille aussitôt.
    std::mutex m;
    std::condition_variable_any cv;
    std::jthread clock;
    if (microseconds > 0) {
        clock = std::jthread([&](std::stop_token st) {
            std::unique_lock lock(m);
            if (!cv.wait_for(lock, st, std::chrono::microseconds(microseconds), [] { return false; })) {
                stop = true;
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
    // (Note : dans `dev_u8`, `cont_age = false` : les tables contextuelles
    // `cont1` et `lmrh` ne sont pas divisées par 2 à chaque coup.)
    for (auto& w : workers) {
        for (auto& by_from : w->history)
            for (auto& by_to : by_from)
                for (int16_t& h : by_to) h /= 2;
    }

    // ── Parallélisme Lazy SMP (L) ────────────────────────────────────────────
    // Les `nb_threads` threads lancent le MÊME approfondissement itératif sur
    // la même position racine :
    //
    //     thread 0 (principal) : prof. 1, 2, 3, 4, 5, 6, 7, 8 …
    //     thread 1             : prof. 1,    3,    5,    7 …   (saute certaines
    //     thread 2             : prof. 1,       4, 5,       8 … profondeurs)
    //                                 ╲      │      ╱
    //                        SearchTable & eval_cache partagés
    //
    // Ils partagent sans verrou la table de transposition et le cache d'éval,
    // mais chacun possède son propre `Worker` (historique, killers, ordre des
    // coups à la racine) : leurs arbres divergent naturellement et explorent
    // en parallèle différentes branches.
    std::vector<std::unique_ptr<Search>> searches;
    for (size_t i = 0; i < nb_threads; i++) {
        auto s = std::make_unique<Search>();
        s->id = int(i);
        s->w  = workers[i].get();

        // Seul l'accumulateur NNUE de la racine est calculé de zéro ; tous les
        // accumulateurs des nœuds descendants seront calculés paresseusement
        // par différence depuis leur parent (`lazy_accumulator`).
        persp->init(yolah, s->accs[yolah.nb_plies()].acc);
        s->acc_ok[yolah.nb_plies()] = true;
        searches.push_back(std::move(s));
    }

    {
        std::vector<std::jthread> helpers;
        for (size_t i = 1; i < nb_threads; i++) {
            helpers.emplace_back([&, i] { iterative_deepening(yolah, *searches[i], max_depth); });
        }
        // Le thread 0 mène la recherche principale ; dès qu'il s'arrête, il
        // stoppe les threads assistants.
        iterative_deepening(yolah, *searches[0], max_depth);
        stop = true;
    }

    if (clock.joinable()) {
        clock.request_stop();
        clock.join();
    }

    // On retient le coup du thread qui a terminé la profondeur la plus élevée
    // (le thread 0 en cas d'égalité).
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
//  Pourquoi des fenêtres d'aspiration adaptatives ?
//    À partir de la profondeur 4, au lieu de chercher dans (−∞, +∞), on cherche
//    dans une fenêtre étroite (v − δ, v + δ) centrée sur la valeur `v` de la
//    profondeur précédente. Une fenêtre étroite provoque beaucoup plus de
//    coupures alpha-bêta.
//
//    Or, avec une demi-largeur fixe δ = 300, la moitié des itérations échouaient
//    (la variation médiane d'une profondeur à la suivante vaut ~630). Dans
//    `dev_u8`, δ s'adapte à la volatilité réelle de la partie :
//
//        δ = max(300, 1.25 × moyenne_glissante(|v_d − v_{d−1}|))
//
//    Lorsqu'une recherche échoue bas (v' ≤ α) ou haut (v' ≥ β) :
//      • On recentre le bord dépassé directement autour de la valeur fail-soft `v'`
//        renvoyée par `root_search` (α = v' − δ en fail-low, β = v' + δ en fail-high).
//      • En cas de fail-low, on abaisse aussi β à (α + β) / 2 pour éviter un
//        rebond trop large.
//      • On multiplie δ par ASPIRATION_GROW = 1.5 avant de relancer.
// =============================================================================

void FinalDevPlayer::iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth) {
    const uint64_t hash = zobrist::hash(yolah);
    Move    res   = Move::none();
    uint8_t depth = 0;
    int     value = 0;

    // Initialisation de la liste des coups à la racine, triée une première fois
    // comme un nœud interne (coup de la table, contre-coup, historique, territoire).
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

    // Motif de saut de profondeurs pour les threads secondaires (Lazy SMP) :
    // le thread 0 (principal) ne saute jamais aucune profondeur.
    static constexpr int SKIP_SIZE[]  = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    static constexpr int SKIP_PHASE[] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
    const int k = s.id % 20;

    for (int d = 1; d <= max_depth && d < 64; d++) {
        if (s.id > 0 && ((d + SKIP_PHASE[k]) / SKIP_SIZE[k]) % 2 == 1 && d < max_depth) {
            continue;
        }

        // Calcul de la demi-largeur initiale δ de la fenêtre d'aspiration
        int delta = ASPIRATION_MIN_DELTA;
        if (s.w->asp_n > 0) {
            delta = std::max(delta, int(ASPIRATION_K * s.w->asp_sum / s.w->asp_n));
        }

        int alpha = -INFINITE, beta = INFINITE;
        // Remise à zéro du compteur de nœuds par coup racine au début de la profondeur `d`
        // (`root_fix` : les nœuds seront cumulés sur toutes les tentatives de la profondeur `d`).
        for (RootMove& rm : s.root_moves) rm.nodes = 0;

        if (d >= 4 && std::abs(value) < WIN) {
            alpha = std::max(value - delta, -INFINITE);
            beta  = std::min(value + delta,  INFINITE);
        }

        // Boucle de résolution de la fenêtre d'aspiration à la profondeur `d`
        for (;;) {
            Move m;
            const int v = root_search(yolah, s, hash, alpha, beta, d, m);

            if (stopped()) {
                // Si l'horloge a interrompu la recherche, mais qu'un nouveau meilleur
                // coup `m` a eu le temps d'être complètement exploré et de battre alpha,
                // on le conserve précieusement !
                if (m != Move::none()) res = m;
                goto done;
            }

            if (v <= alpha && alpha > -INFINITE) {
                // Échec bas (fail-low) : tous les coups ont échoué sous alpha.
                beta  = (alpha + beta) / 2;
                alpha = std::max(v - delta, -INFINITE);
                delta = int(delta * ASPIRATION_GROW);
                continue;
            }
            if (v >= beta && beta < INFINITE) {
                // Échec haut (fail-high) : `m` a dépassé beta.
                if (m != Move::none()) res = m;
                beta  = std::min(v + delta, INFINITE);
                delta = int(delta * ASPIRATION_GROW);
                continue;
            }

            // La valeur `v` est tombée à l'intérieur de (alpha, beta) : valeur exacte !
            res = m;
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
    s.depth = depth;
    s.value = int16_t(value);
    s.move  = res;
}


// =============================================================================
//  3. RECHERCHE À LA RACINE (`root_search`) & TRI DES COUPS RACINE (C, A2)
// =============================================================================
//
//  Particularités de la racine par rapport à un nœud interne :
//    • On ne coupe jamais par la table de transposition ni par élagage avant-coup :
//      on veut un coup légal à jouer et sa valeur.
//    • Seul un coup dont l'exploration s'est terminée AVANT le déclenchement de
//      `stopped()` peut devenir `res`.
//    • Après la recherche, on réordonne `s.root_moves` pour la tentative ou la
//      profondeur suivante (`root_fix = true`) :
//        - En cas de fail-low (`res == Move::none()`), aucun coup n'a battu alpha :
//          on ne touche PAS à l'ordre (l'ancien meilleur coup reste en tête).
//        - En cas de fail-high (`best >= beta`), les coups après la coupure n'ont
//          pas été cherchés : on fait simplement passer le nouveau meilleur coup
//          `res` en première position (`std::rotate`).
//        - En cas de recherche complète (`alpha < best < beta`), `res` passe en
//          tête et tous les autres coups sont triés par `rm.nodes` décroissant :
//          un coup secondaire qui a demandé beaucoup de nœuds pour être réfuté
//          est le principal challenger à tester tôt à la profondeur d + 1.
// =============================================================================

int FinalDevPlayer::root_search(Yolah& yolah, Search& s, uint64_t hash,
                                int alpha, int beta, int depth, Move& res) {
    res = Move::none();
    const int alpha_orig = alpha;
    int best = -INFINITE;
    const uint8_t player = yolah.current_player();

    for (size_t i = 0; i < s.root_moves.size(); i++) {
        RootMove& rm = s.root_moves[i];
        const uint64_t nodes_before = s.nb_nodes;

        // La racine est un nœud PV (`pv_node = true`), sans coup spécial (`special = false`) ;
        // la réduction LMR dépend du rang `i` et de l'historique principal du coup.
        const int r = late_move_reduction_of(
            depth, i, /*pv_node=*/true, /*special=*/false,
            s.w->history[player][rm.move.from_sq()][rm.move.to_sq()]
        );

        const int v = search_move(yolah, s, hash, rm.move, i, r, alpha, beta, depth);
        rm.value  = v;
        rm.nodes += s.nb_nodes - nodes_before;   // cumulé sur les tentatives de la profondeur

        if (stopped()) return best;

        if (v > best) {
            best = v;
            if (v > alpha) {
                res   = rm.move;
                alpha = v;
                if (v >= beta) {
                    break;   // fail-high de la fenêtre d'aspiration
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
        std::stable_sort(s.root_moves.begin(), s.root_moves.end(),
                         [&](const RootMove& a, const RootMove& b) {
                             if ((a.move == res) != (b.move == res)) return a.move == res;
                             return a.nodes > b.nodes;
                         });
    }

    tt_store(hash, yolah, depth, alpha_orig, beta, best, res, /*pv=*/true);
    return best;
}


// =============================================================================
//  4. FORMULE LMR & RECHERCHE D'UN COUP (`search_move` : LMR + PVS)
// =============================================================================

// Calcule la réduction de profondeur de base (en plis) du i-ème coup (i ≥ 0) :
//   • Pas de réduction à depth < 3 ni sur le tout premier coup (i == 0).
//   • Formule de base (en 1/1024 de pli) :
//         r = 1.25 + ln(depth) × ln(i + 1) / 1.5
//   • Ajustements :
//         − 1 pli  si c'est un nœud PV (on cherche la valeur exacte)
//         − 1 pli  si c'est le contre-coup (`special == true`)
//         − hist / 8192 plis (entre −2 et +2 plis selon l'historique principal)
//   • Borné dans [0, depth − 2] pour que la recherche réduite garde au moins 1 pli.
int FinalDevPlayer::late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const {
    if (depth < 3 || i == 0) return 0;
    int r = lmr_table[std::min(depth, 63)][std::min<size_t>(i + 1, Yolah::MAX_NB_MOVES - 1)];
    if (pv_node) r -= 1024;
    if (special) r -= 1024;
    r -= hist * 1024 / 8192;
    return std::clamp(r / 1024, 0, depth - 2);
}

// Exécute le coup `m` (rang `i` dans le nœud) et évalue le sous-arbre enfant
// selon le schéma PVS + LMR en 3 étapes :
//
//   Étape 1 (si `reduction > 0`) :
//     Recherche à profondeur réduite `depth − 1 − reduction` avec fenêtre nulle
//     `(-(alpha + 1), -alpha)`. Si `v <= alpha`, le coup est réputé mauvais et
//     on s'arrête là (économie massive sur les coups tardifs).
//
//   Étape 2 (si `i > 0` et que l'étape 1 a battu `alpha`, ou si `reduction == 0`) :
//     Recherche à pleine profondeur `depth − 1` mais toujours avec fenêtre nulle
//     `(-(alpha + 1), -alpha)` pour vérifier à moindre coût si le coup bat
//     vraiment `alpha`.
//
//   Étape 3 (si `i == 0`, ou si l'étape 2 a donné `alpha < v < beta` sur un nœud PV) :
//     Recherche à pleine profondeur `depth − 1` et fenêtre pleine `(-beta, -alpha)`
//     pour obtenir la valeur exacte.
int FinalDevPlayer::search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i,
                                int reduction, int alpha, int beta, int depth) {
    const uint8_t  player = yolah.current_player();
    const uint64_t child  = zobrist::update(hash, player, m);
    const uint16_t ply    = yolah.nb_plies();

    // Préchargement matériel du cluster de la table de transposition de l'enfant
    yolah_table->prefetch(child);

    // Enregistrement du coup joué à ce pli et invalidation de l'accumulateur enfant
    // (il ne sera calculé que si une feuille sous cet enfant est évaluée).
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

    // Composantes connexes (8-connexité) de cases libres (`free`) accessibles
    // depuis un ensemble de pièces (`pieces`).
    // Comme une dame se déplace aussi en diagonale, deux cases libres qui se
    // touchent par un coin appartiennent à la même région accessible.
    uint64_t region_reach(uint64_t pieces, uint64_t free) {
        auto grow = [](uint64_t b) {
            constexpr uint64_t NOT_A = 0xfefefefefefefefeULL;
            constexpr uint64_t NOT_H = 0x7f7f7f7f7f7f7f7fULL;
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
    // Calcule par recherche exhaustive mémorisée (DFS + table de hachage) le
    // nombre MAXIMUM de coups que les pièces `pieces` peuvent jouer à l'intérieur
    // de la région de cases libres `region` (sans jamais en sortir).
    // Renvoie −1 si le budget de nœuds `budget` est dépassé.
    int harvest_exact_rec(uint64_t pieces, uint64_t region,
                          std::unordered_map<uint64_t, int>& memo, int64_t& budget) {
        if (--budget < 0) return -1;
        const uint64_t key = (pieces * 0x9E3779B97F4A7C15ULL)
                           ^ (region + 0x632BE59BD9B4E019ULL)
                           ^ (region >> 17);
        if (auto it = memo.find(key); it != memo.end()) return it->second;

        int best = 0;
        const int max_possible = std::popcount(region);
        for (uint64_t pb = pieces; pb;) {
            const Square from = pop_lsb(pb);
            for (uint64_t t = attacks_bb(from, ~region) & region; t;) {
                const Square to = pop_lsb(t);
                if (best >= max_possible) break;   // impossible de faire mieux que de tout remplir
                const int v = harvest_exact_rec(
                    pieces ^ square_bb(from) ^ square_bb(to),
                    region & ~square_bb(to),
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
    // noires et des 4 pièces blanches. Deux positions ayant les mêmes pièces au
    // même endroit (mais éventuellement quelques trous différents) partagent le
    // même terme correctif `corr[joueur][corr_key]`.
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
//  Recherche alpha-bêta en convention Negamax fail-soft :
//    • Si la valeur retournée `v` vérifie `alpha < v < beta`, c'est la valeur exacte.
//    • Si `v <= alpha`, tous les coups ont échoué bas : `v` est une borne supérieure.
//    • Si `v >= beta`, un coup a coupé : `v` est une borne inférieure.
// =============================================================================

int FinalDevPlayer::negamax(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth) {
    ++s.nb_nodes;
    if (stopped()) return 0;

    const uint16_t ply_now = yolah.nb_plies();
    s.eval_stack[ply_now] = EVAL_NONE;

    // ── 6.a) Fin de partie exacte ────────────────────────────────────────────
    // Lorsque les deux joueurs n'ont plus aucun coup légal, le score exact est
    // décalé au-delà de ±WIN pour qu'une victoire certaine de +1 point (+30001)
    // batte toujours n'importe quelle évaluation heuristique du réseau (< 30000).
    if (yolah.game_over()) {
        const int score = yolah.score(yolah.current_player());
        if (score == 0) return 0;
        return score + (score > 0 ? WIN : -WIN);
    }

    // ── 6.b) Consultation de la table de transposition (`SearchTable`) ───────
    // Comme chaque coup de Yolah détruit définitivement une case, il n'y a
    // jamais de répétition de position ni de dépendance au chemin : une entrée
    // de profondeur `tt.depth >= depth` est toujours fiable.
    //
    // Règle aux nœuds PV (`tt_cut_pv = false, tt_cut_pv_exact = true`) :
    //   • Aux nœuds non-PV (`beta - alpha == 1`), toute borne suffisante coupe.
    //   • Aux nœuds PV (`beta - alpha > 1`), on n'accepte de couper que si l'entrée
    //     est EXACTE (`tt.lower == tt.upper`) ou prouve une fin de partie (`> WIN`).
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
    // lui ajoute la correction apprise `corr[joueur][corr_key(yolah)]` (`corr_hist = 2`).
    // Important (Fix A) : on ne stocke JAMAIS les feuilles (depth 0) dans la
    // table de transposition, sans quoi elles évinceraient des nœuds internes.
    if (depth <= 0) {
        int leaf = evaluate(yolah, s, hash);
        leaf = std::clamp(leaf + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        return leaf;
    }

    // ── 6.d) Évaluation statique corrigée & Élagage avant-coup (non-PV) ──────
    // Tous les élagages avant-coup (RFP, MPC, Proxy Cut) ne s'appliquent qu'aux
    // nœuds non-PV (`beta == alpha + 1`) et hors scores de fin de partie (`|beta| < WIN`).
    const bool pv_node  = beta - alpha > 1;
    const bool prunable = !pv_node && std::abs(beta) < WIN;

    int static_eval = 0;
    if (prunable) {
        static_eval = evaluate(yolah, s, hash);
        static_eval = std::clamp(static_eval + s.w->corr[yolah.current_player()][corr_key(yolah)], -WIN + 1, WIN - 1);
        s.eval_stack[ply_now] = static_eval;
    }

    // ── F1. Reverse Futility Pruning (RFP, +76 Elo) ──────────────────────────
    // Aux faibles profondeurs (depth ≤ 4), si l'évaluation statique dépasse
    // déjà `beta` d'une marge de sécurité de `3000 × depth`, on coupe sans
    // générer les coups.
    if (prunable && depth <= RFP_DEPTH && static_eval - RFP_MARGIN * depth >= beta) {
        return static_eval;
    }

    // ── O. Multi-ProbCut bidirectionnel (MPC, +57 Elo à 1 s) ─────────────────
    // À `depth >= 5`, on effectue une recherche préliminaire de la position
    // courante à une profondeur réduite `d2 = max(1, 0.40 × depth)` :
    //   • Coupe haute : si `static_eval >= beta` et que la recherche à profondeur `d2`
    //     dépasse `beta + 4000 + 500 × (depth − d2)`, on coupe à `beta`.
    //   • Coupe basse : si `static_eval <= alpha` et que la recherche à profondeur `d2`
    //     reste sous `alpha − (4000 + 500 × (depth − d2))`, on échoue bas à `alpha`.
    if (prunable && depth >= MPC_DEPTH) {
        const int d2     = std::max(1, depth * MPC_RATIO / 100);
        const int margin = MPC_MARGIN + MPC_MARGIN_PER_PLY * (depth - d2);

        if (static_eval >= beta && beta + margin < WIN) {
            const int bound = beta + margin;
            const int v = negamax(yolah, s, hash, bound - 1, bound, d2);
            if (stopped()) return 0;
            if (v >= bound) return beta;
        }
        if (static_eval <= alpha && alpha - margin > -WIN) {
            const int bound = alpha - margin;
            const int v = negamax(yolah, s, hash, bound, bound + 1, d2);
            if (stopped()) return 0;
            if (v <= bound) return alpha;
        }
    }

    // ── 6.e) Préparation de la génération par étapes (`staged = true`, K) ────
    // Si la table de transposition propose un coup légal `tt_move`, on tente
    // l'étape 1 (`tt_first = true`) sans générer la liste des coups.
    // Toutefois, si le Proxy Cut doit s'exécuter ci-dessous, il aura besoin de
    // générer et trier les coups pour trouver son témoin ordinaire.
    Yolah::MoveList moves;
    const bool tt_first = tt_move != Move::none() && is_legal(yolah, tt_move);
    bool generated = false;
    int  scores[Yolah::MAX_NB_MOVES];
    size_t n = 1;

    if (!tt_first) {
        yolah.moves(moves);
        generated = true;
        score_moves(yolah, s, tt_move, moves, scores, depth);
        n = moves.size();
    }

    // ── 6.f) Proxy Cut (N, +47 Elo à 0.2 s, +59 Elo à 1 s) ──────────────────
    // Pourquoi le Proxy Cut au lieu du Null Move Pruning ?
    //   À Yolah, passer son tour (null move) est un avantage dans toutes les
    //   fins de zone (zugzwang : bouger détruit une case de son propre territoire).
    //   Le null move croit donc à tort que la position est gagnante et coupe à
    //   tort (−81 Elo).
    //   À l'inverse, tester un VRAI coup légal `m` est toujours sain vis-à-vis
    //   du zugzwang :
    //       valeur(meilleur_coup) ≥ valeur(m)
    //   Donc si même un coup ordinaire `m` (ni le coup de table, ni le contre-coup),
    //   cherché à profondeur réduite `d = depth − 1 − (2 + depth / 4)`, atteint
    //   `beta`, le meilleur coup du nœud atteindra `beta` a fortiori !
    if (prunable && depth >= PROXY_DEPTH && static_eval >= beta) {
        if (!generated) {
            yolah.moves(moves);
            generated = true;
            score_moves(yolah, s, tt_move, moves, scores, depth);
            n = moves.size();
        }
        // Le témoin est le coup ordinaire (`scores[j] < SCORE_COUNTER`) ayant
        // le meilleur score de tri.
        int best_ordinary = -1;
        for (size_t j = 0; j < n; j++) {
            if (scores[j] < SCORE_COUNTER && (best_ordinary < 0 || scores[j] > scores[best_ordinary])) {
                best_ordinary = int(j);
            }
        }
        if (best_ordinary >= 0) {
            const Move wit = moves[best_ordinary];
            const int  R   = PROXY_REDUCTION + depth / 4 - 1;
            const int  d   = depth - 1 - R;
            const uint8_t  player = yolah.current_player();
            const uint64_t child  = zobrist::update(hash, player, wit);

            s.played[ply_now]     = wit;
            s.acc_ok[ply_now + 1] = false;
            yolah.play(wit);
            const int v = -negamax(yolah, s, child, -beta, -beta + 1, d);
            yolah.undo(wit);

            if (stopped()) return 0;
            if (v >= beta) return v >= WIN ? beta : v;   // pas de victoire non prouvée issue d'une réduction
        }
    }

    // ── Mise à jour de l'historique de correction (`corr_update`, U4/U6) ─────
    const int alpha_orig = alpha;
    int  best      = -INFINITE;
    Move best_move = Move::none();

    auto corr_update = [&](int bestv) {
        if (s.eval_stack[ply_now] == EVAL_NONE || std::abs(bestv) >= WIN || stopped()) return;
        // Une borne inférieure (coupure bêta) située sous static_eval ou une borne
        // supérieure (fail-low) située au-dessus de static_eval ne contredit pas static_eval.
        if (bestv >= beta       && bestv <= static_eval) return;
        if (bestv <= alpha_orig && bestv >= static_eval) return;

        const int b = std::clamp((bestv - static_eval) * depth / CORR_SCALE, -2000, 2000);
        int16_t& e  = s.w->corr[yolah.current_player()][corr_key(yolah)];
        e = int16_t(std::clamp(e + b - e * std::abs(b) / CORR_MAX, -CORR_MAX, CORR_MAX));
    };

    // ── U5 (1) : Détection paresseuse des poches scellées (`reserve = 1`) ────
    // Une « poche scellée » est une composante connexe de cases libres (`comp`,
    // de taille ≤ 20) que NOUS sommes seuls à pouvoir atteindre (`priv`), et
    // telle que les pièces `pieces` qui la touchent n'ont aucun accès au reste
    // du plateau (`sealed == true`).
    //
    // Propriété mathématique exacte :
    //   Ces pièces et cette poche sont totalement isolées du reste du jeu.
    //   Quel que soit le déroulement ailleurs, on pourra toujours y récolter
    //   exactement `h = harvest_exact(pieces, comp)` coups. Il est donc inutile
    //   d'essayer tous les coups de ces pièces dans ce nœud : on trouve UN SEUL
    //   coup `keep` qui laisse une récolte de `h − 1` coups, et on ignore tous
    //   les autres coups de `pieces` sans rien perdre d'exact !
    uint64_t reserve_pieces[4]{};
    Move     reserve_keep[4]{};
    int      n_reserves   = 0;
    bool     reserve_done = false;

    auto compute_reserves = [&]() {
        const uint8_t  us       = yolah.current_player();
        const uint64_t free     = yolah.free_squares();
        const uint64_t mine     = yolah.bitboard(us);
        const uint64_t occupied = yolah.occupied_squares();
        uint64_t priv = region_reach(mine, free) & ~region_reach(yolah.bitboard(Yolah::other_player(us)), free);

        while (priv && n_reserves < 4) {
            const uint64_t seed = priv & -priv;
            const uint64_t comp = region_reach(seed, priv) | seed;
            priv &= ~comp;
            if (std::popcount(comp) > RESERVE_MAX_SQUARES) continue;

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
            if (h <= 0) continue;

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
            if (keep == Move::none()) continue;
            reserve_pieces[n_reserves] = pieces;
            reserve_keep[n_reserves++] = keep;
        }
    };

    // ── 6.g) Boucle principale sur les coups du nœud ─────────────────────────
    for (size_t i = 0; i < n || !generated; i++) {
        // Étape 2 de la génération par étapes : le coup de la table (`i == 0`)
        // n'a pas suffi à couper. On génère maintenant tous les coups, on place
        // `tt_move` en `moves[0]` (déjà exploré) et on note les suivants.
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
            if (i >= n) break;
        }

        // F3. Late Move Pruning (LMP, +35 Elo) :
        // Aux nœuds non-PV peu profonds (depth ≤ 3), dès qu'au moins un coup
        // non perdant (`best > -WIN`) a été trouvé et qu'on a déjà testé
        // `4 + depth²` coups (soit 5, 8 ou 13 coups aux profondeurs 1, 2, 3),
        // on élague le reste de la liste.
        if (prunable && depth <= LMP_DEPTH && i >= size_t(LMP_MOVES + depth * depth) && best > -WIN) {
            break;
        }

        // Sélection paresseuse du meilleur coup restant en position `i`
        // (évite un tri complet O(n log n) alors que la majorité des nœuds
        // coupent dès le 1er ou le 2e coup).
        const Move m = generated ? pick_move(moves, scores, i, n) : tt_move;

        // U5 (1) : Filtrage exact des poches scellées (`reserve = 1`)
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
        int r = !generated ? 0
              : late_move_reduction_of(
                    depth, i,
                    /*pv_node=*/pv_node,
                    /*special=*/scores[i] >= SCORE_COUNTER,
                    s.w->history[player][m.from_sq()][m.to_sq()]
                );

        // Réductions LMR contextuelles (U7 : `intercept_lmr` et `lmrh`) :
        // Si le coup est réduit (`r > 0`) et répond au dernier coup adverse `q1` :
        //   1. `intercept_lmr` : si `m` atterrit sur une ligne de tir ou sur une
        //      case adjacente (3×3) à la pièce adverse qui vient de jouer (`q1.to_sq()`),
        //      on réduit d'un pli de moins (c'est un coup tactique d'obstruction directe).
        //   2. `lmrh` : si la paire `(q1.to_sq(), m.to_sq())` a souvent mis en défaut
        //      la réduction LMR par le passé (`lmrh >= 4000`), on réduit aussi d'un pli de moins.
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

        const int r_used = std::min(r, std::max(0, depth - 2));
        const int v = search_move(yolah, s, hash, m, i, r_used, alpha, beta, depth);

        if (stopped()) return 0;

        // U7 (C5) : Apprentissage des réductions erronées (`lmrh`) :
        // Si le coup a été réduit (`r_used > 0`) mais qu'il a finalement battu
        // `alpha` après re-recherche à pleine profondeur, la réduction était
        // trop optimiste : on augmente `lmrh[player][prev.to][m.to]`.
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
                    // 1. Enregistrement dans la table de transposition et dans `corr`
                    tt_store(hash, yolah, depth, alpha_orig, beta, v, m, /*pv=*/false);
                    corr_update(v);

                    // 2. Mise à jour des deux coups tueurs (`killer1`, `killer2`) du pli
                    if (s.killer1[ply_now] != m) {
                        s.killer2[ply_now] = s.killer1[ply_now];
                        s.killer1[ply_now] = m;
                    }

                    // 3. Mise à jour par « gravité » de l'historique principal (`history`)
                    //    et de l'historique de continuation à 1 pli (`cont1`) :
                    //    • Le coup `m` qui a provoqué la coupure bêta reçoit `+bonus`.
                    //    • Tous les coups `moves[0..i-1]` essayés avant lui sans couper
                    //      reçoivent un malus `-bonus`.
                    const int bonus = std::min(16 * depth * depth + 32 * depth, 2000);
                    update_history(s, player, m, bonus);
                    for (size_t k = 0; k < i; k++) {
                        update_history(s, player, moves[k], -bonus);
                    }

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

                        // 4. Enregistrement du contre-coup (`countermoves`)
                        s.w->countermoves[p1.from_sq()][p1.to_sq()] = m;
                    }

                    return v;
                }
                alpha = v;
            }
        }
    }

    // Fin du nœud sans coupure bêta : valeur exacte (si `best > alpha_orig`)
    // ou borne supérieure fail-soft (si tous les coups ont échoué sous `alpha_orig`).
    tt_store(hash, yolah, depth, alpha_orig, beta, best, best_move, /*pv=*/pv_node);
    corr_update(best);
    return best;
}


// =============================================================================
//  7. ORDONNANCEMENT DES COUPS (`score_moves`) & TERRITOIRE (`territory_after`)
// =============================================================================
//
//  À Yolah, il n'y a ni prises ni promotions : tous les coups sont « calmes ».
//  La qualité de l'ordonnancement repose donc sur la combinaison de :
//    1. La mémoire exacte du nœud : `tt_move` (`SCORE_TT`).
//    2. La réfutation directe du dernier coup adverse : `countermove` (`SCORE_COUNTER`).
//    3. Le score composite des coups ordinaires :
//         • `history[player][m.from][m.to]` : succès global du coup dans la partie
//         • `cont1[player][p1.to][m.to]`    : succès de la case d'arrivée `m.to`
//                                             en réponse à la dernière case d'arrivée
//                                             adverse `p1.to` (+32.6 Elo en U4)
//         • `INTERCEPT_BONUS` (+2000)       : si `m.to` coupe une ligne de tir ou
//                                             touche (3×3) `p1.to` (+17.0 Elo en U7)
//         • `KILLER_BONUS` (+4000)          : si `m` est `killer1` ou `killer2` du pli
//                                             (`killer_mode = 1`, +17.7 Elo en U7)
//         • `512 × territory_after(m)`      : aux nœuds profonds (`depth >= 4`),
//                                             avantage territorial Voronoï à pas de roi
//                                             laissé par le coup (+36 Elo en J).
// =============================================================================

void FinalDevPlayer::score_moves(const Yolah& yolah, const Search& s, Move tt_move,
                                 const Yolah::MoveList& moves, int* scores, int depth) const {
    const bool     territory = depth >= TERRITORY_DEPTH;
    const uint16_t ply       = yolah.nb_plies();
    const uint8_t  player    = yolah.current_player();
    const Move     p1        = ply > 0 ? s.played[ply - 1] : Move::none();
    const Move     counter   = p1 != Move::none() ? s.w->countermoves[p1.from_sq()][p1.to_sq()] : Move::none();

    // Score d'historique composite (historique principal + continuation 1 pli + interception)
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
        } else if (m == counter && counter != Move::none()) {
            scores[i] = SCORE_COUNTER;
        } else if (kil) {
            // `killer_mode = 1` : les killers restent dans la catégorie des coups
            // ordinaires (< SCORE_ARTICULATION) avec un bonus de +4000 sur `hist(m)`.
            scores[i] = std::min(hist(m) + KILLER_BONUS, SCORE_ARTICULATION - 1);
        } else {
            scores[i] = hist(m);
        }

        // Aux nœuds de profondeur ≥ 4, ajout du terme territorial Voronoï (`512 × influence_fast`)
        if (scores[i] < SCORE_ARTICULATION && territory) {
            scores[i] += TERRITORY_WEIGHT * territory_after(yolah, m);
        }
    }
}

// ── Calcul rapide de l'influence Voronoï à pas de roi (`influence_fast`) ─────
// Les deux camps inondent simultanément les cases libres d'un pas de roi à
// chaque étape :
//   • `mf`, `of` : fronts d'onde (uniquement les NOUVELLES cases atteintes à
//     l'étape courante, pour que chaque case ne soit visitée qu'une fois).
//   • Les cases atteintes au même pas par les deux camps (`collision = nm & no`)
//     deviennent neutres et propagent cette neutralité autour d'elles.
//   • Renvoie `(nos_cases_libres_contrôlées − leurs_cases_libres_contrôlées)`.
static inline uint64_t expand_king(uint64_t b) {
    const uint64_t h = b | ((b << 1) & 0xFEFEFEFEFEFEFEFEULL) | ((b >> 1) & 0x7F7F7F7F7F7F7F7FULL);
    return h | (h << 8) | (h >> 8);
}

static int influence_fast(uint64_t me, uint64_t opp, uint64_t free) {
    uint64_t mi = me, oi = opp, mf = me, of = opp, neutral = 0;
    while (mf | of) {
        uint64_t nm = expand_king(mf) & free & ~oi & ~mi;
        uint64_t no = expand_king(of) & free & ~mi & ~oi;
        const uint64_t collision = nm & no;
        if (neutral | collision) {
            neutral |= collision | (expand_king(neutral) & free);
            nm &= ~neutral;
            no &= ~neutral;
        }
        mi |= nm;
        oi |= no;
        mf = nm;
        of = no;
    }
    return std::popcount(mi & free) - std::popcount(oi & free);
}

int FinalDevPlayer::territory_after(const Yolah& yolah, Move m) const {
    if (m == Move::none()) return 0;
    const uint8_t  player = yolah.current_player();
    const uint64_t from   = uint64_t(1) << m.from_sq();
    const uint64_t to     = uint64_t(1) << m.to_sq();
    const uint64_t me     = (yolah.bitboard(player) & ~from) | to;
    const uint64_t opp    = yolah.bitboard(Yolah::other_player(player));
    const uint64_t free   = yolah.free_squares() & ~to;   // la case de départ était déjà occupée, elle devient un trou
    return influence_fast(me, opp, free);
}

bool FinalDevPlayer::is_legal(const Yolah& yolah, Move m) {
    if (m == Move::none()) return false;
    const uint64_t from = uint64_t(1) << m.from_sq();
    const uint64_t to   = uint64_t(1) << m.to_sq();
    if (!(yolah.bitboard(yolah.current_player()) & from)) return false;
    return (attacks_bb(m.from_sq(), yolah.occupied_squares()) & yolah.free_squares() & to) != 0;
}

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
// borné dans [-16384, +16384] sans saturation brutale.
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
//          bits 16..1  : la valeur `int16_t` du réseau
//          bit  0      : 1 si l'emplacement est occupé
//    • Lecture/écriture atomique `relaxed` : même en multithread, une entrée ne
//      peut jamais être « déchirée » (clé d'une position avec valeur d'une autre).
//
//  Fonctionnement des accumulateurs paresseux (`lazy_accumulator`) :
//    • Lorsqu'on descend dans l'arbre (`search_move`, `proxy_cut`), on se contente
//      de noter `s.played[ply] = m` et `s.acc_ok[ply + 1] = false`.
//    • Si le sous-arbre coupe par la table de transposition, par RFP ou par
//      fin de partie, aucune opération NNUE n'aura été payée !
//    • Si `raw_network_value` est appelée au pli `ply`, on remonte au dernier
//      ancêtre `k <= ply` dont `s.acc_ok[k]` est vrai (au pire la racine), puis
//      on redescend de `k` à `ply` en appelant `persp->update(...)` une fois par pli.
// =============================================================================

int FinalDevPlayer::evaluate(const Yolah& yolah, Search& s, uint64_t hash) {
    s.nb_evals++;
    constexpr uint64_t KEY_MASK = ~uint64_t(0) << 17;
    const uint64_t key = hash & KEY_MASK;

    std::atomic<uint64_t>& slot = eval_cache[hash & (eval_cache.size() - 1)];
    const uint64_t word = slot.load(std::memory_order_relaxed);
    if ((word & 1) && (word & KEY_MASK) == key) {
        s.nb_eval_hits++;
        return int16_t(uint16_t(word >> 1));
    }

    const int v = raw_network_value(yolah, s);
    slot.store(key | (uint64_t(uint16_t(int16_t(v))) << 1) | 1, std::memory_order_relaxed);
    return v;
}

int FinalDevPlayer::raw_network_value(const Yolah& yolah, Search& s) {
    const int ply = yolah.nb_plies();
    int k = ply;
    while (!s.acc_ok[k]) k--;   // remonte jusqu'au dernier accumulateur déjà calculé

    for (; k < ply; k++) {
        persp->update(s.accs[k].acc, s.accs[k + 1].acc, uint8_t(k & 1), s.played[k], &s.refresh);
        s.acc_ok[k + 1] = true;
    }
    return int(persp->value(s.accs[ply].acc, yolah) * WIN);
}


// =============================================================================
//  9. ACCÈS À LA TABLE DE TRANSPOSITION (`SearchTable`) & VARIATION PRINCIPALE
// =============================================================================

FinalDevPlayer::TTView FinalDevPlayer::tt_probe(uint64_t hash) const {
    TTView view;
    if (const SearchTable::Entry* e = yolah_table->probe(hash)) {
        view = {true, e->move, e->depth, e->lower, e->upper};
    }
    return view;
}

void FinalDevPlayer::tt_store(uint64_t hash, const Yolah& yolah, int depth,
                              int alpha, int beta, int value, Move move, bool pv) {
    yolah_table->store(hash, std::popcount(yolah.free_squares()), depth, alpha, beta, value, move, pv, /*fragile=*/false);
}

Move FinalDevPlayer::tt_move(uint64_t hash) const {
    return yolah_table->get_move(hash);
}

void FinalDevPlayer::print_pv(Yolah yolah, uint64_t hash, int8_t depth) {
    if (yolah.game_over() || depth <= 0) return;
    const Move m = tt_move(hash);
    if (m == Move::none() || !is_legal(yolah, m)) return;
    const uint8_t player = yolah.current_player();
    cout << m << ' ';
    yolah.play(m);
    print_pv(yolah, zobrist::update(hash, player, m), depth - 1);
}

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
    return "final dev player (dev_u8, " + std::to_string(nb_threads) + " threads)";
}

json FinalDevPlayer::config() {
    json j;
    j["name"]         = "FinalDevPlayer (dev_u8)";
    j["microseconds"] = thinking_time;
    j["tt size"]      = tt_size_mb;
    j["nb threads"]   = nb_threads;
    j["weights"]      = weights_filename;
    return j;
}
