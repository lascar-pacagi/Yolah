#ifndef FINAL_DEV_H
#define FINAL_DEV_H

// =============================================================================
//  FINAL_DEV (dev_u9_spsa) — Version épurée et commentée de MinMaxNNUE_DevPlayer
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (et son compagnon doc/final_dev.cpp) contient le code exact du
//  moteur « Minimax NNUE dev » dans sa configuration `dev_u9_spsa`, celle de
//  l'application Yolah (adversaire « Minimax NNUE dev » et moteur de la barre
//  d'analyse). Le vrai joueur, player/minmax_nnue_dev_player.{h,cpp}, contient
//  en plus une centaine d'options expérimentales testées puis écartées au fil
//  des campagnes A à U (plus de 250 variantes, des centaines de milliers de
//  parties sur le cluster) ; elles sont toutes retirées ici.
//
//  « Code exact » veut dire : compilé à côté du vrai joueur réglé par
//  config/search_study/dev_u9_spsa.cfg (un seul thread, tables vidées avant
//  chaque position), ce code donne la même valeur, le même coup et le même
//  nombre de nœuds sur les 209 positions de test/bench_positions.txt à la
//  profondeur 11 (4 037 052 nœuds de part et d'autre). Toute modification du
//  code doit être revérifiée de cette façon (voir la fin de ce commentaire).
//
//  Le document doc/minmax_nnue_dev_player.html explique les mêmes mécanismes
//  avec des schémas et des widgets interactifs ; ce fichier-ci se lit seul.
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF (ce qu'il faut savoir pour lire le moteur)
//  ---------------------------------------------------------------------------
//    • Plateau 8 × 8. Chaque camp (Noir, qui commence, et Blanc) a 4 pièces.
//    • Un coup déplace une de ses pièces comme une dame d'échecs (ligne, colonne
//      ou diagonale, aussi loin qu'on veut) vers une case LIBRE, sans sauter
//      par-dessus une pièce ni un trou.
//    • La case de départ devient un TROU pour toujours : plus personne ne peut
//      s'y poser ni la traverser. Le nombre de cases libres ne fait donc que
//      baisser : une position ne se répète jamais (pas de cycles), ce que le
//      moteur exploite (table de transposition, section 2 ci-dessous).
//    • Chaque coup joué rapporte 1 point à son camp. Un camp qui n'a plus de
//      coup PASSE (le coup « passe » est `Move::none()`, c'est-à-dire la valeur
//      0 = « a1 → a1 »), et la partie finit quand aucun des deux ne peut jouer.
//      Gagne celui qui a joué le plus de coups. On gagne donc en gardant de
//      l'espace pour ses pièces et en enfermant celles de l'adversaire.
//    • Pas de prises ni de promotions : tous les coups sont « calmes », rien
//      ne signale statiquement un bon coup. L'ordre des coups doit être appris
//      par la recherche elle-même (historiques, section 5).
//    • Conventions du code : une case est un entier 0..63 (a1 = 0, b1 = 1, …,
//      h8 = 63) ; un ensemble de cases est un `uint64_t` (bitboard, bit s =
//      case s) ; `ply` (pli) = nombre de demi-coups joués depuis le début de la
//      PARTIE (pas depuis la racine de la recherche) ; le camp au trait est
//      `ply & 1` (0 = Noir, 1 = Blanc).
//
//  ---------------------------------------------------------------------------
//  PETIT GLOSSAIRE DE LA RECHERCHE
//  ---------------------------------------------------------------------------
//    • Negamax : une seule fonction pour les deux camps ; la valeur d'une
//      position est toujours donnée du point de vue du camp au trait, et la
//      valeur d'un enfant se lit en changeant de signe (−negamax(enfant)).
//    • Fenêtre (alpha, beta) : la recherche ne s'intéresse qu'aux valeurs
//      strictement entre alpha et beta. Une valeur ≥ beta suffit pour savoir
//      que l'adversaire évitera cette position : on arrête (« coupure bêta »).
//    • Fail-soft : quand la recherche sort de sa fenêtre, elle renvoie quand
//      même la meilleure valeur trouvée (pas seulement alpha ou beta). Un
//      résultat v ≤ alpha est une BORNE SUPÉRIEURE de la vraie valeur (« fail
//      low », échec bas) ; v ≥ beta est une BORNE INFÉRIEURE (« fail high »,
//      échec haut) ; entre les deux, c'est la valeur exacte.
//    • Fenêtre nulle : beta = alpha + 1. Elle ne peut rien trouver « entre »
//      les deux : elle répond seulement à « est-ce ≥ beta ? ». C'est beaucoup
//      moins cher qu'une recherche à fenêtre large.
//    • Nœud PV (variation principale) : nœud cherché avec une fenêtre large
//      (beta − alpha > 1), sur le chemin des meilleurs coups ; on y veut la
//      valeur exacte, donc on n'y élague presque rien. Nœud non-PV : fenêtre
//      nulle ; c'est l'immense majorité des nœuds, et c'est là qu'on élague.
//    • Profondeur `depth` : nombre de plis qu'il reste à chercher sous un nœud
//      (0 = feuille, on évalue avec le réseau). Une RÉDUCTION cherche un coup
//      moins profond ; un ÉLAGAGE ne le cherche pas du tout.
//    • Elo : mesure de force relative. +10 Elo ≈ 51,4 % des points ; +100 ≈
//      64 %. « +17.9 Elo à 0.2 s » = résultat d'un match de 2000 parties (sauf
//      mention) à 0,2 s par coup contre la version précédente, avec un
//      intervalle de confiance à 95 % entre crochets quand il est donné.
//    • Les lettres et numéros entre parenthèses (B, A2, H, D, F1, J, K, L, N,
//      O, U4, U7, U19…) sont les noms des étapes et campagnes de mesure du
//      projet où chaque mécanisme a été introduit et mesuré ; leur liste est
//      dans doc/minmax_nnue_dev_player.html, sections 10 et 11.
//
//  ---------------------------------------------------------------------------
//  ÉCHELLE DES VALEURS
//  ---------------------------------------------------------------------------
//    • Le réseau donne une valeur dans [−1, 1] (−1 : perdu, +1 : gagné, pour le
//      camp au trait), multipliée par WIN = 30 000 : les évaluations vivent
//      dans [−29 999, +29 999].
//    • Une fin de partie EXACTE vaut ±(30 000 + écart de points) : une victoire
//      prouvée d'un seul point (30 001) bat donc toute évaluation du réseau.
//      |v| > WIN signifie « résultat prouvé ». 0 = nulle prouvée (ou valeur
//      sans signification quand l'horloge a interrompu la recherche).
//    • INFINITE = 32 000 sert de ±∞ pour les fenêtres.
//
//  ---------------------------------------------------------------------------
//  VUE D'ENSEMBLE DES BRIQUES ACTIVES (et gains Elo mesurés à leur introduction)
//  ---------------------------------------------------------------------------
//  Les gains ont chacun été mesurés contre le joueur de l'époque : ils ne
//  s'additionnent pas.
//
//  1. APPROFONDISSEMENT ITÉRATIF, PVS & FENÊTRES D'ASPIRATION ADAPTATIVES (B, A2)
//     • Approfondissement itératif : on cherche à profondeur 1, puis 2, puis 3…
//       jusqu'à ce que l'horloge arrête la recherche. Les recherches peu
//       profondes coûtent peu et remplissent la table de transposition de
//       meilleurs coups, qui ordonnent si bien la recherche suivante que le
//       total coûte moins qu'une seule recherche profonde.
//     • Recherche à variation principale (PVS) en fail-soft : le premier coup
//       d'un nœud est cherché à fenêtre pleine (alpha, beta), les suivants à
//       fenêtre nulle (alpha, alpha + 1), et re-cherchés à fenêtre pleine
//       seulement s'ils battent alpha (cpp, section 4).
//     • Fenêtres d'aspiration adaptatives à la racine (à partir de depth = 4) :
//       on cherche dans (v − δ, v + δ) autour de la valeur v de la profondeur
//       précédente, avec δ = max(345, 0.92 × moyenne glissante des |v_d − v_{d−1}|).
//       En cas d'échec bas/haut, la fenêtre est recentrée sur la valeur fail-soft
//       et δ est multiplié par 1.34 (−22 % de nœuds à profondeur 12, mesuré
//       avec 300, 1.25 et 1.5 ; valeurs actuelles : SPSA, point 8).
//     • Tri des coups à la racine (« root_fix ») : meilleur coup en tête, les
//       suivants triés par la taille (en nœuds) du sous-arbre qu'il a fallu
//       explorer pour les réfuter, cumulée sur les tentatives de la profondeur.
//
//  2. TABLE DE TRANSPOSITION DÉDIÉE À YOLAH (`SearchTable`, H)
//     • Elle mémorise, pour chaque position déjà cherchée (repérée par son
//       hachage Zobrist de 64 bits), le meilleur coup trouvé et ce que la
//       recherche a prouvé sur sa valeur, pour ne pas refaire le travail quand
//       la même position revient par un autre ordre de coups.
//     • Chaque entrée stocke simultanément les DEUX bornes [lower, upper] de la
//       valeur (au lieu d'écraser une borne inférieure par une borne supérieure,
//       ce que faisait l'ancienne table à une seule borne).
//     • Remplacement : comme une case libre ne revient jamais, une position qui
//       a PLUS de cases libres que la racine actuelle ne peut plus être
//       atteinte ; son entrée est évincée en priorité.
//     • Aux nœuds PV, la table ne coupe que sur une valeur exacte (lower == upper)
//       ou une fin de partie prouvée (|borne| > WIN), jamais sur une simple borne
//       issue d'un nœud non-PV où l'élagage agressif s'est appliqué.
//     • Préchargement matériel (`__builtin_prefetch`) de l'entrée de l'enfant
//       dès qu'un coup est choisi, avant même de jouer le coup : la ligne de
//       cache arrive pendant qu'on fait le reste du travail.
//
//  3. ÉVALUATION NNUE PERSPECTIVE, ACCUMULATEURS PARESSEUX & CORRECTION (D, S, U6)
//     • Réseau NNUE vu du camp au trait (`NNUE_Persp`) : première couche
//       factorisée par phase (4 jeux de poids selon le nombre de trous, avec un
//       cache de rafraîchissement) et 8 têtes de sortie selon le nombre de
//       cases libres. Les poids sont dans
//       models/nnue_persp_k8_them_g046_qat_ib4f_ref2u_az20_distill.quantized.txt.
//     • « NNUE » = la première couche (énorme mais creuse : quelques entrées
//       actives par position) est tenue à jour par DIFFÉRENCE d'un coup à
//       l'autre, dans un « accumulateur », au lieu d'être recalculée.
//     • Produit creux (`sparse_h1`, +61 Elo à 0.2 s, +35 Elo à 1 s) : dans la
//       couche 1024 → 64, seuls les groupes de 4 entrées non nulles après ReLU
//       sont multipliés (AVX2/VNNI). Même résultat, plus vite.
//     • Accumulateurs paresseux : on ne met PAS à jour le réseau à chaque coup
//       joué/déjoué dans l'arbre. On empile simplement les coups joués, et
//       lorsqu'une feuille (ou un élagage statique) a réellement besoin de
//       l'évaluation, on remonte au dernier ancêtre à jour et on redescend
//       pli par pli (aucun « undo » nécessaire, aucun travail pour les
//       sous-arbres coupés avant d'évaluer).
//     • Cache d'évaluations sans verrou (`eval_cache`, 2^20 entrées de 64 bits
//       atomiques) : évite de recalculer le réseau quand la même position est
//       atteinte par un autre ordre de coups (36 % des valeurs de feuilles).
//     • Historique de correction (+10.6 Elo) : table indexée par un hachage des
//       8 pièces (noires + blanches) et par le camp au trait, qui apprend
//       l'écart (valeur_recherche − éval_statique) et corrige l'évaluation du
//       réseau aussi bien pour l'élagage qu'aux feuilles (depth == 0).
//
//  4. ÉLAGAGE AVANT LA BOUCLE DE COUPS (aux nœuds non-PV uniquement)
//     • Reverse Futility Pruning (RFP, F1, +76 Elo) à depth ≤ 5 :
//       si `static_eval − 2609 × depth ≥ beta`, on coupe immédiatement.
//     • Multi-ProbCut bidirectionnel (MPC, O, +57 Elo à 1 s) à depth ≥ 4 :
//       recherche la position courante à profondeur réduite `d' = max(1, 0.46 × d)` ;
//       si elle dépasse `beta + 4054 + 644 × (d − d')` (quand `static_eval ≥ beta`)
//       ou tombe sous `alpha − marge` (quand `static_eval ≤ alpha`), on coupe.
//     • Proxy Cut (N, +47 Elo à 0.2 s, +59 Elo à 1 s) à depth ≥ 4 :
//       à Yolah, le *null move pruning* classique (passer son tour) échoue (−81 Elo)
//       car le jeu est plein de zugzwangs (jouer détruit une case de son propre
//       espace : passer serait souvent mieux que tout coup). Le Proxy Cut le
//       remplace par un VRAI coup ordinaire « témoin » (le mieux classé au tri
//       hors coup de table et contre-coup), cherché à profondeur réduite
//       `d − 1 − (3 + d / 4)` contre `beta`. Si même ce coup ordinaire tient
//       `beta`, alors le meilleur coup du nœud tiendra `beta` a fortiori (même en
//       zugzwang : `valeur(meilleur) ≥ valeur(témoin)` est toujours vrai).
//
//  5. GÉNÉRATION PAR ÉTAPES & ORDONNANCEMENT DES COUPS (C, J, K, U4, U7, U19-U22)
//     • Génération par étapes (`staged`) : s'il existe un coup de table légal
//       (`tt_move`), on le teste en étape 1 AVANT de générer et noter les autres
//       coups (−4 % de nœuds, −6 % de temps) : s'il coupe, on n'a rien généré.
//     • Hiérarchie de tri des coups en étape 2 (`score_moves`) :
//         1. Coup de la table de transposition (`SCORE_TT = 2^30`)
//         2. Contre-coup (`SCORE_COUNTER = 2^29 − 2`) : le dernier coup ayant
//            réfuté le coup que l'adversaire vient de jouer (`played[ply − 1]`).
//         3. Coups ordinaires (y compris les deux coups tueurs), triés par la somme :
//              score(m) = history[us][from][to]                (historique principal)
//                       + cont1[us][prev.to][m.to]             (U4 : +32.6 Elo)
//                       + (intercepte(prev.to, m.to) ? 2000 : 0) (U7 : +17.0 Elo)
//                       + (est_killer(m) ? 4667 : 0)           (U7 : +17.7 Elo)
//                       + 1038 × (couverture(après m) − couverture(avant))
//                                                      (U19-U22, voir ci-dessous)
//            où prev est le dernier coup adverse.
//            La COUVERTURE de notre groupe est le nombre de cases libres que
//            nos pièces atteignent collectivement en un coup (union de leurs
//            déplacements de dame, chaque case comptée une fois, l'adversaire
//            n'étant qu'un obstacle). Un coup qui l'agrandit ouvre de l'espace
//            à notre groupe ; un coup qui la réduit l'enferme. Ce terme REMPLACE
//            l'influence de dev_u8 (`512 × influence` à depth ≥ 4, J : +36 Elo ;
//            l'influence inonde le plateau pas à pas depuis les pièces des deux
//            camps à la fois et compte qui arrive le premier) : contre dev_u8,
//            +17.9 Elo [+7.6, +28.2] à 0.2 s/coup (2000 parties) et +6.9 Elo
//            [+0.9, +13.0] à 1 s/coup (4000 parties). Il est appliqué à TOUTES
//            les profondeurs : il coûte 4 consultations de table d'attaques par
//            coup (une par pièce), bien moins que l'inondation, et paie donc
//            même près des feuilles (seuil 2 ou 3 : ≈ 0 Elo ; ajouté EN PLUS de
//            l'influence : ≈ 0).
//            Les coups tueurs (`killer1`, `killer2` : les deux derniers coups
//            qui ont coupé au même pli dans un nœud voisin) ne sont PAS placés
//            d'office devant tous les coups ordinaires : ils reçoivent un bonus
//            de +4667 sur leur score d'historique, ce qui permet à un coup ayant
//            un fort historique de continuation (`cont1`) et d'interception de
//            passer devant un killer hors contexte.
//
//  6. ÉLAGAGE PAR COUP, POCHES SCELLÉES & RÉDUCTIONS LMR (E, F3, U5, U7)
//     • Late Move Pruning (LMP, F3, +35 Elo) à depth ≤ 4 (non-PV) :
//       au-delà de `2 + depth²` coups essayés (et si un coup non perdant a déjà
//       été trouvé), on ne cherche pas la fin de la liste (les coups les moins
//       bien classés).
//     • Poches scellées comme réserves de tempos (`reserve`, U5, +10.3 Elo, EXACT) :
//       si un groupe de nos pièces est enfermé avec une région de ≤ 20 cases
//       libres que l'adversaire ne peut pas atteindre, l'ordre dans lequel on
//       remplit cette poche n'a aucun effet sur le reste du plateau ; seul
//       compte le nombre de coups qu'elle contient encore. On calcule ce nombre
//       maximal `h` exactement (`harvest_exact`) et on ne garde dans la boucle
//       qu'UN SEUL coup de ces pièces, qui préserve `h − 1` coups futurs.
//     • Late Move Reductions (LMR, E + U7) pour `depth ≥ 3` et `i ≥ 1` (i = rang
//       du coup dans l'ordre, à partir de 0) : un coup tardif est d'abord
//       cherché moins profond, et re-cherché à pleine profondeur seulement s'il
//       semble battre alpha. Réduction de base, en plis :
//           r = 1.1158 + ln(depth) × ln(i + 1) / 1.4152, modulée par :
//         − 1058/1024 pli aux nœuds PV
//         − 829/1024 pli pour le contre-coup (`SCORE_COUNTER`)
//         − history[us][from][to] / 6945 plis (historique principal seul)
//         − 1 pli si le coup « intercepte » le dernier coup adverse (il arrive
//           sur une ligne de la pièce qui vient de bouger, ou à côté d'elle)
//         − 1 pli si ce couple (prev.to → m.to) a souvent contredit sa réduction
//           par le passé (`lmrh[us][prev.to][m.to] ≥ 4453`, U7 +12.3 Elo).
//
//  7. PARALLÉLISME LAZY SMP (L)
//     • `nb_threads` threads lancent chacun l'approfondissement itératif sur la
//       même racine, en partageant sans verrou `SearchTable` et `eval_cache`,
//       avec chacun son `Worker` privé (historiques, killers, fenêtres). Ce que
//       l'un trouve aide les autres par la table. Les threads secondaires
//       sautent certaines profondeurs pour ne pas tous chercher la même chose
//       au même moment (cpp, section 2). Campagne U24 : sans ces sauts, −1 à
//       −4 Elo à 4 et 8 threads (+6 à 8 threads à 0.2 s), rien de significatif :
//       ils sont gardés, sans gain démontré.
//
//  8. RÉGLAGE DES CONSTANTES PAR SPSA (U23)
//     • Les 29 constantes numériques de la recherche (fenêtres d'aspiration,
//       LMR, RFP, MPC, Proxy Cut, LMP, bonus d'ordonnancement, historiques,
//       correction) ont été réglées ENSEMBLE par SPSA (« simultaneous
//       perturbation stochastic approximation », Spall 1992), comme le fait
//       Fishtest pour Stockfish (nnue/spsa.py, nnue/spsa.sh,
//       config/spsa/dev_u9_params.json).
//     • Le problème : régler une constante à la fois coûte un match de 2000
//       parties par valeur essayée, et la meilleure valeur de l'une dépend des
//       autres (les marges du MPC ont été réglées avant que le meilleur ordre
//       des coups ne change ce qui est sûr d'élaguer). Régler les 29 ensemble
//       demande une méthode qui les fait toutes bouger à chaque match.
//     • Un lot (24 parties) se déroule ainsi. Soit θ les valeurs actuelles.
//         1. Pour chaque constante i, on tire à pile ou face Δ_i = +1 ou −1.
//         2. On fabrique deux joueurs : θ+ = θ + c·Δ et θ− = θ − c·Δ, où c_i est
//            un écart choisi par constante (« c_end » du fichier de paramètres,
//            ~5 à 15 % de la valeur : assez pour changer le jeu).
//            Exemple sur 3 constantes : rfp margin 3000 (c 300), lmp moves 4
//            (c 0.8), killer bonus 4000 (c 500), tirage Δ = (+1, −1, +1) :
//              θ+ = (3300, 3.2, 4500)    θ− = (2700, 4.8, 3500).
//            Les constantes entières sont arrondies au hasard sans biais
//            (3.2 → 3 quatre fois sur cinq, 4 une fois sur cinq).
//         3. θ+ joue θ− : 12 ouvertures équilibrées, chacune avec les deux
//            couleurs. W = victoires de θ+, L = victoires de θ−.
//         4. TOUTES les constantes bougent d'un petit pas vers le gagnant :
//              θ_i ← θ_i + pas_i × (W − L) × Δ_i
//            Si θ+ a gagné (W > L) dans l'exemple : rfp margin monte, lmp moves
//            descend, killer bonus monte (chacune va vers sa valeur dans θ+).
//            Si θ− a gagné, c'est l'inverse ; à égalité, rien ne bouge.
//     • Pourquoi ça marche alors qu'un lot ne dit pas QUELLE constante a fait
//       gagner : les signes Δ sont tirés indépendamment à chaque lot. Une
//       constante sans effet sur la force est poussée vers le haut ou vers le
//       bas au hasard, et ses pas s'annulent en moyenne. Une constante dont la
//       hausse rend le joueur plus fort est, plus souvent qu'à son tour, du
//       côté gagnant quand son Δ vaut +1 : elle dérive vers le haut. En
//       moyenne, le pas de chaque constante est proportionnel à la pente du
//       taux de victoire selon cette constante (un gradient estimé), et ce
//       pour les 29 à la fois, au prix d'un seul petit match par lot.
//     • L'écart c et le pas diminuent au fil du run (calendriers de Fishtest :
//       c_k = c / (k+1)^0.101, pas_k ∝ 1 / (A+k+1)^0.602, k = nombre de paires
//       jouées) : on explore large au début, on affine à la fin.
//     • Le run : 4 167 lots, 50 004 paires de parties (100 008 parties) à
//       0.2 s/coup, un thread par joueur, ~15 h sur un nœud de 24 cœurs.
//       θ+ et θ− ont fait jeu égal sur l'ensemble (22 245 victoires contre
//       22 666), ce qui est normal : ils sont symétriques autour de θ ; c'est
//       θ qui a progressé. Le dernier quart du run n'a presque plus rien
//       changé. Les valeurs finales de θ, arrondies, sont celles de ce fichier.
//     • Gain contre dev_u9 (mêmes heuristiques, constantes réglées à la main) :
//       +14.4 Elo [+4.2, +24.7] à 0.2 s et +18.3 Elo [+9.2, +27.3] à 1 s
//       (2000 parties chacun) : le réglage tient à une cadence 5 fois plus
//       longue que celle où il a été fait. Contre dev_u8 (avant la couverture),
//       à 1 s : +19.3 Elo [+10.5, +28.1] (2000 parties). Le moteur cherche
//       24 % de nœuds en moins à profondeur égale (4,04 M contre 5,33 M).
//     • Les changements les plus nets : LMP plus agressif (2 + d² coups au lieu
//       de 4 + d², jusqu'à depth 4), Proxy Cut plus réduit (R = 3 + d/4 au lieu
//       de 2 + d/4), MPC dès depth 4 avec une marge plus large, RFP jusqu'à
//       depth 5 avec une marge plus faible, fenêtres d'aspiration plus serrées.
//       Le poids de la couverture (1038) reste au réglage trouvé à la main.
//       Les anciennes valeurs sont rappelées entre crochets dans les constantes.
//
//  ---------------------------------------------------------------------------
//  COMMENT VÉRIFIER QUE CE CODE JOUE COMME LE VRAI JOUEUR
//  ---------------------------------------------------------------------------
//  Un petit programme temporaire (non versionné) compile doc/final_dev.cpp avec
//  le projet, crée `MinMaxNNUE_DevPlayer` depuis
//  config/search_study/dev_u9_spsa.cfg (avec "nb threads" 1 et "tt size" 64)
//  et `FinalDevPlayer(0, 64, poids, false, 1)`, puis pour chaque position de
//  test/bench_positions.txt : `clear_table()` des deux côtés, `search(position,
//  11, 0)` des deux côtés, et compare `value`, `move` et `nb_nodes`. Les 209
//  positions doivent être identiques. (Avec plusieurs threads, la recherche
//  n'est pas déterministe : la comparaison ne se fait qu'à un thread.)
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
    // Résultat d'une recherche (`search`).
    struct Result {
        Move     move     = Move::none();  // coup recommandé (Move::none() = passe, ou rien trouvé)
        int16_t  value    = 0;     // sa valeur, du point de vue du camp au trait (échelle ci-dessus)
        uint8_t  depth    = 0;     // dernière profondeur entièrement terminée
        uint64_t nb_nodes = 0;     // nœuds visités, tous threads confondus
        uint64_t nb_hits  = 0;     // nœuds où la table de transposition connaissait la position
        double   seconds  = 0;     // durée de la recherche
    };

    // Constructeur : charge le réseau NNUE perspective, alloue la table de
    // transposition Yolah (`tt_size_mb` mégaoctets), le cache d'évaluations et
    // précalcule la table logarithmique des réductions LMR.
    //   microseconds        : temps de réflexion par coup utilisé par `play`
    //   nnue_persp_filename : fichier de poids du réseau (format quantifié)
    //   verbose             : `play` affiche profondeur, valeur, nœuds et variation principale
    //   nb_threads          : nombre de threads de recherche (Lazy SMP), au moins 1
    FinalDevPlayer(uint64_t microseconds,
                   size_t tt_size_mb,
                   const std::string& nnue_persp_filename,
                   bool verbose = false,
                   size_t nb_threads = 4);

    // Choisit un coup en `thinking_time` microsecondes (interface `Player`).
    Move play(Yolah yolah) override;
    // Recherche jusqu'à `max_depth` plis OU jusqu'à `microseconds` (0 = pas de
    // limite de temps : seule la profondeur arrête la recherche).
    Result search(const Yolah& yolah, uint8_t max_depth, uint64_t microseconds);
    // Vide la table de transposition et le cache d'évaluations ; si
    // `keep_ordering` est faux, efface aussi tout ce que les threads ont appris
    // (historiques, contre-coups, correction…). Sert entre deux parties et
    // pour les comparaisons reproductibles.
    void clear_table(bool keep_ordering = false);
    std::string info() override;
    json config() override;
    // La suite des meilleurs coups lue dans la table de transposition depuis
    // `yolah` (au plus `max_len` coups) : ce que le moteur prévoit.
    std::vector<Move> principal_variation(Yolah yolah, int max_len = 16) const;

private:
    // ── Échelle des valeurs de recherche ─────────────────────────────────────
    // Les calculs internes se font en `int` (pas de débordement lors des
    // décalages de fenêtres) et sont stockés en `int16_t` dans la table.
    //   • [-WIN + 1, WIN - 1] : évaluations du réseau NNUE (±30 000 = tanh ±1)
    //   • ±(WIN + écart)      : fin de partie exacte (une vraie victoire d'1 point
    //                           vaut WIN + 1 > toute évaluation heuristique).
    static constexpr int INFINITE = 32000;                  // ±∞ des fenêtres (> toute valeur possible)
    static constexpr int WIN      = heuristic::MAX_VALUE;   // 30 000
    static constexpr int EVAL_NONE = -32768;                // « pas d'évaluation statique à ce pli »

    // ── Constantes des heuristiques de `dev_u9_spsa` ─────────────────────────
    // Les valeurs viennent du SPSA de dev_u9 (U23 : 29 paramètres, 100 008 parties
    // à 0,2 s ; +14,4 Elo à 0,2 s, +18,3 à 1 s contre dev_u9). Entre crochets :
    // la valeur réglée à la main avant le SPSA. Les constantes sans crochets
    // n'ont pas été réglées par le SPSA (ou il les a laissées en place).

    // 1. Fenêtres d'aspiration adaptatives (B, A2) — cpp, section 2
    static constexpr int    ASPIRATION_MIN_DELTA   = 345;     // δ minimal (demi-largeur de la fenêtre) [300]
    static constexpr double ASPIRATION_K           = 0.9187;  // δ = K × variation moyenne de la valeur [1.25]
    static constexpr double ASPIRATION_GROW        = 1.3372;  // δ multiplié par ce facteur à chaque échec [1.5]
    static constexpr int    ASPIRATION_STATS_DEPTH = 4;       // la moyenne n'apprend qu'à partir de cette profondeur

    // 2. Late Move Reductions logarithmiques (E, U7) — cpp, sections 1, 4 et 6.g
    static constexpr double LMR_BASE      = 1.1158;  // r = BASE + ln(depth)·ln(i+1)/DIVISOR plis [1.25]
    static constexpr double LMR_DIVISOR   = 1.4152;  // [1.5]
    static constexpr int    LMRH_THRESHOLD = 4453;  // au-delà, lmrh retire 1 pli de réduction [4000]
    static constexpr int    LMR_PV         = 1058;  // −1058/1024 pli en nœud PV [1024]
    static constexpr int    LMR_SPECIAL    = 829;   // −829/1024 pli pour le contre-coup [1024]
    static constexpr int    LMR_HIST_DIV   = 6945;  // − hist / 6945 plis [8192]

    // 3. Reverse Futility Pruning (F1) — cpp, section 6, « F1 »
    static constexpr int RFP_DEPTH  = 5;              // appliqué si depth ≤ 5 [4]
    static constexpr int RFP_MARGIN = 2609;           // marge par pli de profondeur [3000]

    // 4. Multi-ProbCut bidirectionnel (O) — cpp, section 6, « O »
    static constexpr int MPC_DEPTH          = 4;     // appliqué si depth ≥ 4 [5]
    static constexpr int MPC_RATIO          = 46;    // d' = max(1, depth * 46 / 100) [40]
    static constexpr int MPC_MARGIN         = 4054;  // marge = 4054 + 644 × (depth − d') [4000]
    static constexpr int MPC_MARGIN_PER_PLY = 644;   // [500]

    // 5. Proxy Cut (N) — cpp, section 6.f
    static constexpr int PROXY_DEPTH     = 4;        // appliqué si depth ≥ 4
    static constexpr int PROXY_REDUCTION = 4;        // R = 4 + depth / 4 - 1 plis de moins [3]

    // 6. Late Move Pruning (F3) — cpp, section 6.g
    static constexpr int LMP_DEPTH = 4;              // appliqué si depth ≤ 4 [3]
    static constexpr int LMP_MOVES = 2;              // garde 2 + depth² coups [4]

    // 7. Poches scellées comme réserves de tempos (U5) — cpp, sections 5 et 6
    static constexpr int RESERVE_MAX_SQUARES = 20;   // taille max d'une poche (au-delà, calcul trop cher)
    static constexpr int HARVEST_BUDGET      = 200000; // nœuds max du calcul exact d'une poche (sinon abandon)

    // 8. Ordonnancement des coups et tables d'historique (C, J, U4, U7) — cpp, section 7
    static constexpr int HISTORY_MAX      = 16384;   // les historiques restent dans [−16384, +16384]
    static constexpr int KILLER_BONUS     = 4667;    // bonus des coups tueurs sur leur historique [4000]
    static constexpr int HIST_BONUS_A     = 16;      // bonus d'historique à la coupure :
    static constexpr int HIST_BONUS_B     = 34;      //   min(A·d² + B·d, CAP) [16, 32, 2000]
    static constexpr int HIST_BONUS_CAP   = 2166;
    static constexpr int INTERCEPT_BONUS  = 2000;    // coup sur un rayon ou une case voisine de prev.to
    static constexpr int COVER_DEPTH      = 1;       // couverture à toutes les profondeurs ≥ 1 (U20)
    static constexpr int COVER_WEIGHT     = 1038;    // poids de (couverture après − avant) (U20-U22) [1024]

    // Scores spéciaux de tri. Un historique composite reste sous 2^28 (il vaut
    // au plus quelques dizaines de milliers) : ces scores placent le coup de
    // table puis le contre-coup devant tous les coups ordinaires, quoi qu'il arrive.
    static constexpr int SCORE_TT           = 1 << 30;           // coup de la table : toujours 1er
    static constexpr int SCORE_KILLER1      = 1 << 29;           // (sert de repère ; les killers sont notés comme des coups ordinaires)
    static constexpr int SCORE_COUNTER      = SCORE_KILLER1 - 2; // contre-coup : 2e
    static constexpr int SCORE_ARTICULATION = 1 << 28; // plafond sous lequel vivent les coups ordinaires

    // 9. Historique de correction de l'évaluation statique (U4/U6) — cpp, sections 5 et 6
    static constexpr int CORR_TABLE_SIZE = 16384;    // 2^14 entrées par camp (clé de 14 bits)
    static constexpr int CORR_SCALE      = 8;        // pas = (best − static_eval) × depth / 8 (borné à ±2000)
    static constexpr int CORR_MAX        = 7917;     // la correction reste dans [−7917, +7917] [8000]

    // 10. Cache d'évaluations des feuilles (D) — cpp, section 8
    static constexpr int EVAL_CACHE_BITS = 20;       // 2^20 = 1 048 576 entrées (8 Mo)

    // ── État partagé entre les threads ───────────────────────────────────────
    const uint64_t thinking_time;   // µs par coup pour `play`
    const size_t   tt_size_mb;      // taille de la table de transposition
    const std::string weights_filename;
    const bool     verbose;
    const size_t   nb_threads;
    // Drapeau d'arrêt lu par tous les threads à chaque nœud (`stopped()`).
    // Positionné par l'horloge, ou par le thread principal quand il a fini.
    std::atomic_bool stop = false;

    // Table de transposition dédiée à Yolah (player/search_table.h) : clusters
    // de 4 entrées de 16 octets = 64 octets = une ligne de cache ; chaque
    // entrée garde une borne inférieure et une borne supérieure.
    std::unique_ptr<SearchTable> yolah_table;

    // Réseau NNUE perspective (nnue/nnue_persp.h), produit creux activé
    std::unique_ptr<NNUE_Persp> persp;
    using Accumulator = NNUE_Persp::Accumulator;

    // Table des réductions LMR précalculées en 1/1024 de pli :
    // lmr_table[depth][n] avec n = rang du coup + 1 (n ≥ 1, pour que ln(n) ≥ 0).
    int lmr_table[64][Yolah::MAX_NB_MOVES]{};

    // Cache d'évaluations atomique (1 mot de 64 bits par entrée :
    // clé 47 bits | valeur 16 bits | bit « occupé »), partagé par tous les threads.
    std::vector<std::atomic<uint64_t>> eval_cache;

    // ── Mémoire persistante propre à chaque thread (Worker) ──────────────────
    // Conservée d'un coup de la partie au suivant (avec vieillissement ÷ 2 de
    // l'historique principal à chaque nouveau coup, voir `search`). Ne pas
    // partager ces tables évite tout verrou, et fait diverger utilement les
    // arbres des threads de Lazy SMP (chacun trie ses coups un peu autrement).
    // Effacée par `clear_table(false)`.
    struct Worker {
        // Historique principal : history[joueur][ origine ][ destination ] ∈ [-16384, 16384].
        // Grand = ce coup (from → to) a souvent provoqué des coupures bêta,
        // n'importe où dans l'arbre ; négatif = il a souvent été essayé en vain.
        int16_t history[2][SQUARE_NB][SQUARE_NB]{};

        // Contre-coup : countermoves[ origine_adverse ][ dest_adverse ] = le dernier
        // coup qui a provoqué une coupure juste après ce coup adverse. La passe
        // adverse (Move::none() = a1 → a1) a sa propre case [0][0].
        Move countermoves[SQUARE_NB][SQUARE_NB]{};

        // U4 (7) : Historique de continuation à 1 pli (+32.6 Elo, le plus gros gain de U1–U10) :
        //   cont1[joueur][ destination_adverse_précédente ][ notre_destination ] ∈ [-16384, 16384]
        //
        // Pourquoi indexer par (prev.to, m.to) et non par (prev.from, prev.to, m.from, m.to) ?
        //   • Une table complète 4D (2 × 64^4 = 32 Mo) serait beaucoup trop creuse : au cours
        //     d'une recherche d'une seconde, la plupart des quadruplets ne seraient vus qu'une
        //     seule fois.
        //   • À Yolah, l'urgence créée par le dernier coup adverse dépend avant tout de
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
        // qui ont été réduites par LMR (`r_used > 0`) mais qui ont finalement battu `alpha`
        // (la réduction a donc coûté une re-recherche). Dès que `lmrh >= 4453`
        // (LMRH_THRESHOLD), ce couple reçoit 1 pli de réduction LMR en moins.
        // Elle ne fait que MONTER :
        //   • le seul endroit qui l'écrit (cpp, section 6.g) ajoute un bonus b > 0,
        //     b = min(16 d² + 32 d, 2000), quand un coup réduit a battu alpha ;
        //   • aucun malus quand la réduction était juste (un coup réduit qui échoue
        //     bas n'est jamais re-cherché : on ne sait pas s'il était bon) ;
        //   • elle n'est pas divisée par 2 entre deux coups de la partie (seul
        //     `history` l'est) ; seul `clear_table()` la remet à zéro.
        // Elle reste bornée par la « gravité » : h ← h + b − h·b/16384, soit un
        // pas b × (1 − h/16384), qui rétrécit à mesure que h approche 16384 et
        // vaut 0 en h = 16384 (l'écart à 16384 est multiplié par (1 − b/16384) à
        // chaque mise à jour). Exemple avec b = 2000 (d ≥ 11) : 0 → 2000 → 3756 →
        // 5298 : le seuil 4453 est franchi à la 3e réduction ratée ; avec
        // b = 48 (d = 1), il en faut une centaine.
        // Conséquence : un couple (prev.to, m.to) qui a franchi le seuil y reste
        // jusqu'à la fin de la partie, et ses coups sont réduits d'un pli de moins
        // pour de bon. C'est voulu : ce couple a montré qu'il fallait le chercher.
        int16_t lmrh[2][SQUARE_NB][SQUARE_NB]{};

        // U4/U6 (6) : Historique de correction de l'évaluation statique
        // (+10.6 Elo [+0.4, +20.8] sur dev_u6, campagne U6) :
        //   corr[joueur][ clé ] ∈ [-7917, 7917]
        //
        // LE PROBLÈME. Le réseau se trompe de façon systématique dans certaines
        // configurations de pièces : trop optimiste ici, trop pessimiste là. La
        // recherche paie ces erreurs à chaque nœud, car RFP, MPC et Proxy Cut
        // comparent tous l'évaluation statique à beta : une évaluation trop haute
        // fait couper à tort, une trop basse empêche des coupures justes.
        //
        // L'IDÉE. Pendant la partie, chaque recherche sous un nœud dit ce que le
        // nœud valait vraiment (à la profondeur cherchée). L'écart avec ce que le
        // réseau avait annoncé mesure son erreur. Ces erreurs sont mémorisées
        // par configuration de pièces, et ajoutées au réseau la prochaine fois :
        //     évaluation statique = clamp(réseau + corr[camp au trait][clé], −29999, 29999)
        //   (aux nœuds non-PV pour l'élagage, et aux feuilles : `corr_hist = 2`).
        //
        // LA CLÉ (`corr_key`, cpp, section 5). Un hachage sur 14 bits (0..16383)
        // des positions des 4 pièces noires et des 4 pièces blanches. Les trous
        // sont IGNORÉS : deux positions avec les mêmes pièces aux mêmes endroits
        // partagent la même correction, ce qui la fait revenir assez souvent pour
        // apprendre vite. 16384 cases pour bien plus de configurations possibles :
        // des positions sans rapport peuvent tomber sur la même case et brouiller
        // un peu la correction (accepté, c'est mesuré gagnant). Une table par camp
        // au trait : l'erreur n'est pas la même selon qui doit jouer.
        //
        // L'APPRENTISSAGE (`corr_update`, cpp, section 6), à la fin de chaque nœud
        // non-PV (seuls ceux-là calculent l'évaluation statique), avec la valeur
        // `best` trouvée par la recherche :
        //   • rien si `best` est un résultat prouvé (|best| ≥ 30 000) ;
        //   • rien si `best` n'est qu'une borne qui ne contredit pas l'évaluation :
        //     une coupure (best ≥ beta, donc vraie valeur ≥ best) avec best ≤ éval,
        //     ou un échec bas (vraie valeur ≤ best) avec best ≥ éval ; ces bornes
        //     ne disent pas dans quel sens le réseau s'est trompé ;
        //   • sinon, pas b = (best − éval) × depth / 8, borné à ±2000 (une
        //     recherche plus profonde est plus fiable, elle pèse plus), puis
        //     mise à jour par gravité : e ← e + b − e·|b|/7917, ce qui garde e
        //     dans [−7917, 7917] (même principe que `history`, section 7).
        //   Exemple : nœud de profondeur 6, le réseau (déjà corrigé) dit 2000, la
        //   recherche trouve exactement 5000. b = 3000 × 6 / 8 = 2250 → borné à
        //   2000 ; e passe de 0 à 2000. Si la même erreur se répète : e = 2000 +
        //   2000 − 2000 × 2000 / 7917 = 3495, et ainsi de suite vers 7917. Si
        //   l'erreur disparaît (la correction suffit), b ≈ 0 et e ne bouge plus.
        //
        // DURÉE DE VIE. Propre à chaque thread (dans le Worker), gardée d'un coup
        // de la partie au suivant (pas divisée par 2), remise à zéro par
        // `clear_table()`. Elle n'apprend que pendant la partie : rien n'est
        // sauvegardé d'une partie à l'autre.
        int16_t corr[2][CORR_TABLE_SIZE]{};

        // Fenêtres d'aspiration adaptatives : moyenne glissante (décroissance 0.9
        // par profondeur terminée) des écarts |v_d − v_{d−1}| observés par ce
        // thread, gardée d'un coup de la partie au suivant.
        // Moyenne = asp_sum / asp_n.
        double asp_sum = 0;
        double asp_n   = 0;
    };
    std::vector<std::unique_ptr<Worker>> workers;   // un par thread, workers[i] pour le thread i

    // ── État d'un coup à la racine ───────────────────────────────────────────
    struct RootMove {
        Move     move;
        uint64_t nodes = 0;   // nœuds explorés sous ce coup (cumulés sur la profondeur courante)
        int      value = 0;   // dernière valeur (souvent une borne) renvoyée pour ce coup
    };

    // ── État local d'une recherche pour un thread (`Search`) ─────────────────
    // Recréé à chaque appel de `search` (donc vide au début de chaque coup),
    // contrairement au `Worker`. Les tableaux par pli sont indexés par le pli
    // ABSOLU de la partie (`yolah.nb_plies()`), pas par la distance à la racine.
    struct Search {
        int      id       = 0;         // 0 = thread principal, 1..k = threads assistants (Lazy SMP)
        Worker*  w        = nullptr;   // la mémoire persistante de ce thread
        uint8_t  depth    = 0;         // résultat : dernière profondeur terminée
        int16_t  value    = 0;         //            sa valeur
        Move     move     = Move::none(); //         son coup
        size_t   nb_nodes = 0;         // statistiques
        size_t   nb_hits  = 0;
        uint64_t nb_evals = 0, nb_eval_hits = 0;

        // Deux coups tueurs par pli de l'arbre : les deux derniers coups
        // (différents) qui ont provoqué une coupure bêta à ce pli.
        Move killer1[Yolah::MAX_NB_PLIES]{};
        Move killer2[Yolah::MAX_NB_PLIES]{};

        // Pile des coups joués le long de la branche courante :
        // played[p] est le coup joué au pli p (sert au contre-coup, à cont1, à
        // l'interception, à lmrh et à la mise à jour paresseuse de l'accumulateur).
        // Les plis AVANT la racine valent Move::none() : la recherche ne connaît
        // pas le coup que l'adversaire vient réellement de jouer.
        Move played[Yolah::MAX_NB_PLIES + 1]{};

        // Évaluation statique corrigée à chaque pli (EVAL_NONE si non calculée :
        // nœud PV ou position finale). Sert à `corr_update`.
        int eval_stack[Yolah::MAX_NB_PLIES + 2]{};

        // Pile d'accumulateurs NNUE paresseux : accs[p] est l'accumulateur de la
        // position au pli p, valide ssi acc_ok[p] == true (cpp, section 8).
        Accumulator accs[Yolah::MAX_NB_PLIES + 1];
        bool        acc_ok[Yolah::MAX_NB_PLIES + 1]{};
        // Cache du réseau pour les changements de « jeu de poids » de la
        // première couche (quand le nombre de trous change de tranche) :
        // n'influe que sur la vitesse, jamais sur le résultat.
        NNUE_Persp::RefreshCache refresh;

        // Liste ordonnée des coups à la racine (réordonnée après chaque recherche racine)
        std::vector<RootMove> root_moves;
    };

    // ── Accès à la table de transposition ────────────────────────────────────
    // Vue d'une entrée : `found` faux si la position n'est pas dans la table.
    // lower/upper = ±32767 (SearchTable::NO_BOUND) signifie « rien de connu de ce côté ».
    struct TTView {
        bool found = false;
        Move move  = Move::none();   // meilleur coup connu (à vérifier : peut venir d'une collision)
        int  depth = 0;              // profondeur des recherches qui ont prouvé les bornes
        int  lower = -32767;         // valeur ≥ lower
        int  upper =  32767;         // valeur ≤ upper
    };
    TTView tt_probe(uint64_t hash) const;
    // Enregistre le résultat `value` d'une recherche de profondeur `depth` à la
    // fenêtre (alpha, beta) : la table en déduit elle-même les bornes (v ≤ alpha :
    // borne sup. ; v ≥ beta : borne inf. ; entre les deux : les deux).
    void   tt_store(uint64_t hash, const Yolah& yolah, int depth, int alpha, int beta,
                    int value, Move move, bool pv = false);
    Move   tt_move(uint64_t hash) const;

    // ── Méthodes internes du moteur (définies dans final_dev.cpp) ────────────
    bool stopped() const { return stop.load(std::memory_order_relaxed); }
    void iterative_deepening(Yolah yolah, Search& s, uint8_t max_depth);                       // section 2
    int  root_search(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth, Move& res); // section 3
    int  negamax(Yolah& yolah, Search& s, uint64_t hash, int alpha, int beta, int depth);      // section 6
    int  search_move(Yolah& yolah, Search& s, uint64_t hash, Move m, size_t i,
                     int reduction, int alpha, int beta, int depth);                            // section 4
    int  late_move_reduction_of(int depth, size_t i, bool pv_node, bool special, int hist) const; // section 4

    void score_moves(const Yolah& yolah, const Search& s, Move tt_move,
                     const Yolah::MoveList& moves, int* scores, int depth) const;               // section 7
    static uint64_t coverage(uint64_t mine, uint64_t occupied);                                 // section 7
    static bool is_legal(const Yolah& yolah, Move m);                                           // section 7
    static Move pick_move(Yolah::MoveList& moves, int* scores, size_t i, size_t n);             // section 7
    void update_history(Search& s, uint8_t player, Move m, int bonus);                          // section 7

    int  evaluate(const Yolah& yolah, Search& s, uint64_t hash);                                // section 8
    int  raw_network_value(const Yolah& yolah, Search& s);                                      // section 8
    void print_pv(Yolah yolah, uint64_t hash, int8_t depth);                                    // section 9
};

#endif // FINAL_DEV_H
