#ifndef FINAL_NNUE_H
#define FINAL_NNUE_H

// =============================================================================
//  FINAL_NNUE — Version épurée et commentée du réseau NNUE du joueur dev
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (et son compagnon doc/final_nnue.cpp) contient le réseau de
//  neurones qui évalue les positions pour le joueur « Minimax NNUE dev »
//  (doc/final_dev.{h,cpp}) : à chaque feuille de la recherche, il rend un
//  nombre dans [−1, 1], l'estimation de l'issue de la partie pour le camp au
//  trait (+1 : il gagne, −1 : il perd). Le joueur le multiplie par WIN = 30000
//  (doc/final_dev.cpp, section 8) pour obtenir sa valeur entière.
//
//  Le vrai code est réparti dans trois fichiers :
//    nnue/nnue_persp.{h,cpp}   le réseau (struct NNUE_Persp) : chargement des
//                              poids, accumulateurs, valeur
//    nnue/sparse_matvec.h      le produit « creux » de la 2e couche
//    nnue/space_features.h     les entrées d'« espace » calculées à chaque
//                              évaluation (distance de dame, influence…)
//  Ces fichiers servent à beaucoup d'expériences et ont des options que le
//  joueur dev n'utilise pas : première couche de 512 valeurs, anciens formats
//  de fichier (« PERSP », « PERSPS »), chemin direct (« DIRECT »), tête de
//  coups (NNUE_PolicyHead), produit dense, play/undo en place, 92 colonnes
//  d'espace possibles dont 10 seulement sont lues ici. Il ne reste ici QUE ce
//  que le joueur dev utilise avec son fichier de poids
//      models/nnue_persp_k8_them_g046_qat_ib4f_ref2u_az20_distill.quantized.txt
//  et les opérations qu'il appelle : init (à la racine), update (descente
//  paresseuse, avec le cache de rafraîchissement), value (aux feuilles).
//
//  Une différence de forme, pas de résultat : le vrai code calcule avec les
//  instructions vectorielles AVX2 du processeur (32 octets à la fois) ; ici,
//  ce sont des boucles ordinaires, plus lentes mais lisibles. Les calculs sont
//  des sommes d'ENTIERS, dont l'ordre ne change pas le résultat : les deux
//  versions rendent exactement les mêmes nombres (voir « VÉRIFICATION »).
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF
//  ---------------------------------------------------------------------------
//  (Les règles en détail sont au début de doc/final_dev.h.) Plateau 8 × 8,
//  4 pièces par camp, Noir commence. Une pièce se déplace comme une dame
//  d'échecs vers une case libre, sans rien sauter ; sa case de départ devient
//  un TROU définitif. Un camp bloqué passe (`Move::none()`). Une case = un
//  entier 0..63 (a1 = 0, h8 = 63) ; un ensemble de cases = un `uint64_t`
//  (« bitboard », bit k = case k).
//
//  ---------------------------------------------------------------------------
//  « NNUE » : L'IDÉE
//  ---------------------------------------------------------------------------
//  NNUE = « Efficiently Updatable Neural Network » (réseau mis à jour
//  efficacement), inventé pour le shogi (Yu Nasu, 2018) puis adopté par
//  Stockfish. Deux idées :
//    (1) La première couche est ÉNORME (192 entrées → 1024 neurones) mais ses
//        entrées sont des 0/1 (« il y a une de mes pièces en c3 », « il y a un
//        trou en e5 »…) et un coup n'en change que 3 : la pièce quitte sa
//        case, arrive sur une autre, la case quittée devient un trou. Au lieu
//        de recalculer 192 × 1024 produits, on garde les 1024 sommes (« l'ACCU-
//        MULATEUR ») et on leur ajoute ou retire 3 lignes de poids : 3 × 1024
//        additions. C'est ce qui rend le réseau assez rapide pour évaluer
//        des millions de positions par seconde.
//    (2) Le reste du réseau est petit (2048+10 → 64 → 32 → 1) et calculé en
//        entiers 8 bits (« quantifié »), ce que le processeur fait très vite.
//
//  ---------------------------------------------------------------------------
//  L'ARCHITECTURE (de l'entrée à la sortie)
//  ---------------------------------------------------------------------------
//
//    position ──► 192 entrées 0/1 par POINT DE VUE (deux points de vue)
//                    │  couche 1 : 192 → 1024, poids int8, 4 jeux selon la phase
//                    ▼
//               accumulateur : 2 × 1024 sommes int16 (une moitié par point de vue)
//                    │  bornage [0, 64] (« clipped ReLU »)
//                    ▼
//               h1 = 1024 « nous » + 1024 « eux » + 10 entrées d'espace = 2058 int8
//                    │  couche 2 : 2058 → 64, poids int8, 8 têtes selon la phase
//                    ▼
//               h2 = 64 int8 dans [0, 64]
//                    │  couche 3 : 64 → 32, poids int8 (même tête)
//                    ▼
//               h3 = 32 int8 dans [0, 64]
//                    │  couche 4 : 32 → 1, poids float (même tête)
//                    ▼
//               tanh(·) ∈ [−1, 1] : la valeur pour le camp au trait
//
//  • LES ENTRÉES, POINT DE VUE PAR POINT DE VUE. 192 entrées = 3 plans de 64
//    cases : MES pièces (lignes 0–63), SES pièces (64–127), les TROUS
//    (128–191). Le point de vue NOIR voit « mes pièces » = les noires, le
//    point de vue BLANC « mes pièces » = les blanches. Le point de vue blanc
//    est en plus retourné en miroir a ↔ h, pour que les deux camps partent
//    des mêmes cases (les pièces de départ ne sont pas symétriques par une
//    simple rotation). Les DEUX points de vue sont tenus à jour à chaque
//    coup ; au moment d'évaluer, on lit celui du camp au trait (« nous ») et
//    celui de l'autre (« eux »), toujours dans cet ordre. Le réseau apprend
//    donc une seule fonction « vu par celui qui joue », au lieu d'une pour
//    Noir et une pour Blanc.
//  • LES PHASES. La phase de la partie se lit au nombre de trous (0 au début,
//    au plus 56) :
//      – la couche 1 a 4 jeux de poids : trous 0–14, 15–28, 29–42, 43–56
//        (jeu = trous × 4 / 57) ;
//      – les couches 2 à 4 ont 8 « têtes » : trous 0–7, 8–14, …, 50–56
//        (tête = trous × 8 / 57).
//    Une position d'ouverture et une fin de partie ne se jugent pas pareil :
//    chaque phase a ses propres poids.
//  • LES ENTRÉES D'ESPACE. 10 nombres calculés à chaque évaluation à partir de
//    la position, ajoutés à h1 (ils ne sont pas incrémentaux comme
//    l'accumulateur) : pour chaque camp, le nombre de cases libres atteintes
//    en 1, 2 et 3 coups de dame, celui des cases atteintes avant l'adversaire,
//    et l'INFLUENCE (les cases libres conquises quand les deux camps
//    s'étendent d'un pas de roi à la fois). Voir doc/final_nnue.cpp, section 4.
//
//  ---------------------------------------------------------------------------
//  LA QUANTIFICATION : DES ENTIERS À L'ÉCHELLE 64
//  ---------------------------------------------------------------------------
//  Le réseau est entraîné en nombres réels (PyTorch), puis exporté en entiers.
//  L'échelle est 64 : un nombre réel x est représenté par l'entier
//  round(64 · x).
//    • Poids des couches 1 à 3 : int8 (−128..127), donc des réels dans
//      [−2, 2[ par pas de 1/64.
//    • Activations (h1, h2, h3) : int8 dans [0, 64], c'est-à-dire des réels
//      dans [0, 1] : c'est le bornage (« clipped ReLU ») qui les y met.
//    • Produit de deux nombres à l'échelle 64 = un nombre à l'échelle 64² :
//      après chaque produit matrice × vecteur, on redivise par 64 (décalage
//      de 6 bits, `>> 6`), puis on borne dans [0, 64].
//    • Les biais sont rangés directement à la bonne échelle : 64 pour la
//      couche 1 (ajoutés à l'accumulateur), 64² pour les couches 2 et 3
//      (ajoutés aux sommes avant le décalage).
//    • La dernière couche (32 → 1) est en float : elle est minuscule.
//  Ce réseau a été entraîné « en connaissant » cette quantification (« QAT » :
//  quantization-aware training, le « qat » du nom du fichier) : pendant
//  l'entraînement, PyTorch calcule exactement les mêmes arrondis que ce code,
//  si bien que le réseau exporté rend les mêmes valeurs que le réseau
//  entraîné (sans QAT, l'arrondi des poids coûtait des points de précision et
//  des dizaines d'Elo, campagne P6).
//
//  ---------------------------------------------------------------------------
//  D'OÙ VIENT CE RÉSEAU (et ce que chaque choix a rapporté, en parties)
//  ---------------------------------------------------------------------------
//  Le nom du fichier résume son histoire :
//    • persp, k8, them (étape 1, campagnes P1–P2) : point de vue du camp au
//      trait, 8 têtes par phase, la couche 2 lit aussi le point de vue de
//      l'adversaire. Contre le réseau précédent (193 entrées, une seule
//      tête) : +52,6 Elo à 1 s par coup. Les 8 têtes SANS « them » perdaient
//      75 Elo contre un réseau de contrôle à une tête (réseau plus bruité) :
//      c'est la lecture des deux points de vue qui fait le gain.
//    • le produit creux de la couche 2 (section 5 du .cpp) : même résultat,
//      −34 % de temps par évaluation, +35 Elo à 1 s (campagne S1).
//    • g046 (étape 2, campagnes P3–P7) : les groupes d'entrées d'espace
//      « distance de dame » (qdist), « atteint en premier » (qfirst) et
//      « influence » (infl), 0x46 en hexadécimal = ces trois groupes. Choisis
//      après une ablation de 9 groupes candidats (92 colonnes) : beaucoup
//      améliorent la précision mais coûtent trop de temps par évaluation.
//    • qat (campagne P6) : entraînement conscient de la quantification,
//      environ +20 à +30 Elo à budget égal.
//    • az20 (campagne P7) : 20 époques avec pour professeur le réseau
//      AlphaZero 256 canaux (les valeurs à imiter sur ~1 milliard de
//      positions).
//    • ib4f, ref2u (campagne F1) : 4 jeux de poids de la couche 1 selon la
//      phase, ajoutés à un réseau de référence puis tout ré-entraîné :
//      +32,4 Elo à 1 s contre le réseau précédent du joueur dev.
//    • distill : entraîné par distillation (imiter les valeurs d'un
//      professeur plutôt que les seuls résultats des parties).
//
//  ---------------------------------------------------------------------------
//  L'ACCUMULATEUR ET SON CACHE
//  ---------------------------------------------------------------------------
//  Accumulator = les 2 × 1024 sommes de la couche 1 (avant bornage) d'UNE
//  position, plus cette position elle-même (3 bitboards : noires, blanches,
//  trous). La position est gardée parce que `update` ne la reçoit pas : il a
//  besoin de savoir quand le nombre de trous fait changer de jeu de poids de
//  la couche 1.
//    • Dans le même jeu (le cas courant) : update = 3 lignes de poids par
//      point de vue (≈ 6 × 1024 additions).
//    • Changement de jeu (3 fois par partie : à 15, 29 et 43 trous) : TOUTES
//      les sommes changent de poids, il faut « rafraîchir » l'accumulateur à
//      partir de la position. Pour ne pas tout recalculer (60 à 120 lignes),
//      chaque thread de recherche garde un RefreshCache (les « finny tables »
//      de Stockfish) : pour chaque jeu, le dernier accumulateur rafraîchi et
//      sa position. Un rafraîchissement part de là et n'applique que les
//      cases qui diffèrent, peu nombreuses dans une même recherche. Sans ce
//      cache, 9 % des mises à jour étaient des rafraîchissements complets
//      (+11 % de temps) ; avec lui, le coût disparaît.
//  Le cache ne change JAMAIS le résultat, seulement la vitesse.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_nnue_check.sh compile test/final_nnue_check.cpp avec ce code et
//  le vrai NNUE_Persp (bibliothèque du moteur de l'app), joue des parties au
//  hasard et compare à chaque position : les accumulateurs (mis à jour coup
//  par coup, avec et sans cache, et recalculés de zéro), les valeurs rendues
//  (au bit près) et la valeur entière du joueur (valeur × 30000).
//
//  ---------------------------------------------------------------------------
//  GLOSSAIRE
//  ---------------------------------------------------------------------------
//    accumulateur   les sommes de la couche 1, tenues à jour coup par coup
//    clipped ReLU   « bornage » : x ↦ min(max(x, 0), 64) à l'échelle 64
//    int8 / int16   entiers signés sur 8 / 16 bits
//    jeu (couche 1) un des 4 ensembles de poids de la couche 1, choisi par le
//                   nombre de trous ; on dit « set » dans le vrai code
//    tête           un des 8 ensembles de poids des couches 2 à 4, choisi par
//                   le nombre de trous ; « head » ou « bucket » dans le code
//    point de vue   « view » : la moitié de l'accumulateur vue par un camp
//    quantifier     remplacer des réels par des entiers à une échelle donnée
//    rafraîchir     recalculer un accumulateur depuis la position (« refresh »)
// =============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "game.h"
#include "move.h"

namespace final_nnue {

// ── Tailles du réseau du joueur dev (vérifiées au chargement) ───────────────
constexpr int INPUTS     = 192;    // entrées par point de vue : mes pièces, ses pièces, trous
constexpr int H1         = 1024;   // neurones de la couche 1, par point de vue
constexpr int SPACE      = 10;     // entrées d'espace (section 4 du .cpp)
constexpr int FC2_IN     = 2 * H1 + SPACE;   // entrées de la couche 2 : nous, eux, espace
constexpr int H2         = 64;     // neurones de la couche 2
constexpr int H3         = 32;     // neurones de la couche 3
constexpr int SETS       = 4;      // jeux de poids de la couche 1 (phase)
constexpr int HEADS      = 8;      // têtes des couches 2 à 4 (phase)
constexpr int MAX_HOLES  = 56;     // 64 cases − 8 pièces
constexpr int SCALE      = 64;     // l'échelle de la quantification
constexpr int SHIFT      = 6;      // 64 = 2^6 : diviser par 64 = décaler de 6 bits

// Les sommes de la couche 1 d'une position (avant bornage), pour les deux
// points de vue, et la position elle-même.
struct Accumulator {
    int16_t view[2][H1];   // view[Yolah::BLACK] et view[Yolah::WHITE]
    uint64_t black = 0, white = 0, holes = 0;
};

// Le cache de rafraîchissement d'un thread : pour chaque jeu de poids de la
// couche 1, le dernier accumulateur rafraîchi (valid = il existe).
struct RefreshCache {
    Accumulator entry[SETS];
    bool valid[SETS]{};
    void clear() { for (bool& v : valid) v = false; }
};

class Network {
public:
    // Lit le fichier de poids (format texte écrit par nnue/export_value_nets.py).
    // Refuse tout autre réseau que celui du joueur dev (std::runtime_error).
    explicit Network(const std::string& filename);

    // L'accumulateur de `yolah`, calculé de zéro (à la racine de la recherche).
    void init(const Yolah& yolah, Accumulator& acc) const;
    // out = in + le coup `m` joué par `player` (`in` n'est pas modifié).
    // cache : sert aux changements de jeu de la couche 1 (nullptr : sans cache,
    // même résultat).
    void update(const Accumulator& in, Accumulator& out, uint8_t player, Move m, RefreshCache* cache) const;
    // La valeur dans [−1, 1] pour le camp au trait de `yolah`, dont `acc` est
    // l'accumulateur.
    float value(const Accumulator& acc, const Yolah& yolah) const;

    // La ligne de la case `sq` pour le point de vue `view` (section 2 du .cpp).
    static int row(int view, int sq) { return view == Yolah::BLACK ? 63 - sq : (63 - sq) ^ 7; }

private:
    int set_of(uint64_t holes) const;       // jeu de la couche 1 : trous × 4 / 57
    int head_of(uint64_t holes) const;      // tête des couches 2 à 4 : trous × 8 / 57
    void init_from(uint64_t black, uint64_t white, uint64_t holes, Accumulator& acc) const;
    void refresh(uint64_t black, uint64_t white, uint64_t holes, Accumulator& out, RefreshCache* cache) const;
    void space_inputs(const Yolah& yolah, int8_t out[SPACE]) const;

    // Les poids, rangés pour être lus dans l'ordre des calculs :
    std::vector<int8_t>  fc1;        // SETS × INPUTS × H1 : une LIGNE de H1 poids par entrée
    std::vector<int16_t> fc1_bias;   // SETS × H1                         (échelle 64)
    std::vector<int8_t>  fc2;        // HEADS × H2 × FC2_IN : une ligne par neurone de sortie
    std::vector<int16_t> fc2_bias;   // HEADS × H2                        (échelle 64²)
    std::vector<int8_t>  fc3;        // HEADS × H3 × H2
    std::vector<int16_t> fc3_bias;   // HEADS × H3                        (échelle 64²)
    std::vector<float>   fc4;        // HEADS × H3 (float)
    std::vector<float>   fc4_bias;   // HEADS
    std::array<int, SPACE> space_cols{};   // les colonnes d'espace lues, dans l'ordre du fichier
};

} // namespace final_nnue

#endif
