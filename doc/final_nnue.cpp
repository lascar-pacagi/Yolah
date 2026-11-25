// =============================================================================
//  final_nnue.cpp — le réseau NNUE du joueur dev, épuré et commenté.
//  Lire d'abord doc/final_nnue.h (l'idée du NNUE, l'architecture, la
//  quantification, l'histoire du réseau, le glossaire). Plan :
//    1. Le chargement des poids
//    2. Les entrées : lignes, points de vue, phases
//    3. L'accumulateur : calcul de zéro, rafraîchissement, mise à jour d'un coup
//    4. Les entrées d'espace (distance de dame, atteint en premier, influence)
//    5. La valeur : bornage, couche 2 (produit creux), couches 3 et 4
//  Le vrai code : nnue/nnue_persp.cpp (NNUE_Persp), nnue/sparse_matvec.h,
//  nnue/space_features.h.
// =============================================================================
#include "final_nnue.h"
#include "magic.h"
#include "types.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace final_nnue {

// =============================================================================
//  1. LE CHARGEMENT DES POIDS
// =============================================================================
//
// Le fichier est du texte, écrit par nnue/export_value_nets.py à partir du
// réseau entraîné (PyTorch), poids déjà quantifiés (des entiers, sauf la
// dernière couche). Son début, pour le réseau du joueur dev :
//     PERSPC 8 1 1024 64 32 10 1 2 3 4 9 10 11 12 16 17
//       │    │ │  │   │  │  │  └─ les 10 colonnes d'espace lues (section 4)
//       │    │ │  │   │  │  └─ leur nombre
//       │    │ │  │   │  └─ neurones de la couche 3
//       │    │ │  │   └─ neurones de la couche 2
//       │    │ │  └─ neurones de la couche 1 (par point de vue)
//       │    │ └─ 1 = la couche 2 lit aussi le point de vue de l'adversaire
//       │    └─ nombre de têtes (phases) des couches 2 à 4
//       └─ le format (« PERSP » + « C » : avec une liste de colonnes d'espace)
// puis les matrices, chacune annoncée par « W lignes colonnes » et rangée
// ligne par ligne, et les biais, annoncés par « B taille » :
//     W 4096 192    couche 1 : 4 jeux × 1024 neurones (lignes), 192 entrées
//     B 4096        ses biais
//     W 512 2058    couche 2 : 8 têtes × 64 neurones, 2 × 1024 + 10 entrées
//     B 512         ses biais
//     8 fois : W 32 64 et B 32          couche 3, une tête après l'autre
//     W 8 32 et B 8                     couche 4 (float), les 8 têtes
// Le nombre de jeux de la couche 1 se lit au nombre de lignes de sa matrice
// (4096 / 1024 = 4).

// Lit « W lignes colonnes » (ou « B taille ») et vérifie les tailles annoncées.
static void expect(std::ifstream& in, const char* type, int a, int b, const char* what) {
    std::string t;
    int x = 0, y = 0;
    in >> t >> x;
    if (t == "W") in >> y;
    if (!in || t != type || x != a || (t == "W" && y != b))
        throw std::runtime_error(std::string("final_nnue: bloc inattendu pour ") + what);
}

// Lit `n` nombres et les range dans `out`. Le fichier écrit des entiers (ou des
// réels pour la couche 4) ; comme le vrai code, on lit un double puis on le
// convertit vers le type de rangement.
template<typename T>
static void read_values(std::ifstream& in, size_t n, T* out) {
    for (size_t i = 0; i < n; i++) {
        double v;
        in >> v;
        out[i] = T(v);
    }
}

Network::Network(const std::string& filename) {
    std::ifstream in(filename);
    std::string tag;
    int heads, them, l1, l2, l3, n_space;
    if (!(in >> tag >> heads >> them >> l1 >> l2 >> l3 >> n_space))
        throw std::runtime_error("final_nnue: impossible de lire " + filename);
    // Seul le réseau du joueur dev est accepté : ses tailles sont des constantes
    // du code (final_nnue.h), ce qui rend chaque boucle plus simple à lire.
    if (tag != "PERSPC" || heads != HEADS || them != 1 || l1 != H1 || l2 != H2 || l3 != H3 || n_space != SPACE)
        throw std::runtime_error("final_nnue: " + filename + " n'est pas le réseau du joueur dev");
    for (int& c : space_cols) {
        in >> c;
        // les colonnes que la section 4 sait calculer : Q1, Q2, Q3, QFIRST des
        // deux camps (1–4, 9–12) et l'influence des deux camps (16, 17)
        const bool known = (c >= 1 && c <= 4) || (c >= 9 && c <= 12) || c == 16 || c == 17;
        if (!in || !known) throw std::runtime_error("final_nnue: colonne d'espace inconnue");
    }

    // Couche 1. PyTorch range la matrice « une ligne par neurone » (4096 lignes
    // de 192 entrées). On la range « une ligne par ENTRÉE » : quand une entrée
    // change (une pièce bouge), on ajoute ou retire d'un bloc ses 1024 poids
    // contigus, ce qu'un processeur fait bien plus vite que de sauter de ligne
    // en ligne. Le neurone i du fichier est le neurone i % 1024 du jeu i / 1024.
    expect(in, "W", SETS * H1, INPUTS, "la couche 1");
    fc1.resize(size_t(SETS) * INPUTS * H1);
    for (int i = 0; i < SETS * H1; i++) {
        int8_t* set = fc1.data() + size_t(i / H1) * INPUTS * H1;
        for (int j = 0; j < INPUTS; j++) {
            double v;
            in >> v;
            set[size_t(j) * H1 + i % H1] = int8_t(v);
        }
    }
    expect(in, "B", SETS * H1, 0, "les biais de la couche 1");
    fc1_bias.resize(size_t(SETS) * H1);
    read_values(in, fc1_bias.size(), fc1_bias.data());

    // Couche 2 : gardée comme dans le fichier, une ligne de 2058 poids par
    // neurone de sortie, la tête b occupant les lignes b · 64 … b · 64 + 63.
    expect(in, "W", HEADS * H2, FC2_IN, "la couche 2");
    fc2.resize(size_t(HEADS) * H2 * FC2_IN);
    read_values(in, fc2.size(), fc2.data());
    expect(in, "B", HEADS * H2, 0, "les biais de la couche 2");
    fc2_bias.resize(size_t(HEADS) * H2);
    read_values(in, fc2_bias.size(), fc2_bias.data());

    // Couche 3 : une matrice et un biais par tête, l'une après l'autre.
    fc3.resize(size_t(HEADS) * H3 * H2);
    fc3_bias.resize(size_t(HEADS) * H3);
    for (int b = 0; b < HEADS; b++) {
        expect(in, "W", H3, H2, "la couche 3");
        read_values(in, size_t(H3) * H2, fc3.data() + size_t(b) * H3 * H2);
        expect(in, "B", H3, 0, "les biais de la couche 3");
        read_values(in, H3, fc3_bias.data() + b * H3);
    }

    // Couche 4 (float) : une ligne de 32 poids par tête, un biais par tête.
    expect(in, "W", HEADS, H3, "la couche 4");
    fc4.resize(size_t(HEADS) * H3);
    read_values(in, fc4.size(), fc4.data());
    expect(in, "B", HEADS, 0, "les biais de la couche 4");
    fc4_bias.resize(HEADS);
    read_values(in, fc4_bias.size(), fc4_bias.data());
    if (!in) throw std::runtime_error("final_nnue: fichier tronqué " + filename);
    // Le vrai format permet encore une section « DIRECT » (un chemin direct des
    // entrées d'espace vers la sortie) : essayée en campagne P6, inutile, et
    // absente de ce fichier. On vérifie qu'il n'y a plus rien.
    if (std::string rest; in >> rest)
        throw std::runtime_error("final_nnue: section inattendue « " + rest + " » dans " + filename);
}

// =============================================================================
//  2. LES ENTRÉES : LIGNES, POINTS DE VUE, PHASES
// =============================================================================
//
// La ligne de poids d'une case `sq` pour un point de vue (Network::row, dans
// final_nnue.h) :
//     point de vue NOIR  : 63 − sq
//     point de vue BLANC : (63 − sq) ^ 7
// • « 63 − sq » : l'ordre des cases du cache d'entraînement, qui écrit le
//   bitboard du bit de poids fort au bit de poids faible (h8 d'abord). Le
//   réseau a appris dans cet ordre : il faut le respecter.
// • « ^ 7 » : changer les 3 bits bas d'un numéro de case = changer sa colonne
//   c en 7 − c, le miroir a ↔ h. Le point de vue blanc est ce miroir.
//   Exemple : c1 = case 2. Point de vue noir : ligne 61. Point de vue blanc :
//   61 ^ 7 = 58, la ligne de f1 pour Noir (f1 = case 5, 63 − 5 = 58).
// Une entrée est (plan, case) : plan 0 = mes pièces, 64 = ses pièces, 128 =
// trous. Sa ligne de poids dans un jeu de la couche 1 : (plan + row) × 1024.
//
// Les phases (voir final_nnue.h) : `holes` = le nombre de trous, 0 à 56.
//   jeu de la couche 1 = holes · 4 / 57 : 0–14 → 0, 15–28 → 1, 29–42 → 2, 43–56 → 3
//   tête des couches 2–4 = holes · 8 / 57 : 0–7 → 0, 8–14 → 1, …, 50–56 → 7
// (le `min` ne sert que si une position avait plus de 56 trous, impossible en
// jeu : il garde l'indice dans le tableau quoi qu'il arrive.)
int Network::set_of(uint64_t holes) const {
    return std::min(SETS - 1, std::popcount(holes) * SETS / (MAX_HOLES + 1));
}

int Network::head_of(uint64_t holes) const {
    return std::min(HEADS - 1, std::popcount(holes) * HEADS / (MAX_HOLES + 1));
}

// =============================================================================
//  3. L'ACCUMULATEUR
// =============================================================================
//
// Un point de vue de l'accumulateur = les 1024 sommes
//     biais[j] + Σ (sur les entrées à 1) poids[entrée][j]
// du jeu de poids de la position. Les sommes sont des int16 et peuvent en
// théorie déborder ; elles le font « modulo 2^16 », comme le vrai code. Une
// addition modulo 2^16 ne dépend pas de l'ordre des opérations : un
// accumulateur mis à jour coup par coup est donc EXACTEMENT celui recalculé
// de zéro (test/final_nnue_check le vérifie).

// Calcul de zéro : le biais, puis les lignes de chaque pièce et de chaque trou,
// pour les deux points de vue. Range aussi la position.
void Network::init_from(uint64_t black, uint64_t white, uint64_t holes, Accumulator& acc) const {
    const int set = set_of(holes);
    const int8_t* w1 = fc1.data() + size_t(set) * INPUTS * H1;
    const int16_t* b1 = fc1_bias.data() + size_t(set) * H1;
    const uint64_t pieces[2] = { black, white };    // indice = la couleur (Yolah::BLACK = 0)
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        int16_t* a = acc.view[v];
        for (int j = 0; j < H1; j++) a[j] = b1[j];
        auto add = [&](uint64_t bb, int plane) {   // ajoute la ligne de chaque case de bb
            for (; bb; bb &= bb - 1) {
                const int8_t* w = w1 + size_t(plane + row(v, std::countr_zero(bb))) * H1;
                for (int j = 0; j < H1; j++) a[j] = int16_t(a[j] + w[j]);
            }
        };
        add(pieces[v], 0);          // mes pièces : celles de la couleur du point de vue
        add(pieces[1 - v], 64);     // ses pièces
        add(holes, 128);            // les trous (les mêmes pour les deux points de vue)
    }
    acc.black = black;
    acc.white = white;
    acc.holes = holes;
}

void Network::init(const Yolah& yolah, Accumulator& acc) const {
    init_from(yolah.bitboard(Yolah::BLACK), yolah.bitboard(Yolah::WHITE), yolah.empty_bitboard(), acc);
}

// Rafraîchissement : l'accumulateur de la position (black, white, holes) dans
// SON jeu de poids, quand un coup vient de faire changer de jeu.
//   • sans cache : calcul de zéro ;
//   • avec cache, première fois pour ce jeu : calcul de zéro, gardé dans le
//     cache ;
//   • sinon : on part de l'accumulateur gardé pour ce jeu et on n'applique que
//     les DIFFÉRENCES de position : pour chaque plan (noires, blanches, trous),
//     on retire la ligne des cases qui ne sont plus occupées et on ajoute
//     celle des nouvelles. Dans une même recherche, les positions qui
//     changent de jeu se ressemblent : quelques cases au lieu de 60 à 120.
// Puis on recopie le résultat dans `out`.
void Network::refresh(uint64_t black, uint64_t white, uint64_t holes, Accumulator& out, RefreshCache* cache) const {
    if (!cache) {
        init_from(black, white, holes, out);
        return;
    }
    const int set = set_of(holes);
    Accumulator& e = cache->entry[set];
    if (!cache->valid[set]) {
        init_from(black, white, holes, e);
        cache->valid[set] = true;
    } else {
        const int8_t* w1 = fc1.data() + size_t(set) * INPUTS * H1;
        const uint64_t old_pos[3] = { e.black, e.white, e.holes };
        const uint64_t new_pos[3] = { black, white, holes };
        for (int v : { Yolah::BLACK, Yolah::WHITE }) {
            int16_t* a = e.view[v];
            // le plan de chaque bitboard pour ce point de vue : les pièces de la
            // couleur du point de vue sont « mes pièces » (0), les autres « ses
            // pièces » (64) ; les trous, 128
            const int plane[3] = { v == Yolah::BLACK ? 0 : 64, v == Yolah::BLACK ? 64 : 0, 128 };
            for (int c = 0; c < 3; c++) {
                for (uint64_t bb = old_pos[c] & ~new_pos[c]; bb; bb &= bb - 1) {      // disparues
                    const int8_t* w = w1 + size_t(plane[c] + row(v, std::countr_zero(bb))) * H1;
                    for (int j = 0; j < H1; j++) a[j] = int16_t(a[j] - w[j]);
                }
                for (uint64_t bb = new_pos[c] & ~old_pos[c]; bb; bb &= bb - 1) {      // apparues
                    const int8_t* w = w1 + size_t(plane[c] + row(v, std::countr_zero(bb))) * H1;
                    for (int j = 0; j < H1; j++) a[j] = int16_t(a[j] + w[j]);
                }
            }
        }
        e.black = black;
        e.white = white;
        e.holes = holes;
    }
    out = e;
}

// Mise à jour d'un coup : out = in + le coup `m` de `player`.
//   • Une passe ne change aucune entrée : out = in. (Elle change le camp au
//     trait, donc le point de vue que `value` lira « nous », mais
//     l'accumulateur garde toujours les deux.)
//   • Sinon, la nouvelle position : la pièce va de `from` à `to`, `from`
//     devient un trou. Si le nombre de trous fait changer de jeu de poids,
//     toutes les sommes changent : rafraîchissement.
//   • Sinon, dans chaque point de vue, 3 lignes de poids :
//         − la pièce sur `from`   (plan « mes pièces » si le point de vue est
//         + la pièce sur `to`      celui du joueur, « ses pièces » sinon)
//         + le trou sur `from`    (plan des trous)
//     soit 3 × 1024 additions par point de vue, au lieu de recalculer
//     192 × 1024 produits.
void Network::update(const Accumulator& in, Accumulator& out, uint8_t player, Move m, RefreshCache* cache) const {
    if (m == Move::none()) {
        out = in;
        return;
    }
    const uint64_t from = uint64_t(1) << m.from_sq(), to = uint64_t(1) << m.to_sq();
    uint64_t black = in.black, white = in.white;
    (player == Yolah::BLACK ? black : white) ^= from | to;   // la pièce quitte from, arrive sur to
    const uint64_t holes = in.holes | from;
    const int set = set_of(holes);
    if (set != set_of(in.holes)) {
        refresh(black, white, holes, out, cache);
        return;
    }
    const int8_t* w1 = fc1.data() + size_t(set) * INPUTS * H1;
    for (int v : { Yolah::BLACK, Yolah::WHITE }) {
        const int plane = v == player ? 0 : 64;
        const int8_t* w_from = w1 + size_t(plane + row(v, m.from_sq())) * H1;
        const int8_t* w_to   = w1 + size_t(plane + row(v, m.to_sq())) * H1;
        const int8_t* w_hole = w1 + size_t(128 + row(v, m.from_sq())) * H1;
        const int16_t* a = in.view[v];
        int16_t* b = out.view[v];
        for (int j = 0; j < H1; j++) b[j] = int16_t(a[j] - w_from[j] + w_to[j] + w_hole[j]);
    }
    out.black = black;
    out.white = white;
    out.holes = holes;
}

// =============================================================================
//  4. LES ENTRÉES D'ESPACE
// =============================================================================
//
// 10 nombres calculés à chaque évaluation, « nous » = le camp au trait, « eux »
// = l'autre. Chacun est un entier borné dans [0, 64], donné au réseau tel quel
// dans h1 : à l'échelle 64, l'entier n représente le réel n / 64, exactement
// ce que le réseau a vu à l'entraînement (aucune erreur de quantification).
// Les colonnes, dans la numérotation de nnue/space_features.h (0–7 : nous,
// 8–15 : eux, 16–17 : influence) :
//     1, 9    Q1      cases libres atteintes en 1 coup de dame
//     2, 10   Q2      … en au plus 2 coups de dame
//     3, 11   Q3      … en au plus 3 coups de dame
//     4, 12   QFIRST  cases libres atteintes en STRICTEMENT moins de coups de
//                     dame que l'adversaire
//     16, 17  INFL    influence : les cases libres que chaque camp possède quand
//                     les deux s'étendent ensemble d'un pas de roi à la fois
// (Les autres colonnes du vrai code, essayées pendant l'ablation, ne sont pas
// lues par ce réseau : elles améliorent la précision mais coûtent trop de
// temps par évaluation pour gagner des parties.)
// Un « coup de dame » glisse à travers les cases LIBRES (ni pièce ni trou) :
// c'est le déplacement des pièces du jeu, sans tenir compte du tour de jeu.

// Les cases libres atteintes en un coup de dame depuis les cases de `from`.
// attacks_bb(s, occupées) = les cases que voit une dame en s, case bloquante
// de chaque rayon comprise ; « & free » la retire et garde les cases libres.
static uint64_t queen_reach(uint64_t from, uint64_t free) {
    uint64_t r = 0;
    for (uint64_t b = from; b; b &= b - 1) r |= attacks_bb(Square(std::countr_zero(b)), ~free);
    return r & free;
}

// levels[k] = les cases libres à exactement k + 1 coups de dame des pièces
// `pieces` (parcours en largeur : chaque niveau part du précédent et ne garde
// que les cases jamais vues). Rend le nombre de niveaux.
static int queen_levels(uint64_t pieces, uint64_t free, uint64_t levels[64]) {
    int n = 0;
    uint64_t seen = 0, frontier = pieces;
    for (;;) {
        const uint64_t next = queen_reach(frontier, free) & ~seen;
        if (!next) return n;
        levels[n++] = next;
        seen |= next;
        frontier = next;
    }
}

// Influence (comme heuristic::influence du joueur) : les deux camps s'étendent
// d'un pas de roi (les 8 cases voisines) à la fois, à travers les cases
// libres. Une case atteinte par les deux au même pas est NEUTRE, et les cases
// neutres s'étendent aussi, en bloquant les deux camps. On s'arrête quand plus
// rien ne bouge ; chaque camp possède alors les cases libres qu'il a atteintes.
// Exemple : deux pièces face à face sur une rangée vide, à 4 cases l'une de
// l'autre : chacune prend les cases de son côté ; celle du milieu, atteinte par
// les deux au 2e pas, est neutre et coupe le passage.
static void influence(uint64_t us, uint64_t them, uint64_t free, int out[2]) {
    auto step = [&](uint64_t b) { return shift_all_directions(b) & free; };
    uint64_t own[2] = { us, them }, front[2] = { us, them }, neutral = 0, prev[2] = { 0, 0 };
    while (prev[0] != own[0] || prev[1] != own[1]) {
        front[0] = step(front[0]) & ~own[1];
        front[1] = step(front[1]) & ~own[0];
        neutral |= step(neutral) | (front[0] & front[1]);
        front[0] &= ~neutral;
        front[1] &= ~neutral;
        prev[0] = own[0];
        prev[1] = own[1];
        own[0] |= front[0];
        own[1] |= front[1];
    }
    out[0] = std::popcount(own[0] & free);
    out[1] = std::popcount(own[1] & free);
}

void Network::space_inputs(const Yolah& yolah, int8_t out[SPACE]) const {
    const uint8_t stm = yolah.current_player();
    const uint64_t us = yolah.bitboard(stm), them = yolah.bitboard(Yolah::other_player(stm));
    const uint64_t free = yolah.free_squares();
    int f[18] = {};                   // f[colonne] (numérotation de space_features.h)
    uint64_t lv[2][64];
    const int nl[2] = { queen_levels(us, free, lv[0]), queen_levels(them, free, lv[1]) };
    for (int side = 0; side < 2; side++) {
        // Q1, Q2, Q3 : cases couvertes après 1, 2, 3 niveaux. S'il y a moins de
        // 3 niveaux, les suivants valent le dernier (plus rien de nouveau) ;
        // sans aucun niveau, 0.
        int q[3] = { 0, 0, 0 };
        uint64_t cover = 0;
        for (int k = 0; k < nl[side] && k < 3; k++) {
            cover |= lv[side][k];
            q[k] = std::popcount(cover);
        }
        for (int k = std::max(nl[side], 1); k < 3; k++) q[k] = q[k - 1];
        for (int k = 0; k < 3; k++) f[side * 8 + 1 + k] = q[k];
    }
    // QFIRST : niveau par niveau, une case du niveau k de « nous » compte si
    // elle n'est pas au même niveau chez « eux » et n'a pas été atteinte plus
    // tôt par eux ; et réciproquement.
    {
        uint64_t seen0 = 0, seen1 = 0;
        int first0 = 0, first1 = 0;
        for (int k = 0; k < std::max(nl[0], nl[1]); k++) {
            const uint64_t a = k < nl[0] ? lv[0][k] : 0, b = k < nl[1] ? lv[1][k] : 0;
            first0 += std::popcount(a & ~b & ~seen1);
            first1 += std::popcount(b & ~a & ~seen0);
            seen0 |= a;
            seen1 |= b;
        }
        f[4] = first0;
        f[12] = first1;
    }
    int infl[2];
    influence(us, them, free, infl);
    f[16] = infl[0];
    f[17] = infl[1];
    // Bornage dans [0, 64] (ces comptes de cases libres ne dépassent jamais 56,
    // mais la règle est celle de toutes les entrées d'espace), dans l'ordre des
    // colonnes du fichier.
    for (int j = 0; j < SPACE; j++) out[j] = int8_t(std::clamp(f[space_cols[j]], 0, SCALE));
}

// =============================================================================
//  5. LA VALEUR
// =============================================================================
//
// h1 (2058 int8 dans [0, 64]) = bornage du point de vue « nous » (1024), du
// point de vue « eux » (1024), puis les 10 entrées d'espace. La tête b est
// choisie par le nombre de trous.
//
// Couche 2, et pourquoi elle est « creuse ». Chaque neurone de sortie o :
//     h2[o] = borne( (Σ_i fc2[b][o][i] · h1[i] + biais[o]) >> 6 )
// Or h1 est surtout fait de ZÉROS : le bornage met à 0 toutes les sommes
// négatives de l'accumulateur (avec l'ancien réseau, 87 % des valeurs). Un
// zéro n'ajoute rien à aucune somme : il suffit de parcourir les entrées NON
// nulles et d'ajouter, pour chacune, sa colonne de 64 poids aux 64 sommes.
// Le vrai code (nnue/sparse_matvec.h) fait cela par groupes de 4 entrées
// (l'unité des instructions AVX2 : 4 octets par case de registre) et ne saute
// que les groupes entièrement nuls ; ici on saute chaque zéro, ce qui revient
// au même : des sommes d'entiers, dans un autre ordre, avec les mêmes termes
// non nuls. Gain mesuré : −34 % de temps par évaluation, +35 Elo à 1 s.
//
// « borne(x >> 6) » : la somme est à l'échelle 64² (poids × activation) ; le
// décalage arithmétique de 6 bits la ramène à l'échelle 64 en arrondissant
// vers −∞ (comme l'instruction du vrai code), puis le bornage la met dans
// [0, 64].
//
// Couche 3 : la même chose, dense (64 entrées, rarement nulles), 64 → 32.
// Couche 4 : en float, s = Σ fc4[b][i] · h3[i] ; h3 étant à l'échelle 64, on
// divise s par 64, on ajoute le biais, et tanh ramène le tout dans ]−1, 1[.
float Network::value(const Accumulator& acc, const Yolah& yolah) const {
    const int stm = yolah.current_player();
    const int b = head_of(yolah.empty_bitboard());   // la tête : les trous de la position évaluée
    int8_t h1[FC2_IN];
    for (int i = 0; i < H1; i++) h1[i] = int8_t(std::clamp<int>(acc.view[stm][i], 0, SCALE));            // nous
    for (int i = 0; i < H1; i++) h1[H1 + i] = int8_t(std::clamp<int>(acc.view[1 - stm][i], 0, SCALE));   // eux
    space_inputs(yolah, h1 + 2 * H1);

    // Couche 2 (creuse) : on accumule les colonnes des entrées non nulles.
    int32_t sum2[H2];
    const int16_t* bias2 = fc2_bias.data() + b * H2;
    const int8_t* w2 = fc2.data() + size_t(b) * H2 * FC2_IN;
    for (int o = 0; o < H2; o++) sum2[o] = 0;
    for (int i = 0; i < FC2_IN; i++) {
        if (h1[i] == 0) continue;                       // un zéro n'ajoute rien
        for (int o = 0; o < H2; o++) sum2[o] += int32_t(w2[size_t(o) * FC2_IN + i]) * h1[i];
    }
    int8_t h2[H2];
    for (int o = 0; o < H2; o++) h2[o] = int8_t(std::clamp((sum2[o] + bias2[o]) >> SHIFT, 0, SCALE));

    // Couche 3 (dense).
    int8_t h3[H3];
    const int8_t* w3 = fc3.data() + size_t(b) * H3 * H2;
    const int16_t* bias3 = fc3_bias.data() + b * H3;
    for (int o = 0; o < H3; o++) {
        int32_t s = 0;
        for (int i = 0; i < H2; i++) s += int32_t(w3[o * H2 + i]) * h2[i];
        h3[o] = int8_t(std::clamp((s + bias3[o]) >> SHIFT, 0, SCALE));
    }

    // Couche 4 (float) et tanh.
    const float* w4 = fc4.data() + b * H3;
    float s = 0;
    for (int i = 0; i < H3; i++) s += w4[i] * h3[i];
    return std::tanh(s / SCALE + fc4_bias[b]);
}

} // namespace final_nnue
