// =============================================================================
//  final_retro.cpp — les régions privées et leur table, commentées
// =============================================================================
//  Lire d'abord doc/final_retro.h (pourquoi des régions, la grappe, la valeur,
//  la table à rebours). Plan de ce fichier :
//    1. les bitboards      voisines, symétries, coin
//    2. canonical          la forme canonique (8 symétries + translation)
//    3. regions, clusters  les régions, les grappes (union-find)
//    4. split              les grappes privées d'une position
//    5. Solver             la valeur d'une grappe (recherche avec mémoire)
//    6. build_table        la table, couche par couche
// =============================================================================

#include "final_retro.h"
#include "magic.h"
#include <algorithm>
#include <bit>
#include <unordered_set>

namespace final_retro {

// ── 1. Les bitboards ─────────────────────────────────────────────────────────
// Une case = un bit (a1 = bit 0, h1 = bit 7, a2 = bit 8, …, h8 = bit 63).
namespace {
constexpr uint64_t NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;

// b et ses 8 voisines. Décaler d'un bit vers la gauche (<< 1) avance d'une
// colonne vers h ; les cases qui débordent de la colonne h arriveraient en
// colonne a de la rangée suivante : le masque NOT_A les efface (et NOT_H pour
// l'autre sens). La ligne obtenue est ensuite décalée d'une rangée (<< 8, >> 8).
inline uint64_t grow(uint64_t b) {
    const uint64_t row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
    return row | (row << 8) | (row >> 8);
}

// Les 8 symétries du carré s'obtiennent en combinant trois réflexions :
//   flip_vertical      rangée 1 ↔ rangée 8 : les 8 octets dans l'ordre inverse
//                      (un octet = une rangée) ;
//   mirror_horizontal  colonne a ↔ colonne h : les 8 bits de chaque octet dans
//                      l'ordre inverse (échanges de 1, puis 2, puis 4 bits) ;
//   flip_diagonal      la diagonale a1-h8 : colonne ↔ rangée (transposition par
//                      trois échanges de blocs, la méthode classique des
//                      programmes d'échecs).
inline uint64_t flip_vertical(uint64_t b) { return __builtin_bswap64(b); }
inline uint64_t mirror_horizontal(uint64_t b) {
    constexpr uint64_t k1 = 0x5555555555555555ULL, k2 = 0x3333333333333333ULL, k4 = 0x0f0f0f0f0f0f0f0fULL;
    b = ((b >> 1) & k1) | ((b & k1) << 1);
    b = ((b >> 2) & k2) | ((b & k2) << 2);
    b = ((b >> 4) & k4) | ((b & k4) << 4);
    return b;
}
inline uint64_t flip_diagonal(uint64_t b) {
    uint64_t t;
    constexpr uint64_t k1 = 0x5500550055005500ULL, k2 = 0x3333000033330000ULL, k4 = 0x0f0f0f0f00000000ULL;
    t = k4 & (b ^ (b << 28)); b ^= t ^ (t >> 28);
    t = k2 & (b ^ (b << 14)); b ^= t ^ (t >> 14);
    t = k1 & (b ^ (b << 7));  b ^= t ^ (t >> 7);
    return b;
}

// La forme ramenée dans le coin a1 : décalée vers le bas jusqu'à toucher la
// rangée 1, puis vers la gauche jusqu'à toucher la colonne a. La rangée la plus
// basse : celle du bit le plus faible (countr_zero / 8). La colonne la plus à
// gauche : on superpose les 8 rangées en un octet (OU des décalages de 32, 16,
// 8 bits) et on prend son bit le plus faible. Décaler à droite de `file` bits
// recule chaque case de `file` colonnes ; aucune ne passe à la rangée d'avant
// puisqu'aucune n'est à gauche de `file`.
inline void to_corner(uint64_t& f, uint64_t& p) {
    const uint64_t all = f | p;
    if (!all) return;
    const int rank = std::countr_zero(all) / 8;
    f >>= 8 * rank;
    p >>= 8 * rank;
    uint64_t cols = (f | p);
    cols |= cols >> 32; cols |= cols >> 16; cols |= cols >> 8;
    const int file = std::countr_zero(cols & 0xFF);
    f >>= file;
    p >>= file;
}
}  // namespace

// ── 2. La forme canonique ────────────────────────────────────────────────────
// On parcourt les 8 symétries : la boucle intérieure applique deux fois la
// réflexion horizontale (rien, puis réfléchie), la boucle du milieu deux fois
// la verticale, la boucle extérieure deux fois la diagonale. Chaque
// combinaison est ramenée au coin et comparée (cases libres d'abord, puis
// pièces) : la plus petite est la forme canonique. Les cases libres et les
// pièces subissent toujours la même transformation.
Key canonical(uint64_t free, uint64_t pieces) {
    Key best{~uint64_t(0), ~uint64_t(0)};
    uint64_t f = free, p = pieces;
    for (int d = 0; d < 2; d++) {
        for (int v = 0; v < 2; v++) {
            for (int h = 0; h < 2; h++) {
                uint64_t ff = f, pp = p;
                to_corner(ff, pp);
                if (ff < best.free || (ff == best.free && pp < best.pieces)) best = {ff, pp};
                f = mirror_horizontal(f);
                p = mirror_horizontal(p);
            }
            f = flip_vertical(f);
            p = flip_vertical(p);
        }
        f = flip_diagonal(f);
        p = flip_diagonal(p);
    }
    return best;
}

// ── 3. Les régions et les grappes ────────────────────────────────────────────
// Les régions : on prend la case libre la plus basse, on l'« inonde » (grow
// restreint aux cases libres, jusqu'à ce que plus rien ne change), on retire
// la région obtenue, et on recommence.
int regions(uint64_t free, uint64_t out[64]) {
    int n = 0;
    while (free) {
        uint64_t r = free & -free, prev = 0;   // free & -free : le bit le plus faible
        while (r != prev) { prev = r; r = grow(r) & free; }
        out[n++] = r;
        free &= ~r;
    }
    return n;
}

// Les grappes : les régions reliées par une pièce qui en touche plusieurs.
// Union-find : chaque région pointe vers un « parent » ; deux régions sont dans
// la même grappe si elles mènent à la même racine (find, avec raccourcissement
// des chemins). Pour chaque pièce, on réunit toutes les régions qu'elle touche.
// Les grappes sans pièce sont abandonnées (personne n'y jouera : valeur 0).
int clusters(uint64_t free, uint64_t pieces, Cluster out[64]) {
    uint64_t reg[64];
    const int nr = regions(free, reg);
    int parent[64];
    for (int i = 0; i < nr; i++) parent[i] = i;
    auto find = [&](int i) { while (parent[i] != i) i = parent[i] = parent[parent[i]]; return i; };
    uint64_t touch[64];
    for (int i = 0; i < nr; i++) touch[i] = grow(reg[i]);   // une région et son pourtour
    for (uint64_t ps = pieces; ps; ps &= ps - 1) {
        const uint64_t bit = ps & -ps;
        int first = -1;
        for (int i = 0; i < nr; i++)
            if (touch[i] & bit) {
                if (first < 0) first = i;
                else parent[find(i)] = find(first);
            }
    }
    int n = 0, index[64];
    for (int i = 0; i < nr; i++) index[i] = -1;
    for (int i = 0; i < nr; i++) {
        const int r = find(i);
        if (index[r] < 0) { index[r] = n; out[n++] = Cluster{}; }
        out[index[r]].free |= reg[i];
    }
    int m = 0;
    for (int c = 0; c < n; c++) {
        out[c].pieces = grow(out[c].free) & pieces;   // les pièces qui la touchent
        if (out[c].pieces) out[m++] = out[c];
    }
    return m;
}

// ── 4. Les grappes privées d'une position ────────────────────────────────────
// Chaque région est classée : touchée par les deux camps (disputée), par un
// seul (privée), par personne (perdue). Une pièce qui touche à la fois une
// région privée de son camp et une région disputée relie les deux : la région
// privée n'est alors pas indépendante (la pièce peut aller de l'un ou l'autre
// côté), et sa grappe est laissée de côté.
Split split(const Yolah& y) {
    Split s;
    uint64_t reg[64];
    const uint64_t free = y.free_squares();
    const int nr = regions(free, reg);
    uint64_t priv_free[2] = {0, 0};
    for (int i = 0; i < nr; i++) {
        const uint64_t t = grow(reg[i]);
        const bool b = t & y.bitboard(Yolah::BLACK), w = t & y.bitboard(Yolah::WHITE);
        if (b && w) s.contested |= reg[i];
        else if (b) priv_free[0] |= reg[i];
        else if (w) priv_free[1] |= reg[i];
    }
    for (int side = 0; side < 2; side++) {
        Cluster c[64];
        const int n = clusters(priv_free[side], y.bitboard(side), c);
        for (int i = 0; i < n; i++)
            if (!(grow(s.contested) & c[i].pieces)) s.priv[side].push_back(c[i]);
    }
    return s;
}

// ── 5. La valeur d'une grappe ────────────────────────────────────────────────
// value : la somme des valeurs des grappes (elles sont indépendantes).
int Solver::value(uint64_t free, uint64_t pieces) {
    Cluster c[64];
    const int n = clusters(free, pieces, c);
    int v = 0;
    for (int i = 0; i < n; i++) v += cluster_value(c[i].free, c[i].pieces);
    return v;
}

// cluster_value : la mémoire d'abord (par la forme canonique), sinon tous les
// coups. attacks_bb(from, ~free) donne les cases qu'une dame en `from` atteint
// en glissant, avec comme obstacles tout ce qui n'est pas libre ; « & free »
// garde les destinations libres. Après le coup, la destination n'est plus
// libre et la case de départ devient un trou : il reste free sans `to`, et les
// pièces avec `to` au lieu de `from`. L'arrêt anticipé : `best < most` dans les
// deux boucles (most = le nombre de cases libres, le plafond).
int Solver::cluster_value(uint64_t free, uint64_t pieces) {
    const Key k = canonical(free, pieces);
    if (auto it = memo.find(k); it != memo.end()) return it->second;
    const uint64_t occupied = ~free;
    const int most = std::popcount(free);
    int best = 0;
    for (uint64_t ps = pieces; ps && best < most; ps &= ps - 1) {
        const Square from = Square(std::countr_zero(ps));
        const uint64_t from_bit = ps & -ps;
        for (uint64_t to = attacks_bb(from, occupied) & free; to && best < most; to &= to - 1) {
            const uint64_t to_bit = to & -to;
            best = std::max(best, 1 + value(free & ~to_bit, (pieces & ~from_bit) | to_bit));
        }
    }
    memo.emplace(k, int8_t(best));
    return best;
}

// ── 6. La table, couche par couche ───────────────────────────────────────────
namespace {
// Largeur et hauteur de la boîte englobante d'une forme (pour savoir si elle
// peut encore être décalée d'une colonne ou d'une rangée sans sortir du plateau).
int width(uint64_t b) {
    uint64_t c = b; c |= c >> 32; c |= c >> 16; c |= c >> 8;
    c &= 0xFF;
    return c ? 64 - std::countl_zero(c) - std::countr_zero(c) : 0;
}
int height(uint64_t b) { return b ? (63 - std::countl_zero(b)) / 8 - std::countr_zero(b) / 8 + 1 : 0; }
// Une forme du coin, et ses placements à une colonne et/ou une rangée du coin
// (là où elle tient) : les pièces autour d'elle doivent pouvoir être sur le
// plateau. Dans une vraie position, la forme et ses pièces tiennent dans le
// plateau ; ramenées au coin de leur boîte commune, les cases libres sont au
// plus à une colonne et une rangée du coin (les pièces sont voisines) : ces
// quatre placements suffisent.
std::vector<uint64_t> placements(uint64_t s) {
    std::vector<uint64_t> v{s};
    const bool right = width(s) < 8, up = height(s) < 8;
    if (right) v.push_back(s << 1);
    if (up) v.push_back(s << 8);
    if (right && up) v.push_back(s << 9);
    return v;
}
}  // namespace

// Couche n : chaque forme de n cases, à chaque placement, avec 1 ou 2 pièces
// sur les cases autour ; dédupliquée par la forme canonique. Puis les valeurs,
// dans l'ordre des couches : le solveur trouve en mémoire les états des
// couches plus basses (un coup remplit une case), sauf les grappes de deux
// régions reliées par une pièce, qu'il calcule et garde au passage. Puis les
// formes de la couche suivante : chaque forme plus une case voisine.
Table build_table(int max_free, int max_pieces) {
    Table t;
    Solver solver;
    t.layer_size.assign(max_free + 1, 0);
    std::vector<uint64_t> shapes = {1};   // couche 1 : une case
    for (int n = 1; n <= max_free; n++) {
        std::unordered_set<Key, KeyHash> states;
        for (uint64_t sh : shapes)
            for (uint64_t f : placements(sh)) {
                std::vector<uint64_t> ring;
                for (uint64_t r = grow(f) & ~f; r; r &= r - 1) ring.push_back(r & -r);
                const int m = int(ring.size());
                for (int a = 0; a < m; a++) {
                    states.insert(canonical(f, ring[a]));
                    if (max_pieces >= 2)
                        for (int b = a + 1; b < m; b++) states.insert(canonical(f, ring[a] | ring[b]));
                }
            }
        std::vector<Key> layer(states.begin(), states.end());
        std::sort(layer.begin(), layer.end(), [](const Key& x, const Key& y) { return x.free != y.free ? x.free < y.free : x.pieces < y.pieces; });
        for (const Key& k : layer) {
            t.keys.push_back(k);
            t.values.push_back(int8_t(solver.value(k.free, k.pieces)));
        }
        t.layer_size[n] = layer.size();
        if (n == max_free) break;
        std::unordered_set<uint64_t> next;
        for (uint64_t sh : shapes)
            for (uint64_t f : placements(sh))
                for (uint64_t add = grow(f) & ~f; add; add &= add - 1) next.insert(canonical(f | (add & -add), 0).free);
        shapes.assign(next.begin(), next.end());
    }
    return t;
}

}  // namespace final_retro
