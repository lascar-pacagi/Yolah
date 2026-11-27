#ifndef FINAL_RETRO_H
#define FINAL_RETRO_H

// =============================================================================
//  FINAL_RETRO — L'analyse rétrograde des régions privées, commentée
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (avec doc/final_retro.cpp) contient le code du chantier
//  « analyse rétrograde » : la valeur EXACTE des régions de cases libres qu'un
//  seul camp peut encore atteindre, et la table de toutes ces régions jusqu'à
//  9 cases, construite à rebours, couche par couche. Le vrai code est dans :
//    player/solo_region.{h,cpp}   les grappes, la forme canonique, le solveur
//    test/solo_retro_main.cpp     la table, sur CPU
//    test/solo_retro_gpu.cu       la même table sur GPU (doc/final_retro_gpu.cu
//                                 en est la version commentée)
//  et il sert à trois endroits :
//    player/endgame_solver.cpp    option `regions` : le solveur exact (« Vérifier »
//                                 de l'éditeur) ne cherche plus dans les régions
//                                 privées : 2,5 fois plus vite sur un thread,
//                                 4,4 fois sur huit
//    player/minmax_nnue_dev_player.cpp   option « solo » (campagne U29)
//    player/alphazero_mcts.cpp    option « regions » (az_regions_match.sh)
//
//  ---------------------------------------------------------------------------
//  POURQUOI DES RÉGIONS, ET PAS DES POSITIONS ENTIÈRES
//  ---------------------------------------------------------------------------
//  Aux échecs, une table de finales est petite parce qu'il reste peu de pièces.
//  À Yolah, il reste toujours 8 pièces : ce qui diminue, c'est l'espace libre,
//  et les trous qui le bornent peuvent être n'importe où. Une table de toutes
//  les positions à 4 cases libres aurait déjà environ 1,4 × 10^16 entrées
//  (doc/solving.qmd). Impossible.
//
//  Mais les fins de partie se découpent. Un coup de dame ne traverse que des
//  cases libres, et deux cases consécutives d'une ligne sont voisines (8
//  directions) : une pièce ne va jamais que dans les RÉGIONS (groupes de cases
//  libres reliées par des pas de roi) qu'elle touche, et deux régions ne se
//  rejoignent jamais (les cases ne font que cesser d'être libres).
//
//      . . # # . .      Les cases libres « . » forment ici deux régions :
//      . N # # . .      celle de gauche, que seule la pièce noire N touche,
//      # # # # B .      est PRIVÉE : Blanc ne pourra jamais y entrer. Sa
//                       valeur pour Noir : le plus grand nombre de coups que
//                       N peut encore y jouer (ici 3, en remplissant tout).
//
//  Une région PRIVÉE (touchée par les pièces d'un seul camp) le reste pour
//  toujours : une pièce adverse ne peut aller que dans les cases libres de ses
//  propres régions, et une case voisine de la nôtre appartiendrait à la nôtre.
//
//  Mesuré sur 12 000 parties réelles : des régions privées apparaissent vers
//  35 cases libres ; à 20 cases libres, 44 % de l'espace libre est privé ; à
//  14, la moitié des positions sont ENTIÈREMENT privées (plus rien n'est
//  disputé : le résultat est joué). Et elles sont petites : 97 % ont au plus 6
//  cases, 99,8 % au plus 10 ; 85 % ne touchent qu'une pièce, 13 % deux.
//
//  ---------------------------------------------------------------------------
//  LA GRAPPE : L'UNITÉ VRAIMENT INDÉPENDANTE
//  ---------------------------------------------------------------------------
//  Une pièce peut toucher deux régions : elle ira dans l'une OU dans l'autre,
//  pas dans les deux. Ces régions ne sont donc pas indépendantes. On appelle
//  GRAPPE un ensemble de régions reliées par les pièces qui en touchent
//  plusieurs, avec ces pièces (calculé par union-find dans clusters()). Une
//  grappe privée est un casse-tête à UN joueur : sa VALEUR est le plus grand
//  nombre de coups que son camp peut encore y faire.
//
//  Cette valeur ne dépend ni du reste du plateau ni du MOMENT où les coups
//  sont joués (personne d'autre ne peut y toucher). Un camp dont les grappes
//  privées valent M au total possède donc M points ET M « temps » qu'il peut
//  jouer quand il veut : une réserve de coups.
//
//  Deux conséquences exactes, utilisées par les joueurs :
//    • une position où plus rien n'est disputé a une valeur connue sans
//      chercher : restant = M(camp au trait) − M(adversaire) ;
//    • dans une grappe privée du camp au trait, un seul coup mérite d'être
//      cherché : un coup qui garde la plus grande réserve (valeur − 1 après
//      lui). Deux tels coups mènent à des positions équivalentes (seule la
//      réserve compte) ; les autres coups de la grappe réduisent la réserve.
//
//  ---------------------------------------------------------------------------
//  LA VALEUR D'UNE GRAPPE
//  ---------------------------------------------------------------------------
//      valeur(grappe) = max sur les coups m de  1 + valeur(ce qui reste après m)
//      valeur(grappe sans coup) = 0
//  Après un coup, la grappe peut se scinder en plusieurs grappes : leurs
//  valeurs s'ADDITIONNENT (elles sont indépendantes). La recherche garde ses
//  résultats en mémoire.
//
//  La forme canonique. Les lignes de dame ne dépendent ni de l'endroit où se
//  trouve une forme sur le plateau, ni des 8 symétries du carré (4 rotations,
//  4 réflexions) : la mémoire est indexée par la plus petite des 8 formes
//  transformées, ramenée dans le coin a1. Une grappe et sa réflexion
//  partagent ainsi la même entrée.
//
//  L'arrêt anticipé. Chaque coup remplit une case : la valeur ne peut pas
//  dépasser le nombre de cases libres. Dès qu'une suite de coups remplit
//  TOUT, on s'arrête : impossible de faire mieux. C'est le cas le plus
//  fréquent (82 % des états de la table) : la recherche est alors très courte.
//  Mesuré : 0,2 µs par position de partie réelle pour toutes ses grappes.
//
//  ---------------------------------------------------------------------------
//  LA TABLE, À REBOURS (L'ANALYSE RÉTROGRADE)
//  ---------------------------------------------------------------------------
//  build_table() calcule TOUTES les grappes d'une seule région jusqu'à
//  `max_free` cases, avec 1 ou 2 pièces. Les états sont rangés en COUCHES par
//  nombre de cases libres. Un coup remplit une case : la valeur d'un état de la
//  couche n ne dépend que de couches plus basses. On calcule donc la couche 1,
//  puis 2, … : chaque couche trouve tout ce qu'il lui faut déjà calculé. C'est
//  le principe de l'analyse rétrograde (aux échecs : à partir des mats).
//
//  Les formes de la couche n viennent de celles de la couche n − 1 en ajoutant
//  une case voisine (les « polyplets » ; 18 770 formes de 8 cases à symétrie
//  près, comme dans la littérature), à chaque placement près du coin où elles
//  tiennent sur le plateau ; les pièces sur les cases autour.
//
//  Mesuré (CPU, un cœur) : jusqu'à 9 cases et 2 pièces, 57,6 millions d'états
//  en 3 minutes, 4,7 Go de mémoire. 82 % ont la valeur maximale : la table ne
//  garde que les EXCEPTIONS (10,3 millions), la valeur par défaut étant le
//  nombre de cases. Sur GPU (doc/final_retro_gpu.cu) : la même table, plus
//  complète (88,8 millions d'états), en 11 secondes.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_retro_check.cpp (construit par test/final_retro_check.sh)
//  compare ce code au vrai (player/solo_region.cpp) : mêmes formes canoniques,
//  mêmes grappes, mêmes valeurs sur des positions de parties réelles ; la table
//  de ce fichier aux valeurs du vrai solveur ; et, sur des positions de
//  parties réelles entièrement privées, la différence des réserves au restant
//  du solveur exact de fin de partie (le vrai test de l'idée).
// =============================================================================

#include "game.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace final_retro {

// Une grappe : ses cases libres, et les pièces (d'un seul camp) qui les touchent.
struct Cluster {
    uint64_t free = 0, pieces = 0;
};

// La forme canonique d'une grappe (deux bitboards, comparés comme une paire).
struct Key {
    uint64_t free = 0, pieces = 0;
    bool operator==(const Key& o) const { return free == o.free && pieces == o.pieces; }
};
struct KeyHash {
    size_t operator()(const Key& k) const { return size_t(k.free * 0x9E3779B97F4A7C15ull ^ (k.pieces + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full); }
};

Key canonical(uint64_t free, uint64_t pieces);
int regions(uint64_t free, uint64_t out[64]);
int clusters(uint64_t free, uint64_t pieces, Cluster out[64]);

// Pour une position : les grappes privées de chaque camp, et les cases des
// régions disputées (touchées par les deux camps).
struct Split {
    std::vector<Cluster> priv[2];
    uint64_t contested = 0;
};
Split split(const Yolah& y);

class Solver {
public:
    // Le plus grand nombre de coups de `pieces` dans `free` (toutes grappes confondues).
    int value(uint64_t free, uint64_t pieces);
    std::unordered_map<Key, int8_t, KeyHash> memo;   // forme canonique → valeur
private:
    int cluster_value(uint64_t free, uint64_t pieces);
};

// La table de toutes les grappes d'une région jusqu'à `max_free` cases, avec 1 à
// `max_pieces` pièces (1 ou 2) : état canonique → valeur.
struct Table {
    std::vector<Key> keys;           // les états, couche après couche
    std::vector<int8_t> values;
    std::vector<size_t> layer_size;  // nombre d'états de chaque couche (indice = cases libres)
};
Table build_table(int max_free, int max_pieces);

}  // namespace final_retro

#endif
