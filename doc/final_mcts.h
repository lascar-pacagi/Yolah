#ifndef FINAL_MCTS_H
#define FINAL_MCTS_H

// =============================================================================
//  FINAL_MCTS — Version épurée et commentée du joueur « MCTS ResNet » de l'app
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (avec doc/final_mcts.cpp) regroupe tout le code du joueur
//  « MCTS ResNet » de l'application Yolah, dans sa configuration réelle
//  (config/hybrid/resnet.cfg) : une recherche Monte-Carlo à la AlphaZero
//  enrichie des améliorations de KataGo, guidée par un réseau résiduel
//  (ResNet) à deux têtes (valeur + politique) sur GPU.
//
//  Dans le dépôt, ce code est réparti sur plusieurs modules :
//    player/alphazero_mcts.{h,cpp}        la recherche (az::Search)
//    player/alphazero_mcts_player.{h,cpp} le joueur et la lecture JSON
//    player/nn_cache.h                    le cache des sorties du réseau
//    nnue/nn_evaluator.h                  l'interface recherche ↔ réseau
//    nnue/resnet_torch.{h,cpp}            l'inférence GPU via libtorch
//    nnue/batched_evaluator.{h,cpp}       le regroupement des lots inter-threads
//    nnue/cnn_resnet_value_policy.py      l'architecture et l'entraînement
//    nnue/resnet_export.py                l'export PyTorch → C++
//  Tout ce qui ne sert pas à l'app (auto-apprentissage : bruit de Dirichlet,
//  tirage avec température, « forced playouts », « playout cap randomization » ;
//  mode arbre pur ; backend CPU Eigen) a été retiré pour que ces deux fichiers
//  se lisent d'un seul tenant.
//
//  Équivalence exacte : à 1 thread et nombre fixe de simulations, ce code
//  produit exactement les mêmes visites, les mêmes Q et le même coup que le
//  vrai joueur sur toutes les positions de test (voir « VÉRIFICATION »). Avec
//  plusieurs threads, aucune des deux versions n'est déterministe (l'ordre
//  d'accès des threads varie), mais le code exécuté est identique.
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF
//  ---------------------------------------------------------------------------
//  (Règles complètes au début de doc/final_dev.h.) Plateau 8 × 8, 4 pièces par
//  camp, Noir commence. Une pièce glisse comme une dame d'échecs vers une case
//  libre sans rien sauter, et sa case de départ devient un TROU définitif.
//  Chaque coup rapporte 1 point ; un camp bloqué passe (`Move::none()`, valeur
//  0 = a1 → a1) ; la partie s'arrête quand les deux camps sont bloqués, et le
//  meilleur score l'emporte.
//
//  Conséquence majeure pour la recherche : hors passe, chaque coup ajoute un
//  trou, donc une position ne revient JAMAIS (graphe acyclique, DAG), mais une
//  même position est atteinte par de nombreux ordres de coups (les 4 pièces
//  d'un camp jouent indépendamment) → beaucoup de transpositions.
//  Conventions : case = entier 0..63 (a1 = 0, b1 = 1, …, h8 = 63) ; ensemble
//  de cases = `uint64_t` (bitboard) ; `ply` = demi-coups joués ; camp au trait
//  = `ply & 1`.
//
//  ---------------------------------------------------------------------------
//  PRINCIPE DE LA RECHERCHE (MCTS « à la AlphaZero »)
//  ---------------------------------------------------------------------------
//  À l'inverse du minimax (doc/final_dev.h), qui explore largement jusqu'à une
//  profondeur fixe avec un petit réseau rapide (NNUE), le MCTS construit un
//  graphe très asymétrique — profond sur les bons coups, quasi vide ailleurs —
//  et évalue chaque nouvelle position avec un GROS réseau (~36 M paramètres,
//  ~1 000× plus coûteux par position que le NNUE) qui renvoie :
//    • une VALEUR v ∈ (−1, 1) : résultat espéré pour le camp au trait
//      (+1 victoire, −1 défaite, 0 nul) ;
//    • une POLITIQUE P(s, a) : probabilité a priori de chaque coup légal a,
//      qui indique où chercher en priorité.
//
//  Chaque simulation suit 3 étapes :
//    1. SÉLECTION : on descend depuis la racine en prenant à chaque nœud le
//       coup qui maximise PUCT = Q + U (section 3 du .cpp), jusqu'à une
//       position jamais évaluée (une FEUILLE).
//    2. ÉVALUATION : le réseau renvoie v et les a priori P de la feuille, qui
//       est alors développée (création de ses arêtes).
//    3. REMONTÉE (« backup ») : v est ajoutée à tous les nœuds du chemin, en
//       inversant son signe à chaque pli (un gain pour un camp est une perte
//       pour l'autre).
//  Il n'y a aucune partie aléatoire jusqu'au bout (« rollout ») : v la
//  remplace, sauf quand une vraie fin de partie est atteinte dans le graphe.
//
//  Ordres de grandeur (app : 1 s/coup en campagne, 16 threads × 16 feuilles,
//  RTX 3080 en fp16, mesure du 23/11/2026 campagne H2) : le GPU évalue
//  ~7 500 à 10 000 positions/s par lots de 256 ; la recherche crée ~2 500 à
//  2 900 nouvelles feuilles/s en milieu de partie — soit ~1 000× moins de
//  positions que le minimax dans le même temps.
//
//  ---------------------------------------------------------------------------
//  GLOSSAIRE
//  ---------------------------------------------------------------------------
//  Les notations s'appuient sur ce fragment de graphe (A, B, C sont des coups
//  de Noir déplaçant des pièces différentes, X un coup de Blanc ; les nombres
//  sur les flèches sont les visites de chaque arête) :
//
//                           r              Noir au trait (racine)
//                       A ╱   ╲ B
//                       ╱       ╲
//                     ▼           ▼
//                     p           q          Blanc au trait
//                   X │           │ X
//                     ▼           ▼
//                     s           t          Noir au trait
//                  C ╱ ╲ B       ╱ A
//                 6 ╱   ╲ 10    ╱ 30
//                  ▼     ╲     ╱
//                  d      ▼   ▼
//                           c                Blanc au trait
//
//      Ici : N(s, C) = 6, N(s, B) = 10, N(t, A) = 30 ; N(c) = 40, N(d) = 6.
//
//    • Nœud (`Node`) / Arête (`Edge`) : un nœud représente une position
//      (r, p, q, s, t, c, d) ; une arête représente un coup joué depuis un
//      parent (s ──B──► c), avec son a priori P(s, B) et un pointeur vers le
//      nœud fils.
//    • Graphe sans cycle (DAG) : « A, X, B » et « B, X, A » mènent à la même
//      position, donc au même nœud c (deux parents : s et t). Tous les chemins
//      partagent ainsi les statistiques de c (section 2 du .cpp).
//    • N(c) vs N(s, B) :
//        – N(c) (`Node::n`) compte toutes les visites terminées de c, quel que
//          soit le parent (ici 10 + 30 = 40).
//        – N(s, B) (`Edge::n`) ne compte que celles passées par l'arête
//          s ──B──► c (ici 10). Dans un arbre, les deux seraient égaux.
//    • W(c), Q(c), W2(c) :
//        – W(c) est la somme des valeurs remontées dans c, du point de vue du
//          camp qui a JOUÉ le coup menant à c (donc l'adversaire du camp au
//          trait dans c). En s (Noir au trait), Q(c) = W(c) / N(c) et Q(d)
//          sont donc tous deux du point de vue de Noir : Noir veut le plus
//          grand.
//        – Une remontée met à jour uniquement son propre chemin (avec
//          changement de signe à chaque pli) : une visite r → q → t → c met à
//          jour c, t, q, r, mais pas s. N(s) vaut donc 1 (son évaluation
//          initiale) + 10 + 6 = 17.
//        – W2(c) est la somme des carrés des valeurs remontées ; avec W(c),
//          elle donne la variance des valeurs (utilisée par σ̂ et par la LCB).
//    • PUCT (« Predictor + Upper Confidence bounds applied to Trees ») :
//      à chaque nœud s de la descente, on choisit le coup qui maximise :
//          score(B) = Q(c) + E · P(s, B) / (1 + N(s, B))
//                     ────   ──────────────────────────
//                     valeur      bonus d'exploration
//      Le facteur d'exploration vaut E = c · σ̂ · √(W_tot + 0.01) (section 3) :
//        – W_tot = Σ_a N(s, a), le total des visites des coups de s ;
//        – c = 1.0 + 0.45 · ln((W_tot + 500) / 500), constante d'exploration ;
//        – σ̂ = 1.0 + 0.85 · (σ(s) / 0.40 − 1), qui adapte l'exploration à la
//          dispersion σ(s) des valeurs en s (valeurs dispersées → plus d'explo).
//      Point subtil du graphe — les deux termes n'utilisent pas le même N :
//        – Q(c) agrège les 40 visites de c (`N(c)`), car la position c vaut la
//          même chose qu'on arrive par s ou par t ;
//        – le bonus divise par `1 + N(s, B)` (= 11), PAS par `1 + N(c)` (= 41).
//      Pourquoi ? Le bonus sert à partager les visites DE s entre B et C :
//      chaque visite de s vers B réduit le bonus de B pendant que celui de C
//      monte avec √W_tot (où W_tot = N(s, B) + N(s, C)). Si l'on divisait par
//      1 + N(c), les 30 visites venues de t écraseraient le bonus de B alors
//      que C (dont le fils d n'a pas d'autre parent) garderait le sien entier.
//      À Q proches, s finirait par envoyer la majorité de ses visites vers le
//      coup le moins connu (C) plutôt que vers le meilleur (B), ce qui
//      fausserait le coup joué à la racine, la cible de politique en self-play
//      et la valeur moyenne remontée de s vers p.
//    • FPU (« First Play Urgency ») : valeur supposée d'un coup jamais essayé
//      à la place de Q(c) — en gros la valeur du parent moins une pénalité
//      (section 3 du .cpp).
//    • Perte virtuelle (« virtual loss ») : tant qu'une descente attend la
//      réponse du réseau, elle compte provisoirement comme une défaite et
//      comme une visite sur son chemin, afin que les autres threads (ou les
//      descentes suivantes du même lot) explorent d'autres coups (section 5).
//    • Lot (« batch ») : ensemble de feuilles envoyées d'un coup au GPU ;
//      évaluer 256 positions en une passe coûte à peine plus qu'en évaluer 1.
//    • Elo : échelle des écarts de force cités ici. Un écart d correspond à un
//      score moyen attendu de 1 / (1 + 10^(−d / 400)) pour le plus fort
//      (+10 Elo ≈ 51,4 %, +100 ≈ 64 %, +400 ≈ 91 %).
//
//  ---------------------------------------------------------------------------
//  ÉCHELLE DES VALEURS
//  ---------------------------------------------------------------------------
//    • Toutes les valeurs vivent dans [−1, 1] (du point de vue du camp au
//      trait ou du camp qui vient de jouer, toujours précisé).
//    • Le réseau sort `tanh(…) ∈ (−1, 1)`, entraîné sur z ∈ {−1, 0, +1} :
//      v ≈ E[z], soit P(gain) ≈ (v + 1) / 2 quand les nuls sont rares.
//    • Une vraie fin de partie atteinte dans le graphe vaut (« utilité de
//      score » à la KataGo, u = 0.05, échelle 10) :
//          (1 − u) · signe(d) + u · tanh(d / 10),   d = écart de points final
//      Une victoire vaut entre 0.95 et 1 (plus l'écart est grand, plus elle
//      vaut), et bat toujours le nul (0). Sans ce terme, tous les coups
//      gagnants se valent et la recherche joue mollement en finale (+4 Elo en
//      campagne). Le réseau, lui, ne prédit pas le score.
//
//  ---------------------------------------------------------------------------
//  LE RÉSEAU (ResNet 256 × 30, nnue/cnn_resnet_value_policy.py, classe Net)
//  ---------------------------------------------------------------------------
//  ENTRÉE : tenseur (4, 8, 8) de 0 et 1 :
//      plan 0 : pièces noires          plan 2 : trous (« empty »)
//      plan 1 : pièces blanches        plan 3 : camp au trait (0 Noir, 1 Blanc)
//  Les scores courants ne sont pas fournis (ils sont presque déterminés par le
//  nombre de trous, hors passes). Les colonnes sont encodées en miroir dans
//  `encode_planes` (section 9 du .cpp) : sans importance pour le réseau tant
//  que l'entraînement et le C++ suivent la même convention.
//
//  TRONC (256 canaux, 30 blocs résiduels « pré-activation » ResNet v2) :
//      (4,8,8) ─► conv 3×3 (4→256) ─► BN ─► ReLU                   « stem »
//              ─► 30 × [BN → ReLU → conv 3×3 → BN → ReLU → conv 3×3 + identité]
//              ─► BN ─► ReLU ─► carte (256, 8, 8)
//  Chaque conv 3×3 a un padding de 1 (la taille reste 8 × 8) ; après 60
//  convolutions, chaque case voit tout le plateau. Les sauts d'identité
//  permettent au gradient de traverser les 30 blocs.
//
//  DEUX TÊTES (AlphaZero) :
//    • Valeur    : conv 1×1 (256→1) → BN → ReLU → aplatir (64)
//                  → linéaire 64→256 → ReLU → linéaire 256→1 → tanh → v
//    • Politique : conv 1×1 (256→2) → BN → ReLU → aplatir (128)
//                  → linéaire 128→4096 → 4096 logits
//      Un logit par paire (départ, arrivée), indice = départ × 64 + arrivée
//      (0..63, a1 = 0). Les coups illégaux ne sont pas masqués à
//      l'entraînement ; en recherche, seul le softmax sur les coups légaux est
//      calculé (section 4). La passe (a1 → a1) utilise l'indice 0.
//
//  TAILLE ET COÛT :
//    • ~35,9 millions de paramètres (~72 Mo en fp16) : tronc 35,4 M, tête de
//      politique 528 k, tête de valeur ~17 k, BN ~31 k, stem ~9 k.
//    • ~2,3 milliards de MAC (~4,5 GFLOP) par position → exécution sur GPU en
//      fp16, par lots.
//
//  ENTRAÎNEMENT (supervisé, nnue/cnn_resnet_value_policy_chunked.py) :
//    • Données : parties générées par data/generate_games.cpp (hors coups
//      d'ouverture aléatoires et parties purement aléatoires), pré-encodées par
//      nnue/preprocess.py.
//    • Perte : MSE(v, z) + entropie croisée(logits, coup joué), poids 1 et 1.
//    • Optimiseur : Adam, pas 1e-3 → 1e-5 en cosinus, weight decay 1e-4,
//      multi-GPU DDP en précision mixte (bf16 sur Ampere+, fp16 sinon).
//
//  EXPORT (nnue/resnet_export.py) :
//    • <nom>.ts       : module TorchScript tracé en mode eval (BN figées),
//                       chargé par TorchNet (section 9). Sortie : (v (n,),
//                       logits (n, 4096)) pour une entrée (n, 4, 8, 8).
//    • <nom>.bin      : poids à plat (BN repliées) pour l'ancien backend CPU
//                       Eigen (~150 pos/s, retiré ici).
//    • <nom>.test.bin : positions de référence pour test/alphazero_check_main.cpp.
//
//  ---------------------------------------------------------------------------
//  PIPELINE C++ ↔ GPU (sections 8 à 10 du .cpp)
//  ---------------------------------------------------------------------------
//  La recherche ne manipule aucun tenseur : elle envoie des `Request` (une
//  position + ses coups légaux) et reçoit des `Result` (valeur + logits de ces
//  seuls coups) via l'interface `Evaluator`.
//
//     threads de recherche              Batcher                      TorchNet
//     ────────────────────              ───────                      ────────
//     thread 0  : 16 feuilles ──┐
//     thread 1  : 16 feuilles ──┼──► file d'attente ──► lot ≤ 256 ──► GPU (1 passe)
//     …                         │    (attend tous les                 (n,4,8,8) → v, logits
//     thread 15 : 16 feuilles ──┘    threads ou 0,5 ms)  ◄── résultats redistribués
//
//    • TorchNet : encode dans un tampon CPU épinglé, copie sur GPU en fp16,
//      fait une passe avant et extrait les logits des coups légaux.
//    • Batcher  : fusionne les lots de 16 feuilles des 16 threads en un lot de
//      256, divisant par 16 le nombre d'appels GPU.
//    • NNCache  : mémorise (64 Mo) valeur et a priori indexés par Zobrist pour
//      éviter de réévaluer les positions déjà vues (transpositions / coups
//      suivants).
//
//  ---------------------------------------------------------------------------
//  CONFIGURATION DE L'APP (config/hybrid/resnet.cfg)
//  ---------------------------------------------------------------------------
//    microseconds 1 000 000 (selon niveau)   nb threads 16   batch size 16
//    cpuct exploration 1.0, log 0.45, base 500
//    cpuct utility stdev scale 0.85, prior 0.40, prior weight 2.0
//    fpu reduction 0.2, root fpu reduction 0.1  (pondération masse a priori : oui)
//    policy temperature 1.0   graph search oui   reuse tree oui   nn cache 64 Mo
//    use lcb oui, lcb stdevs 5, min visit prop for lcb 0.2
//    policy target pruning oui   score utility 0.05, score scale 10
//    backend torch, device cuda, fp16 oui   merged batch size = 16 × 16
//  Réglages repris de KataGo (cpp/search/searchparams.cpp), sauf l'utilité de
//  score (spécifique à Yolah) ; non réoptimisés pour Yolah.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_mcts_check.cpp (compilé par test/final_mcts_check.sh, lancé
//  depuis config/) exécute cette version et le vrai joueur côte à côte sur
//  test/bench_positions.txt (1 thread, 800 simulations, fp32) et vérifie
//  l'égalité exacte du coup, des visites, de Q et des valeurs de jeu.
//  Dernière exécution (23/11/2026, RTX 3080) : 209 / 209 positions identiques,
//  411 recherches (position initiale + deux plis plus loin sur graphe réutilisé).
// =============================================================================

#include "game.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <torch/script.h>

namespace final_mcts {

// =============================================================================
//  DÉCLARATIONS — Vue d'ensemble d'un appel Player::play(position)
// =============================================================================
//
//    Player::play(position)
//      └─► Search::search(position)                         sections 1–7 du .cpp
//            ├─ racine : reprise du graphe précédent si possible, sinon créée
//            │  et évaluée seule immédiatement (1 appel réseau)
//            ├─ 16 threads run_worker, en boucle jusqu'à épuisement du budget :
//            │    1. RASSEMBLER jusqu'à 16 descentes PUCT (ou 16 collisions) :
//            │         • fin de partie      → valeur exacte remontée aussitôt ;
//            │         • hit NNCache        → développé et remonté sans GPU ;
//            │         • feuille inédite    → marquée EXPANDING, ajoutée au
//            │           lot du thread (perte virtuelle posée sur le chemin) ;
//            │         • déjà dans CE lot   → la descente s'y joint ;
//            │         • lot d'un AUTRE thread → descente annulée (collision).
//            │    2. ÉVALUER le lot via Evaluator::evaluate (le Batcher regroupe
//            │       jusqu'à 256 feuilles avant l'appel GPU de TorchNet).
//            │    3. DÉVELOPPER chaque feuille, la mettre en NNCache, puis
//            │       REMONTER chaque descente le long de son propre chemin.
//            └─► make_result : calcule lcb, target, play_value et choisit le coup
//
//  La recherche ne connaît le réseau qu'à travers `Evaluator` : elle ignore la
//  présence du `Batcher` (multi-thread) ou d'un appel direct à `TorchNet`
//  (1 thread / vérification).

// ── Interface recherche ↔ réseau ────────────────────────────────────────────
// Une position à évaluer, accompagnée de ses coups légaux (déjà générés pour
// créer les arêtes) afin que le réseau ne renvoie que leurs logits.
struct Request {
    const Yolah* state;     // position à évaluer (vit dans le tampon du thread, Worker::Leaf)
    const Move*  moves;     // coups légaux dans l'ordre de Yolah::moves (au moins 1, la passe sinon)
    uint16_t     nb_moves;  // nombre de coups légaux (≤ Yolah::MAX_NB_MOVES = 75)
};

// Sortie brute du réseau pour une requête (taille fixe pour éviter toute
// allocation pendant la recherche ; le softmax est fait côté recherche).
struct Result {
    float value;                        // valeur pour le camp au trait, dans (−1, 1)
    float logits[Yolah::MAX_NB_MOVES];  // logits bruts des coups moves[0..nb_moves)
};

// Contrat unique pour évaluer un lot de positions (implémenté par TorchNet et
// par Batcher).
class Evaluator {
public:
    virtual ~Evaluator() = default;
    // Évalue `in` et écrit les résultats dans `out` (out.size() >= in.size()).
    // Bloquant et sûr entre threads. Un `in` vide est autorisé (sert de
    // rendez-vous au Batcher, section 10).
    virtual void evaluate(std::span<const Request> in, std::span<Result> out) = 0;
};

// ── 9. Inférence GPU via libtorch (nnue/resnet_torch.cpp) ───────────────────
// Charge un module TorchScript (.ts), encode un lot en tenseur (n, 4, 8, 8)
// dans une mémoire CPU épinglée, lance une passe avant sur GPU et extrait les
// logits des coups légaux de chaque requête.
class TorchNet final : public Evaluator {
public:
    // device : "cuda" (repli sur "cpu" si absent) ; fp16 : demi-précision GPU ;
    // warm_batch : plus grand lot préparé au démarrage (256 = 16 × 16) via des
    // passes à blanc de tailles 1, 8, 32, 128 et warm_batch pour cuDNN.
    TorchNet(const std::string& filename, const std::string& device, bool fp16, int64_t warm_batch);
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    torch::Device device;               // périphérique de calcul (GPU en pratique)
    bool fp16;                          // demi-précision activée (GPU uniquement)
    torch::jit::script::Module module;  // réseau TorchScript chargé depuis le .ts
    std::mutex mutex;                   // sérialise les passes avant sur le GPU
    torch::Tensor input_cpu;            // tampon CPU épinglé (copie DMA vers GPU), agrandi au besoin
};

// ── 10. Regroupeur inter-threads (nnue/batched_evaluator.cpp) ───────────────
// Fusionne les petits lots (16) des threads de recherche en un gros lot (≤ 256)
// pour saturer le GPU :
//
//    thread 1  ─ evaluate(16) ─┐  file   ┌─────────────┐
//    …                         ├───────► │ worker_loop │ ─► TorchNet (≤ 256)
//    thread 16 ─ evaluate(16) ─┘ (FIFO)  └─────────────┘
//          ◄── chaque thread reçoit sa tranche et est réveillé
//
// Un lot part dès que : (1) `max_batch` positions sont en file, (2) tous les
// threads (`nb_clients`) attendent déjà, ou (3) `max_wait` s'est écoulé depuis
// la première demande en attente.
class Batcher final : public Evaluator {
public:
    Batcher(std::unique_ptr<Evaluator> net, size_t nb_clients, size_t max_batch,
            std::chrono::microseconds max_wait = std::chrono::microseconds(500));
    ~Batcher() override;  // vide la file et arrête le thread regroupeur
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    // Demande posée sur la pile du thread appelant (aucune allocation) ; le
    // thread dort jusqu'à ce que `done` passe à vrai.
    struct Job {
        std::span<const Request> in;
        std::span<Result> out;
        bool done = false;
    };
    void worker_loop();  // boucle du thread regroupeur
    std::unique_ptr<Evaluator> net;
    const size_t nb_clients, max_batch;
    const std::chrono::microseconds max_wait;
    std::mutex mutex;                    // protège l'état partagé ci-dessous jusqu'à `stop`
    std::condition_variable cv_worker;   // réveille le regroupeur (nouvelle demande / arrêt)
    std::condition_variable cv_clients;  // réveille les threads servis
    std::deque<Job*> queue;              // demandes en attente (FIFO)
    size_t queued_positions = 0;         // positions actuellement en file (condition 1)
    size_t waiting_clients = 0;          // threads en attente, demandes vides incluses (condition 2)
    bool stop = false;                   // signal d'arrêt du destructeur
    std::vector<Request> batch_in;       // lot fusionné (thread regroupeur uniquement)
    std::vector<Result> batch_out;       // résultats avant redistribution
    std::thread worker;                  // déclaré en dernier : démarré une fois tout prêt
};

// ── 8. Cache des sorties du réseau (player/nn_cache.h) ──────────────────────
// Table à accès direct (indice = clé & masque, écrasement sur collision) qui
// stocke la valeur et les a priori après softmax pour une position Zobrist.
// 64 Mo ≈ 400 000 positions ; évite de repasser par le GPU sur transpositions
// (après ramasse-miettes) et d'un coup à l'autre de la partie.
class NNCache {
public:
    // Arrondit `size_bytes` au nombre d'entrées puissance de 2 inférieur.
    explicit NNCache(size_t size_bytes);
    // Remplit `value` et `priors[0..nb_moves)` et renvoie true en cas de hit.
    bool lookup(uint64_t key, size_t nb_moves, float& value, float* priors);
    // Enregistre la sortie réseau (écrase l'entrée existante à cet indice).
    void store(uint64_t key, size_t nb_moves, float value, const float* priors);
private:
    struct Entry {
        uint64_t key = 0;                             // hachage Zobrist complet (vérifié à la lecture)
        float    value = 0.0f;                        // valeur réseau pour le camp au trait
        uint16_t nb_moves = 0;                        // vérifié aussi contre une collision de hachage
        uint16_t priors[Yolah::MAX_NB_MOVES] = {};    // a priori quantifiés sur [0, 65535]
    };                                                // 164 octets (168 alignés)
    std::vector<Entry> table;                         // vide si cache désactivé
    size_t mask = 0;                                  // table.size() − 1
    // 256 verrous en bandes (indice & 255) : contention quasi nulle à 16 threads
    // pour un coût mémoire négligeable.
    std::array<std::mutex, 256> locks;
};

// ── Réglages de la recherche (valeurs par défaut = config/hybrid/resnet.cfg) ─
// Les noms entre guillemets sont les clés des fichiers .cfg.
struct Params {
    // Budgets (arrêt au premier atteint ; 0 = illimité). L'app utilise le temps
    // seul ; la vérification, 800 simulations seules.
    uint64_t microseconds = 1'000'000;   // « microseconds » : temps par coup
    uint64_t nb_simulations = 0;         // « nb simulations » : nombre de simulations
    size_t nb_threads = 16;              // « nb threads » : threads de recherche
    size_t batch_size = 16;              // « batch size » : feuilles par thread et par lot
    size_t nn_cache_mb = 64;             // « nn cache » : taille du NNCache en Mo
    // Exploration PUCT (section 3) : c(s) = exploration + log · ln((W_tot + base) / base).
    float cpuct_exploration = 1.0f, cpuct_exploration_log = 0.45f, cpuct_exploration_base = 500.0f;
    // Adaptation à la dispersion (section 3) : σ̂ = 1 + scale · (écart-type / prior − 1),
    // l'écart-type étant régularisé vers `prior` avec `prior_weight` visites fictives.
    float cpuct_utility_stdev_scale = 0.85f, cpuct_utility_stdev_prior = 0.40f, cpuct_utility_stdev_prior_weight = 2.0f;
    // Pénalité FPU d'un coup jamais visité (ailleurs dans le graphe et à la racine).
    float fpu_reduction = 0.2f, root_fpu_reduction = 0.1f;
    // Température du softmax sur les logits de politique (1.0 = sortie brute).
    float policy_temperature = 1.0f;
    // Choix final par LCB (section 7) : rayon de `lcb_stdevs` écarts-types ;
    // candidats ayant au moins `min_visit_prop_for_lcb` de la valeur de jeu du leader.
    bool  use_lcb = true;
    float lcb_stdevs = 5.0f, min_visit_prop_for_lcb = 0.20f;
    // Élagage de la cible (section 7) : retire les visites d'exploration périmées.
    bool  policy_target_pruning = true;
    // Utilité de score des fins de partie : (1 − u)·signe(d) + u·tanh(d / scale).
    float score_utility = 0.05f, score_scale = 10.0f;
    // Réutilisation du sous-graphe d'un coup au suivant (section 2).
    bool  reuse_tree = true;
};

// ── Bilan de la racine : ChildStat ──────────────────────────────────────────
// Un ChildStat résume UN coup légal de la racine à la fin de search(). Ce
// commentaire se lit en quatre passes, chacune plus précise que la précédente :
//   niveau 1  l'idée en quelques lignes ;
//   niveau 2  les quatre étapes en mots, sur un exemple chiffré ;
//   niveau 3  les formules exactes du code, et leurs calculs sur l'exemple ;
//   niveau 4  les cas limites et à quoi servent les champs ensuite.
//
// ═══ Niveau 1 : l'idée ═══════════════════════════════════════════════════════
//
// À la fin de la recherche, on doit choisir un coup. Le critère d'AlphaZero
// est simple : jouer le coup le plus VISITÉ, car PUCT envoie ses visites là
// où la valeur est bonne. Mais les visites ont deux défauts :
//   • une partie d'entre elles vient de l'EXPLORATION (le bonus d'a priori),
//     pas de la valeur : on les retire (l'élagage) ;
//   • elles réagissent en retard : un coup très visité au début peut s'être
//     révélé moins bon qu'un coup découvert plus tard, encore peu visité.
//     On regarde donc aussi une valeur PRUDENTE de chaque coup (sa moyenne
//     moins une marge d'incertitude, la LCB) ; si le coup de meilleure valeur
//     prudente n'est pas le plus visité, on lui donne assez de « visites
//     fictives » pour qu'il passe devant.
// Le résultat est un score par coup, play_value, sur l'échelle des visites ;
// le coup joué est celui de plus grand play_value.
//
// ═══ Niveau 2 : les quatre étapes sur un exemple ═════════════════════════════
//
// Exemple (réglages par défaut), 1 520 visites terminées sous la racine R :
//
//                                  R  (au trait : nous)
//          ┌───────────────┬───────┴───────┬───────────────┐
//        c3-c6           e4-e7           b2-b5           h1-h3   (+ les autres
//        P 0.40          P 0.20          P 0.09          P 0.01   coups légaux :
//        n 1 000         n 400           n 120           n 0      0.30 d'a priori
//        q +0.10         q +0.20         q +0.05                  au total,
//        q2 0.132        q2 0.290        q2 0.200                 0 visite)
//
//   P = prior (réseau), n = visites de l'arête, q = moyenne des valeurs
//   remontées (de notre point de vue), q2 = moyenne de leurs carrés.
//
// Étape 1 — Moyenne et dispersion (q, q2). Chaque visite a remonté une
//   valeur dans [−1, 1]. q est leur moyenne ; q2 sert à savoir si ces valeurs
//   étaient proches les unes des autres ou très éparpillées (variance).
//   e4-e7 : moyenne +0.20 mais valeurs très éparpillées (écart-type 0.50) ;
//   c3-c6 : moyenne +0.10, un peu moins éparpillées (0.35).
//
// Étape 2 — Élagage (target). c3-c6 est le coup « de référence » b (le plus
//   visité). Pour chaque autre coup, on se demande : avec les moyennes
//   finales, combien de visites PUCT lui aurait-il données ? Celles en trop
//   ne sont dues qu'à l'exploration et sont retirées.
//   b2-b5 (q +0.05, nettement sous c3-c6) : PUCT ne l'aurait visité que
//   ~76 fois, au lieu de 120, à cause de son petit a priori → target 76.
//   e4-e7 (q +0.20, au-dessus de c3-c6) : sa valeur suffit, il garde 400.
//
// Étape 3 — Valeur prudente (lcb). lcb = q − marge, la marge étant grande
//   si les valeurs sont éparpillées ou les visites peu nombreuses :
//   e4-e7 : +0.20 − 0.125 = +0.075   (400 visites, éparpillées)
//   c3-c6 : +0.10 − 0.055 = +0.045   (1 000 visites)
//   b2-b5 : +0.05 − 0.203 = −0.153   (120 visites seulement)
//   Même avec sa grande marge, e4-e7 reste le meilleur coup « prudent ».
//
// Étape 4 — Score de décision (play_value). On part de target. Le meilleur
//   coup prudent (e4-e7) n'a que 400 contre 1 000 : il reçoit un score
//   d'autant plus grand que son avance de lcb est nette, ici ≈ 1 945. Il
//   passe devant c3-c6 et c'est lui qu'on joue.
//
//   coup    prior  visits      q     q2      lcb   target   play_value
//   e4-e7    0.20     400  +0.20  0.290   +0.075      400       ≈ 1 945  ← joué
//   c3-c6    0.40   1 000  +0.10  0.132   +0.045    1 000         1 000
//   b2-b5    0.09     120  +0.05  0.200   −0.153       76            76
//   h1-h3    0.01       0      0      0       −2        0             0
//
// ═══ Niveau 3 : les formules exactes ═════════════════════════════════════════
//
// Fonctions : make_result remplit prior, visits, q, q2 ; root_play_values
// calcule target, lcb et play_value.
//
// Étape 1 — q et q2 (make_result).
//   ATTENTION, deux compteurs différents (le graphe, cf. glossaire) :
//     n = visits = Edge::n, les visites passées par l'arête racine → coup ;
//     N = Node::n, les visites du nœud enfant, tous parents confondus (N ≥ n
//         si d'autres chemins mènent à la même position).
//   Le nœud a accumulé W = Σ v_i et W2 = Σ v_i² sur ses N valeurs, d'où
//       q = W / N,   q2 = W2 / N      (0 et 0 si N = 0).
//   Pourquoi q2 donne la variance : la variance est la moyenne des carrés
//   des écarts à la moyenne. En développant (v_i − q)² = v_i² − 2·q·v_i + q²
//   et en moyennant terme à terme : q2 − 2·q·q + q² = q2 − q² (König-Huygens).
//       e4-e7 : 0.290 − 0.20² = 0.250 → écart-type √0.250 = 0.500
//       c3-c6 : 0.132 − 0.10² = 0.122 → écart-type 0.349
//       b2-b5 : 0.200 − 0.05² = 0.1975 → écart-type 0.444
//
// Étape 2 — target (élagage de la cible de politique).
//   a) La référence b maximise max(0, visits − 1) + 2·prior : le plus visité,
//      départagé par l'a priori ; une visite unique compte comme zéro, pour
//      qu'un coup essayé une fois ne détrône pas un coup de meilleur a priori.
//      Ici b = c3-c6 (999 + 0.8).
//   b) Le score PUCT d'un coup à la racine est q + E·prior / (1 + n), avec
//          E = cpuct · σ̂(racine) · √(Σ visits + 0.01)
//          cpuct = 1 + 0.45 · ln((1 520 + 500) / 500) = 1.628
//          σ̂(racine) = 1 + 0.85 · (écart-type racine / 0.40 − 1) = 1.011
//            (écart-type racine ≈ 0.405, calculé par utility_stdev_factor
//             sur toutes les valeurs remontées à la racine)
//          E = 1.628 · 1.011 · √1 520 ≈ 64.2
//      score(b) = 0.10 + 64.2 · 0.40 / 1 001 = 0.10 + 0.0256 = 0.1256.
//   c) Pour un autre coup de moyenne q, l'écart à combler est d = score(b) − q.
//      Son bonus E·prior / (1 + n) décroît avec n ; PUCT le visitait tant que
//      ce bonus dépassait d, c'est-à-dire jusqu'à
//          n* = E·prior / d − 1.
//      Si d > 0 : target = min(visits, ⌈max(0, n*)⌉) ; si d ≤ 0 (le coup vaut
//      déjà score(b) par sa seule moyenne) : target = visits. b garde tout.
//          b2-b5 : d = 0.1256 − 0.05 = 0.0756, n* = 64.2·0.09/0.0756 − 1
//                  ≈ 75.3 → target = 76 (au lieu de 120)
//          e4-e7 : d = 0.1256 − 0.20 < 0 → target = 400
//          h1-h3 : 0 visite → target = 0
//
// Étape 3 — lcb (borne inférieure de confiance).
//   a) Principe : q est une moyenne de n valeurs d'écart-type σ = √(q2 − q²).
//      L'incertitude sur une MOYENNE est σ/√n (Var(moyenne) = σ²/n : les
//      variances de n valeurs indépendantes s'ajoutent, puis on divise la
//      somme par n, donc la variance par n²). On retranche 5 fois cette
//      incertitude (lcb_stdevs = 5, très prudent) :
//          lcb ≈ q − 5·σ/√n
//          e4-e7 : 0.20 − 5·0.500/√400  = 0.20 − 0.125 = +0.075
//          c3-c6 : 0.10 − 5·0.349/√1000 = 0.10 − 0.055 = +0.045
//      Quatre fois plus de visites → marge divisée par deux.
//   b) Formule exacte : avec très peu de visites, σ observé peut être nul
//      par hasard (une seule valeur). On mélange donc au second moment un
//      a priori « tout est possible » (second moment + 1) de poids a = 1/n² :
//          m2  = max(q2, q² + 10⁻⁸)
//          m2' = (m2·n + (m2 + 1)·a) / (n + a)      = m2 + a/(n + a)
//          ess = (n + a)² / (n + a²)                (nombre effectif de visites)
//          r   = 5 · √(max(0, m2' − q²) / ess)
//          lcb = q − r
//      Dès quelques visites, a est négligeable et on retrouve a). À une seule
//      visite (a = 1, ess = 2, m2' − q² = 0.5) : r = 5·√0.25 = 2.5, la borne
//      est hors de [−1, 1] : le coup ne peut pas gagner sur sa lcb.
//          b2-b5 : r = 5·0.444/√120 ≈ 0.203 → lcb = −0.153
//
// Étape 4 — play_value.
//   a) Départ : play_value = target.
//   b) Candidats à la promotion : les coups dont target > 0 et
//      target ≥ 20 % du target de b (min_visit_prop_for_lcb = 0.2) ; ici
//      ≥ 200 : c3-c6 et e4-e7 (b2-b5, 76, est exclu). Le seuil porte sur le
//      target, donc après élagage : un coup trop peu visité « pour sa valeur »
//      ne peut pas gagner par une borne chanceuse. Parmi les candidats, le
//      gagnant g est celui de plus grande lcb : e4-e7.
//   c) g reçoit assez de score pour battre chaque coup i de borne ≤ la sienne.
//      Avance e = lcb(g) − lcb(i), marge r_i du coup i :
//          k_i = (r_i + e) / (r_i + 0.2·e)            (1 ≤ k_i < 5)
//          play_value(g) = max(target(g), max_i k_i² · target(i))
//      Lecture : pour que i rattrape g, il faudrait que sa marge grandisse
//      d'un facteur k_i environ ; la marge décroît en 1/√visites, donc g
//      « vaut » k_i² fois les visites de i. Une avance minuscule (e ≪ r_i)
//      donne k ≈ 1 ; une avance énorme (e ≫ r_i) donne k → 5, soit au plus
//      25 × target(i).
//          vs c3-c6 : e = 0.075 − 0.045 = 0.030, r = 0.055
//                     k = 0.085 / 0.061 ≈ 1.395, k² · 1 000 ≈ 1 945
//          vs b2-b5 : e = 0.228, r = 0.203
//                     k = 0.431 / 0.248 ≈ 1.73,  k² · 76 ≈ 228
//          vs h1-h3 : target 0 → 0
//      → play_value(e4-e7) = max(400, 1 945, 228, 0) ≈ 1 945.
//      Les autres coups gardent play_value = target.
//
// ═══ Niveau 4 : cas limites et usages ════════════════════════════════════════
//
//   • lcb = −2 (sentinelle) pour un coup sans visite, ou si use_lcb est
//     désactivé (aucun calcul de borne).
//   • Un coup dont la borne dépasse celle du gagnant, mais exclu par le seuil
//     des 20 %, est ignoré au calcul c) (e < 0) : il garde son target.
//   • Sans élagage (policy_target_pruning = false), target = visits.
//   • make_result trie par play_value décroissant, puis q décroissant (tri
//     stable : à égalité complète, ordre de Yolah::moves) et joue le premier.
//     Si Σ target = 0 (aucune simulation terminée), il joue le plus grand
//     prior.
//   • target dans le joueur de l'app ne sert qu'à ce dernier cas. Dans le
//     joueur complet (self-play), target / Σ target est la cible de la tête
//     de politique, et les coups d'ouverture sont tirés ∝ target^(1/τ).
//     play_value ne sert jamais d'apprentissage : seulement à choisir.
struct ChildStat {
    Move  move;
    // Probabilité donnée par le réseau avant la recherche (softmax légal).
    float prior;
    // Visites terminées de l'arête, y compris celles héritées du graphe.
    uint32_t visits;
    // Moyenne du nœud enfant, vue du camp qui joue le coup (dans [−1, 1]).
    float q;
    // Moyenne des carrés dans le nœud enfant (voir étape 1).
    float q2;
    // Borne inférieure de confiance, q moins la marge de l'étape 3.
    float lcb;
    // Score de sélection après la promotion LCB (étape 4).
    float play_value;
    // Visites retenues après l'élagage PUCT (étape 2).
    float target;
};

struct SearchResult {
    // Coup choisi. make_result trie `children` par `play_value` décroissante
    // avant de choisir : le coup de plus grand `play_value` est donc le premier
    // de la liste, children[0].move (pas besoin de chercher le maximum). Seule
    // exception : si Σ target == 0 (aucune simulation terminée), c'est le coup
    // de plus grand `prior`, qui peut être ailleurs dans la liste.
    // Move::none() représente aussi la passe quand c'est le seul coup légal.
    Move best_move = Move::none();
    // Simulations terminées durant cet appel : les descentes qui ont remonté
    // une valeur jusqu'à la racine. Une descente se termine de quatre façons :
    //   • sur une fin de partie : le résultat exact est remonté aussitôt ;
    //   • sur une feuille dont la position est dans le cache du réseau : la
    //     valeur du cache est remontée aussitôt ;
    //   • sur une feuille nouvelle : elle entre dans le lot du thread et
    //     attend la réponse du réseau ;
    //   • sur une feuille DÉJÀ dans le lot de ce même thread (une descente
    //     précédente du lot l'a choisie, souvent par le même chemin, faute de
    //     changement dans les statistiques) : on ne la redemande pas au réseau.
    //     La descente attend la même réponse, puis remonte la même valeur par
    //     son propre chemin. C'est une visite de plus, mais pas une évaluation
    //     de plus.
    // Les quatre comptent. Ne comptent pas : une descente qui tombe sur une
    // feuille en cours d'évaluation dans le lot d'un AUTRE thread (une
    // « collision » : elle est annulée, ses visites en attente retirées), et
    // les visites héritées de la recherche du coup précédent (arbre réutilisé).
    uint64_t nb_simulations = 0;
    // Positions effectivement envoyées au réseau par les threads : le 3e cas
    // ci-dessus seulement (hors cache, fins de partie, feuilles partagées dans
    // un lot, et hors évaluation initiale de la racine). Donc ≤ nb_simulations.
    uint64_t nb_evaluations = 0;
    // Un ChildStat par coup légal de la racine, trié par `play_value`
    // décroissante, puis par `q` (tri stable : ordre de Yolah::moves sinon).
    std::vector<ChildStat> children;
};

// ── La recherche MCTS (sections 1 à 7 du .cpp) ──────────────────────────────
// Possède le graphe (alloué dans le pool `memory` et indexé par la table de
// transpositions `shards`) ainsi que le `NNCache`, et emprunte l'`Evaluator`.
// Le graphe survit d'un `search()` au suivant (`reuse_tree`) ; `reset()` le
// vide en conservant le `NNCache` (qui ne dépend que du réseau).
class Search {
public:
    // Lève std::invalid_argument si les deux budgets (temps et simulations)
    // sont nuls ; borne `nb_threads` et `batch_size` à 1 au minimum.
    Search(Evaluator& evaluator, const Params& params, std::pmr::memory_resource* memory);
    ~Search();  // libère tout le graphe
    // Lance une recherche bloquante depuis `state` (position non terminale) :
    //   1. reprend ou crée la racine (et la développe si inédite) ;
    //   2. fait tourner `nb_threads` workers jusqu'à épuisement du budget
    //      (sans lancer de thread supplémentaire si nb_threads == 1) ;
    //   3. construit et retourne le bilan (`make_result`).
    SearchResult search(const Yolah& state);
    // Efface le graphe (à appeler entre deux parties) ; le cache est conservé.
    void reset();
private:
    struct Node;      // position : N, W, W2, pending, état atomique, arêtes (section 1)
    struct Edge;      // coup : action, prior, N(s, a), pending, nœud fils (section 1)
    struct Worker;    // tampons privés d'un thread : feuilles, chemins, lots (section 1)
    struct PathStep;  // un pli d'une descente : (nœud atteint, arête prise) (section 1)
    // Cycle de vie atomique d'un nœud (`Node::state`) :
    //
    //     UNEXPANDED ──(compare_exchange par 1 thread)──► EXPANDING
    //         │                                               │
    //         │ partie finie                    réponse GPU   │
    //         ▼                                 ou hit cache  ▼
    //      TERMINAL                                        EXPANDED
    //
    //   • UNEXPANDED : créé mais jamais évalué ;
    //   • EXPANDING  : pris par un thread en attente du GPU (un autre chemin
    //                  du même lot s'y joint ; un autre thread subit une
    //                  collision, section 6) ;
    //   • EXPANDED   : arêtes prêtes, traversable définitivement ;
    //   • TERMINAL   : fin de partie, valeur exacte remontée sans GPU.
    enum : uint8_t { UNEXPANDED = 0, EXPANDING = 1, EXPANDED = 2, TERMINAL = 3 };
    // Table de transpositions (clé de position → Node*), découpée en 64
    // tranches verrouillées indépendamment (clé % 64) pour limiter la
    // contention entre threads. Consultée uniquement lors de la première
    // traversée d'une arête (ensuite `Edge::child` pointe directement dessus).
    static constexpr size_t NB_SHARDS = 64;
    struct Shard {
        std::mutex mutex;
        std::unordered_map<uint64_t, Node*> nodes;  // clé = Zobrist + score de Noir (section 2)
    };
    // ── section 2 : gestion du graphe et ramasse-miettes ──
    Node* new_node();                // alloue un nœud UNEXPANDED dans le pool
    Node* lookup(uint64_t key);      // cherche un nœud par clé (sous verrou de tranche)
    void  destroy_all();             // détruit tous les nœuds et vide la table
    void  mark(Node* node);          // marque `node` et ses descendants (DFS)
    void  collect(Node* keep);       // libère tout nœud inatteignable depuis `keep`
    bool  reuse(uint64_t key);       // promeut `key` en racine + collect(), si présent
    void  new_root(uint64_t key);    // repart d'un graphe neuf réduit à une racine UNEXPANDED
    void  expand_root(const Yolah& state);  // évalue seule et développe la racine avant les threads
    // ── section 6 : boucle d'un thread de recherche ──
    void  run_worker(Worker& w, const Yolah& root_state);
    // ── section 3 : sélection PUCT ──
    uint32_t select_child(const Node& node, bool is_root) const;
    float utility_stdev_factor(const Node& node, uint32_t visits) const;
    float fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const;
    // ── section 4 : développement d'une feuille ──
    void  priors_from_logits(const Result& result, size_t nb_moves, float* priors) const;
    void  expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value);
    // ── section 5 : remontée et valeur terminale ──
    void  backup(const PathStep* path, size_t length, float leaf_value);
    void  undo_pending(const PathStep* path, size_t length);
    float terminal_value(const Yolah& state) const;
    // ── section 7 : bilan et choix du coup à la racine ──
    void  root_play_values(std::vector<ChildStat>& children) const;
    SearchResult make_result();

    Evaluator& evaluator;
    Params prm;
    std::pmr::polymorphic_allocator<> alloc;  // alloue nœuds et arêtes dans le pool du Player
    Shard shards[NB_SHARDS];
    Node* root = nullptr;
    std::unique_ptr<NNCache> cache;
    uint32_t mark_counter = 0;                // génération du ramasse-miettes (évite de réinitialiser les marques)
    // Compteurs atomiques remis à 0 au début de chaque `search()` :
    std::atomic<uint64_t> simulations{0};     // descentes terminées (→ SearchResult::nb_simulations)
    std::atomic<uint64_t> reserved{0};        // budget de simulations réservé d'avance par les threads (mode simulations fixes)
    std::atomic<uint64_t> evaluations{0};     // positions envoyées au réseau (→ SearchResult::nb_evaluations)
    std::atomic<bool> stop{false};            // signal d'arrêt commun aux threads
    std::chrono::steady_clock::time_point deadline;  // échéance = début de search() + microseconds
};

// ── Le joueur de l'app (player/alphazero_mcts_player.cpp) ───────────────────
// Assemble le réseau (`TorchNet`), le `Batcher` si `nb_threads > 1`, le pool
// mémoire et `Search`.
class Player {
public:
    // weights : fichier .ts ; nb_threads > 1 intercale un Batcher (lots ≤ 256).
    Player(const std::string& weights, const Params& params, bool fp16 = true, const std::string& device = "cuda");
    ~Player();
    Move play(const Yolah& yolah);                             // lance une recherche et renvoie le coup choisi
    void game_over();                                          // oublie le graphe en fin de partie (cache conservé)
    const SearchResult& last_result() const { return last; }   // bilan de la dernière recherche
private:
    // Ordre de déclaration critique (destruction en ordre INVERSE) :
    // `searcher` doit être détruit avant `memory` (qui contient ses nœuds) et
    // avant `net` (qu'il référence).
    std::pmr::synchronized_pool_resource memory;  // pool mémoire thread-safe des nœuds
    std::unique_ptr<Evaluator> net;               // TorchNet, ou Batcher enveloppant TorchNet
    std::unique_ptr<Search> searcher;
    SearchResult last;
};

} // namespace final_mcts

#endif
