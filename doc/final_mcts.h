#ifndef FINAL_MCTS_H
#define FINAL_MCTS_H

// =============================================================================
//  FINAL_MCTS — Version épurée et commentée du joueur « MCTS ResNet » de l'app
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (et son compagnon doc/final_mcts.cpp) contient le code du
//  joueur « MCTS ResNet » de l'application Yolah, dans sa configuration de
//  l'application (config/hybrid/resnet.cfg, identique aux réglages de l'app) :
//  une recherche arborescente Monte-Carlo à la AlphaZero, avec les
//  raffinements de KataGo, guidée par un réseau de neurones résiduel (ResNet)
//  à deux têtes (valeur + politique) qui tourne sur la carte graphique.
//
//  Le vrai code est réparti dans plusieurs fichiers :
//    player/alphazero_mcts.{h,cpp}        la recherche (az::Search)
//    player/alphazero_mcts_player.{h,cpp} le joueur (lecture du JSON)
//    player/nn_cache.h                    le cache des sorties du réseau
//    nnue/nn_evaluator.h                  l'interface recherche ↔ réseau, l'encodage
//    nnue/resnet_torch.{h,cpp}            le réseau exécuté par libtorch (GPU)
//    nnue/batched_evaluator.{h,cpp}       le regroupement des requêtes des threads
//    nnue/cnn_resnet_value_policy.py      l'architecture et l'entraînement (Python)
//    nnue/resnet_export.py                l'export du réseau vers le C++
//  Ils contiennent en plus tout ce qui sert à l'auto-apprentissage (bruit de
//  Dirichlet, tirage des coups avec température, « forced playouts »,
//  « playout cap randomization »), un mode arbre (sans graphe), un backend CPU
//  (Eigen) et des options jamais utilisées par l'app. Tout cela est retiré ici,
//  et ces fichiers-ci se lisent seuls.
//
//  « Même code » veut dire : compilé à côté du vrai joueur, avec un seul thread
//  de recherche et un nombre fixe de simulations, ce code donne exactement les
//  mêmes visites, les mêmes valeurs et le même coup sur les positions de test
//  (voir « VÉRIFICATION » à la fin de ce commentaire). Avec plusieurs threads,
//  aucune des deux versions n'est reproductible d'une exécution à l'autre (les
//  threads se partagent l'arbre dans un ordre qui dépend du système), mais le
//  code exécuté est le même.
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF
//  ---------------------------------------------------------------------------
//  (Les règles en détail sont au début de doc/final_dev.h.) Plateau 8 × 8,
//  4 pièces par camp, Noir commence. Une pièce se déplace comme une dame
//  d'échecs vers une case libre, sans rien sauter ; sa case de départ devient
//  un TROU définitif. Chaque coup rapporte 1 point ; un camp bloqué passe
//  (`Move::none()`, la valeur 0 = « a1 → a1 ») ; la partie finit quand les deux
//  camps sont bloqués, et gagne celui qui a le plus de points. Conséquence
//  essentielle pour la recherche : le nombre de trous augmente de 1 à chaque
//  coup (hors passe), donc une position ne se répète JAMAIS (graphe sans
//  cycle), alors que la même position est atteinte par de très nombreux ordres
//  de coups (les 4 pièces d'un camp bougent indépendamment). Une case = un
//  entier 0..63 (a1 = 0, b1 = 1, …, h8 = 63) ; un ensemble de cases = un
//  `uint64_t` (« bitboard », bit s = case s) ; le pli `ply` compte les
//  demi-coups depuis le début de la partie ; le camp au trait est `ply & 1`.
//
//  ---------------------------------------------------------------------------
//  L'IDÉE DE LA RECHERCHE (MCTS « à la AlphaZero »)
//  ---------------------------------------------------------------------------
//  Le minimax (doc/final_dev.h) explore tous les coups jusqu'à une profondeur
//  fixe et évalue les feuilles avec un petit réseau rapide (NNUE). Le MCTS
//  fait l'inverse : il construit un arbre très déséquilibré, profond sur les
//  coups prometteurs et presque vide ailleurs, et chaque position nouvelle est
//  évaluée par un GROS réseau (environ 36 millions de paramètres, 1000 fois
//  plus de calcul par position que le NNUE) qui donne deux choses :
//    • une VALEUR v ∈ (−1, 1) : le résultat espéré pour le camp au trait
//      (+1 gagné, −1 perdu, 0 nul) ;
//    • une POLITIQUE : une probabilité a priori P(s, a) pour chaque coup a,
//      « la chance que ce soit le meilleur coup », qui dit où chercher.
//  Une SIMULATION (ou « visite ») fait :
//    1. SÉLECTION : on descend de la racine en choisissant à chaque nœud le
//       coup qui maximise PUCT = Q + U (exploitation + exploration, section 3
//       du .cpp), jusqu'à une position jamais évaluée (une FEUILLE) ;
//    2. ÉVALUATION : le réseau donne v et les P de la feuille ; elle est
//       « développée » (ses coups deviennent des arêtes, avec leurs P) ;
//    3. REMONTÉE (« backup ») : v est ajouté à la somme des valeurs de chaque
//       nœud du chemin, avec un changement de signe à chaque pli (ce qui est
//       bon pour l'un est mauvais pour l'autre).
//  Q(s, a) est la moyenne des valeurs remontées par le coup a : plus un coup a
//  été visité, plus son Q est fiable. Le coup joué à la fin est en gros le
//  plus visité (section 7 du .cpp : la borne inférieure de confiance, LCB).
//  Il n'y a pas de « partie aléatoire jusqu'au bout » comme dans le MCTS
//  classique : la valeur du réseau la remplace (seules les fins de partie
//  réellement atteintes dans l'arbre donnent un résultat exact).
//
//  Dans l'app, la recherche dure le temps donné par le niveau (1 s par coup
//  dans les campagnes), avec 16 threads qui envoient chacun 16 feuilles à la
//  fois au réseau. Ordres de grandeur mesurés (RTX 3080, fp16) : le réseau
//  évalue ~7 500 à 10 000 positions par seconde par lots de 256 ; la recherche
//  fait ~2 500 à 2 900 nouvelles feuilles par seconde en milieu de partie
//  (mesure du 23/11/2026, campagne H2), donc ~2 500 évaluations par coup à
//  1 s — 1000 fois moins de positions que le minimax dev dans le même temps.
//
//  ---------------------------------------------------------------------------
//  PETIT GLOSSAIRE
//  ---------------------------------------------------------------------------
//    • Nœud (`Node`) : une position. Arête (`Edge`) : un coup depuis une
//      position, avec son a priori P et un pointeur vers le nœud d'arrivée.
//    • N(s) : nombre de visites terminées du nœud s (tous parents confondus).
//      N(s, a) : visites passées par l'arête (s, a) (la part de CE parent).
//    • W : somme des valeurs remontées dans un nœud, du point de vue du camp
//      qui a JOUÉ le coup menant à ce nœud (donc l'adversaire du camp au trait
//      dans ce nœud). Q = W / N : la valeur moyenne, même point de vue. Ainsi,
//      à un nœud parent, le camp au trait compare directement les Q de ses
//      enfants : il veut le plus grand.
//    • W2 : somme des carrés des valeurs remontées (pour l'écart-type).
//    • PUCT : « Predictor + Upper Confidence bounds applied to Trees », la
//      formule de choix du coup (section 3 du .cpp).
//    • Perte virtuelle (« virtual loss ») : une descente en cours (pas encore
//      remontée) compte provisoirement comme une défaite dans les nœuds
//      qu'elle traverse, pour que les autres descentes (autres threads, ou la
//      suite du même lot) aillent ailleurs (section 5).
//    • FPU (« First Play Urgency ») : la valeur supposée d'un coup jamais
//      essayé (section 3).
//    • Lot (« batch ») : plusieurs feuilles envoyées ensemble au réseau. Un GPU
//      ne va vite que sur des lots : évaluer 256 positions coûte à peine plus
//      cher que d'en évaluer une.
//    • Recherche en graphe : un seul nœud par position, partagé par tous les
//      chemins qui y mènent (section 2).
//    • Elo : +10 Elo ≈ 51,4 % des points, +100 ≈ 64 %.
//
//  ---------------------------------------------------------------------------
//  ÉCHELLE DES VALEURS
//  ---------------------------------------------------------------------------
//    • Toutes les valeurs de la recherche sont dans [−1, 1], du point de vue
//      indiqué à chaque fois (camp au trait, ou camp qui vient de jouer).
//    • Le réseau donne tanh(…) ∈ (−1, 1), entraîné sur le résultat z ∈
//      {−1, 0, +1} : v ≈ E[z], et P(gain) ≈ (v + 1) / 2 si les nuls sont rares.
//    • Une fin de partie réellement atteinte vaut (« utilité de score », à la
//      KataGo, u = 0.05, échelle 10) :
//          (1 − u)·signe(d) + u·tanh(d / 10),     d = écart de points final
//      pour le camp au trait : une victoire vaut entre 0.95 et 1, une victoire
//      large plus qu'une courte, et une victoire bat toujours un nul (0). Le
//      réseau, lui, ne prédit pas de score : seules les fins réelles en ont
//      (campagne de l'utilité de score : +4 Elo, des victoires plus nettes ;
//      sans elle, une fois la partie gagnée, tous les coups gagnants se
//      valent et la recherche joue n'importe lequel).
//
//  ---------------------------------------------------------------------------
//  LE RÉSEAU (ResNet 256 × 30, nnue/cnn_resnet_value_policy.py, classe Net)
//  ---------------------------------------------------------------------------
//  ENTRÉE : 4 plans de 8 × 8 nombres (0 ou 1), soit un tenseur (4, 8, 8) :
//      plan 0 : pièces noires,  plan 1 : pièces blanches,
//      plan 2 : trous (cases vidées, « empty » dans le code),
//      plan 3 : le camp au trait (tout à 0 = Noir, tout à 1 = Blanc).
//  Les points déjà marqués ne sont pas donnés : ils se déduisent du nombre de
//  trous de chaque camp... à peu près (une passe ne fait pas de trou, et le
//  réseau ne sait pas de quel camp vient un trou). L'ordre des cases dans un
//  plan est particulier (encode_planes dans le .cpp, section 9) : les colonnes
//  y sont en miroir. C'est sans importance pour le réseau, mais l'entraînement
//  et le C++ doivent faire exactement la même chose.
//
//  TRONC (les « 256 × 30 » : 256 canaux, 30 blocs résiduels) :
//
//      (4,8,8) ─► conv 3×3, 4→256 ─► BN ─► ReLU                    « stem »
//              ─► 30 × bloc résiduel                                (ci-dessous)
//              ─► BN ─► ReLU ─► carte de caractéristiques (256, 8, 8)
//
//    Bloc résiduel « pré-activation » (ResNet v2) :
//                          ┌─────────────── identité ───────────────┐
//      x ─► BN ─► ReLU ─► conv 3×3 ─► BN ─► ReLU ─► conv 3×3 ─────► (+) ─► sortie
//    Les convolutions 3×3 ont un « padding » de 1 : la carte reste 8 × 8 tout
//    du long, chaque case voit ses 8 voisines à chaque couche ; après 60
//    convolutions, toute case « voit » tout le plateau (il suffit de 7 pour
//    aller d'un coin à l'autre). L'identité (le saut) permet d'entraîner un
//    réseau aussi profond : le gradient passe directement d'un bloc à l'autre.
//    BN = BatchNorm : normalise chaque canal (moyenne 0, variance 1 sur le lot
//    pendant l'entraînement ; des constantes apprises ensuite). ReLU(x) =
//    max(0, x).
//
//  TÊTE DE VALEUR (AlphaZero) :
//      (256,8,8) ─► conv 1×1, 256→1 ─► BN ─► ReLU ─► aplatir (64)
//                ─► linéaire 64→256 ─► ReLU ─► linéaire 256→1 ─► tanh ─► v
//  TÊTE DE POLITIQUE (AlphaZero) :
//      (256,8,8) ─► conv 1×1, 256→2 ─► BN ─► ReLU ─► aplatir (128)
//                ─► linéaire 128→4096 ─► 4096 « logits »
//    Un logit par paire (départ, arrivée) : indice = départ × 64 + arrivée,
//    départ et arrivée en 0..63 (a1 = 0). Les trois quarts sont des coups
//    impossibles : on ne les masque pas à l'entraînement (ils ne sont jamais la
//    cible, leur probabilité tombe toute seule) ; la recherche ne lit que les
//    logits des coups légaux et fait le softmax sur eux seuls (section 4 du
//    .cpp). La passe (a1 → a1) a l'indice 0, qui n'est jamais un vrai coup.
//
//  TAILLE ET COÛT :
//    • stem 4·256·9 = 9 216 poids ; chaque conv 3×3 du tronc 256·256·9 =
//      589 824 poids, 60 convolutions : 35,4 millions ; tête de politique
//      128·4096 + 4096 = 528 384 ; tête de valeur ~16 900 ; BN ~31 000.
//      Total ≈ 35,9 millions de paramètres (≈ 72 Mo en fp16).
//    • Calcul par position : chaque conv 3×3 fait 589 824 multiplications-
//      additions par case, × 64 cases = 37,7 millions ; × 60 ≈ 2,3 milliards
//      de multiplications-additions par position (~4,5 GFLOP). C'est pourquoi
//      le réseau tourne sur le GPU, en demi-précision (fp16), par lots.
//
//  ENTRAÎNEMENT (supervisé, nnue/cnn_resnet_value_policy*.py) :
//    • Données : les positions des parties des fichiers games_* (écrits par
//      data/generate_games.cpp : quelques coups aléatoires au début, puis deux
//      joueurs programmés ; les positions des coups aléatoires et les parties
//      entièrement aléatoires sont exclues), encodées une fois pour toutes par
//      nnue/preprocess.py.
//    • Cibles : pour la valeur, le résultat final z ∈ {−1, 0, +1} du point de
//      vue du camp au trait ; pour la politique, le coup réellement joué dans
//      la partie (indice départ × 64 + arrivée).
//    • Perte : MSE(v, z) + entropie croisée(logits, coup joué), poids 1 et 1
//      (comme AlphaZero). Optimiseur Adam, pas 0.001 qui décroît en cosinus
//      jusqu'à 1e-5, « weight decay » 1e-4 (version par blocs,
//      cnn_resnet_value_policy_chunked.py, celle que lance le .sh).
//      Entraînement sur plusieurs GPU (DDP), en précision mixte (bf16 sur les
//      GPU Ampere et plus récents, fp16 sinon).
//
//  EXPORT VERS LE C++ (nnue/resnet_export.py) :
//    • <nom>.ts : le réseau en TorchScript, « tracé » en mode évaluation
//      (les BatchNorm deviennent des constantes). C'est ce fichier que charge
//      l'app (models/ ou nnue/cnn_resnet_256x30_value_policy.ts) et que lit
//      la classe TorchNet du .cpp (section 9). Sortie : un tuple (valeurs (n,),
//      logits (n, 4096)) pour une entrée (n, 4, 8, 8).
//    • <nom>.bin : les poids à plat, BN repliées dans les convolutions, pour un
//      backend C++ pur (Eigen) sans GPU, très lent (~150 positions/s), retiré ici.
//    • <nom>.test.bin : quelques positions avec les sorties exactes de PyTorch,
//      pour vérifier les backends C++ (test/alphazero_check_main.cpp).
//
//  ---------------------------------------------------------------------------
//  L'INTERFACE AVEC LE C++ (sections 8 à 10 du .cpp)
//  ---------------------------------------------------------------------------
//  La recherche ne manipule jamais de tenseur. Elle donne une liste de
//  REQUÊTES (une position + ses coups légaux) et reçoit pour chacune un
//  RÉSULTAT (la valeur, et les logits de ces coups-là, dans le même ordre) :
//
//     threads de recherche              regroupeur (Batcher)        réseau (TorchNet)
//     ───────────────────              ────────────────────        ─────────────────
//     thread 0 : 16 feuilles ──┐
//     thread 1 : 16 feuilles ──┼──►  file d'attente ──► un lot de ≤ 256 ──► GPU, une passe
//     …                        │     (attend que tous les                  (n,4,8,8) → v, logits
//     thread 15: 16 feuilles ──┘      threads soient là,       ◄── résultats redistribués
//                                     ou 0,5 ms au plus)            à chaque thread
//
//    • TorchNet (copie commentée de nnue/resnet_torch.cpp) : encode les
//      positions dans un tampon CPU « épinglé » (copie DMA rapide vers le GPU),
//      le copie sur le GPU en fp16, fait UNE passe avant, recopie les deux
//      sorties, et extrait pour chaque position les logits de ses coups légaux.
//    • Batcher (copie de nnue/batched_evaluator.cpp) : 16 threads × 16
//      feuilles = des lots de 256 au lieu de 16. Sans lui, chaque thread
//      appellerait le GPU avec 16 positions : 16 fois plus d'appels.
//    • NNCache (copie de player/nn_cache.h) : la sortie du réseau ne dépend que
//      du plateau et du camp au trait ; ces paires reviennent souvent par
//      transposition (et d'un coup à l'autre), on les garde en mémoire (64 Mo).
//
//  ---------------------------------------------------------------------------
//  CONFIGURATION DE L'APP (config/hybrid/resnet.cfg) ET VALEURS PAR DÉFAUT
//  ---------------------------------------------------------------------------
//    microseconds 1 000 000 (selon le niveau)   nb threads 16   batch size 16
//    cpuct exploration 1.0, log 0.45, base 500  (section 3)
//    cpuct utility stdev scale 0.85, prior 0.40, prior weight 2.0
//    fpu reduction 0.2, root fpu reduction 0.1 (pondération par la masse
//      d'a priori visitée : oui, puissance 1 ; « loss prop » 0)
//    policy temperature 1.0   graph search oui   reuse tree oui   nn cache 64 Mo
//    use lcb oui, lcb stdevs 5, min visit prop for lcb 0.2
//    policy target pruning oui (il modifie aussi le choix du coup, section 7)
//    score utility 0.05, score scale 10
//    backend torch, device cuda, fp16 oui ; « merged batch size » = 16 × 16
//  Ces valeurs sont celles de KataGo (cpp/search/searchparams.cpp), sauf
//  l'utilité de score (spécifique à Yolah). Elles n'ont pas été réglées pour
//  Yolah.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_mcts_check.cpp (hors de la compilation normale ; compilé par
//  test/final_mcts_check.sh, lancé depuis config/) joue les deux versions côte à côte sur les positions
//  de test/bench_positions.txt : un thread, 800 simulations, réseau en fp32,
//  et compare pour chaque position le coup, les visites et le Q de chaque
//  coup de la racine. Toute modification de ce code doit être revérifiée.
//
//  Dernière vérification (23/11/2026, RTX 3080) : 209 / 209 positions
//  identiques, 411 recherches (la position, puis deux plis plus loin depuis
//  le graphe réutilisé) : mêmes coups, visites, Q et valeurs de jeu.
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

// ── Requête et résultat : l'interface entre la recherche et le réseau ──────
struct Request {
    const Yolah* state;     // la position à évaluer
    const Move*  moves;     // ses coups légaux (dans l'ordre de Yolah::moves), au moins 1
    uint16_t     nb_moves;
};
struct Result {
    float value;                        // valeur pour le camp au trait de la position, dans (−1, 1)
    float logits[Yolah::MAX_NB_MOVES];  // logits de politique des coups moves[0..nb_moves), même ordre
};

// Ce que la recherche sait d'un réseau : évaluer un lot. Deux réalisations :
// TorchNet (le réseau lui-même) et Batcher (qui regroupe les lots de plusieurs
// threads et les passe à un TorchNet).
class Evaluator {
public:
    virtual ~Evaluator() = default;
    // Évalue in.size() positions ; out.size() >= in.size(). Appelable par
    // plusieurs threads à la fois.
    virtual void evaluate(std::span<const Request> in, std::span<Result> out) = 0;
};

// ── 9. Le réseau, exécuté par libtorch (copie de nnue/resnet_torch.cpp) ────
class TorchNet final : public Evaluator {
public:
    // fichier : le .ts de resnet_export.py ; device : "cuda" (ou "cpu") ;
    // fp16 : demi-précision (seulement sur GPU) ; warm_batch : la plus grande
    // taille de lot préparée au chargement (256 = 16 threads × 16 feuilles).
    TorchNet(const std::string& filename, const std::string& device, bool fp16, int64_t warm_batch);
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    torch::Device device;
    bool fp16;
    torch::jit::script::Module module;
    std::mutex mutex;          // une passe à la fois (les appels sont déjà regroupés)
    torch::Tensor input_cpu;   // tampon d'entrée en mémoire épinglée, agrandi si besoin
};

// ── 10. Le regroupeur des requêtes (copie de nnue/batched_evaluator.cpp) ───
class Batcher final : public Evaluator {
public:
    // net : le réseau (possédé) ; nb_clients : le nombre de threads de
    // recherche (un lot part dès qu'ils attendent tous) ; max_batch : la
    // taille maximale d'un lot fusionné ; max_wait : l'attente maximale des
    // retardataires avant de lancer un lot partiel.
    Batcher(std::unique_ptr<Evaluator> net, size_t nb_clients, size_t max_batch,
            std::chrono::microseconds max_wait = std::chrono::microseconds(500));
    ~Batcher() override;
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    struct Job {               // la demande d'un thread, sur sa pile pendant qu'il attend
        std::span<const Request> in;
        std::span<Result> out;
        bool done = false;
    };
    void worker_loop();
    std::unique_ptr<Evaluator> net;
    const size_t nb_clients, max_batch;
    const std::chrono::microseconds max_wait;
    std::mutex mutex;
    std::condition_variable cv_worker;   // réveille le thread du regroupeur
    std::condition_variable cv_clients;  // réveille les threads de recherche servis
    std::deque<Job*> queue;
    size_t queued_positions = 0;         // positions en file
    size_t waiting_clients = 0;          // threads en attente (demandes vides comprises)
    bool stop = false;
    std::vector<Request> batch_in;       // le lot fusionné (au thread du regroupeur)
    std::vector<Result> batch_out;
    std::thread worker;                  // déclaré en dernier : démarré quand le reste est prêt
};

// ── 8. Le cache des sorties du réseau (copie de player/nn_cache.h) ─────────
class NNCache {
public:
    explicit NNCache(size_t size_bytes);
    // true (et value, priors[0..nb_moves)) si la position est en cache.
    bool lookup(uint64_t key, size_t nb_moves, float& value, float* priors);
    void store(uint64_t key, size_t nb_moves, float value, const float* priors);
private:
    struct Entry {
        uint64_t key = 0;                       // le hachage Zobrist complet (vérification)
        float    value = 0.0f;
        uint16_t nb_moves = 0;                  // vérification aussi
        uint16_t priors[Yolah::MAX_NB_MOVES] = {};   // a priori en virgule fixe (1/65535)
    };
    std::vector<Entry> table;
    size_t mask = 0;
    std::array<std::mutex, 256> locks;           // verrous « en bandes »
};

// ── Les réglages (ceux de l'app ; voir la liste plus haut) ─────────────────
struct Params {
    uint64_t microseconds = 1'000'000;   // budget de temps par coup (0 = pas de limite de temps)
    uint64_t nb_simulations = 0;         // budget de simulations (0 = pas de limite ; la vérification : 800)
    size_t nb_threads = 16;              // threads de recherche
    size_t batch_size = 16;              // feuilles par thread et par appel au réseau
    size_t nn_cache_mb = 64;
    float cpuct_exploration = 1.0f, cpuct_exploration_log = 0.45f, cpuct_exploration_base = 500.0f;
    float cpuct_utility_stdev_scale = 0.85f, cpuct_utility_stdev_prior = 0.40f, cpuct_utility_stdev_prior_weight = 2.0f;
    float fpu_reduction = 0.2f, root_fpu_reduction = 0.1f;
    float policy_temperature = 1.0f;
    bool  use_lcb = true;
    float lcb_stdevs = 5.0f, min_visit_prop_for_lcb = 0.20f;
    bool  policy_target_pruning = true;
    float score_utility = 0.05f, score_scale = 10.0f;
    bool  reuse_tree = true;
};

// Ce que la recherche rapporte de la racine (pour le coup et pour la vérification).
struct ChildStat {
    Move  move;
    float prior;        // P(racine, coup)
    uint32_t visits;    // N(racine, coup) : visites de l'ARÊTE
    float q;            // Q du nœud enfant, pour le camp au trait à la racine
    float q2;           // moyenne des carrés des valeurs remontées dans l'enfant
    float lcb;          // borne inférieure de confiance (section 7)
    float play_value;   // les visites après élagage de la cible et ajustement LCB : le coup joué a la plus grande
    float target;       // les visites après l'élagage seul (avant la LCB) : leur somme nulle = aucune visite utile
};
struct SearchResult {
    Move best_move = Move::none();
    uint64_t nb_simulations = 0, nb_evaluations = 0;
    std::vector<ChildStat> children;     // triés par play_value décroissante
};

// ── La recherche (sections 1 à 7 du .cpp) ──────────────────────────────────
class Search {
public:
    Search(Evaluator& evaluator, const Params& params, std::pmr::memory_resource* memory);
    ~Search();
    SearchResult search(const Yolah& state);   // `state` ne doit pas être finie
    void reset();                              // oublie le graphe (nouvelle partie) ; le cache reste
private:
    struct Node;
    struct Edge;
    struct Worker;
    struct PathStep;
    enum : uint8_t { UNEXPANDED = 0, EXPANDING = 1, EXPANDED = 2, TERMINAL = 3 };
    static constexpr size_t NB_SHARDS = 64;
    struct Shard {                             // une tranche de la table des nœuds
        std::mutex mutex;
        std::unordered_map<uint64_t, Node*> nodes;
    };
    Node* new_node();
    Node* lookup(uint64_t key);
    void  destroy_all();
    void  mark(Node* node);
    void  collect(Node* keep);
    bool  reuse(uint64_t key);
    void  new_root(uint64_t key);
    void  expand_root(const Yolah& state);
    void  run_worker(Worker& w, const Yolah& root_state);
    uint32_t select_child(const Node& node, bool is_root) const;
    float utility_stdev_factor(const Node& node, uint32_t visits) const;
    float fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const;
    void  priors_from_logits(const Result& result, size_t nb_moves, float* priors) const;
    void  expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value);
    void  backup(const PathStep* path, size_t length, float leaf_value);
    void  undo_pending(const PathStep* path, size_t length);
    float terminal_value(const Yolah& state) const;
    void  root_play_values(std::vector<ChildStat>& children) const;
    SearchResult make_result();

    Evaluator& evaluator;
    Params prm;
    std::pmr::polymorphic_allocator<> alloc;   // les nœuds vivent dans un « pool » mémoire
    Shard shards[NB_SHARDS];
    Node* root = nullptr;
    std::unique_ptr<NNCache> cache;
    uint32_t mark_counter = 0;
    std::atomic<uint64_t> simulations{0};      // simulations terminées pendant cette recherche
    std::atomic<uint64_t> reserved{0};         // simulations réservées par les threads (budget)
    std::atomic<uint64_t> evaluations{0};      // positions envoyées au réseau
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point deadline;
};

// ── Le joueur (copie de player/alphazero_mcts_player.cpp, réglages de l'app) ─
class Player {
public:
    // weights : le .ts ; nb_threads > 1 : les requêtes passent par un Batcher.
    Player(const std::string& weights, const Params& params, bool fp16 = true, const std::string& device = "cuda");
    ~Player();
    Move play(const Yolah& yolah);
    void game_over();                           // à appeler à la fin de chaque partie
    const SearchResult& last_result() const { return last; }
private:
    std::pmr::synchronized_pool_resource memory;   // avant searcher : détruit après lui
    std::unique_ptr<Evaluator> net;
    std::unique_ptr<Search> searcher;
    SearchResult last;
};

} // namespace final_mcts

#endif
