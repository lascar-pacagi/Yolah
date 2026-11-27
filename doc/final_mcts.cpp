// =============================================================================
//  final_mcts.cpp — le joueur MCTS ResNet de l'app, épuré et commenté.
//  Lire d'abord doc/final_mcts.h (le jeu, l'idée du MCTS, le réseau, le
//  glossaire, les réglages). Plan :
//    1. Nœuds, arêtes, chemins, état d'un thread
//    2. Le graphe : clé d'une position, table des nœuds, réutilisation d'un
//       coup à l'autre, ramasse-miettes
//    3. Le choix d'un coup pendant la descente : PUCT, facteur σ̂, FPU
//    4. Développement d'un nœud : softmax de la politique
//    5. Remontée des valeurs et perte virtuelle ; valeur d'une fin de partie
//    6. La boucle d'un thread de recherche (rassembler un lot, l'évaluer,
//       développer, remonter)
//    7. Le choix final du coup à la racine (élagage de la cible, LCB) et la
//       fonction search()
//    8. Le cache des sorties du réseau
//    9. Le réseau exécuté par libtorch, et l'encodage d'une position
//   10. Le regroupeur des requêtes des threads
//   11. Le joueur
// =============================================================================
#include "final_mcts.h"
#include "zobrist.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace final_mcts {

// =============================================================================
//  1. NŒUDS, ARÊTES, CHEMINS, ÉTAT D'UN THREAD
// =============================================================================
//
// Toutes les statistiques sont des `std::atomic` modifiés en ordre « relaxed »
// (sans barrière mémoire) : 16 threads les lisent et les écrivent en même
// temps, sans verrou. Une lecture légèrement périmée ne fait que changer la
// feuille choisie par une descente — c'est une heuristique — et ne peut pas
// casser le graphe. Les deux seules publications qui comptent utilisent
// acquire/release :
//   • Node::state : un thread qui lit EXPANDED (acquire) voit à coup sûr le
//     vecteur des arêtes entièrement construit (écrit avant le store release) ;
//   • Edge::child : un thread qui lit un pointeur non nul voit le nœud créé.
struct Search::Edge {
    Move  action = Move::none();        // le coup
    float prior  = 0.0f;                // P(s, a), de la tête de politique (softmax, section 4)
    std::atomic<Search::Node*> child{nullptr};   // le nœud d'arrivée, créé à la première descente par ce coup
    std::atomic<uint32_t> n{0};         // N(s, a) : visites TERMINÉES passées par cette arête
    std::atomic<uint32_t> pending{0};   // descentes EN COURS par cette arête (perte virtuelle, section 5)

    Edge() = default;
    Edge(Move m, float p) : action(m), prior(p) {}
    // Un std::atomic ne se copie pas ; std::pmr::vector exige pourtant un
    // constructeur de copie (pour reserve/emplace_back). Il ne sert qu'à la
    // construction du vecteur, avant toute publication : pas de concurrence.
    Edge(const Edge& o)
        : action(o.action), prior(o.prior), child(o.child.load()), n(o.n.load()), pending(o.pending.load()) {}
    Edge& operator=(const Edge& o) {
        action = o.action; prior = o.prior;
        child = o.child.load(); n = o.n.load(); pending = o.pending.load();
        return *this;
    }
};

struct Search::Node {
    std::atomic<uint32_t> n{0};         // N(s) : visites terminées, TOUS parents confondus
    std::atomic<uint32_t> pending{0};   // descentes en cours qui passent ici (chacune compte comme une défaite)
    std::atomic<float>    w{0.0f};      // W : Σ des valeurs remontées, pour le camp qui a joué le coup MENANT ici
    std::atomic<float>    w2{0.0f};     // Σ de leurs carrés (le signe disparaît : sert à l'écart-type)
    std::atomic<uint8_t>  state{UNEXPANDED};   // UNEXPANDED → EXPANDING (évaluation en cours) → EXPANDED, ou TERMINAL
    uint32_t mark = 0;                  // ramasse-miettes : dernier numéro de marquage qui l'a atteint (section 2)
    float    net_v = 0.0f;              // la valeur du réseau ICI, pour le camp au trait ici (sert à la FPU)
    std::pmr::vector<Edge> children;    // une arête par coup légal, dans l'ordre de Yolah::moves

    // Le nœud est créé par l'allocateur « pool » de la recherche
    // (alloc.new_object<Node>()), qui passe son allocateur au constructeur :
    // le vecteur des arêtes est alors alloué dans le même pool.
    using allocator_type = std::pmr::polymorphic_allocator<>;
    explicit Node(allocator_type a = {}) : children(a) {}
};

// Un pli d'une descente : le nœud atteint et l'arête prise pour l'atteindre
// (nullptr pour la racine, où aucune arête ne mène).
struct Search::PathStep {
    Node* node;
    Edge* edge;
};

// Worker : les tableaux de travail d'UN thread de recherche.
//
// Chaque thread fait des lots : il lance jusqu'à batch_size descentes, note
// où chacune s'est arrêtée, envoie les feuilles au réseau en une fois, puis
// remonte les valeurs (section 6). Pour un lot, il lui faut de la place :
//   leaves    les feuilles à évaluer (position, coups légaux) ;
//   descents  pour chaque descente, sa feuille et la longueur de son chemin ;
//   paths     le chemin de chaque descente (les nœuds et arêtes traversés,
//             pour la remontée) ;
//   requests  ce qu'on envoie au réseau, results ce qu'il répond.
//
// « Propre à un thread » : chaque thread a SON Worker. Personne d'autre n'y
// écrit, donc aucun verrou n'est nécessaire pour ces tableaux.
//
// « Réutilisés d'un lot à l'autre » : ils sont alloués UNE fois, à la
// création du Worker, à la taille d'un lot (reserve, ou taille fixe pour
// paths et results). À chaque lot, on les vide (clear garde la mémoire déjà
// réservée) et on les remplit de nouveau. Il n'y a donc aucun new ni malloc
// pendant la recherche : une allocation coûte cher (des centaines de
// nanosecondes, et plusieurs threads qui allouent en même temps se gênent
// dans l'allocateur), alors qu'un lot ne dure que quelques millisecondes et
// qu'il y en a des milliers par coup.
struct Search::Worker {
    // Un chemin racine → feuille a au plus autant de plis que la partie (120)
    // plus la racine ; +8 de marge.
    static constexpr size_t MAX_PATH = Yolah::MAX_NB_PLIES + 8;
    // Une feuille à évaluer (distincte : deux descentes qui tombent sur la
    // même feuille dans un lot n'en font qu'une).
    struct Leaf {
        Node*    node;
        uint64_t hash;           // son hachage Zobrist (plateau + trait), la clé du cache
        Yolah    state;          // la position (le réseau en a besoin)
        Yolah::MoveList moves;   // ses coups légaux, dans l'ordre des arêtes à venir
    };
    // Une descente terminée sur la feuille `leaf` : sa valeur remontera le
    // long de SON chemin. Deux descentes d'un même lot peuvent atteindre la
    // même feuille par des chemins différents (même par des parents
    // différents, en graphe) : chacune garde le sien.
    struct Descent {
        uint32_t leaf;           // indice dans `leaves`
        uint32_t path_len;       // longueur de son chemin dans `paths`
    };
    std::vector<Leaf> leaves;
    std::vector<Descent> descents;
    std::vector<PathStep> paths;         // (batch_size + 1) chemins de MAX_PATH plis, bout à bout
    std::vector<Request> requests;       // les requêtes du lot (pointent dans `leaves`)
    std::vector<Result> results;         // et leurs résultats

    explicit Worker(size_t batch_size)
        : paths((batch_size + 1) * MAX_PATH, PathStep{nullptr, nullptr}), results(batch_size) {
        leaves.reserve(batch_size);
        descents.reserve(batch_size);
        requests.reserve(batch_size);
    }
    // Le chemin de la k-ième descente du lot.
    PathStep* path(size_t k) { return paths.data() + k * MAX_PATH; }
};

// =============================================================================
//  2. LE GRAPHE : CLÉ D'UNE POSITION, TABLE DES NŒUDS, RÉUTILISATION
// =============================================================================
//
// RECHERCHE EN GRAPHE (et pas en arbre). Dans Yolah, la même position est
// atteinte par un grand nombre d'ordres de coups : jouer a puis b, ou b puis
// a, avec deux pièces différentes, donne la même position. Un arbre aurait un
// nœud par CHEMIN et referait le même travail sur chaque copie ; ici il y a un
// nœud par POSITION, dans une table de hachage partagée par tous les threads,
// et tous les chemins qui y mènent partagent ses statistiques : une suite
// réfutée par un chemin l'est pour tous. Mesure (CPU, 3200 simulations) : 5 à
// 15 % plus lent par simulation (verrous de la table), mais 20 à 25 % de
// positions distinctes examinées en plus.
//
// Pas de cycles : chaque coup (hors passe) ajoute un trou, une position ne
// revient jamais, et deux passes de suite finissent la partie. Les méthodes
// récursives (mark) terminent donc sans précaution particulière.
//
// Ce que le graphe change dans les formules (KataGo, « graph search ») :
//   • Q d'un coup = W / N du NŒUD enfant (tout ce que la recherche sait de
//     cette position, quel que soit le parent qui a payé ces visites) ;
//   • le dénominateur de l'exploration de PUCT = N(s, a), les visites de
//     l'ARÊTE (la part de CE parent), sinon un enfant très visité par un autre
//     parent ne serait jamais exploré depuis celui-ci.
namespace {
    // Un « mélangeur » 64 bits (splitmix64 / Stafford variante 13) : transforme
    // un petit entier en 64 bits bien répartis.
    uint64_t mix(uint64_t x) {
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27; x *= 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    // La clé d'une position dans la table des nœuds. zobrist::hash couvre le
    // plateau et le camp au trait : exactement ce que voit le réseau, d'où la
    // clé du cache (section 8). Mais il ne couvre PAS la répartition des points
    // déjà marqués entre les deux camps : deux positions de même plateau
    // peuvent la différer (une passe ne marque rien et ne fait pas de trou),
    // et la valeur d'une fin de partie en dépend. Un nœud porte des
    // statistiques de résultats, pas seulement la sortie du réseau : sa clé
    // mélange donc aussi le score de Noir (le score de Blanc s'en déduit,
    // le total = le nombre de coups joués hors passe).
    uint64_t node_key(uint64_t zobrist_hash, const Yolah& state) {
        return zobrist_hash ^ mix(uint64_t(state.score().first) + 1);
    }
}

Search::Search(Evaluator& ev, const Params& params, std::pmr::memory_resource* memory)
    : evaluator(ev), prm(params), alloc(memory) {
    if (prm.microseconds == 0 && prm.nb_simulations == 0)
        throw std::invalid_argument("final_mcts::Search: microseconds or nb_simulations must be > 0");
    prm.nb_threads = std::max<size_t>(1, prm.nb_threads);
    prm.batch_size = std::max<size_t>(1, prm.batch_size);
    cache = std::make_unique<NNCache>(prm.nn_cache_mb << 20);   // Mo → octets
}

Search::~Search() { destroy_all(); }

Search::Node* Search::new_node() { return alloc.new_object<Node>(); }

// La table est coupée en 64 tranches (« shards »), chacune avec son verrou :
// deux threads ne se bloquent que s'ils touchent la même tranche (1 fois sur
// 64). La tranche est choisie par la clé.
Search::Node* Search::lookup(uint64_t key) {
    Shard& sh = shards[key % NB_SHARDS];
    std::lock_guard lock(sh.mutex);
    auto it = sh.nodes.find(key);
    return it == sh.nodes.end() ? nullptr : it->second;
}

void Search::destroy_all() {
    for (Shard& sh : shards) {
        for (auto& [key, node] : sh.nodes) alloc.delete_object(node);
        sh.nodes.clear();
    }
    root = nullptr;
}

void Search::reset() { destroy_all(); }   // nouvelle partie : le graphe s'en va, le cache reste (il ne dépend que du réseau)

// Ramasse-miettes « marquer et balayer » : après un coup, seule la partie du
// graphe atteignable depuis la nouvelle racine peut encore servir. mark()
// parcourt en profondeur depuis la racine et numérote ce qu'il atteint ;
// collect() libère tout nœud de la table qui n'a pas le numéro courant.
void Search::mark(Node* node) {
    if (!node || node->mark == mark_counter) return;   // déjà marqué (atteint par un autre chemin)
    node->mark = mark_counter;
    if (node->state.load(std::memory_order_relaxed) != EXPANDED) return;   // pas d'arêtes
    for (Edge& e : node->children) mark(e.child.load(std::memory_order_relaxed));
}

void Search::collect(Node* keep) {
    ++mark_counter;
    mark(keep);
    for (Shard& sh : shards)
        for (auto it = sh.nodes.begin(); it != sh.nodes.end();) {
            if (it->second->mark != mark_counter) {
                alloc.delete_object(it->second);
                it = sh.nodes.erase(it);
            } else {
                ++it;
            }
        }
}

// Nouvelle racine sans rien garder (première recherche de la partie, ou
// position absente du graphe).
void Search::new_root(uint64_t key) {
    destroy_all();
    Shard& sh = shards[key % NB_SHARDS];
    root = sh.nodes[key] = new_node();
}

// RÉUTILISATION D'UN COUP À L'AUTRE (« reuse tree ») : entre deux coups du
// joueur, deux plis ont été joués (le sien puis celui de l'adversaire). Si la
// recherche précédente avait déjà créé la nouvelle position, toutes ses
// statistiques sont là : la recherche repart de ce nœud au lieu de zéro (à 1 s
// par coup, souvent des centaines de visites gagnées). En graphe, la table des
// nœuds suffit : la position est retrouvée par sa clé, à n'importe quelle
// profondeur de l'ancien graphe.
bool Search::reuse(uint64_t key) {
    if (!root) return false;
    Node* node = lookup(key);
    if (!node) return false;
    root = node;
    collect(root);      // tout ce que la nouvelle racine n'atteint pas est libéré
    return true;
}

// La racine doit être développée avant que les threads ne démarrent : un seul
// appel au réseau, d'une position.
void Search::expand_root(const Yolah& state) {
    if (state.game_over()) {
        root->state.store(TERMINAL, std::memory_order_release);
        return;
    }
    Yolah::MoveList moves;
    state.moves(moves);
    Request request{&state, moves.begin(), static_cast<uint16_t>(moves.size())};
    Result result;
    evaluator.evaluate(std::span<const Request>(&request, 1), std::span<Result>(&result, 1));
    root->state.store(EXPANDING, std::memory_order_relaxed);
    float priors[Yolah::MAX_NB_MOVES];
    priors_from_logits(result, moves.size(), priors);
    expand(*root, moves, priors, result.value);
}

// =============================================================================
//  3. LE CHOIX D'UN COUP PENDANT LA DESCENTE : PUCT, FACTEUR σ̂, FPU
// =============================================================================
//
// À un nœud s développé, la descente prend le coup a qui maximise
//
//     score(a) = Q(s, a) + E · P(s, a) / (1 + N(s, a))
//     E        = c(s) · σ̂(s) · √(W_tot + 0.01)
//     c(s)     = c_expl + c_log · ln((W_tot + c_base) / c_base)
//
//   • Q(s, a) : la valeur moyenne de l'enfant (pour le camp au trait en s) ;
//     pour un enfant jamais visité, la FPU (plus bas).
//   • P(s, a) : l'a priori du réseau. Le terme d'exploration est grand pour un
//     coup que le réseau aime et que la recherche a peu visité ; il décroît en
//     1 / (1 + N(s, a)) à mesure qu'on le visite : la recherche finit par se
//     fier à Q.
//   • W_tot = Σ_a (N(s, a) + pending(s, a)) : le poids total des arêtes qui
//     partent de s (KataGo, totalChildWeight ; la visite du nœud lui-même n'y
//     est pas). √W_tot fait croître l'exploration avec le nombre de visites,
//     moins vite que les visites elles-mêmes : un coup peu visité finit
//     toujours par être réessayé. Le +0.01 évite √0 au premier passage (tous
//     les scores seraient égaux à la FPU, et l'a priori ne départagerait plus).
//   • c(s) : c_expl = 1.0, c_log = 0.45, c_base = 500 (KataGo). Le terme en
//     logarithme augmente doucement l'exploration quand les visites deviennent
//     nombreuses (≈ +0.31 à 500 visites, +1.08 à 5000).
//   • σ̂(s) : le facteur d'écart-type de KataGo (utility_stdev_factor) : on
//     explore plus là où les valeurs remontées se contredisent (position
//     incertaine), moins là où elles concordent.
//
// Tous les comptes incluent les descentes en cours (pending) : c'est ce qui
// fait que 16 descentes lancées en même temps ne prennent pas toutes le même
// chemin (section 5).

// σ̂ = 1 + 0.85 · (écart-type / 0.40 − 1). L'écart-type des valeurs remontées
// dans le nœud est « régularisé » vers l'a priori 0.40 avec un poids de 2
// visites fictives, pour qu'un nœud très peu visité n'ait pas un écart-type
// absurde (0 avec une visite) :
//     écart-type² = ((m² + 0.4²)·2 + moy(v²)·n) / (2 + n − 1) − m²
// m = W / n la moyenne, moy(v²) = W2 / n. Écart-type 0.40 → σ̂ = 1 ; une
// position où tout concorde (écart-type ~0.05) → σ̂ ≈ 0.26 ; très contestée
// (0.8) → σ̂ = 1.85.
float Search::utility_stdev_factor(const Node& node, uint32_t visits) const {
    const float prior = prm.cpuct_utility_stdev_prior;
    float stdev = prior;
    if (visits > 1) {
        const float n = static_cast<float>(visits);
        // Le signe de la moyenne est sans importance, seul son carré sert.
        const float mean    = node.w.load(std::memory_order_relaxed) / n;
        const float mean_sq = mean * mean;
        // W et W2 sont lus séparément, sans verrou : un autre thread a pu
        // remonter entre les deux lectures. On borne pour que la variance ne
        // sorte jamais négative.
        const float sq_avg  = std::max(node.w2.load(std::memory_order_relaxed) / n, mean_sq);
        const float pw = prm.cpuct_utility_stdev_prior_weight;
        stdev = std::sqrt(std::max(0.0f, ((mean_sq + prior * prior) * pw + sq_avg * n) / (pw + n - 1.0f) - mean_sq));
    }
    return 1.0f + prm.cpuct_utility_stdev_scale * (stdev / prior - 1.0f);
}

// FPU (« First Play Urgency ») : la valeur supposée d'un coup jamais visité.
// AlphaZero prenait 0 (« nul »), ce qui pousse à tout essayer dans une
// position perdue et à ne rien essayer dans une gagnée. KataGo prend la
// valeur du PARENT, diminuée :
//     fpu = base − réduction · √(masse d'a priori déjà visitée)
//   • base : un mélange de la moyenne Q du parent (ce que la recherche en sait)
//     et de la valeur brute du réseau au parent, pondéré par la masse d'a
//     priori des coups déjà essayés : tant qu'on n'a essayé que des coups peu
//     probables, on croit le réseau ; quand les coups probables ont été vus,
//     la moyenne de la recherche.
//   • réduction : 0.2 (0.1 à la racine, plus « généreuse » : on y veut voir
//     plus de coups). Multipliée par √(masse visitée) : au premier coup essayé,
//     pas de réduction ; quand les coups les plus probables ont été vus, les
//     autres sont supposés nettement moins bons — c'est ce qui concentre la
//     recherche, au lieu de l'étaler sur les 30 à 70 coups légaux.
// Tout est du point de vue du camp au trait dans `node` : node.w est pour le
// camp qui y a joué, d'où le signe −, et node.net_v pour le camp au trait.
float Search::fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const {
    const float parent_q = visits > 0 ? -node.w.load(std::memory_order_relaxed) / visits : node.net_v;
    const float avg_weight = std::min(1.0f, visited_policy_mass);
    const float baseline = avg_weight * parent_q + (1.0f - avg_weight) * node.net_v;
    const float reduction = (is_root ? prm.root_fpu_reduction : prm.fpu_reduction) * std::sqrt(visited_policy_mass);
    return baseline - reduction;
}

uint32_t Search::select_child(const Node& node, bool is_root) const {
    const auto& children = node.children;
    const uint32_t nb = static_cast<uint32_t>(children.size());
    const uint32_t n_parent = node.n.load(std::memory_order_relaxed);

    // Premier passage : W_tot, et la masse d'a priori des coups essayés au
    // moins une fois (visite terminée ou en cours).
    float total_child_weight = 0.0f, visited_mass = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const Edge& e = children[i];
        const uint32_t w = e.n.load(std::memory_order_relaxed) + e.pending.load(std::memory_order_relaxed);
        total_child_weight += static_cast<float>(w);
        if (w > 0) visited_mass += e.prior;
    }
    const float cpuct = prm.cpuct_exploration + prm.cpuct_exploration_log *
        std::log((total_child_weight + prm.cpuct_exploration_base) / prm.cpuct_exploration_base);
    const float explore_scaling = cpuct * utility_stdev_factor(node, n_parent) * std::sqrt(total_child_weight + 0.01f);
    const float fpu = fpu_value(node, n_parent, visited_mass, is_root);

    // Second passage : le meilleur score. À égalité, le premier coup dans
    // l'ordre de Yolah::moves (comparaison stricte).
    uint32_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < nb; i++) {
        const Edge& e = children[i];
        const uint32_t n_eff = e.n.load(std::memory_order_relaxed) + e.pending.load(std::memory_order_relaxed);
        // Q vient du NŒUD enfant (section 2) ; un enfant pas encore créé, ou
        // créé mais sans aucune visite (terminée ou en cours), vaut la FPU.
        float q = fpu;
        const Node* child = e.child.load(std::memory_order_acquire);
        if (child) {
            const uint32_t cn = child->n.load(std::memory_order_relaxed);
            const uint32_t cp = child->pending.load(std::memory_order_relaxed);
            // Perte virtuelle : chaque descente en cours compte comme une
            // défaite (−1) du camp qui joue ce coup. child->w est pour ce camp
            // (celui qui joue vers l'enfant = le camp au trait ici) : on
            // retranche cp. Une descente en cours rend donc ce coup moins
            // attirant pour la descente suivante.
            if (cn + cp > 0) q = (child->w.load(std::memory_order_relaxed) - static_cast<float>(cp)) / (cn + cp);
        }
        const float score = q + explore_scaling * e.prior / (1.0f + n_eff);
        if (score > best_score) { best_score = score; best = i; }
    }
    return best;
}

// =============================================================================
//  4. DÉVELOPPEMENT D'UN NŒUD : SOFTMAX DE LA POLITIQUE
// =============================================================================
//
// Le réseau donne un logit ℓ_a par coup légal. L'a priori est le softmax sur
// les coups LÉGAUX seulement (les 4096 − k autres logits sont ignorés) :
//     P(s, a) = exp((ℓ_a − ℓ_max) / T) / Σ_b exp((ℓ_b − ℓ_max) / T),   T = 1
// Retrancher ℓ_max ne change rien au résultat mais évite que exp() déborde.
// Une température T > 1 aplatirait les a priori (non utilisé : T = 1).
void Search::priors_from_logits(const Result& result, size_t nb_moves, float* priors) const {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t m = 0; m < nb_moves; m++) max_logit = std::max(max_logit, result.logits[m]);
    float sum = 0.0f;
    const float inv_t = 1.0f / prm.policy_temperature;
    for (size_t m = 0; m < nb_moves; m++) {
        priors[m] = std::exp((result.logits[m] - max_logit) * inv_t);
        sum += priors[m];
    }
    for (size_t m = 0; m < nb_moves; m++) priors[m] /= sum;
}

// Les arêtes sont créées toutes ensemble (une par coup légal, sans nœud
// d'arrivée : ceux-ci ne seront créés qu'à la première descente par le coup),
// puis publiées par le store release de `state`.
void Search::expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value) {
    const size_t nb = moves.size();
    node.children.reserve(nb);
    for (size_t m = 0; m < nb; m++) node.children.emplace_back(moves[m], priors[m]);
    node.net_v = value;
    node.state.store(EXPANDED, std::memory_order_release);
}

// =============================================================================
//  5. REMONTÉE DES VALEURS, PERTE VIRTUELLE, VALEUR D'UNE FIN DE PARTIE
// =============================================================================
//
// La valeur d'une feuille est pour le camp AU TRAIT à la feuille. Le W d'un
// nœud est pour le camp qui a joué le coup menant à ce nœud, c'est-à-dire
// l'autre : d'où le premier changement de signe, puis un autre à chaque pli
// en remontant (negamax). La même boucle termine les descentes en cours :
// pending −1 partout où la descente avait mis +1 en descendant.
void Search::backup(const PathStep* path, size_t length, float leaf_value) {
    float v = -leaf_value;
    const float v2 = leaf_value * leaf_value;   // le carré est le même à chaque pli
    for (size_t i = length; i-- > 0;) {         // de la feuille (i = length − 1) à la racine (i = 0)
        Node* node = path[i].node;
        node->w.fetch_add(v, std::memory_order_relaxed);
        node->w2.fetch_add(v2, std::memory_order_relaxed);
        node->n.fetch_add(1, std::memory_order_relaxed);
        node->pending.fetch_sub(1, std::memory_order_relaxed);
        if (Edge* e = path[i].edge) {
            e->n.fetch_add(1, std::memory_order_relaxed);
            e->pending.fetch_sub(1, std::memory_order_relaxed);
        }
        v = -v;
    }
}

// Une descente abandonnée (elle est tombée sur une feuille qu'un AUTRE thread
// est en train d'évaluer) : on retire ses « pending » sans rien remonter.
void Search::undo_pending(const PathStep* path, size_t length) {
    for (size_t i = 0; i < length; i++) {
        path[i].node->pending.fetch_sub(1, std::memory_order_relaxed);
        if (Edge* e = path[i].edge) e->pending.fetch_sub(1, std::memory_order_relaxed);
    }
}

// Une fin de partie atteinte dans l'arbre : sa valeur exacte, pour le camp au
// trait, avec l'utilité de score (doc/final_mcts.h, « Échelle des valeurs ») :
// d = écart de points final pour le camp au trait,
//     valeur = 0.95·signe(d) + 0.05·tanh(d / 10).
float Search::terminal_value(const Yolah& state) const {
    const auto [black, white] = state.score();
    const int diff = (state.current_player() == Yolah::BLACK ? 1 : -1) * (int(black) - int(white));
    const float outcome = diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
    if (prm.score_utility <= 0.0f) return outcome;
    return (1.0f - prm.score_utility) * outcome + prm.score_utility * std::tanh(float(diff) / prm.score_scale);
}

// =============================================================================
//  6. LA BOUCLE D'UN THREAD DE RECHERCHE
// =============================================================================
//
// Chaque thread répète, jusqu'à la fin du temps (ou du budget de simulations) :
//   1. RASSEMBLER : jusqu'à batch_size (16) descentes depuis la racine.
//      Chacune met ses « pending » en descendant (perte virtuelle), ce qui
//      écarte la suivante du même chemin ; elle s'arrête sur :
//        • une fin de partie déjà connue (TERMINAL) : sa valeur exacte est
//          remontée tout de suite, sans réseau ;
//        • un nœud jamais développé (UNEXPANDED) : il passe en EXPANDING (il
//          est à ce thread), devient une feuille du lot — sauf si la partie
//          y est finie (TERMINAL, valeur remontée tout de suite) ou s'il est
//          dans le cache du réseau (développé et remonté tout de suite) ;
//        • un nœud EXPANDING : si c'est une feuille de CE lot, la descente
//          « monte dans le même bus » (elle remontera la même valeur, par son
//          propre chemin) ; si c'est celle d'un AUTRE thread, la descente est
//          annulée (une « collision »).
//      On arrête de rassembler après 16 visites, ou après 16 collisions
//      (l'arbre est alors trop petit, ou les autres threads ont pris toutes
//      les feuilles intéressantes : mieux vaut évaluer ce qu'on a).
//   2. ÉVALUER : un appel au réseau pour toutes les feuilles du lot (à
//      travers le regroupeur, qui fusionne avec les lots des autres threads).
//   3. DÉVELOPPER chaque feuille (ses arêtes, avec les a priori), la mettre en
//      cache, puis REMONTER une fois par descente, chacune par son chemin.
void Search::run_worker(Worker& w, const Yolah& root_position) {
    Node* const root_node = root;
    const uint64_t root_hash = zobrist::hash(root_position);
    const size_t max_collisions = prm.batch_size;
    float priors[Yolah::MAX_NB_MOVES];

    while (!stop.load(std::memory_order_relaxed)) {
        // ── 1. rassembler ──
        w.leaves.clear();
        w.descents.clear();
        w.requests.clear();
        size_t budget = prm.batch_size;
        if (prm.nb_simulations) {
            // Budget de simulations : chaque thread RÉSERVE ses visites dans
            // le compteur commun avant de descendre, pour que tous ensemble ne
            // dépassent jamais le budget (les réservations inutilisées sont
            // rendues plus bas). compare_exchange : une réservation atomique.
            uint64_t claimed = reserved.load(std::memory_order_relaxed);
            for (;;) {
                if (claimed >= prm.nb_simulations) { budget = 0; break; }
                const uint64_t want = std::min<uint64_t>(budget, prm.nb_simulations - claimed);
                if (reserved.compare_exchange_weak(claimed, claimed + want, std::memory_order_relaxed)) {
                    budget = want;
                    break;
                }
            }
            if (budget == 0) {
                if (simulations.load(std::memory_order_relaxed) >= prm.nb_simulations) break;
                // D'autres threads ont encore des réservations en cours :
                // attendre leur lot (une requête vide = rejoindre le prochain
                // lot du regroupeur, section 10) plutôt que tourner à vide.
                evaluator.evaluate({}, {});
                std::this_thread::yield();
                continue;
            }
        }
        size_t batch_collisions = 0, gathered_visits = 0;
        while (gathered_visits < budget && batch_collisions <= max_collisions) {
            // Une descente.
            PathStep* path = w.path(w.descents.size());
            size_t len = 0;
            Node* node = root_node;
            Yolah state = root_position;
            uint64_t hash = root_hash;
            node->pending.fetch_add(1, std::memory_order_relaxed);
            path[len++] = {node, nullptr};
            for (;;) {
                uint8_t st = node->state.load(std::memory_order_acquire);
                if (st == EXPANDED) {
                    // Choisir le coup (section 3), le jouer, mettre à jour le hachage.
                    Edge& e = node->children[select_child(*node, node == root_node)];
                    const uint8_t mover = state.current_player();
                    state.play(e.action);
                    hash = zobrist::update(hash, mover, e.action);
                    e.pending.fetch_add(1, std::memory_order_relaxed);
                    Node* child = e.child.load(std::memory_order_acquire);
                    if (!child) {
                        // Première descente par ce coup : relier l'arête au nœud de la
                        // position d'arrivée, en le créant s'il n'existe pas encore —
                        // ou en partageant celui qu'un autre chemin a déjà créé (le
                        // graphe). Le verrou de la tranche sérialise cette liaison ;
                        // on relit child sous le verrou (un autre thread a pu relier
                        // l'arête entre-temps).
                        const uint64_t key = node_key(hash, state);
                        Shard& sh = shards[key % NB_SHARDS];
                        std::unique_lock lock(sh.mutex);
                        child = e.child.load(std::memory_order_acquire);
                        if (!child) {
                            auto [it, inserted] = sh.nodes.try_emplace(key, nullptr);
                            if (inserted) it->second = new_node();
                            child = it->second;
                            e.child.store(child, std::memory_order_release);
                        }
                    }
                    child->pending.fetch_add(1, std::memory_order_relaxed);
                    path[len++] = {child, &e};
                    node = child;
                    continue;
                }
                if (st == TERMINAL) {
                    backup(path, len, terminal_value(state));
                    simulations.fetch_add(1, std::memory_order_relaxed);
                    ++gathered_visits;
                    break;
                }
                if (st == UNEXPANDED) {
                    // Prendre le nœud : UNEXPANDED → EXPANDING, atomiquement (si
                    // un autre thread le prend au même moment, un seul réussit).
                    uint8_t expected = UNEXPANDED;
                    if (node->state.compare_exchange_strong(expected, EXPANDING, std::memory_order_acq_rel,
                                                            std::memory_order_acquire)) {
                        ++gathered_visits;
                        if (state.game_over()) {
                            node->state.store(TERMINAL, std::memory_order_release);
                            backup(path, len, terminal_value(state));
                            simulations.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                        Worker::Leaf& leaf = w.leaves.emplace_back();
                        leaf.node = node;
                        leaf.state = state;
                        state.moves(leaf.moves);
                        leaf.hash = hash;
                        // La position est-elle dans le cache du réseau (une
                        // transposition déjà évaluée, ce coup-ci ou un précédent) ?
                        float value;
                        if (cache->lookup(leaf.hash, leaf.moves.size(), value, priors)) {
                            expand(*node, leaf.moves, priors, value);
                            backup(path, len, value);
                            simulations.fetch_add(1, std::memory_order_relaxed);
                            w.leaves.pop_back();
                        } else {
                            w.descents.push_back({static_cast<uint32_t>(w.leaves.size() - 1), static_cast<uint32_t>(len)});
                        }
                        break;
                    }
                    // Raté : le nœud est devenu EXPANDING, EXPANDED ou TERMINAL entre la lecture et
                    // l'échange. EXPANDED ou TERMINAL : on relit son état et on continue la descente.
                    if (expected != EXPANDING) continue;
                }
                // st == EXPANDING : la feuille est en cours d'évaluation.
                uint32_t mine = UINT32_MAX;
                for (uint32_t i = 0; i < w.leaves.size(); i++)
                    if (w.leaves[i].node == node) { mine = i; break; }
                if (mine != UINT32_MAX) {
                    // Une feuille de ce lot : la descente comptera comme une visite de plus.
                    w.descents.push_back({mine, static_cast<uint32_t>(len)});
                    ++gathered_visits;
                } else {
                    // Celle d'un autre thread : on annule la descente.
                    undo_pending(path, len);
                    ++batch_collisions;
                }
                break;
            }
        }
        if (prm.nb_simulations && gathered_visits < budget)
            reserved.fetch_sub(budget - gathered_visits, std::memory_order_relaxed);   // rendre le reste de la réservation

        // ── 2. évaluer ──
        const size_t nb = w.leaves.size();
        if (nb == 0 && gathered_visits == 0) {
            // Toutes les descentes ont heurté des feuilles des autres threads :
            // attendre le prochain lot (requête vide), plutôt que de
            // redescendre aussitôt sur les mêmes feuilles encore en évaluation.
            evaluator.evaluate({}, {});
            std::this_thread::yield();
        }
        if (nb > 0) {
            for (Worker::Leaf& leaf : w.leaves)
                w.requests.push_back({&leaf.state, leaf.moves.begin(), static_cast<uint16_t>(leaf.moves.size())});
            evaluator.evaluate(w.requests, std::span<Result>(w.results.data(), nb));
            evaluations.fetch_add(nb, std::memory_order_relaxed);
            // ── 3. développer ... ──
            for (size_t i = 0; i < nb; i++) {
                Worker::Leaf& leaf = w.leaves[i];
                const Result& r = w.results[i];
                priors_from_logits(r, leaf.moves.size(), priors);
                cache->store(leaf.hash, leaf.moves.size(), r.value, priors);
                expand(*leaf.node, leaf.moves, priors, r.value);
            }
            // ... et remonter, une fois par descente, chacune par son chemin.
            for (size_t d = 0; d < w.descents.size(); d++) {
                const Worker::Descent& desc = w.descents[d];
                backup(w.path(d), desc.path_len, w.results[desc.leaf].value);
            }
            simulations.fetch_add(w.descents.size(), std::memory_order_relaxed);
        }

        // L'horloge est regardée une fois par lot (un lot dure quelques ms).
        if (prm.microseconds && std::chrono::steady_clock::now() >= deadline) stop.store(true, std::memory_order_relaxed);
        if (prm.nb_simulations && simulations.load(std::memory_order_relaxed) >= prm.nb_simulations)
            stop.store(true, std::memory_order_relaxed);
    }
}

// =============================================================================
//  7. LE CHOIX FINAL DU COUP À LA RACINE, ET search()
// =============================================================================
//
// La « valeur de jeu » (play value) de chaque coup de la racine part de ses
// visites, puis subit deux corrections de KataGo ; le coup joué est celui de
// plus grande valeur de jeu.
//
//   a. ÉLAGAGE DE LA CIBLE (« policy target pruning ») : pour chaque coup
//      autre que le « meilleur » b, combien de visites PUCT lui aurait-il
//      données s'il n'avait pas été poussé par l'exploration ? On inverse la
//      formule de PUCT au score de b :
//          Q_c + E · P_c / (1 + n) = score(b)  ⇒  n = E·P_c / (score(b) − Q_c) − 1
//      et on garde le plus petit de ce n (arrondi au-dessus) et des visites
//      réelles. Utile surtout pour l'apprentissage (la cible de la politique),
//      mais il change aussi le choix du coup (via les seuils de la LCB).
//   b. LCB (« lower confidence bound ») : parmi les coups qui ont au moins
//      20 % de la valeur de jeu du meilleur, celui qui a la meilleure BORNE
//      INFÉRIEURE Q − 5·σ(Q) reçoit juste assez de valeur de jeu pour être
//      choisi. Un coup très visité mais à la valeur moyenne médiocre perd
//      contre un coup un peu moins visité mais nettement meilleur et sûr.
void Search::root_play_values(std::vector<ChildStat>& children) const {
    const size_t nb = children.size();
    if (nb == 0) return;
    float total = 0.0f;
    for (const ChildStat& c : children) total += static_cast<float>(c.visits);
    for (ChildStat& c : children) c.play_value = static_cast<float>(c.visits);

    // Le « meilleur » b : le coup qui maximise max(0, v − 1) + 2·P (écrit ici
    // v·max(0, v − 1)/max(1, v) comme dans KataGo) — le plus visité, départagé
    // par l'a priori P, mais où v = 1 compte comme 0 afin qu'une unique visite
    // ne suffise pas à détrôner un coup de meilleur a priori. (« non-LCB best ».)
    size_t best = 0;
    {
        float best_goodness = -1e30f;
        for (size_t i = 0; i < nb; i++) {
            const float v = static_cast<float>(children[i].visits);
            const float g = v * std::max(0.0f, v - 1.0f) / std::max(1.0f, v) + 2.0f * children[i].prior;
            if (g > best_goodness) { best_goodness = g; best = i; }
        }
    }

    // a. élagage de la cible
    if (prm.policy_target_pruning && total > 0.0f) {
        const float cpuct = prm.cpuct_exploration + prm.cpuct_exploration_log *
            std::log((total + prm.cpuct_exploration_base) / prm.cpuct_exploration_base);
        const float explore_scaling = cpuct * utility_stdev_factor(*root, root->n.load(std::memory_order_relaxed))
                                    * std::sqrt(total + 0.01f);
        const ChildStat& b = children[best];
        const float best_value = b.q + explore_scaling * b.prior / (1.0f + static_cast<float>(b.visits));
        for (size_t i = 0; i < nb; i++) {
            if (i == best) continue;
            ChildStat& c = children[i];
            if (c.visits == 0) { c.play_value = 0.0f; continue; }
            const float excess = best_value - c.q;     // l'exploration qu'il lui aurait fallu
            if (excess <= 0.0f) continue;              // sa seule valeur le met devant : on le garde entier
            const float wanted = explore_scaling * c.prior / excess - 1.0f;
            c.play_value = std::min(c.play_value, std::ceil(std::max(0.0f, wanted)));
        }
    }
    for (ChildStat& c : children) c.target = c.play_value;

    // b. LCB
    if (!prm.use_lcb || total <= 0.0f) return;
    std::vector<float> radius(nb, 0.0f);
    const float best_play = children[best].play_value;
    float best_lcb = -1e10f;
    size_t best_lcb_index = SIZE_MAX;
    for (size_t i = 0; i < nb; i++) {
        ChildStat& c = children[i];
        if (c.visits == 0) { c.lcb = -2.0f; radius[i] = 2.0f * prm.lcb_stdevs; continue; }
        // L'écart-type de la MOYENNE Q : σ(Q) = √(variance / n_eff). La
        // machinerie des échantillons pondérés de KataGo, avec des poids tous
        // égaux à 1 (chaque visite pèse 1) : somme des poids = somme des carrés
        // des poids = n, donc n_eff = n. S'y ajoute un a priori de variance
        // (le rayon pourrait être de toute l'échelle, +1 sur la moyenne des
        // carrés) de poids 1/n², qui s'efface vite avec les visites.
        const float n = static_cast<float>(c.visits);
        const float mean_sq = c.q * c.q;
        float sq_avg = std::max(c.q2, mean_sq + 1e-8f);
        const float prior_w = 1.0f / (n * n);
        sq_avg = (sq_avg * n + (sq_avg + 1.0f) * prior_w) / (n + prior_w);
        const float weight_sum = n + prior_w;
        const float ess = weight_sum * weight_sum / (n + prior_w * prior_w);
        const float var = std::max(0.0f, sq_avg - mean_sq);
        radius[i] = std::sqrt(var / ess) * prm.lcb_stdevs;
        c.lcb = c.q - radius[i];
        if (c.play_value > 0.0f && c.play_value >= prm.min_visit_prop_for_lcb * best_play)
            if (c.lcb > best_lcb) { best_lcb = c.lcb; best_lcb_index = i; }
    }
    if (best_lcb_index == SIZE_MAX) return;
    // Le gagnant reçoit juste assez de valeur de jeu pour battre chacun des
    // autres : un coup dont la borne est inférieure de `excess` aurait besoin
    // que son rayon grandisse d'un facteur (r + excess) / (r + 0.2·excess) pour
    // le rattraper, et le rayon décroît en 1/√visites : d'où le carré. (Facteur
    // au plus 5, donc au plus 25 × la valeur de jeu de l'autre.)
    float adjusted = children[best_lcb_index].play_value;
    for (size_t i = 0; i < nb; i++) {
        if (i == best_lcb_index) continue;
        const float excess = best_lcb - children[i].lcb;
        if (excess < 0.0f) continue;   // ce coup n'a perdu que sur le seuil des 20 %
        const float factor = (radius[i] + excess) / (radius[i] + 0.20f * excess);
        adjusted = std::max(adjusted, factor * factor * children[i].play_value);
    }
    children[best_lcb_index].play_value = adjusted;
}

SearchResult Search::make_result() {
    SearchResult res;
    res.nb_simulations = simulations.load();
    res.nb_evaluations = evaluations.load();
    const auto& children = root->children;
    res.children.reserve(children.size());
    for (size_t i = 0; i < children.size(); i++) {
        const Edge& e = children[i];
        const Node* c = e.child.load(std::memory_order_acquire);
        const uint32_t cn = c ? c->n.load() : 0;
        ChildStat st{};
        st.move   = e.action;
        st.prior  = e.prior;
        st.visits = e.n.load();                 // les visites de l'ARÊTE (celles de ce parent)
        st.q      = cn > 0 ? c->w.load() / cn : 0.0f;    // le Q du NŒUD (pour le camp qui joue ce coup = au trait à la racine)
        st.q2     = cn > 0 ? c->w2.load() / cn : 0.0f;
        st.lcb    = -2.0f;
        res.children.push_back(st);
    }
    root_play_values(res.children);
    // Tri par valeur de jeu décroissante, Q pour départager ; stable : à
    // égalité complète, l'ordre de Yolah::moves.
    std::stable_sort(res.children.begin(), res.children.end(), [](const ChildStat& a, const ChildStat& b) {
        return a.play_value != b.play_value ? a.play_value > b.play_value : a.q > b.q;
    });
    if (res.children.empty()) return res;
    // Aucune visite utile (le temps n'a pas suffi à une seule simulation) : le
    // coup que le réseau préfère.
    float total = 0.0f;
    for (const ChildStat& c : res.children) total += c.target;
    size_t pick = 0;
    if (total == 0.0f)
        for (size_t i = 1; i < res.children.size(); i++)
            if (res.children[i].prior > res.children[pick].prior) pick = i;
    res.best_move = res.children[pick].move;
    return res;
}

SearchResult Search::search(const Yolah& state) {
    using namespace std::chrono;
    const auto start = steady_clock::now();
    // La racine : réutilisée si possible (section 2), sinon neuve, et développée.
    const uint64_t key = node_key(zobrist::hash(state), state);
    if (!(prm.reuse_tree && reuse(key))) new_root(key);
    if (root->state.load(std::memory_order_acquire) == UNEXPANDED) expand_root(state);

    std::vector<std::unique_ptr<Worker>> workers;
    for (size_t t = 0; t < prm.nb_threads; t++) workers.push_back(std::make_unique<Worker>(prm.batch_size));
    simulations = 0;
    reserved = 0;
    evaluations = 0;
    stop = false;
    if (root->state.load(std::memory_order_acquire) == EXPANDED) {
        deadline = prm.microseconds ? start + microseconds(prm.microseconds) : steady_clock::time_point::max();
        if (prm.nb_threads == 1) {
            run_worker(*workers[0], state);
        } else {
            // std::jthread : chaque thread est rejoint (join) à la destruction
            // du vecteur, à la fin de ce bloc : quand on en sort, tous ont fini.
            std::vector<std::jthread> threads;
            for (size_t t = 0; t < prm.nb_threads; t++)
                threads.emplace_back([this, &workers, t, &state] { run_worker(*workers[t], state); });
        }
    }
    return make_result();
}

// =============================================================================
//  8. LE CACHE DES SORTIES DU RÉSEAU
// =============================================================================
//
// La sortie du réseau ne dépend que du plateau et du camp au trait, c'est-à-
// dire du hachage Zobrist. Les positions reviennent souvent : par
// transposition quand le nœud a été libéré par le ramasse-miettes, et d'un coup
// à l'autre. Le cache garde (valeur, a priori des coups légaux) — les a priori
// déjà passés par le softmax, en virgule fixe sur 16 bits (précision
// 1/65535), dans l'ordre de Yolah::moves, qui est toujours le même pour une
// position. Une entrée fait 8 + 4 + 2 + 75·2 = 164 octets (168 avec
// l'alignement) : 64 Mo ≈ 400 000 positions.
//
// Table à accès direct : indice = hachage & masque, une seule place par
// indice, on écrase toujours. Le hachage complet ET le nombre de coups sont
// vérifiés à la lecture : une collision de 64 bits ne peut pas donner des
// a priori pour un autre nombre de coups. 256 verrous « en bandes » : un
// verrou par groupe d'indices, assez pour que 16 threads ne s'attendent
// presque jamais. Le cache n'est jamais vidé par l'app (il ne dépend que du
// réseau, qui ne change pas en cours de partie).
NNCache::NNCache(size_t size_bytes) {
    const size_t n = size_bytes / sizeof(Entry);
    size_t pow2 = 1;
    while (pow2 * 2 <= n) pow2 *= 2;        // arrondi à la puissance de 2 inférieure (pour le masque)
    if (n >= 2) {
        table.resize(pow2);
        mask = pow2 - 1;
    }
}

bool NNCache::lookup(uint64_t key, size_t nb_moves, float& value, float* priors) {
    if (table.empty()) return false;
    const Entry& e = table[key & mask];
    std::lock_guard lock(locks[key & (locks.size() - 1)]);
    if (e.key != key || e.nb_moves != nb_moves) return false;
    value = e.value;
    for (size_t m = 0; m < nb_moves; m++) priors[m] = e.priors[m] * (1.0f / 65535.0f);
    return true;
}

void NNCache::store(uint64_t key, size_t nb_moves, float value, const float* priors) {
    if (table.empty()) return;
    Entry& e = table[key & mask];
    std::lock_guard lock(locks[key & (locks.size() - 1)]);
    e.key = key;
    e.value = value;
    e.nb_moves = static_cast<uint16_t>(nb_moves);
    for (size_t m = 0; m < nb_moves; m++) e.priors[m] = static_cast<uint16_t>(priors[m] * 65535.0f + 0.5f);
}

// =============================================================================
//  9. LE RÉSEAU EXÉCUTÉ PAR LIBTORCH, ET L'ENCODAGE D'UNE POSITION
// =============================================================================
//
// L'ENCODAGE (doit être identique à celui de l'entraînement : nnue/
// preprocess.py et encode_cnn dans nnue/cnn_resnet_value_policy.py).
// 4 plans de 64 nombres, plan après plan (l'ordre « NCHW » de PyTorch :
// lot, canal, ligne, colonne). La cellule i d'un plan (i = 8·ligne + colonne)
// contient le bit (63 − i) du bitboard, c'est-à-dire la case (63 − i).
// D'où vient ce 63 − i : preprocess.py construit un plan avec
//     np.unpackbits(np.array([bb], dtype='>u8').view(np.uint8)).reshape(8, 8)
// '>u8' écrit l'entier en gros-boutiste (l'octet de poids fort d'abord) et
// unpackbits sort le bit de poids fort de chaque octet d'abord : les bits
// sortent dans l'ordre 63, 62, …, 0. Géométriquement, avec s = 8·rangée +
// colonne : ligne = 7 − rangée, colonne du plan = 7 − colonne. Les rangées
// sont dans l'ordre habituel (la 8e en haut) mais les COLONNES sont en miroir
// (h à gauche) : la cellule 0 est h8, la 63 est a1. Pour le réseau c'est un
// simple renommage des cases ; mais il a été entraîné ainsi : ne pas « corriger ».
namespace {
    constexpr int PLANES = 4, PLANE_CELLS = 64, INPUT_FLOATS = PLANES * PLANE_CELLS, NUM_ACTIONS = 64 * 64;

    void encode_planes(const Yolah& y, float* out) {
        const uint64_t bbs[3] = {y.bitboard(Yolah::BLACK), y.bitboard(Yolah::WHITE), y.empty_bitboard()};
        for (int c = 0; c < 3; c++) {
            float* plane = out + c * PLANE_CELLS;
            for (int i = 0; i < PLANE_CELLS; i++) plane[i] = static_cast<float>((bbs[c] >> (63 - i)) & 1);
        }
        const float turn = y.current_player() == Yolah::WHITE ? 1.0f : 0.0f;   // plan 3 : le trait
        float* plane = out + 3 * PLANE_CELLS;
        for (int i = 0; i < PLANE_CELLS; i++) plane[i] = turn;
    }

    // L'indice du logit d'un coup : départ × 64 + arrivée (cases 0..63, a1 = 0,
    // SANS le miroir des plans : c'est l'indice de preprocess.py, pol = s1·64 + s2).
    // La passe (a1 → a1) a l'indice 0, comme à l'entraînement.
    constexpr int action_index(Move m) { return static_cast<int>(m.from_sq()) * 64 + static_cast<int>(m.to_sq()); }
}

// Le chargement. Le fichier .ts est un module TorchScript : le graphe de calcul
// du réseau et ses poids, exécutable sans Python.
TorchNet::TorchNet(const std::string& filename, const std::string& device_name, bool half, int64_t warm_batch)
    : device(device_name == "cuda" && !torch::cuda::is_available() ? torch::Device(torch::kCPU) : torch::Device(device_name)),
      fp16(half && device.is_cuda()) {   // la demi-précision n'a de sens que sur le GPU
    // Par défaut, l'exécuteur de TorchScript « profile » le graphe et le
    // re-spécialise pour chaque nouvelle forme d'entrée (plusieurs secondes à
    // chaque fois, et la recherche envoie des lots de toutes les tailles de 1
    // à 256) : on prend l'exécuteur simple. Ce sont des réglages globaux du
    // processus, à faire avant le chargement.
    torch::jit::getProfilingMode() = false;
    torch::jit::setGraphExecutorOptimize(false);
    module = torch::jit::load(filename, device);   // les poids vont directement sur le GPU
    module.eval();                                 // BatchNorm en mode évaluation (constantes apprises)
    if (fp16) module.to(torch::kHalf);             // poids en fp16 : 2 fois moins de mémoire, cœurs « tensor » du GPU
    // Préchauffage : cuDNN choisit son algorithme de convolution à la première
    // passe de chaque forme ; on le fait au chargement pour les tailles
    // courantes plutôt que pendant le premier coup.
    torch::NoGradGuard no_grad;
    c10::InferenceMode inference;
    for (int64_t n : {int64_t(1), int64_t(8), int64_t(32), int64_t(128), warm_batch}) {
        torch::Tensor x = torch::zeros({n, PLANES, 8, 8}, torch::TensorOptions().dtype(fp16 ? torch::kHalf : torch::kFloat32).device(device));
        module.forward({x});
    }
    if (device.is_cuda()) torch::cuda::synchronize();
}

void TorchNet::evaluate(std::span<const Request> in, std::span<Result> out) {
    const int64_t n = static_cast<int64_t>(in.size());
    if (n == 0) return;
    std::lock_guard lock(mutex);
    // Pas de gradient (on ne s'entraîne pas) et le mode « inférence » de
    // libtorch, qui évite toute la comptabilité de l'autograd.
    torch::NoGradGuard no_grad;
    c10::InferenceMode inference;

    // 1. Le tampon d'entrée, sur le CPU en mémoire « épinglée » (non paginable :
    //    le GPU peut la lire directement par DMA, et la copie peut être
    //    asynchrone). Créé à la première taille, agrandi si un lot plus gros arrive.
    if (!input_cpu.defined() || input_cpu.size(0) < n) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device.is_cuda());
        input_cpu = torch::empty({n, PLANES, 8, 8}, options);
    }
    float* dst = input_cpu.data_ptr<float>();
    for (int64_t i = 0; i < n; i++) encode_planes(*in[i].state, dst + i * INPUT_FLOATS);
    // 2. Copie vers le GPU (et conversion en fp16) des n premières positions.
    torch::Tensor x = input_cpu.narrow(0, 0, n).to(device, fp16 ? torch::kHalf : torch::kFloat32, /*non_blocking=*/true);
    // 3. UNE passe avant pour tout le lot. Le module rend un tuple
    //    (valeurs (n,), logits (n, 4096)).
    auto outputs = module.forward({x}).toTuple();
    // 4. Les deux sorties reviennent sur le CPU en fp32 (une copie chacune ;
    //    .to(kCPU) attend la fin du calcul sur le GPU).
    torch::Tensor values = outputs->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
    torch::Tensor logits = outputs->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32).contiguous();
    const float* v = values.data_ptr<float>();
    const float* p = logits.data_ptr<float>();
    // 5. Pour chaque position : sa valeur, et parmi ses 4096 logits ceux de
    //    ses coups légaux seulement, dans l'ordre de la requête.
    for (int64_t i = 0; i < n; i++) {
        const Request& rq = in[i];
        Result& rs = out[i];
        rs.value = v[i];
        const float* row = p + i * NUM_ACTIONS;
        for (uint16_t m = 0; m < rq.nb_moves; m++) rs.logits[m] = row[action_index(rq.moves[m])];
    }
}

// =============================================================================
//  10. LE REGROUPEUR DES REQUÊTES DES THREADS
// =============================================================================
//
// Un GPU n'est efficace que sur de gros lots, alors qu'un thread de recherche
// ne peut pas rassembler beaucoup de feuilles à la fois sans dégrader la
// recherche (les pertes virtuelles le poussent vers des coups de moins en
// moins bons). Solution : 16 threads × 16 feuilles, et ce regroupeur fusionne
// leurs demandes en un seul lot de ≤ 256 positions.
//
// Fonctionnement : chaque thread de recherche dépose sa demande (un `Job` sur
// sa propre pile) dans une file et s'endort. Un thread dédié (worker_loop) se
// réveille dès que
//   • la file contient au moins max_batch positions, ou
//   • tous les threads de recherche attendent (aucun autre ne peut venir), ou
//   • 0,5 ms se sont écoulées avec quelque chose en file (un retardataire ne
//     bloque pas les autres),
// concatène les demandes, appelle le réseau UNE fois, redistribue les
// résultats et réveille les threads servis. File FIFO : personne n'est oublié.
//
// Une demande VIDE est un « rendez-vous » : le thread n'a rien à évaluer
// (toutes ses descentes ont heurté les feuilles des autres) mais veut dormir
// jusqu'au prochain lot ; il compte parmi les threads en attente, donc ne
// retarde pas le lot.
Batcher::Batcher(std::unique_ptr<Evaluator> backend, size_t clients, size_t max_b, std::chrono::microseconds wait)
    : net(std::move(backend)), nb_clients(std::max<size_t>(1, clients)), max_batch(std::max<size_t>(1, max_b)), max_wait(wait) {
    batch_in.reserve(max_batch);
    batch_out.resize(max_batch);
    worker = std::thread([this] { worker_loop(); });
}

Batcher::~Batcher() {
    {
        std::lock_guard lock(mutex);
        stop = true;
    }
    cv_worker.notify_all();
    worker.join();
}

void Batcher::evaluate(std::span<const Request> in, std::span<Result> out) {
    Job job{in, out};
    std::unique_lock lock(mutex);
    queue.push_back(&job);
    queued_positions += in.size();
    ++waiting_clients;
    cv_worker.notify_one();
    cv_clients.wait(lock, [&] { return job.done; });
    // (waiting_clients est décrémenté par le regroupeur quand la demande est
    // servie, pas ici après le réveil : sinon le regroupeur pourrait déjà
    // décider du lot suivant en comptant encore des threads servis comme en
    // attente, et partir trop tôt avec un lot incomplet.)
}

void Batcher::worker_loop() {
    std::unique_lock lock(mutex);
    for (;;) {
        auto ready = [&] {
            return stop || queued_positions >= max_batch || (!queue.empty() && waiting_clients >= nb_clients);
        };
        if (!ready()) {
            if (!queue.empty()) cv_worker.wait_for(lock, max_wait, ready);   // quelque chose en file : attendre au plus 0,5 ms
            else                cv_worker.wait(lock, [&] { return stop || !queue.empty(); });
        }
        if (stop && queue.empty()) return;
        if (queue.empty()) continue;

        // Prendre des demandes entières, dans l'ordre, tant que le lot ne
        // dépasse pas max_batch. La première est toujours prise, même si elle
        // dépasse seule (le réseau accepte toute taille ; max_batch n'est
        // qu'une cible de fusion).
        std::vector<Job*> taken;
        batch_in.clear();
        while (!queue.empty()) {
            Job* j = queue.front();
            if (!taken.empty() && batch_in.size() + j->in.size() > max_batch) break;
            taken.push_back(j);
            batch_in.insert(batch_in.end(), j->in.begin(), j->in.end());
            queued_positions -= j->in.size();
            queue.pop_front();
        }
        if (batch_out.size() < batch_in.size()) batch_out.resize(batch_in.size());
        // Le réseau tourne SANS le verrou : les threads de recherche peuvent
        // continuer à déposer leurs demandes pour le lot suivant.
        if (!batch_in.empty()) {
            lock.unlock();
            net->evaluate(batch_in, std::span<Result>(batch_out.data(), batch_in.size()));
            lock.lock();
        }
        // Redistribuer : chaque demande reçoit sa tranche des résultats.
        size_t offset = 0;
        for (Job* j : taken) {
            std::copy_n(batch_out.begin() + offset, j->in.size(), j->out.begin());
            offset += j->in.size();
            j->done = true;
        }
        waiting_clients -= taken.size();
        cv_clients.notify_all();
    }
}

// =============================================================================
//  11. LE JOUEUR
// =============================================================================
//
// Le réseau (sur le GPU, fp16), le regroupeur s'il y a plusieurs threads (un
// lot fusionné = batch_size × nb_threads = 256 positions au plus), la
// recherche. play() lance une recherche du temps imparti et rend son coup ;
// game_over() oublie le graphe (le cache reste).
Player::Player(const std::string& weights, const Params& params, bool fp16, const std::string& device) {
    const int64_t merged = static_cast<int64_t>(params.batch_size * params.nb_threads);
    auto backend = std::make_unique<TorchNet>(weights, device, fp16, std::max<int64_t>(merged, 256));
    if (params.nb_threads > 1) net = std::make_unique<Batcher>(std::move(backend), params.nb_threads, merged);
    else                       net = std::move(backend);
    searcher = std::make_unique<Search>(*net, params, &memory);
}

Player::~Player() { searcher.reset(); }   // le graphe vit dans `memory` : détruit avant le pool

Move Player::play(const Yolah& yolah) {
    last = searcher->search(yolah);
    return last.best_move;
}

void Player::game_over() { searcher->reset(); }

} // namespace final_mcts
