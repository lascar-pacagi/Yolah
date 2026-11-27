// =============================================================================
//  final_retro_gpu.cu — la table des régions privées sur GPU, commentée
// =============================================================================
//  Lire d'abord doc/final_retro.h (les régions privées, la grappe, la valeur,
//  la table à rebours sur CPU). Ce fichier calcule la MÊME table sur la carte
//  graphique, et sert aussi d'introduction à la programmation GPU (CUDA). Le
//  vrai code est test/solo_retro_gpu.cu (mêmes instructions, commentaires en
//  anglais) ; le contrôle : test/final_retro_gpu_check.sh.
//
//  ---------------------------------------------------------------------------
//  LE GPU EN DIX LIGNES
//  ---------------------------------------------------------------------------
//  Un processeur (CPU) a quelques cœurs rapides ; une carte graphique (GPU) en
//  a des milliers, plus lents, qui exécutent le MÊME programme sur des données
//  différentes. La RTX 3080 en a 8 704. Un programme GPU (un « noyau »,
//  `__global__`) est lancé sur une GRILLE de BLOCS de THREADS :
//      noyau<<<nombre_de_blocs, threads_par_bloc>>>(arguments);
//  Chaque thread connaît sa place (blockIdx.x : son bloc, blockDim.x : la
//  taille d'un bloc, threadIdx.x : sa place dans le bloc) et en déduit QUEL
//  élément traiter : i = blockIdx.x · blockDim.x + threadIdx.x. Ici : un thread
//  par état de la table.
//  La carte a sa propre mémoire (10 Go ici) : les données y sont copiées
//  (thrust::device_vector, cudaMemcpy) et les résultats en reviennent. Les
//  copies coûtent cher : on les évite le plus possible.
//  Une fonction marquée `__host__ __device__` est compilée deux fois : pour le
//  CPU et pour le GPU. thrust est la bibliothèque standard de CUDA pour trier,
//  dédupliquer, fusionner… des tableaux entiers sur le GPU, en parallèle.
//
//  ---------------------------------------------------------------------------
//  LE CALCUL, EN TROIS TEMPS
//  ---------------------------------------------------------------------------
//    1. Les GRAINES (CPU) : toutes les grappes d'une seule région jusqu'à N
//       cases et 1 ou 2 pièces, canoniques (comme doc/final_retro.cpp).
//    2. La DESCENTE (GPU), de la couche N à la couche 1 (couche = états à n
//       cases libres) : un thread par état joue tous ses coups et écrit ses
//       ENFANTS, normalisés (sans les cases que plus aucune pièce n'atteint ni
//       les pièces qui ne peuvent plus bouger : ils ne changent rien) et
//       canoniques. Triés et dédupliqués sur le GPU, ils rejoignent la couche
//       de leur nombre de cases. Chaque couche contient ainsi tout ce dont les
//       couches au-dessus auront besoin.
//    3. La REMONTÉE (GPU), de la couche 1 à la couche N : un thread par état
//       essaie tous ses coups, cherche chaque enfant dans sa couche (déjà
//       calculée : un coup remplit une case) par dichotomie, et garde le
//       maximum de 1 + sa valeur. C'est l'ordre de l'analyse rétrograde.
//  Pourquoi pas comme sur CPU (une recherche récursive avec une mémoire) ? La
//  récursion et une table de hachage partagée se prêtent mal au GPU. Couche par
//  couche, chaque état ne dépend que de tableaux TRIÉS déjà finis : des milliers
//  de threads peuvent travailler sans jamais s'attendre.
//
//  Mesuré (RTX 3080), jusqu'à 9 cases et 2 pièces : 88,8 millions d'états, la
//  descente en 11 s, la remontée en 0,14 s (contre 181 s pour le CPU, sur un
//  cœur, qui ne calculait que les grappes d'une région). Une première version
//  mettait 556 s : elle triait 108 cases par état, presque toutes vides, et
//  renvoyait au CPU les doublons d'un paquet à l'autre. Le compactage avant le
//  tri (remove_if) et la fusion sur le GPU (merge + unique) l'ont ramenée à 11 s.
//
//  Contrôles : la forme canonique calculée par le GPU de chaque graine est la
//  graine elle-même (GPU et CPU calculent les mêmes formes), aucun enfant n'est
//  introuvable, et 2 000 états de chaque couche sont recalculés par le solveur
//  du CPU (player/solo_region.cpp) : les mêmes valeurs.
// =============================================================================
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

// ── Un état : ses cases libres et ses pièces (deux bitboards) ────────────────
// Rangés comme une paire (cases libres d'abord) : c'est l'ordre des tableaux
// triés, et celui de la dichotomie. Les deux opérateurs servent sur le CPU et
// sur le GPU (thrust trie avec operator<).
struct K {
    unsigned long long f, p;
    __host__ __device__ bool operator<(const K& o) const { return f != o.f ? f < o.f : p < o.p; }
    __host__ __device__ bool operator==(const K& o) const { return f == o.f && p == o.p; }
};

// ── Les fonctions de bitboards, pour le CPU et le GPU ────────────────────────
// Les mêmes que doc/final_retro.cpp (expliquées là-bas). HD = compilée pour les
// deux. Deux différences seulement : compter les bits et trouver le bit le plus
// faible ont des instructions propres au GPU (__popcll, __ffsll) ; et
// __builtin_bswap64 n'existe pas sur GPU : l'inversion des octets est écrite à
// la main (échanges de 32, puis 16, puis 8 bits). __CUDA_ARCH__ n'est défini
// que pendant la compilation pour le GPU.
#define HD __host__ __device__ inline
constexpr unsigned long long NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
HD unsigned long long grow(unsigned long long b) {   // b et ses 8 voisines
    const unsigned long long row = b | ((b << 1) & NOT_A) | ((b >> 1) & NOT_H);
    return row | (row << 8) | (row >> 8);
}
HD int popc(unsigned long long b) {
#ifdef __CUDA_ARCH__
    return __popcll(b);
#else
    return std::popcount(b);
#endif
}
HD int ctz(unsigned long long b) {
#ifdef __CUDA_ARCH__
    return __ffsll((long long)b) - 1;   // __ffsll : position du premier bit à 1, à partir de 1
#else
    return std::countr_zero(b);
#endif
}
HD unsigned long long bswap(unsigned long long b) {   // rangée 1 ↔ rangée 8
    b = ((b & 0x00000000FFFFFFFFULL) << 32) | (b >> 32);
    b = ((b & 0x0000FFFF0000FFFFULL) << 16) | ((b >> 16) & 0x0000FFFF0000FFFFULL);
    b = ((b & 0x00FF00FF00FF00FFULL) << 8) | ((b >> 8) & 0x00FF00FF00FF00FFULL);
    return b;
}
HD unsigned long long mirror(unsigned long long b) {   // colonne a ↔ colonne h
    const unsigned long long k1 = 0x5555555555555555ULL, k2 = 0x3333333333333333ULL, k4 = 0x0f0f0f0f0f0f0f0fULL;
    b = ((b >> 1) & k1) | ((b & k1) << 1);
    b = ((b >> 2) & k2) | ((b & k2) << 2);
    b = ((b >> 4) & k4) | ((b & k4) << 4);
    return b;
}
HD unsigned long long diag(unsigned long long b) {   // colonne ↔ rangée
    unsigned long long t;
    const unsigned long long k1 = 0x5500550055005500ULL, k2 = 0x3333000033330000ULL, k4 = 0x0f0f0f0f00000000ULL;
    t = k4 & (b ^ (b << 28)); b ^= t ^ (t >> 28);
    t = k2 & (b ^ (b << 14)); b ^= t ^ (t >> 14);
    t = k1 & (b ^ (b << 7));  b ^= t ^ (t >> 7);
    return b;
}
HD void to_corner(unsigned long long& f, unsigned long long& p) {   // dans le coin a1
    const unsigned long long all = f | p;
    if (!all) return;
    const int rank = ctz(all) / 8;
    f >>= 8 * rank;
    p >>= 8 * rank;
    unsigned long long c = f | p;
    c |= c >> 32; c |= c >> 16; c |= c >> 8;
    const int file = ctz(c & 0xFF);
    f >>= file;
    p >>= file;
}
// La plus petite des 8 formes symétriques, ramenée au coin (comme le CPU).
HD K canonical(unsigned long long f, unsigned long long p) {
    K best{~0ULL, ~0ULL};
    for (int d = 0; d < 2; d++) {
        for (int v = 0; v < 2; v++) {
            for (int h = 0; h < 2; h++) {
                unsigned long long ff = f, pp = p;
                to_corner(ff, pp);
                if (ff < best.f || (ff == best.f && pp < best.p)) best = {ff, pp};
                f = mirror(f); p = mirror(p);
            }
            f = bswap(f); p = bswap(p);
        }
        f = diag(f); p = diag(p);
    }
    return best;
}
// Normalisé : seulement les cases libres qu'une pièce atteint (inondation depuis
// les pièces), seulement les pièces qui les touchent. Un état et sa forme
// normalisée ont la même valeur ; la normalisation rend la table plus petite.
HD K normalise(unsigned long long f, unsigned long long p) {
    unsigned long long r = grow(p) & f, prev = 0;
    while (r != prev) { prev = r; r = grow(r) & f; }
    return {r, p & grow(r)};
}
// Les cases où une pièce en `sq` peut glisser, à travers les cases libres `f`.
// Le CPU utilise les tables « magic » du jeu ; sur le GPU, une boucle simple par
// direction (avancer tant que la case est libre et sur le plateau) suffit.
HD unsigned long long slides(int sq, unsigned long long f) {
    const int dr[8] = {1, 1, 0, -1, -1, -1, 0, 1}, dc[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    unsigned long long out = 0;
    const int r0 = sq / 8, c0 = sq % 8;
    for (int d = 0; d < 8; d++) {
        int r = r0 + dr[d], c = c0 + dc[d];
        while (r >= 0 && r < 8 && c >= 0 && c < 8) {
            const unsigned long long b = 1ULL << (r * 8 + c);
            if (!(f & b)) break;
            out |= b;
            r += dr[d]; c += dc[d];
        }
    }
    return out;
}
// L'enfant d'un état après un coup : la destination n'est plus libre, la pièce y
// est ; puis normalisé et canonique (c'est sous cette forme qu'il est rangé).
HD K child_of(K s, int from, int to) {
    const unsigned long long tb = 1ULL << to;
    const K n = normalise(s.f & ~tb, (s.p & ~(1ULL << from)) | tb);
    return canonical(n.f, n.p);
}

// ── Les noyaux ───────────────────────────────────────────────────────────────
// Au plus 4 pièces, et au plus 27 destinations de dame par pièce : chaque état
// a au plus 108 enfants. Le GPU n'a pas de tableaux qui grandissent : chaque
// thread reçoit 108 cases à lui, et marque les cases inutilisées (~0, la plus
// grande valeur possible) ; elles sont retirées après coup (remove_if).
constexpr int MAX_CHILDREN = 4 * 27;

// La descente : les enfants de chaque état (un thread par état).
__global__ void children_kernel(const K* layer, size_t n, K* out) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;   // l'état de ce thread
    if (i >= n) return;   // la grille a des threads en trop dans son dernier bloc
    const K s = layer[i];
    int k = 0;
    for (unsigned long long ps = s.p; ps; ps &= ps - 1) {   // chaque pièce (ps & ps − 1 retire le bit le plus faible)
        const int from = ctz(ps);
        for (unsigned long long t = slides(from, s.f); t; t &= t - 1)   // chaque destination
            if (k < MAX_CHILDREN) out[i * MAX_CHILDREN + k++] = child_of(s, from, ctz(t));
    }
    for (; k < MAX_CHILDREN; k++) out[i * MAX_CHILDREN + k] = K{~0ULL, ~0ULL};   // cases vides
}

// Les couches déjà calculées, pour la remontée : par couche, ses états triés et
// leurs valeurs (des adresses dans la mémoire du GPU). Passée au noyau par
// valeur : chaque thread en a une copie.
struct Layers {
    const K* keys[65];
    const signed char* values[65];
    size_t sizes[65];
};
// La valeur d'un enfant : par dichotomie dans le tableau trié de sa couche (la
// couche = son nombre de cases libres). Couche 0 : plus de case libre, valeur 0.
// −100 : introuvable (ne doit jamais arriver : la descente a mis chaque enfant
// dans sa couche ; compté comme une erreur).
__device__ int lookup(const Layers& L, K c) {
    const int m = popc(c.f);
    if (m == 0) return 0;
    const K* a = L.keys[m];
    size_t lo = 0, hi = L.sizes[m];
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (a[mid] < c) lo = mid + 1;
        else hi = mid;
    }
    return lo < L.sizes[m] && a[lo] == c ? L.values[m][lo] : -100;
}
// La remontée : la valeur de chaque état d'une couche (un thread par état), à
// partir des couches plus basses ; l'arrêt anticipé (best < most) comme sur CPU.
// atomicAdd : plusieurs threads peuvent ajouter au même compteur en même temps
// sans se marcher dessus.
__global__ void value_kernel(const K* layer, size_t n, Layers L, signed char* values, int* missing) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const K s = layer[i];
    const int most = popc(s.f);
    int best = 0;
    for (unsigned long long ps = s.p; ps && best < most; ps &= ps - 1) {
        const int from = ctz(ps);
        for (unsigned long long t = slides(from, s.f); t && best < most; t &= t - 1) {
            const int v = lookup(L, child_of(s, from, ctz(t)));
            if (v < 0) { atomicAdd(missing, 1); continue; }
            if (1 + v > best) best = 1 + v;
        }
    }
    values[i] = (signed char)best;
}
// Contrôle : la forme canonique calculée par le GPU de chaque graine (calculée
// par le CPU) doit être la graine elle-même.
__global__ void canon_kernel(const K* in, size_t n, int* differ) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    if (!(canonical(in[i].f, in[i].p) == in[i])) atomicAdd(differ, 1);
}

// ── Le côté CPU ──────────────────────────────────────────────────────────────
// Le solveur du CPU et sa forme canonique (test/solo_retro_gpu_cpu.cpp, compilé
// par g++ : nvcc ne sait pas compiler les en-têtes du jeu).
void cpu_init();
int cpu_value(unsigned long long free, unsigned long long pieces);
void cpu_canonical(unsigned long long free, unsigned long long pieces, unsigned long long* f, unsigned long long* p);
// Les placements d'une forme près du coin (comme doc/final_retro.cpp).
static int width(unsigned long long b) {
    unsigned long long c = b; c |= c >> 32; c |= c >> 16; c |= c >> 8;
    c &= 0xFF;
    return c ? 64 - std::countl_zero(c) - std::countr_zero(c) : 0;
}
static int height(unsigned long long b) { return b ? (63 - std::countl_zero(b)) / 8 - std::countr_zero(b) / 8 + 1 : 0; }
static std::vector<unsigned long long> placements(unsigned long long s) {
    std::vector<unsigned long long> v{s};
    const bool right = width(s) < 8, up = height(s) < 8;
    if (right) v.push_back(s << 1);
    if (up) v.push_back(s << 8);
    if (right && up) v.push_back(s << 9);
    return v;
}
// Trier et dédupliquer un tableau du GPU (thrust : en parallèle, sur la carte).
static void sort_unique(thrust::device_vector<K>& v) {
    thrust::sort(v.begin(), v.end());
    v.erase(thrust::unique(v.begin(), v.end()), v.end());
}
// Les prédicats de thrust : une case vide ; un état de la couche m.
struct IsEmpty { __host__ __device__ bool operator()(const K& k) const { return k.f == ~0ULL; } };
struct InLayer {
    int m;
    __host__ __device__ bool operator()(const K& k) const { return popc(k.f) == m; }
};
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    cpu_init();
    const int N = argc > 1 ? atoi(argv[1]) : 8, KP = argc > 2 ? atoi(argv[2]) : 2;
    const double t0 = now();
    // 1. Les graines (CPU) : chaque forme de n cases, à chaque placement, avec 1
    //    ou 2 pièces autour ; puis les formes de n + 1 cases. pending[n] : les
    //    états de la couche n en attente (les graines, puis les enfants).
    std::vector<std::vector<K>> pending(N + 1);
    std::vector<size_t> nb_seeds(N + 1, 0);
    {
        std::vector<unsigned long long> shapes = {1};
        for (int n = 1; n <= N; n++) {
            std::unordered_set<unsigned long long> next;
            for (unsigned long long sh : shapes)
                for (unsigned long long f : placements(sh)) {
                    std::vector<unsigned long long> ring;
                    for (unsigned long long r = grow(f) & ~f; r; r &= r - 1) ring.push_back(r & -r);
                    const int m = int(ring.size());
                    for (int a = 0; a < m; a++) {
                        auto add = [&](unsigned long long p) { K k; cpu_canonical(f, p, &k.f, &k.p); pending[n].push_back(k); };
                        add(ring[a]);
                        if (KP >= 2) for (int b = a + 1; b < m; b++) add(ring[a] | ring[b]);
                    }
                    if (n < N)
                        for (unsigned long long add = grow(f) & ~f; add; add &= add - 1) {
                            K k;
                            cpu_canonical(f | (add & -add), 0, &k.f, &k.p);
                            next.insert(k.f);
                        }
                }
            std::sort(pending[n].begin(), pending[n].end());
            pending[n].erase(std::unique(pending[n].begin(), pending[n].end()), pending[n].end());
            nb_seeds[n] = pending[n].size();
            shapes.assign(next.begin(), next.end());
        }
    }
    std::printf("seeds (CPU): %.1f s\n", now() - t0);
    // Contrôle : les formes canoniques du GPU sont celles du CPU. Le nombre de
    // blocs : de quoi couvrir tous les états, par blocs de 256 threads (arrondi
    // au-dessus). cudaMalloc / cudaMemcpy : une allocation et une copie à la main.
    {
        int differ = 0, *d_differ;
        cudaMalloc(&d_differ, sizeof(int));
        for (int n = 1; n <= N; n++) {
            thrust::device_vector<K> d(pending[n].begin(), pending[n].end());   // copie vers le GPU
            cudaMemset(d_differ, 0, sizeof(int));
            canon_kernel<<<(d.size() + 255) / 256, 256>>>(thrust::raw_pointer_cast(d.data()), d.size(), d_differ);
            int x;
            cudaMemcpy(&x, d_differ, sizeof(int), cudaMemcpyDeviceToHost);   // attend la fin du noyau, puis copie
            differ += x;
        }
        cudaFree(d_differ);
        std::printf("check: the GPU's canonical form differs from the CPU's on %d seeds\n", differ);
    }
    // 2. La descente, de la couche N à la couche 1.
    std::vector<thrust::device_vector<K>> layer(N + 1);
    const double t1 = now();
    for (int n = N; n >= 1; n--) {
        // La couche n : ses graines et les enfants des couches au-dessus, sur le GPU.
        layer[n] = thrust::device_vector<K>(pending[n].begin(), pending[n].end());
        std::vector<K>().swap(pending[n]);   // libère la mémoire du CPU
        sort_unique(layer[n]);
        // Ses enfants, par paquets d'un million d'états (108 cases chacun : 1,7 Go).
        // La plupart tombent dans la couche n − 1 : ils restent sur le GPU (acc),
        // dédupliqués au fur et à mesure ; les autres (les coups qui coupent des
        // cases du reste) retournent au CPU, dans la liste de leur couche.
        const size_t chunk = size_t(1) << 20;
        thrust::device_vector<K> acc;
        for (size_t start = 0; start < layer[n].size(); start += chunk) {
            const size_t m = std::min(chunk, layer[n].size() - start);
            thrust::device_vector<K> out(m * MAX_CHILDREN);
            children_kernel<<<(m + 255) / 256, 256>>>(thrust::raw_pointer_cast(layer[n].data()) + start, m, thrust::raw_pointer_cast(out.data()));
            // D'abord compacter (retirer les cases vides, l'immense majorité), PUIS
            // trier : trier 108 cases par état coûtait 50 fois plus.
            out.erase(thrust::remove_if(out.begin(), out.end(), IsEmpty{}), out.end());
            sort_unique(out);
            // Ceux de la couche n − 1 devant, les autres derrière (partition stable :
            // l'ordre trié est gardé dans chaque partie).
            auto mid = thrust::stable_partition(out.begin(), out.end(), InLayer{n - 1});
            thrust::host_vector<K> rest(mid, out.end());   // copie vers le CPU
            for (const K& c : rest) {
                const int m2 = std::popcount(c.f);
                if (m2 >= 1) pending[m2].push_back(c);   // couche 0 : valeur 0, pas besoin de table
            }
            out.erase(mid, out.end());
            // Fusion avec ceux des paquets précédents : deux tableaux triés se
            // fusionnent en un tableau trié (merge), puis unique retire les doublons.
            thrust::device_vector<K> merged(acc.size() + out.size());
            thrust::merge(acc.begin(), acc.end(), out.begin(), out.end(), merged.begin());
            merged.erase(thrust::unique(merged.begin(), merged.end()), merged.end());
            acc.swap(merged);
        }
        if (n >= 2) {   // dans la liste de la couche n − 1
            thrust::host_vector<K> h = acc;
            pending[n - 1].insert(pending[n - 1].end(), h.begin(), h.end());
        }
        std::printf("layer %2d: %10zu states (%zu seeds), %.1f s\n", n, layer[n].size(), nb_seeds[n], now() - t1);
        std::fflush(stdout);
    }
    // 3. La remontée, de la couche 1 à la couche N. Après chaque couche, ses
    //    adresses rejoignent `L` : la couche suivante y trouvera ses enfants.
    //    cudaDeviceSynchronize : attendre que le noyau ait fini (un lancement de
    //    noyau rend la main tout de suite au CPU).
    std::vector<thrust::device_vector<signed char>> value(N + 1);
    Layers L{};
    int* d_missing;
    cudaMalloc(&d_missing, sizeof(int));
    cudaMemset(d_missing, 0, sizeof(int));
    const double t2 = now();
    long long full = 0, states = 0;
    for (int n = 1; n <= N; n++) {
        value[n] = thrust::device_vector<signed char>(layer[n].size());
        value_kernel<<<(layer[n].size() + 255) / 256, 256>>>(thrust::raw_pointer_cast(layer[n].data()), layer[n].size(), L,
                                                            thrust::raw_pointer_cast(value[n].data()), d_missing);
        cudaDeviceSynchronize();
        L.keys[n] = thrust::raw_pointer_cast(layer[n].data());
        L.values[n] = thrust::raw_pointer_cast(value[n].data());
        L.sizes[n] = layer[n].size();
        thrust::host_vector<signed char> hv = value[n];
        for (signed char v : hv) full += v == n;   // les états qui remplissent toute leur région
        states += hv.size();
    }
    int missing;
    cudaMemcpy(&missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("values (GPU): %.2f s; %lld states, %lld fill their region (%.1f %%); children not found: %d\n",
                now() - t2, states, full, 100.0 * full / states, missing);
    // 4. Contrôle : 2 000 états de chaque couche, recalculés par le solveur du CPU.
    std::mt19937_64 rng(1);
    int wrong = 0, checked = 0;
    for (int n = 1; n <= N; n++) {
        thrust::host_vector<K> hk = layer[n];
        thrust::host_vector<signed char> hv = value[n];
        for (int i = 0; i < 2000; i++) {
            const size_t j = rng() % hk.size();
            checked++;
            if (cpu_value(hk[j].f, hk[j].p) != hv[j]) wrong++;
        }
    }
    std::printf("check: %d states against the CPU solver, %d wrong; total %.1f s\n", checked, wrong, now() - t0);
    return wrong || missing;
}
