// solo_retro_gpu — the table of private regions by retrograde analysis, on the GPU.
//
// The same values as test/solo_retro_main.cpp (CPU), computed in bulk:
//
//   1. seeds: every single-region state (a region of free squares and 1..K pieces
//      of one side next to it) up to N free squares, canonical, enumerated on the
//      CPU as in solo_retro_main.cpp;
//   2. top-down, layer N to 1 (a layer = the states with n free squares): the GPU
//      plays every move of every state of the layer, one thread per state, and
//      writes the children, normalised (the free squares no piece can reach and
//      the pieces with nowhere to go removed: they change nothing) and canonical
//      (the smallest of the 8 symmetries, translated to the corner). A child
//      falls in a lower layer; sorted and deduplicated on the GPU (thrust), it
//      joins that layer. So every layer holds all it needs below it: the seeds
//      and everything reachable from the layers above;
//   3. bottom-up, layer 1 to N: one thread per state tries every move, finds the
//      child in its layer by binary search (the layers are sorted) and keeps
//      max(1 + its value). A layer only needs the layers below: the retrograde
//      order. Layer 0 (no free square) is worth 0.
//
// Checks: the GPU's canonical form of every seed is the seed itself (the GPU and
// the CPU compute the same canonical forms), and a sample of each layer against
// the CPU cluster solver (player/solo_region.cpp).
//
//   g++ -c -O2 -std=c++20 -march=native -Igame -Iplayer -Imisc test/solo_retro_gpu_cpu.cpp player/solo_region.cpp game/magic.cpp misc/misc.cpp
//   nvcc -O3 -std=c++20 -arch=native test/solo_retro_gpu.cu solo_retro_gpu_cpu.o solo_region.o magic.o misc.o -o solo_retro_gpu
// (test/solo_retro_gpu.sh does both: nvcc cannot compile the game's headers, so
// the CPU reference lives in test/solo_retro_gpu_cpu.cpp, behind plain functions.)
//   solo_retro_gpu [N=8] [K=2]
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

// ── A state: its free squares and its pieces (two bitboards), ordered as a pair ──
struct K {
    unsigned long long f, p;
    __host__ __device__ bool operator<(const K& o) const { return f != o.f ? f < o.f : p < o.p; }
    __host__ __device__ bool operator==(const K& o) const { return f == o.f && p == o.p; }
};

// ── Bitboard functions, on both sides (the same code as player/solo_region.cpp) ──
#define HD __host__ __device__ inline
constexpr unsigned long long NOT_A = 0xfefefefefefefefeULL, NOT_H = 0x7f7f7f7f7f7f7f7fULL;
HD unsigned long long grow(unsigned long long b) {   // b and its 8 neighbours
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
    return __ffsll((long long)b) - 1;
#else
    return std::countr_zero(b);
#endif
}
HD unsigned long long bswap(unsigned long long b) {   // rank 1 ↔ rank 8
    b = ((b & 0x00000000FFFFFFFFULL) << 32) | (b >> 32);
    b = ((b & 0x0000FFFF0000FFFFULL) << 16) | ((b >> 16) & 0x0000FFFF0000FFFFULL);
    b = ((b & 0x00FF00FF00FF00FFULL) << 8) | ((b >> 8) & 0x00FF00FF00FF00FFULL);
    return b;
}
HD unsigned long long mirror(unsigned long long b) {   // file a ↔ file h
    const unsigned long long k1 = 0x5555555555555555ULL, k2 = 0x3333333333333333ULL, k4 = 0x0f0f0f0f0f0f0f0fULL;
    b = ((b >> 1) & k1) | ((b & k1) << 1);
    b = ((b >> 2) & k2) | ((b & k2) << 2);
    b = ((b >> 4) & k4) | ((b & k4) << 4);
    return b;
}
HD unsigned long long diag(unsigned long long b) {   // file ↔ rank
    unsigned long long t;
    const unsigned long long k1 = 0x5500550055005500ULL, k2 = 0x3333000033330000ULL, k4 = 0x0f0f0f0f00000000ULL;
    t = k4 & (b ^ (b << 28)); b ^= t ^ (t >> 28);
    t = k2 & (b ^ (b << 14)); b ^= t ^ (t >> 14);
    t = k1 & (b ^ (b << 7));  b ^= t ^ (t >> 7);
    return b;
}
HD void to_corner(unsigned long long& f, unsigned long long& p) {
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
// The smallest of the 8 symmetric shapes, at the corner (as solo::canonical).
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
// Normalised: only the free squares some piece can reach, only the pieces next to them.
HD K normalise(unsigned long long f, unsigned long long p) {
    unsigned long long r = grow(p) & f, prev = 0;
    while (r != prev) { prev = r; r = grow(r) & f; }
    return {r, p & grow(r)};
}
// The squares a piece on `sq` can slide to through the free squares `f` (8 directions).
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
// The child of a state after one move, normalised and canonical.
HD K child_of(K s, int from, int to) {
    const unsigned long long tb = 1ULL << to;
    const K n = normalise(s.f & ~tb, (s.p & ~(1ULL << from)) | tb);
    return canonical(n.f, n.p);
}

// ── Kernels ──
constexpr int MAX_CHILDREN = 4 * 27;   // ≤ 4 pieces, ≤ 27 queen destinations each

// Top-down: the children of every state (up to MAX_CHILDREN each; empty slots: ~0).
__global__ void children_kernel(const K* layer, size_t n, K* out) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const K s = layer[i];
    int k = 0;
    for (unsigned long long ps = s.p; ps; ps &= ps - 1) {
        const int from = ctz(ps);
        for (unsigned long long t = slides(from, s.f); t; t &= t - 1)
            if (k < MAX_CHILDREN) out[i * MAX_CHILDREN + k++] = child_of(s, from, ctz(t));
    }
    for (; k < MAX_CHILDREN; k++) out[i * MAX_CHILDREN + k] = K{~0ULL, ~0ULL};
}

// The layers below, for the lookups: per layer, its sorted keys and their values.
struct Layers {
    const K* keys[65];
    const signed char* values[65];
    size_t sizes[65];
};
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
    return lo < L.sizes[m] && a[lo] == c ? L.values[m][lo] : -100;   // −100: missing (a bug)
}
// Bottom-up: the value of every state of a layer, from the layers below.
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
// The GPU's canonical form of each seed (it must be the seed itself).
__global__ void canon_kernel(const K* in, size_t n, int* differ) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    if (!(canonical(in[i].f, in[i].p) == in[i])) atomicAdd(differ, 1);
}

// ── Host ──
// The CPU side (test/solo_retro_gpu_cpu.cpp).
void cpu_init();
int cpu_value(unsigned long long free, unsigned long long pieces);
void cpu_canonical(unsigned long long free, unsigned long long pieces, unsigned long long* f, unsigned long long* p);
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
static void sort_unique(thrust::device_vector<K>& v) {
    thrust::sort(v.begin(), v.end());
    v.erase(thrust::unique(v.begin(), v.end()), v.end());
}
struct IsEmpty { __host__ __device__ bool operator()(const K& k) const { return k.f == ~0ULL; } };
struct InLayer {   // the children that fall in layer m
    int m;
    __host__ __device__ bool operator()(const K& k) const { return popc(k.f) == m; }
};
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    cpu_init();
    const int N = argc > 1 ? atoi(argv[1]) : 8, KP = argc > 2 ? atoi(argv[2]) : 2;
    const double t0 = now();
    // 1. Seeds (CPU).
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
    // Check: the GPU's canonical forms are the CPU's.
    {
        int differ = 0, *d_differ;
        cudaMalloc(&d_differ, sizeof(int));
        for (int n = 1; n <= N; n++) {
            thrust::device_vector<K> d(pending[n].begin(), pending[n].end());
            cudaMemset(d_differ, 0, sizeof(int));
            canon_kernel<<<(d.size() + 255) / 256, 256>>>(thrust::raw_pointer_cast(d.data()), d.size(), d_differ);
            int x;
            cudaMemcpy(&x, d_differ, sizeof(int), cudaMemcpyDeviceToHost);
            differ += x;
        }
        cudaFree(d_differ);
        std::printf("check: the GPU's canonical form differs from the CPU's on %d seeds\n", differ);
    }
    // 2. Top-down: every layer gets the children of the layers above.
    std::vector<thrust::device_vector<K>> layer(N + 1);
    const double t1 = now();
    for (int n = N; n >= 1; n--) {
        layer[n] = thrust::device_vector<K>(pending[n].begin(), pending[n].end());
        std::vector<K>().swap(pending[n]);
        sort_unique(layer[n]);
        // The children, by chunks (MAX_CHILDREN slots per state). Most fall in layer
        // n − 1: they are gathered on the GPU (acc), deduplicated as they come; the
        // few others (the moves that cut off squares) go to the CPU.
        const size_t chunk = size_t(1) << 20;
        thrust::device_vector<K> acc;
        for (size_t start = 0; start < layer[n].size(); start += chunk) {
            const size_t m = std::min(chunk, layer[n].size() - start);
            thrust::device_vector<K> out(m * MAX_CHILDREN);
            children_kernel<<<(m + 255) / 256, 256>>>(thrust::raw_pointer_cast(layer[n].data()) + start, m, thrust::raw_pointer_cast(out.data()));
            out.erase(thrust::remove_if(out.begin(), out.end(), IsEmpty{}), out.end());   // compact first: most slots are empty
            sort_unique(out);
            // The children of layer n − 1 stay on the GPU, the others go to the CPU.
            auto mid = thrust::stable_partition(out.begin(), out.end(), InLayer{n - 1});
            thrust::host_vector<K> rest(mid, out.end());
            for (const K& c : rest) {
                const int m2 = std::popcount(c.f);
                if (m2 >= 1) pending[m2].push_back(c);   // m2 = 0: worth 0, no table needed
            }
            out.erase(mid, out.end());
            thrust::device_vector<K> merged(acc.size() + out.size());
            thrust::merge(acc.begin(), acc.end(), out.begin(), out.end(), merged.begin());
            merged.erase(thrust::unique(merged.begin(), merged.end()), merged.end());
            acc.swap(merged);
        }
        if (n >= 2) {   // into layer n − 1's pending list
            thrust::host_vector<K> h = acc;
            pending[n - 1].insert(pending[n - 1].end(), h.begin(), h.end());
        }
        std::printf("layer %2d: %10zu states (%zu seeds), %.1f s\n", n, layer[n].size(), nb_seeds[n], now() - t1);
        std::fflush(stdout);
    }
    // 3. Bottom-up: the values, each layer from those below.
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
        for (signed char v : hv) full += v == n;
        states += hv.size();
    }
    int missing;
    cudaMemcpy(&missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("values (GPU): %.2f s; %lld states, %lld fill their region (%.1f %%); children not found: %d\n",
                now() - t2, states, full, 100.0 * full / states, missing);
    // 4. Check: a sample of each layer against the CPU cluster solver.
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
