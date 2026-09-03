// resnet.cpp — see resnet.h for the design overview.
#define EIGEN_DONT_PARALLELIZE   // we parallelise over positions ourselves (OpenMP)
#include "resnet.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <omp.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

namespace nn {

namespace {

using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapRow      = Eigen::Map<RowMat, Eigen::Aligned32>;
using MapConstRow = Eigen::Map<const RowMat, Eigen::Aligned32>;

// SRC[p][tap] = source pixel of output pixel p for kernel tap (kh, kw),
// or -1 when the tap falls outside the board (zero padding).
// p = 8*row + col, tap = 3*kh + kw, kh/kw in 0..2.
struct TapTable {
    int src[64][9];
    TapTable() {
        for (int p = 0; p < 64; p++) {
            const int r = p / 8, c = p % 8;
            for (int tap = 0; tap < 9; tap++) {
                const int rr = r + tap / 3 - 1, cc = c + tap % 3 - 1;
                src[p][tap] = (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) ? rr * 8 + cc : -1;
            }
        }
    }
};
const TapTable TAPS;

// Little-endian binary reader for the .bin format documented in resnet_export.py.
struct Reader {
    FILE* f;
    std::string name;
    explicit Reader(const std::string& filename) : f(std::fopen(filename.c_str(), "rb")), name(filename) {
        if (!f) throw std::runtime_error("ResNetCPU: cannot open " + filename);
    }
    ~Reader() { if (f) std::fclose(f); }
    uint32_t u32() {
        uint32_t v;
        if (std::fread(&v, 4, 1, f) != 1) throw std::runtime_error("ResNetCPU: truncated file " + name);
        return v;
    }
    void floats(ResNetCPU::AlignedVec& dst, size_t n) {
        dst.resize(n);
        if (std::fread(dst.data(), sizeof(float), n, f) != n)
            throw std::runtime_error("ResNetCPU: truncated file " + name);
    }
};

// y = max(0, scale[c] * x + shift[c]) over a (rows × C) row-major block.
inline void bn_relu(const float* __restrict x, float* __restrict y, size_t rows, int C,
                    const float* __restrict scale, const float* __restrict shift) {
    for (size_t r = 0; r < rows; r++) {
        const float* xr = x + r * C;
        float* yr = y + r * C;
        for (int c = 0; c < C; c++) {
            yr[c] = std::max(0.0f, scale[c] * xr[c] + shift[c]);
        }
    }
}

// Col[(b*64 + p) × (tap*C + ch)] = H[b][src(p, tap)][ch]  (0 when padded).
inline void im2col(const float* __restrict h, float* __restrict col, size_t n, int C) {
    const size_t row_len = 9 * static_cast<size_t>(C);
    for (size_t b = 0; b < n; b++) {
        const float* hb = h + b * 64 * C;
        float* cb = col + b * 64 * row_len;
        for (int p = 0; p < 64; p++) {
            float* crow = cb + p * row_len;
            for (int tap = 0; tap < 9; tap++) {
                float* dst = crow + tap * C;
                const int src = TAPS.src[p][tap];
                if (src < 0) std::memset(dst, 0, C * sizeof(float));
                else         std::memcpy(dst, hb + src * C, C * sizeof(float));
            }
        }
    }
}

// The trained network produces activations in the denormal range (1e-41 …),
// and denormal arithmetic is ~100× slower on x86. Flush-to-zero / denormals-
// are-zero are per-thread MXCSR flags, so set them on every worker thread of
// the parallel region instead of relying on the link flags (-ffast-math adds
// crtfastmath.o, but only for the main thread and only if it is on the link
// line — the CMake LTO link line does not have it).
inline void flush_denormals() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

} // namespace

// Per-thread scratch: activations X, H, Y (rows × C), im2col buffer (rows × 9C),
// the stem im2col (rows × 36) and the head maps. Grown lazily; thread_local.
struct ResNetCPU::Scratch {
    int C = 0;
    size_t rows = 0;
    AlignedVec X, H, Y, Col, Col0, VMap, PMap, V64, HV;
    void ensure(int channels, size_t n_positions, int F) {
        const size_t need = n_positions * 64;
        if (channels == C && need <= rows) return;
        C = channels; rows = std::max(rows, need);
        X.assign(rows * C, 0.f); H.assign(rows * C, 0.f); Y.assign(rows * C, 0.f);
        Col.assign(rows * 9 * C, 0.f);
        Col0.assign(rows * 36, 0.f);
        VMap.assign(rows, 0.f); PMap.assign(rows * 2, 0.f);
        V64.assign(64, 0.f); HV.assign(F, 0.f);
    }
};

ResNetCPU::ResNetCPU(const std::string& weights_filename, int nb_threads, size_t preferred_batch)
    : filename(weights_filename), threads(nb_threads), preferred_batch(preferred_batch) {
    load(weights_filename);
}

ResNetCPU::~ResNetCPU() = default;

void ResNetCPU::load(const std::string& weights_filename) {
    Reader r(weights_filename);
    char magic[8];
    if (std::fread(magic, 1, 8, r.f) != 8 || std::memcmp(magic, "YOLAHRSN", 8) != 0)
        throw std::runtime_error("ResNetCPU: bad magic in " + weights_filename +
                                 " (expected a file written by resnet_export.py)");
    const uint32_t version = r.u32();
    if (version != 1) throw std::runtime_error("ResNetCPU: unsupported version " + std::to_string(version));
    const int in_channels = static_cast<int>(r.u32());
    if (in_channels != PLANES) throw std::runtime_error("ResNetCPU: expected 4 input planes");
    const int c = static_cast<int>(r.u32()), b = static_cast<int>(r.u32());
    const int f = static_cast<int>(r.u32()), a = static_cast<int>(r.u32());
    if (C != 0 && (c != C || b != B || f != F || a != A))
        throw std::runtime_error("ResNetCPU::reload: architecture mismatch");
    if (a != NUM_ACTIONS) throw std::runtime_error("ResNetCPU: expected 4096 actions");
    C = c; B = b; F = f; A = a;

    r.floats(stem_w, static_cast<size_t>(C) * 9 * PLANES);
    r.floats(stem_b, C);
    blocks.assign(B, Block{});
    for (Block& blk : blocks) {
        r.floats(blk.bn1_scale, C); r.floats(blk.bn1_shift, C);
        r.floats(blk.conv1_w, static_cast<size_t>(C) * 9 * C);
        r.floats(blk.bn2_scale, C); r.floats(blk.bn2_shift, C);
        r.floats(blk.conv2_w, static_cast<size_t>(C) * 9 * C);
    }
    r.floats(out_scale, C); r.floats(out_shift, C);
    r.floats(vconv_w, C); r.floats(vconv_b, 1);
    r.floats(vfc1_w, static_cast<size_t>(F) * 64); r.floats(vfc1_b, F);
    r.floats(vfc2_w, F); r.floats(vfc2_b, 1);
    r.floats(pconv_w, 2 * static_cast<size_t>(C)); r.floats(pconv_b, 2);
    r.floats(pfc_w, static_cast<size_t>(A) * 128); r.floats(pfc_b, A);
    // Anything left over means the exporter and this reader disagree.
    char probe;
    if (std::fread(&probe, 1, 1, r.f) == 1)
        throw std::runtime_error("ResNetCPU: trailing bytes in " + weights_filename);
    filename = weights_filename;
}

void ResNetCPU::reload(const std::string& weights_filename) {
    load(weights_filename);
}

std::string ResNetCPU::info() const {
    std::ostringstream os;
    os << "ResNetCPU " << C << "x" << B << " (" << filename << ", "
       << (threads ? std::to_string(threads) : std::string("omp default")) << " threads)";
    return os.str();
}

float ResNetCPU::policy_logit(const float* policy_vec, int a) const {
    const float* w = pfc_w.data() + static_cast<size_t>(a) * 128;
    float s = pfc_b[a];
    for (int k = 0; k < 128; k++) s += w[k] * policy_vec[k];
    return s;
}

void ResNetCPU::forward_chunk(Scratch& s, const float* planes, size_t n,
                              float* values, float* policy_vec) const {
    s.ensure(C, std::max(n, MAX_CHUNK), F);
    const size_t rows = n * 64;

    // ── stem: im2col over the 4 input planes (NCHW → pixel-major 36-wide rows) ──
    for (size_t b = 0; b < n; b++) {
        const float* pl = planes + b * INPUT_FLOATS;
        for (int p = 0; p < 64; p++) {
            float* crow = s.Col0.data() + (b * 64 + p) * 36;
            for (int tap = 0; tap < 9; tap++) {
                const int src = TAPS.src[p][tap];
                for (int c = 0; c < PLANES; c++) {
                    crow[tap * PLANES + c] = src < 0 ? 0.f : pl[c * 64 + src];
                }
            }
        }
    }
    MapRow X(s.X.data(), rows, C);
    {
        MapConstRow col0(s.Col0.data(), rows, 36);
        MapConstRow w(stem_w.data(), C, 36);
        X.noalias() = col0 * w.transpose();
        X.rowwise() += Eigen::Map<const Eigen::RowVectorXf>(stem_b.data(), C);
        X = X.cwiseMax(0.0f);                     // the stem ends with a ReLU
    }

    // ── residual tower ──
    MapRow Y(s.Y.data(), rows, C);
    MapConstRow col(s.Col.data(), rows, 9 * C);
    for (const Block& blk : blocks) {
        bn_relu(s.X.data(), s.H.data(), rows, C, blk.bn1_scale.data(), blk.bn1_shift.data());
        im2col(s.H.data(), s.Col.data(), n, C);
        MapConstRow w1(blk.conv1_w.data(), C, 9 * C);
        Y.noalias() = col * w1.transpose();
        bn_relu(s.Y.data(), s.H.data(), rows, C, blk.bn2_scale.data(), blk.bn2_shift.data());
        im2col(s.H.data(), s.Col.data(), n, C);
        MapConstRow w2(blk.conv2_w.data(), C, 9 * C);
        X.noalias() += col * w2.transpose();      // skip connection
    }
    bn_relu(s.X.data(), s.H.data(), rows, C, out_scale.data(), out_shift.data());
    MapConstRow H(s.H.data(), rows, C);

    // ── heads ──
    // 1x1 convs are plain matrix products over the pixel rows.
    Eigen::Map<Eigen::VectorXf> vmap(s.VMap.data(), rows);
    vmap.noalias() = H * Eigen::Map<const Eigen::VectorXf>(vconv_w.data(), C);
    MapRow pmap(s.PMap.data(), rows, 2);
    pmap.noalias() = H * MapConstRow(pconv_w.data(), 2, C).transpose();

    Eigen::Map<const RowMat> Wfc1(vfc1_w.data(), F, 64);
    Eigen::Map<const Eigen::VectorXf> bfc1(vfc1_b.data(), F);
    Eigen::Map<const Eigen::VectorXf> wfc2(vfc2_w.data(), F);
    Eigen::Map<Eigen::VectorXf> v64(s.V64.data(), 64);
    Eigen::Map<Eigen::VectorXf> hv(s.HV.data(), F);
    for (size_t b = 0; b < n; b++) {
        // value: relu(conv+bias) over the 64 pixels → FC → ReLU → FC → tanh
        for (int p = 0; p < 64; p++) v64[p] = std::max(0.f, s.VMap[b * 64 + p] + vconv_b[0]);
        hv.noalias() = Wfc1 * v64 + bfc1;
        hv = hv.cwiseMax(0.f);
        values[b] = std::tanh(wfc2.dot(hv) + vfc2_b[0]);
        // policy: relu(conv+bias), flattened channel-major (c*64 + p) as in torch's flatten
        float* pv = policy_vec + b * 128;
        for (int p = 0; p < 64; p++) {
            pv[p]      = std::max(0.f, s.PMap[(b * 64 + p) * 2 + 0] + pconv_b[0]);
            pv[64 + p] = std::max(0.f, s.PMap[(b * 64 + p) * 2 + 1] + pconv_b[1]);
        }
    }
}

void ResNetCPU::forward(const float* planes, size_t n, float* values, float* logits) const {
    const size_t nb_chunks = (n + MAX_CHUNK - 1) / MAX_CHUNK;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : omp_get_max_threads())
    for (size_t k = 0; k < nb_chunks; k++) {
        thread_local Scratch scratch;
        flush_denormals();
        const size_t start = k * MAX_CHUNK, cnt = std::min(MAX_CHUNK, n - start);
        float pvec[MAX_CHUNK * 128];
        forward_chunk(scratch, planes + start * INPUT_FLOATS, cnt, values + start, pvec);
        for (size_t b = 0; b < cnt; b++) {
            float* out = logits + (start + b) * NUM_ACTIONS;
            for (int a = 0; a < NUM_ACTIONS; a++) out[a] = policy_logit(pvec + b * 128, a);
        }
    }
}

void ResNetCPU::evaluate(std::span<const Request> in, std::span<Result> out) {
    const size_t n = in.size();
    if (out.size() < n) throw std::invalid_argument("ResNetCPU::evaluate: output span too small");
    // Chunk so that every OpenMP thread gets work even for small batches.
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    const size_t chunk = std::clamp<size_t>(n / static_cast<size_t>(nthreads), 1, MAX_CHUNK);
    const size_t nb_chunks = (n + chunk - 1) / chunk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (size_t k = 0; k < nb_chunks; k++) {
        thread_local Scratch scratch;
        flush_denormals();
        const size_t start = k * chunk, cnt = std::min(chunk, n - start);
        float planes[MAX_CHUNK * INPUT_FLOATS];
        float values[MAX_CHUNK];
        float pvec[MAX_CHUNK * 128];
        for (size_t b = 0; b < cnt; b++) encode_planes(*in[start + b].state, planes + b * INPUT_FLOATS);
        forward_chunk(scratch, planes, cnt, values, pvec);
        for (size_t b = 0; b < cnt; b++) {
            const Request& rq = in[start + b];
            Result& rs = out[start + b];
            rs.value = values[b];
            for (uint16_t m = 0; m < rq.nb_moves; m++) {
                rs.logits[m] = policy_logit(pvec + b * 128, action_index(rq.moves[m]));
            }
        }
    }
}

} // namespace nn
