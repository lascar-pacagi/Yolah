// nets_check_main.cpp — checks the C++ minimax networks against PyTorch.
//
//   ./nets_check dump positions.bin [N]      random positions + their features
//   python3 ../nnue/check_value_nets.py positions.bin refs.bin   (PyTorch values)
//   ./nets_check check positions.bin refs.bin ../models
//   python3 ../nnue/check_persp_net.py positions.bin refs.bin model.pt   (a NetPersp)
//   ./nets_check check_persp positions.bin refs.bin model.quantized.txt [old.quantized.txt]
//   ./nets_check check_sparse positions.bin model.quantized.txt   (sparse product = dense, and speed)
//   ./nets_check values positions.bin first.quantized.txt other.quantized.txt...   (value scale per phase)
//
// For every position and every network — the old NNUE (3 outputs) and the
// value networks (NNUE and features, plain and distilled), each in float and
// quantized — it compares the value for the side to move computed in C++ with
// PyTorch's, and reports the largest error. The NNUE accumulators are built
// the way the minimax builds them: from the initial position, updated move by
// move with play() (and checked against init() on the final position).
//
// positions.bin, one record per position:
//   uint64 black, white, empty; uint8 turn; uint8 features[NB_FEATURES]
//   int32 nb_moves; nb_moves × (uint8 from, uint8 to)   — the game from the start
// refs.bin: for each network (in the order of NETWORKS in check_value_nets.py),
//   N float32 values for the side to move.
#include "nnue.h"
#include "nnue_quantized.h"
#include "nnue_persp.h"
#include "ffnn.h"
#include "ffnn_value.h"
#include <array>
#include "yolah_features.h"
#include "magic.h"
#include "zobrist.h"
#include "misc.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using std::string, std::vector;

namespace {

struct Record {
    uint64_t black, white, empty;
    uint8_t turn;
    uint8_t features[YolahFeatures::NB_FEATURES];
    vector<Move> moves;
};

void write_record(std::ofstream& out, const Yolah& y, const vector<Move>& moves) {
    const uint64_t b = y.bitboard(Yolah::BLACK), w = y.bitboard(Yolah::WHITE), e = y.empty_bitboard();
    const uint8_t turn = y.current_player();
    alignas(64) uint8_t features[128]{};
    YolahFeatures::set_features(features, y);
    out.write((const char*)&b, 8); out.write((const char*)&w, 8); out.write((const char*)&e, 8);
    out.write((const char*)&turn, 1);
    out.write((const char*)features, YolahFeatures::NB_FEATURES);
    const int32_t n = (int32_t)moves.size();
    out.write((const char*)&n, 4);
    for (Move m : moves) {
        const uint8_t f = (uint8_t)m.from_sq(), t = (uint8_t)m.to_sq();
        out.write((const char*)&f, 1); out.write((const char*)&t, 1);
    }
}

vector<Record> read_records(const string& path) {
    std::ifstream in(path, std::ios::binary);
    vector<Record> rs;
    Record r;
    while (in.read((char*)&r.black, 8)) {
        in.read((char*)&r.white, 8); in.read((char*)&r.empty, 8);
        in.read((char*)&r.turn, 1);
        in.read((char*)r.features, YolahFeatures::NB_FEATURES);
        int32_t n; in.read((char*)&n, 4);
        r.moves.clear();
        for (int i = 0; i < n; i++) {
            uint8_t f, t; in.read((char*)&f, 1); in.read((char*)&t, 1);
            r.moves.emplace_back(Square(f), Square(t));
        }
        rs.push_back(r);
    }
    return rs;
}

int dump(const string& path, size_t n) {
    std::ofstream out(path, std::ios::binary);
    std::mt19937_64 rng(12345);
    size_t written = 0;
    while (written < n) {
        Yolah y;
        vector<Move> moves;
        const size_t plies = rng() % 56;
        for (size_t p = 0; p < plies && !y.game_over(); p++) {
            Yolah::MoveList ml;
            y.moves(ml);
            const Move m = ml[rng() % ml.size()];
            moves.push_back(m);
            y.play(m);
        }
        if (y.game_over()) continue;
        if (std::getenv("NO_PASS")) {
            bool pass = false;
            for (Move m : moves) pass |= (m == Move::none());
            if (pass) continue;
        }
        write_record(out, y, moves);
        written++;
    }
    std::cout << "wrote " << written << " positions to " << path << "\n";
    return 0;
}

// Value of every record with an NNUE-like network, the accumulator being
// updated incrementally along the game as the minimax does.
template <class Net>
vector<float> nnue_values(Net& net, const vector<Record>& rs, double& acc_error) {
    vector<float> out;
    acc_error = 0;
    for (const Record& r : rs) {
        Yolah y;
        typename Net::Accumulator acc;
        net.init(y, acc);
        for (Move m : r.moves) {
            net.play(y.current_player(), m, acc);
            y.play(m);
        }
        typename Net::Accumulator fresh;
        net.init(y, fresh);
        for (int i = 0; i < Net::H1_SIZE; i++)
            acc_error = std::max(acc_error, (double)std::abs(float(acc.acc[i]) - float(fresh.acc[i])));
        out.push_back(net.value(acc, y.current_player()));
    }
    return out;
}

template <class Net>
vector<float> features_values(const Net& net, const vector<Record>& rs) {
    vector<float> out;
    for (const Record& r : rs) {
        alignas(64) uint8_t input[128]{};
        std::memcpy(input, r.features, YolahFeatures::NB_FEATURES);
        out.push_back(net.value(input));
    }
    return out;
}

int check(const string& positions, const string& refs_path, const string& models) {
    const vector<Record> rs = read_records(positions);
    const size_t n = rs.size();
    std::ifstream refs(refs_path, std::ios::binary);
    auto next_refs = [&] {
        vector<float> v(n);
        refs.read((char*)v.data(), n * sizeof(float));
        return v;
    };
    bool ok = true;
    // Float networks must match PyTorch to rounding (max error). Quantized
    // ones carry the intrinsic error of 6-bit activations: judged on the MEAN
    // error, against the level of the original quantized NNUE (~0.04).
    constexpr double QUANTIZED_MEAN_TOL = 0.08;
    auto report = [&](const string& name, const vector<float>& got, const vector<float>& ref, double tol,
                      double acc_err = -1, bool on_mean = false) {
        double max_err = 0, sum_err = 0;
        size_t sign_flips = 0;
        for (size_t i = 0; i < n; i++) {
            const double e = std::abs(got[i] - ref[i]);
            max_err = std::max(max_err, e); sum_err += e;
            if ((got[i] > 0.05 && ref[i] < -0.05) || (got[i] < -0.05 && ref[i] > 0.05)) sign_flips++;
        }
        const bool pass = (on_mean ? sum_err / n : max_err) <= tol && acc_err <= 1e-3;
        ok &= pass;
        std::printf("  %-44s max |err| %.5f  mean %.5f  sign flips %zu%s  %s\n", name.c_str(), max_err,
                    sum_err / n, sign_flips,
                    acc_err >= 0 ? (string("  acc drift ") + std::to_string(acc_err)).c_str() : "",
                    pass ? "OK" : "FAIL");
    };
    std::printf("%zu positions\n", n);
    for (const string base : {"nnue", "nnue_193x1024x64x32x1", "nnue_193x1024x64x32x1_distill"}) {
        const vector<float> ref = next_refs();
        double acc_err;
        {
            auto net = std::make_unique<NNUE>();
            net->load(models + "/" + base + ".float.txt");
            const vector<float> v = nnue_values(*net, rs, acc_err);
            report(base + " (float)", v, ref, 1e-3, acc_err);
        }
        {
            const string q = base == "nnue" ? models + "/nnue_quantized.txt" : models + "/" + base + ".quantized.txt";
            auto net = std::make_unique<NNUE_Quantized>();
            net->load(q);
            const vector<float> v = nnue_values(*net, rs, acc_err);
            report(base + " (quantized)", v, ref, QUANTIZED_MEAN_TOL, acc_err, true);
        }
    }
    for (const string base : {"features_119x256x64x1", "features_119x256x64x1_distill"}) {
        const vector<float> ref = next_refs();
        auto f = std::make_unique<FFNNValueFloat<YolahFeatures::NB_FEATURES, 256, 64>>(models + "/" + base + ".float.txt");
        report(base + " (float)", features_values(*f, rs), ref, 1e-4);
        auto q = std::make_unique<FFNNValue<YolahFeatures::NB_FEATURES, 256, 64>>(models + "/" + base + ".quantized.txt");
        report(base + " (quantized)", features_values(*q, rs), ref, QUANTIZED_MEAN_TOL, -1, true);
    }
    // Speed: evaluations per second of each features network (the NNUE ones
    // are dominated by the incremental accumulator updates, measured in play).
    for (const string base : {"features_119x256x64x1"}) {
        auto f = std::make_unique<FFNNValueFloat<YolahFeatures::NB_FEATURES, 256, 64>>(models + "/" + base + ".float.txt");
        auto q = std::make_unique<FFNNValue<YolahFeatures::NB_FEATURES, 256, 64>>(models + "/" + base + ".quantized.txt");
        auto bench = [&](auto& net, const char* label) {
            const auto t0 = std::chrono::steady_clock::now();
            float sink = 0;
            const int reps = 50;
            for (int r = 0; r < reps; r++) sink += features_values(net, rs)[r % n];
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  speed %-10s %8.0f evaluations/s  (%.2f µs each)%s\n", label, reps * n / dt,
                        1e6 * dt / (reps * n), sink == 12345.f ? " " : "");
        };
        bench(*f, "float");
        bench(*q, "int16");
    }
    std::cout << (ok ? "ALL NETWORKS OK" : "SOME NETWORKS FAILED") << "\n";
    return ok ? 0 : 1;
}

// NNUE_Persp (nnue_persp.h) against PyTorch. The accumulators are built three
// ways — play() move by move, update() move by move (the lazy search's way)
// and init() on the final position — and must be identical. Then the cost of
// value() and of one update, compared with NNUE_Quantized if a 193 network is given.
int check_persp(const string& positions, const string& refs_path, const string& weights, const string& old_weights) {
    const vector<Record> rs = read_records(positions);
    const size_t n = rs.size();
    vector<float> ref(n);
    std::ifstream(refs_path, std::ios::binary).read((char*)ref.data(), n * sizeof(float));
    auto net = std::make_unique<NNUE_Persp>(weights);
    std::printf("%zu positions, %s: K = %d, them = %d\n", n, weights.c_str(), net->k, int(net->them));
    double max_err = 0, sum_err = 0, acc_drift = 0;
    size_t sign_flips = 0;
    vector<Yolah> finals;
    for (const Record& r : rs) {
        Yolah y;
        NNUE_Persp::Accumulator played, fresh;
        vector<std::unique_ptr<NNUE_Persp::Accumulator>> lazy;
        lazy.push_back(std::make_unique<NNUE_Persp::Accumulator>());
        net->init(y, played);
        net->init(y, *lazy.back());
        for (Move m : r.moves) {
            net->play(y.current_player(), m, played);
            lazy.push_back(std::make_unique<NNUE_Persp::Accumulator>());
            net->update(lazy[lazy.size() - 2]->acc, lazy.back()->acc, y.current_player(), m);
            y.play(m);
        }
        net->init(y, fresh);
        for (int i = 0; i < NNUE_Persp::ACC_SIZE; i++) {
            acc_drift = std::max(acc_drift, (double)std::abs(played.acc[i] - fresh.acc[i]));
            acc_drift = std::max(acc_drift, (double)std::abs(lazy.back()->acc[i] - fresh.acc[i]));
        }
        const float v = net->value(played, y);
        const size_t i = finals.size();
        const double e = std::abs(v - ref[i]);
        max_err = std::max(max_err, e); sum_err += e;
        if ((v > 0.05 && ref[i] < -0.05) || (v < -0.05 && ref[i] > 0.05)) sign_flips++;
        finals.push_back(y);
    }
    const bool ok = sum_err / n <= 0.08 && acc_drift == 0;
    std::printf("  value vs PyTorch: max |err| %.5f  mean %.5f  sign flips %zu  acc drift %g  %s\n",
                max_err, sum_err / n, sign_flips, acc_drift, ok ? "OK" : "FAIL");
    // Speed, on the final positions.
    const int reps = 200;
    auto timed = [&](const char* label, auto&& body) {
        const auto t0 = std::chrono::steady_clock::now();
        float sink = 0;
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < n; i++) sink += body(i);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %-34s %7.1f ns%s\n", label, 1e9 * dt / (double(reps) * n), sink == 1.2345f ? " " : "");
    };
    vector<std::unique_ptr<NNUE_Persp::Accumulator>> accs;
    for (const Yolah& y : finals) { accs.push_back(std::make_unique<NNUE_Persp::Accumulator>()); net->init(y, *accs.back()); }
    NNUE_Persp::Accumulator out;
    const Move probe(SQ_A1, SQ_B2);
    timed("NNUE_Persp value()", [&](size_t i) { return net->value(*accs[i], finals[i]); });
    timed("NNUE_Persp update() (2 views)", [&](size_t i) { net->update(accs[i]->acc, out.acc, uint8_t(i & 1), probe); return float(out.acc[i & 1023]); });
    if (!old_weights.empty()) {
        auto old = std::make_unique<NNUE_Quantized>();
        old->load(old_weights);
        vector<std::unique_ptr<NNUE_Quantized::Accumulator>> oaccs;
        for (const Yolah& y : finals) { oaccs.push_back(std::make_unique<NNUE_Quantized::Accumulator>()); old->init(y, *oaccs.back()); }
        timed("NNUE_Quantized value()", [&](size_t i) { return old->value(*oaccs[i], finals[i].current_player()); });
        timed("NNUE_Quantized play()+undo()", [&](size_t i) { old->play(uint8_t(i & 1), probe, *oaccs[i]); old->undo(uint8_t(i & 1), probe, *oaccs[i]); return float(oaccs[i]->acc[i & 1023]); });
    }
    return ok ? 0 : 1;
}

// The sparse product (nnue/sparse_matvec.h) against the dense one, for a 193
// network (NNUE_Quantized) or a PERSP one: at every position of the games of
// positions.bin, the two values must be EXACTLY equal; then the cost of each.
int check_sparse(const string& positions, const string& weights) {
    const vector<Record> rs = read_records(positions);
    vector<Yolah> ys;
    for (const Record& r : rs) {
        Yolah y;
        for (Move m : r.moves) { if (!y.game_over()) ys.push_back(y); y.play(m); }
        if (!y.game_over()) ys.push_back(y);
    }
    const size_t n = ys.size();
    const bool is_persp = NNUE_Persp::is_persp_file(weights);
    std::unique_ptr<NNUE_Persp> persp;
    std::unique_ptr<NNUE_Quantized> old;
    vector<std::unique_ptr<NNUE_Persp::Accumulator>> paccs;
    vector<std::unique_ptr<NNUE_Quantized::Accumulator>> oaccs;
    if (is_persp) {
        persp = std::make_unique<NNUE_Persp>(weights);
        for (const Yolah& y : ys) { paccs.push_back(std::make_unique<NNUE_Persp::Accumulator>()); persp->init(y, *paccs.back()); }
    } else {
        old = std::make_unique<NNUE_Quantized>();
        old->load(weights);
        for (const Yolah& y : ys) { oaccs.push_back(std::make_unique<NNUE_Quantized::Accumulator>()); old->init(y, *oaccs.back()); }
    }
    auto set_sparse = [&](bool s) { if (is_persp) persp->sparse = s; else old->sparse = s; };
    auto value = [&](size_t i) { return is_persp ? persp->value(*paccs[i], ys[i]) : old->value(*oaccs[i], ys[i].current_player()); };
    std::printf("%zu positions (all the positions of %zu games), %s\n", n, rs.size(), weights.c_str());
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) {
        set_sparse(false);
        const float d = value(i);
        set_sparse(true);
        const float s = value(i);
        if (d != s) diff++;
    }
    std::printf("  sparse vs dense: %zu different values  %s\n", diff, diff == 0 ? "OK" : "FAIL");
    const int reps = 50;
    for (bool s : {false, true}) {
        set_sparse(s);
        const auto t0 = std::chrono::steady_clock::now();
        float sink = 0;
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < n; i++) sink += value(i);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  value() %-6s %7.1f ns%s\n", s ? "sparse" : "dense", 1e9 * dt / (double(reps) * n), sink == 1.2345f ? " " : "");
    }
    return diff == 0 ? 0 : 1;
}

// Value scale of several networks (193 or PERSP, quantized), for the side to
// move, at every position of the games of positions.bin, per phase (free
// squares 40–56 / 20–39 / 0–19): spread of the values (the search's margins,
// MPC 4000 + 500/ply and RFP 4500/ply, are in units of value × 30000), and how
// far each network is from the FIRST one (mean |difference|, correlation).
int values(const string& positions, const vector<string>& weights) {
    const vector<Record> rs = read_records(positions);
    vector<Yolah> ys;
    // VALUES_OUT=prefix: also writes every position (prefix.pos, positions.bin
    // format, for check_persp_net.py) and each network's values (prefix.<k>.bin).
    const char* out_prefix = std::getenv("VALUES_OUT");
    std::ofstream pos_out;
    if (out_prefix) pos_out.open(string(out_prefix) + ".pos", std::ios::binary);
    for (const Record& r : rs) {
        Yolah y;
        vector<Move> prefix;
        for (Move m : r.moves) {
            if (!y.game_over()) { ys.push_back(y); if (out_prefix) write_record(pos_out, y, prefix); }
            y.play(m); prefix.push_back(m);
        }
        if (!y.game_over()) { ys.push_back(y); if (out_prefix) write_record(pos_out, y, prefix); }
    }
    const size_t n = ys.size();
    vector<vector<float>> v(weights.size(), vector<float>(n));
    for (size_t k = 0; k < weights.size(); k++) {
        if (NNUE_Persp::is_persp_file(weights[k])) {
            auto net = std::make_unique<NNUE_Persp>(weights[k]);
            NNUE_Persp::Accumulator a;
            for (size_t i = 0; i < n; i++) { net->init(ys[i], a); v[k][i] = net->value(a, ys[i]); }
        } else {
            auto net = std::make_unique<NNUE_Quantized>();
            net->load(weights[k]);
            NNUE_Quantized::Accumulator a;
            for (size_t i = 0; i < n; i++) { net->init(ys[i], a); v[k][i] = net->value(a, ys[i].current_player()); }
        }
    }
    if (out_prefix)
        for (size_t k = 0; k < weights.size(); k++)
            std::ofstream(string(out_prefix) + "." + std::to_string(k) + ".bin", std::ios::binary)
                .write((const char*)v[k].data(), n * sizeof(float));
    auto phase = [](const Yolah& y) { const int f = std::popcount(y.free_squares()); return f >= 40 ? 0 : f >= 20 ? 1 : 2; };
    const char* names[] = {"opening 40-56", "middle 20-39", "endgame 0-19"};
    std::printf("%zu positions (all the positions of %zu games); values × 30000\n", n, rs.size());
    for (int p = 0; p < 3; p++) {
        vector<size_t> idx;
        for (size_t i = 0; i < n; i++) if (phase(ys[i]) == p) idx.push_back(i);
        std::printf("\n%s (%zu positions)\n  %-48s %7s %7s %7s %7s %9s %6s\n", names[p], idx.size(),
                    "network", "mean", "std", "|v|", "p90|v|", "|v-first|", "corr");
        for (size_t k = 0; k < weights.size(); k++) {
            double s = 0, s2 = 0, sa = 0, sd = 0, s0 = 0, s02 = 0, s0k = 0;
            vector<float> a;
            for (size_t i : idx) {
                const double x = v[k][i], x0 = v[0][i];
                s += x; s2 += x * x; sa += std::abs(x); sd += std::abs(x - x0);
                s0 += x0; s02 += x0 * x0; s0k += x0 * x;
                a.push_back(std::abs(v[k][i]));
            }
            const double m = idx.size(), mean = s / m, var = s2 / m - mean * mean, mean0 = s0 / m, var0 = s02 / m - mean0 * mean0;
            const double corr = (s0k / m - mean * mean0) / std::sqrt(var * var0);
            std::nth_element(a.begin(), a.begin() + a.size() * 9 / 10, a.end());
            string name = weights[k].substr(weights[k].find_last_of('/') + 1);
            std::printf("  %-48s %+7.0f %7.0f %7.0f %7.0f %9.0f %6.3f\n", name.c_str(), 30000 * mean,
                        30000 * std::sqrt(var), 30000 * sa / m, 30000 * a[a.size() * 9 / 10], 30000 * sd / m, corr);
        }
    }
    // Consistency along the games: a move adds one hole, so |v(parent) + v(child)|
    // (the side to move changes) when the 8-head bucket holes · 8 / 57 changes
    // between them, against when it does not (positions in the middle game only).
    std::printf("\nmean |v(parent) + v(child)| × 30000, middle game, child's head bucket (holes·8/57) changed or not\n");
    for (size_t k = 0; k < weights.size(); k++) {
        double s[2]{}; size_t c[2]{};
        for (size_t i = 0; i + 1 < n; i++) {
            const int h0 = std::popcount(ys[i].empty_bitboard()), h1 = std::popcount(ys[i + 1].empty_bitboard());
            if (h1 != h0 + 1 || phase(ys[i]) != 1) continue;   // not parent → child (next game)
            const int changed = h0 * 8 / 57 != h1 * 8 / 57;
            s[changed] += std::abs(v[k][i] + v[k][i + 1]); c[changed]++;
        }
        // Signed, per side to move of the parent: a network that misjudges whose
        // turn it is shows a bias here (Black wins ~63 % of the games).
        double b[2]{}; size_t cb[2]{};
        for (size_t i = 0; i + 1 < n; i++) {
            const int h0 = std::popcount(ys[i].empty_bitboard()), h1 = std::popcount(ys[i + 1].empty_bitboard());
            if (h1 != h0 + 1 || phase(ys[i]) != 1) continue;
            const int t = ys[i].current_player();
            b[t] += v[k][i] + v[k][i + 1]; cb[t]++;
        }
        string name = weights[k].substr(weights[k].find_last_of('/') + 1);
        std::printf("  %-48s same %6.0f (%zu)  changed %6.0f (%zu)  signed: black to move %+6.0f, white %+6.0f\n", name.c_str(),
                    30000 * s[0] / c[0], c[0], 30000 * s[1] / c[1], c[1], 30000 * b[0] / cb[0], 30000 * b[1] / cb[1]);
    }
    return 0;
}

// Cost of one leaf evaluation of the features players: the features, then
// each kind of network (int8 = FFNN of ffnn.h, the format of the original
// networks; its weights file only needs the right sizes).
int bench(const string& int8_file, const string& int16_file, const string& float_file) {
    constexpr int NB = YolahFeatures::NB_FEATURES;
    std::mt19937_64 rng(1);
    vector<Yolah> pos;
    while (pos.size() < 2000) {
        Yolah y;
        const size_t k = rng() % 50;
        for (size_t p = 0; p < k && !y.game_over(); p++) { Yolah::MoveList ml; y.moves(ml); y.play(ml[rng() % ml.size()]); }
        if (!y.game_over()) pos.push_back(y);
    }
    vector<std::array<uint8_t, 128>> feats(pos.size());
    const int reps = 100;
    const double N = double(reps) * pos.size();
    auto timed = [&](const char* label, auto&& body) {
        const auto t0 = std::chrono::steady_clock::now();
        float sink = 0;
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < pos.size(); i++) sink += body(i, r);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %-26s %6.3f µs%s\n", label, 1e6 * dt / N, sink == 1.2345f ? " " : "");
    };
    timed("set_features", [&](size_t i, int r) {
        alignas(64) uint8_t f[128]{};
        YolahFeatures::set_features(f, pos[i]);
        if (r == 0) std::copy(f, f + 128, feats[i].begin());
        return float(f[r % NB]);
    });
    auto i8 = std::make_unique<FFNN<NB, 256, 64, 3>>(int8_file);
    auto i16 = std::make_unique<FFNNValue<NB, 256, 64>>(int16_file);
    auto fl = std::make_unique<FFNNValueFloat<NB, 256, 64>>(float_file);
    timed("network int8 (FFNN)", [&](size_t i, int) { alignas(64) uint8_t x[128]; std::copy(feats[i].begin(), feats[i].end(), x); return std::get<0>((*i8)(x)); });
    timed("network int16 (value)", [&](size_t i, int) { alignas(64) uint8_t x[128]; std::copy(feats[i].begin(), feats[i].end(), x); return i16->value(x); });
    timed("network float (value)", [&](size_t i, int) { alignas(64) uint8_t x[128]; std::copy(feats[i].begin(), feats[i].end(), x); return fl->value(x); });
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    const string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "dump" && argc >= 3) return dump(argv[2], argc > 3 ? std::stoul(argv[3]) : 2000);
    if (cmd == "check" && argc >= 5) return check(argv[2], argv[3], argv[4]);
    if (cmd == "check_persp" && argc >= 5) return check_persp(argv[2], argv[3], argv[4], argc > 5 ? argv[5] : "");
    if (cmd == "check_sparse" && argc >= 4) return check_sparse(argv[2], argv[3]);
    if (cmd == "values" && argc >= 4) return values(argv[2], vector<string>(argv + 3, argv + argc));
    if (cmd == "bench" && argc >= 5) return bench(argv[2], argv[3], argv[4]);
    std::cerr << "usage: nets_check dump positions.bin [N] | check positions.bin refs.bin models_dir\n";
    return 2;
}
