// nets_check_main.cpp — checks the C++ minimax networks against PyTorch.
//
//   ./nets_check dump positions.bin [N]      random positions + their features
//   python3 ../nnue/check_value_nets.py positions.bin refs.bin   (PyTorch values)
//   ./nets_check check positions.bin refs.bin ../models
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
    if (cmd == "bench" && argc >= 5) return bench(argv[2], argv[3], argv[4]);
    std::cerr << "usage: nets_check dump positions.bin [N] | check positions.bin refs.bin models_dir\n";
    return 2;
}
