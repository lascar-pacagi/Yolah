// phase_fit_dump_main.cpp — the data of the "fast evaluation per phase" study
// (nnue/phase_fit.py): can a cheap function of hand-made features replace the
// NNUE in some phases of the game (a range of numbers of holes)?
//
//   ./phase_fit_dump CACHE_DIR nnue.quantized.txt OUT_DIR [--stride S]
//
// Reads the positions of a training cache (inputs.u8: 193 bytes per position,
// black / white / empty MSB-first + turn, see nnue/distill_common.py), one
// every S, and writes to OUT_DIR, for each of them:
//   index.i64        its index in the cache (to read the teacher's and the outcome's values)
//   hf.f32           heuristic_features::extract (24 floats, black − white)
//   yf.u8            YolahFeatures::set_features (119 bytes, black then white)
//   heur.f32         heuristic::eval (the old minimax's evaluation, GA weights), black's view
//   nnue.f32         the 193 NNUE's value for the side to move
//   meta.txt         sizes, and the mean cost of each computation (ns)
// Scores are not in the cache: set as if nobody had passed (they do not
// change any of these features but SCORE_DIFF).
#include "game.h"
#include "heuristic.h"
#include "heuristic_features.h"
#include "yolah_features.h"
#include "nnue_quantized.h"
#include "magic.h"
#include "zobrist.h"
#include <bit>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::string, std::vector;

static uint64_t bits(const uint8_t* row) {      // 64 bytes, MSB-first → bitboard
    uint64_t b = 0;
    for (int i = 0; i < 64; i++) b = (b << 1) | (row[i] & 1);
    return b;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: phase_fit_dump CACHE_DIR nnue.quantized.txt OUT_DIR [--stride S]\n";
        return 1;
    }
    magic::init();
    zobrist::init();
    const string cache = argv[1], weights = argv[2], out = argv[3];
    size_t stride = 1;
    for (int i = 4; i + 1 < argc; i++) if (string(argv[i]) == "--stride") stride = std::stoul(argv[i + 1]);

    std::ifstream in(cache + "/inputs.u8", std::ios::binary | std::ios::ate);
    const size_t n_all = size_t(in.tellg()) / 193;
    in.seekg(0);
    auto nnue = std::make_unique<NNUE_Quantized>();
    nnue->load(weights);
    NNUE_Quantized::Accumulator acc;

    std::ofstream f_idx(out + "/index.i64", std::ios::binary), f_hf(out + "/hf.f32", std::ios::binary),
        f_yf(out + "/yf.u8", std::ios::binary), f_heur(out + "/heur.f32", std::ios::binary),
        f_nnue(out + "/nnue.f32", std::ios::binary);
    double t_hf = 0, t_yf = 0, t_heur = 0, t_nnue = 0;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ns = [](auto a, auto b) { return std::chrono::duration<double, std::nano>(b - a).count(); };
    vector<uint8_t> row(193);
    size_t n = 0;
    for (size_t i = 0; i < n_all; i++) {
        in.read((char*)row.data(), 193);
        if (i % stride) continue;
        const uint64_t black = bits(&row[0]), white = bits(&row[64]), empty = bits(&row[128]);
        const int turn = row[192];
        const int holes = std::popcount(empty);
        const uint16_t ply = uint16_t(holes + ((holes & 1) != turn));
        Yolah y;
        y.set_state(black, white, empty, uint16_t((holes + 1) / 2), uint16_t(holes / 2), ply);

        auto t0 = now();
        const heuristic_features::Features hf = heuristic_features::extract(y);
        auto t1 = now();
        alignas(64) uint8_t yf[128]{};
        YolahFeatures::set_features(yf, y);
        auto t2 = now();
        const float heur = heuristic::eval(Yolah::BLACK, y);
        auto t3 = now();
        nnue->init(y, acc);
        auto t4 = now();
        const float v = nnue->value(acc, y.current_player());
        auto t5 = now();
        t_hf += ns(t0, t1); t_yf += ns(t1, t2); t_heur += ns(t2, t3); t_nnue += ns(t4, t5);

        const int64_t idx = int64_t(i);
        f_idx.write((const char*)&idx, 8);
        f_hf.write((const char*)hf.data(), sizeof(float) * hf.size());
        f_yf.write((const char*)yf, YolahFeatures::NB_FEATURES);
        f_heur.write((const char*)&heur, 4);
        f_nnue.write((const char*)&v, 4);
        n++;
    }
    std::ofstream meta(out + "/meta.txt");
    meta << "n " << n << "\nhf " << heuristic_features::NB_FEATURES << "\nyf " << YolahFeatures::NB_FEATURES
         << "\nns_hf " << t_hf / n << "\nns_yf " << t_yf / n << "\nns_heur " << t_heur / n
         << "\nns_nnue_value " << t_nnue / n << "\n";
    std::printf("%zu positions of %zu; mean cost: heuristic_features %.0f ns, YolahFeatures %.0f ns, "
                "heuristic::eval %.0f ns, NNUE value() %.0f ns (dense, without the accumulator update)\n",
                n, n_all, t_hf / n, t_yf / n, t_heur / n, t_nnue / n);
    return 0;
}
