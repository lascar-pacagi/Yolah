// space_features_main.cpp — the space features (nnue/space_features.h) of every
// position of a training cache, for the NNUE whose second layer reads them
// (NetPersp with a space mask, nnue/nnue_persp.py).
//
//   ./space_features compute CACHE_DIR [--threads 16]
//       reads CACHE_DIR/inputs.u8 (N × 193: black, white, holes MSB first, turn)
//       and writes CACHE_DIR/space.u8 (N × 16, row i = position i) and
//       CACHE_DIR/space_meta.json. The features are from the side to move.
//   ./space_features bench CACHE_DIR [--n 200000]
//       cost of the features per position: all, without the corrected flood
//       fill, the corrected flood fill alone.
//   ./space_features make-cache GAMES_DIR OUT_DIR [--max-positions N] [--stride S]
//       a small cache from data/games for local tests (inputs.u8, values.i8 =
//       result for the side to move, meta.json) with FAKE teacher labels
//       (0.6 · result): only to check that the training runs.
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include "space_features.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

constexpr int ROW = 193;

static uint64_t bits(const uint8_t* row) {      // MSB first: byte i is square 63 − i
    uint64_t b = 0;
    for (int i = 0; i < 64; i++) if (row[i]) b |= uint64_t(1) << (63 - i);
    return b;
}

static void features_of_row(const uint8_t* row, uint8_t out[space::NB], uint32_t mask = space::ALL) {
    const uint64_t black = bits(row), white = bits(row + 64), holes = bits(row + 128);
    const bool white_to_move = row[192] != 0;
    const uint64_t us = white_to_move ? white : black, them = white_to_move ? black : white;
    space::compute(us, them, ~(black | white | holes), out, mask);
}

struct Mapped {
    const uint8_t* data = nullptr;
    size_t size = 0;
    explicit Mapped(const string& path) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::perror(path.c_str()); std::exit(1); }
        struct stat st;
        fstat(fd, &st);
        size = size_t(st.st_size);
        data = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        madvise((void*)data, size, MADV_SEQUENTIAL);
    }
    ~Mapped() { munmap((void*)data, size); }
};

static int compute(const string& dir, int threads) {
    Mapped in(dir + "/inputs.u8");
    const size_t n = in.size / ROW;
    if (n * ROW != in.size) { std::cerr << "inputs.u8: size not a multiple of 193\n"; return 1; }
    const string out_path = dir + "/space.u8";
    const int fd = open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, off_t(n) * space::NB) != 0) { std::perror(out_path.c_str()); return 1; }
    constexpr size_t BLOCK = 1 << 18;
    std::atomic<size_t> next{0}, done{0};
    const auto t0 = std::chrono::steady_clock::now();
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&] {
            vector<uint8_t> buf(BLOCK * space::NB);
            for (size_t b; (b = next++) * BLOCK < n;) {
                const size_t lo = b * BLOCK, hi = std::min(n, lo + BLOCK);
                for (size_t i = lo; i < hi; i++) features_of_row(in.data + i * ROW, &buf[(i - lo) * space::NB]);
                const size_t bytes = (hi - lo) * space::NB;
                if (pwrite(fd, buf.data(), bytes, off_t(lo) * space::NB) != ssize_t(bytes)) { std::perror("pwrite"); std::exit(1); }
                const size_t d = done += hi - lo;
                if (t == 0 && b % 64 == 0) {
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::fprintf(stderr, "\r%zu / %zu positions (%.0f k/s)", d, n, d / s / 1000);
                }
            }
        });
    for (auto& t : pool) t.join();
    close(fd);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream meta(dir + "/space_meta.json");
    meta << "{\n  \"n_positions\": " << n << ",\n  \"path\": \"space.u8\",\n  \"dtype\": \"uint8\",\n"
         << "  \"shape\": [" << n << ", " << space::NB << "],\n  \"features\": [";
    for (int i = 0; i < space::NB; i++)
        meta << (i ? ", " : "") << "\"" << (i < space::PER_SIDE ? "us_" : "them_") << space::name(i) << "\"";
    meta << "],\n  \"scale\": 64,\n  \"source\": \"test/space_features_main.cpp, nnue/space_features.h\"\n}\n";
    std::fprintf(stderr, "\n");
    std::printf("%zu positions in %.1f s → %s, %s/space_meta.json\n", n, s, out_path.c_str(), dir.c_str());
    return 0;
}

static int bench(const string& dir, size_t n_max) {
    Mapped in(dir + "/inputs.u8");
    const size_t n = std::min(n_max, in.size / ROW);
    uint8_t out[space::NB];
    uint64_t sink = 0;
    const uint32_t corr_only = (1u << space::CORR) | (1u << (space::PER_SIDE + space::CORR));
    for (auto [label, mask] : {std::pair{"all 16 features", space::ALL}, {"without corr (14)", space::NO_CORR},
                               {"corr only (2)", corr_only}}) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 3; rep++)
            for (size_t i = 0; i < n; i++) { features_of_row(in.data + i * ROW, out, mask); sink += out[i & 15]; }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %-20s %7.1f ns per position (row decoding included)%s\n", label, 1e9 * s / (3.0 * n),
                    sink == 12345 ? " " : "");
    }
    // the row decoding alone
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 3; rep++)
        for (size_t i = 0; i < n; i++) { const uint8_t* r = in.data + i * ROW; sink += bits(r) ^ bits(r + 64) ^ bits(r + 128); }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %-20s %7.1f ns per position%s\n", "row decoding alone", 1e9 * s / (3.0 * n), sink == 12345 ? " " : "");
    return 0;
}

static int make_cache(const string& games_dir, const string& out_dir, size_t max_positions, size_t stride) {
    fs::create_directories(out_dir);
    vector<string> files;
    for (const auto& d : fs::directory_iterator(games_dir)) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());
    std::ofstream inputs(out_dir + "/inputs.u8", std::ios::binary), values(out_dir + "/values.i8", std::ios::binary);
    vector<uint16_t> teacher;     // float16 bits
    size_t n = 0, games = 0;
    auto f16 = [](float x) {      // float → IEEE half (|x| < 1, enough for fake labels)
        uint32_t u; std::memcpy(&u, &x, 4);
        const uint32_t sign = (u >> 16) & 0x8000;
        const int exp = int((u >> 23) & 0xFF) - 127 + 15;
        if (exp <= 0) return uint16_t(sign);
        return uint16_t(sign | (exp << 10) | ((u >> 13) & 0x3FF));
    };
    for (const string& path : files) {
        std::ifstream in(path, std::ios::binary);
        vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
        for (size_t i = 0; i + 1 < e.size() && n < max_positions; games++) {
            const int len = e[i], nb_random = e[i + 1];
            if (i + 2 * len + 4 > e.size()) break;
            const int black_score = e[i + 2 + 2 * len], white_score = e[i + 3 + 2 * len];
            const int black_result = black_score > white_score ? 1 : black_score < white_score ? -1 : 0;
            Yolah y;
            for (int k = 0; k <= len && n < max_positions; k++) {
                if (k >= nb_random && (k + games) % stride == 0 && !y.game_over()) {
                    uint8_t row[ROW];
                    const uint64_t bb[3] = { y.bitboard(Yolah::BLACK), y.bitboard(Yolah::WHITE), y.empty_bitboard() };
                    for (int p = 0; p < 3; p++)
                        for (int j = 0; j < 64; j++) row[p * 64 + j] = uint8_t(bb[p] >> (63 - j) & 1);
                    row[192] = y.current_player();
                    inputs.write((const char*)row, ROW);
                    const int8_t z = int8_t(y.current_player() == Yolah::BLACK ? black_result : -black_result);
                    values.write((const char*)&z, 1);
                    teacher.push_back(f16(0.6f * z));
                    n++;
                }
                if (k == len) break;
                y.play(Move(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k])));
            }
            i += 2 * len + 4;
        }
        if (n >= max_positions) break;
    }
    std::ofstream(out_dir + "/teacher_values.f16", std::ios::binary).write((const char*)teacher.data(), teacher.size() * 2);
    const size_t chunks = (n + (1 << 20) - 1) >> 20;
    vector<uint8_t> done(chunks, 1);
    std::ofstream(out_dir + "/teacher_done.u8", std::ios::binary).write((const char*)done.data(), chunks);
    std::ofstream(out_dir + "/teacher_meta.json") << "{\"fake\": \"0.6 * result, space_features make-cache (local test only)\"}\n";
    std::ofstream(out_dir + "/meta.json") << "{\n  \"n_positions\": " << n << ",\n  \"n_games\": " << games
        << ",\n  \"inputs\": {\"path\": \"inputs.u8\", \"dtype\": \"uint8\", \"shape\": [" << n << ", 193]},\n"
        << "  \"values\": {\"path\": \"values.i8\", \"dtype\": \"int8\", \"shape\": [" << n << "]},\n"
        << "  \"input_size\": 193,\n  \"value_perspective\": \"current_player\",\n  \"value_range\": [-1, 0, 1]\n}\n";
    std::printf("%zu positions of %zu games → %s (fake teacher labels: local tests only)\n", n, games, out_dir.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    const string cmd = argc > 1 ? argv[1] : "";
    auto opt = [&](const string& name, size_t def) {
        for (int i = 2; i + 1 < argc; i++) if (argv[i] == name) return size_t(std::stoull(argv[i + 1]));
        return def;
    };
    if (cmd == "compute" && argc >= 3) return compute(argv[2], int(opt("--threads", 16)));
    if (cmd == "bench" && argc >= 3) return bench(argv[2], opt("--n", 200000));
    if (cmd == "make-cache" && argc >= 4) return make_cache(argv[2], argv[3], opt("--max-positions", 4u << 20), opt("--stride", 1));
    std::cerr << "usage: space_features compute CACHE_DIR [--threads T] | bench CACHE_DIR [--n N] |"
                 " make-cache GAMES_DIR OUT_DIR [--max-positions N] [--stride S]\n";
    return 2;
}
