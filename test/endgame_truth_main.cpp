// endgame_truth_main.cpp — how right are the teacher and the NNUE in the
// endgame? (retrograde analysis study, nnue/endgame_truth.py)
//
//   ./endgame_truth CACHE_DIR nnue.quantized.txt OUT.bin [--free 14,16,18,20,22] [--per 2000]
//                   [--threads 12] [--timeout-ms 5000] [--wld 1]
//
// Picks --per positions of the training cache for each number of free squares
// in --free (evenly spread over the cache), solves each one exactly with the
// endgame solver (player/endgame_solver.h, final score difference) and writes
// one record per position (--wld 1: only win / draw / loss, much cheaper; the
// difference written is then only the sign of the exact one):
//   int64 index in the cache, int8 free squares, int8 solved (0: timeout),
//   int16 exact final score difference for the side to move,
//   float NNUE value for the side to move, float solve time (ms)
// The teacher's value and the game's outcome are read by the Python script
// from the cache with the index.
//
// Scores are not in the cache: set as if nobody had passed (the teacher and
// the NNUE do not see them either). Positions whose turn does not match the
// number of holes (a pass happened) are skipped.
#include "game.h"
#include "endgame_solver.h"
#include "nnue_quantized.h"
#include "magic.h"
#include "zobrist.h"
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;

static uint64_t bits(const uint8_t* row) {
    uint64_t b = 0;
    for (int i = 0; i < 64; i++) b = (b << 1) | (row[i] & 1);
    return b;
}

#pragma pack(push, 1)
struct Out {
    int64_t index;
    int8_t free;
    int8_t solved;
    int16_t diff;
    float nnue;
    float ms;
};
#pragma pack(pop)

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: endgame_truth CACHE_DIR nnue.quantized.txt OUT.bin [--free 14,16,...] [--per N] "
                     "[--threads T] [--timeout-ms MS] [--wld 1]\n";
        return 1;
    }
    magic::init();
    zobrist::init();
    const string cache = argv[1], weights = argv[2], out_path = argv[3];
    vector<int> frees = {14, 16, 18, 20, 22};
    size_t per = 2000;
    int threads = 12, timeout_ms = 5000;
    bool wld = false;
    for (int i = 4; i + 1 < argc; i += 2) {
        const string k = argv[i], v = argv[i + 1];
        if (k == "--free") { frees.clear(); std::stringstream ss(v); string t; while (std::getline(ss, t, ',')) frees.push_back(std::stoi(t)); }
        else if (k == "--per") per = std::stoul(v);
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--timeout-ms") timeout_ms = std::stoi(v);
        else if (k == "--wld") wld = v == "1";
    }

    // The candidate positions, per number of free squares.
    std::ifstream in(cache + "/inputs.u8", std::ios::binary | std::ios::ate);
    const size_t n_all = size_t(in.tellg()) / 193;
    in.seekg(0);
    vector<vector<std::pair<int64_t, Yolah>>> cand(65);
    vector<uint8_t> row(193);
    for (size_t i = 0; i < n_all; i++) {
        in.read((char*)row.data(), 193);
        const uint64_t black = bits(&row[0]), white = bits(&row[64]), empty = bits(&row[128]);
        const int holes = std::popcount(empty), turn = row[192];
        const int free = 64 - 8 - holes;
        if ((holes & 1) != turn) continue;                   // a pass happened: scores unknown
        bool wanted = false;
        for (int f : frees) wanted |= f == free;
        if (!wanted) continue;
        Yolah y;
        y.set_state(black, white, empty, uint16_t((holes + 1) / 2), uint16_t(holes / 2), uint16_t(holes));
        if (y.game_over()) continue;
        cand[free].push_back({int64_t(i), y});
    }
    vector<std::pair<int64_t, Yolah>> jobs;
    for (int f : frees) {
        const auto& c = cand[f];
        const size_t take = std::min(per, c.size());
        for (size_t j = 0; j < take; j++) jobs.push_back(c[j * c.size() / take]);
        std::printf("%2d free squares: %zu positions in the cache, %zu taken\n", f, c.size(), take);
    }

    auto nnue = std::make_unique<NNUE_Quantized>();
    nnue->load(weights);
    vector<Out> results(jobs.size());
    std::atomic<size_t> next{0}, done{0};
    auto worker = [&] {
        EndgameSolver solver;
        NNUE_Quantized::Accumulator acc;
        for (size_t j; (j = next++) < jobs.size();) {
            const auto& [idx, y] = jobs[j];
            Out o{};
            o.index = idx;
            o.free = int8_t(std::popcount(y.free_squares()));
            nnue->init(y, acc);
            o.nnue = nnue->value(acc, y.current_player());
            // A watchdog stops the solver after the timeout.
            std::atomic_bool stop{false};
            std::mutex m;
            std::condition_variable cv;
            bool finished = false;
            std::thread dog([&] {
                std::unique_lock lk(m);
                if (!cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return finished; })) stop = true;
            });
            const auto t0 = std::chrono::steady_clock::now();
            solver.clear();
            const EndgameSolver::Result r = solver.solve(y, wld, &stop);
            o.ms = float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            { std::lock_guard lk(m); finished = true; }
            cv.notify_one();
            dog.join();
            o.solved = r.complete;
            // The solver's value is the final difference, current scores included.
            o.diff = int16_t(r.value);
            results[j] = o;
            const size_t d = ++done;
            if (d % 1000 == 0) std::fprintf(stderr, "  %zu / %zu\n", d, jobs.size());
        }
    };
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    std::ofstream out(out_path, std::ios::binary);
    out.write((const char*)results.data(), results.size() * sizeof(Out));
    size_t solved = 0;
    for (const Out& o : results) solved += o.solved;
    std::printf("%zu positions, %zu solved within %d ms\n", results.size(), solved, timeout_ms);
    return 0;
}
