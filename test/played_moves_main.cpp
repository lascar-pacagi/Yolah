// played_moves_main.cpp — do some moves stand out in the games, per phase?
// (move ordering study: a table of the moves the players choose, to put
// before the history-ordered moves)
//
//   ./played_moves GAMES_DIR [--threads 12] [--smooth 10] [--prior OUT.bin]
//
// The games of data/games (binary: nb_moves, nb_random_moves, from/to per
// move, black score, white score). Every position after the random opening
// moves where the side to move has a choice: for each legal move (per phase,
// colour, from, to) how often it was available and how often it was chosen.
// The files are split in two: the table is learnt on the even ones, then on
// the odd ones each position's legal moves are ranked by the table's rate
//     rate = (chosen + smooth · m) / (available + smooth)   (m: the phase's mean rate)
// and we look at the rank of the move actually played: how often the table
// puts it first (top 1), in the first 3, against chance (1 / number of moves).
// Phase: free squares · 8 / 57, as the history prior (step T).
// --prior: the table written as a history prior for the dev player (same
// format as nnue/history_prior.py: int16 [phase][colour][from][to],
// 1500 · (logit(rate) − logit(mean)), clipped to ±8192).
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

constexpr int PH = 8;
static int phase(const Yolah& y) { return std::popcount(y.free_squares()) * PH / 57; }

struct Table {
    vector<uint64_t> avail = vector<uint64_t>(PH * 2 * 64 * 64), chosen = vector<uint64_t>(PH * 2 * 64 * 64);
    static size_t at(int ph, int c, Move m) { return ((size_t(ph) * 2 + c) * 64 + m.from_sq()) * 64 + m.to_sq(); }
    void add(const Table& o) {
        for (size_t i = 0; i < avail.size(); i++) { avail[i] += o.avail[i]; chosen[i] += o.chosen[i]; }
    }
};

struct Eval {
    uint64_t positions[PH]{}, moves[PH]{};
    double top1[PH]{}, top3[PH]{}, chance[PH]{}, top1_global[PH]{};
    void add(const Eval& o) {
        for (int p = 0; p < PH; p++) {
            positions[p] += o.positions[p]; moves[p] += o.moves[p]; top1[p] += o.top1[p];
            top3[p] += o.top3[p]; chance[p] += o.chance[p]; top1_global[p] += o.top1_global[p];
        }
    }
};

// Calls f(y, played move, legal moves) for every position with a choice after the random moves.
template <class F> static uint64_t for_each_position(const string& path, F&& f) {
    std::ifstream in(path, std::ios::binary);
    vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
    uint64_t games = 0;
    for (size_t i = 0; i + 1 < e.size();) {
        const int n = e[i], nb_random = e[i + 1];
        if (i + 2 * n + 4 > e.size()) break;
        Yolah y;
        Yolah::MoveList moves;
        for (int k = 0; k < n; k++) {
            const Move m(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k]));
            y.moves(moves);
            if (std::find(moves.begin(), moves.end(), m) == moves.end()) break;   // corrupt: stop the game
            if (k >= nb_random && m != Move::none() && moves.size() > 1) f(y, m, moves);
            y.play(m);
        }
        i += 2 * n + 4;
        games++;
    }
    return games;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "usage: played_moves GAMES_DIR [--threads T] [--smooth S]\n"; return 1; }
    magic::init();
    zobrist::init();
    int threads = 12;
    double smooth = 10;
    string prior_out;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (string(argv[i]) == "--threads") threads = std::stoi(argv[i + 1]);
        else if (string(argv[i]) == "--smooth") smooth = std::stod(argv[i + 1]);
        else if (string(argv[i]) == "--prior") prior_out = argv[i + 1];
    }
    vector<string> files;
    for (const auto& d : fs::directory_iterator(argv[1])) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());
    vector<string> train, test;
    for (size_t i = 0; i < files.size(); i++) (i % 2 ? test : train).push_back(files[i]);

    // 1. The table, on the even files.
    Table table;
    std::mutex mtx;
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> games{0};
    auto run = [&](auto&& body) {
        vector<std::thread> pool;
        for (int t = 0; t < threads; t++) pool.emplace_back(body);
        for (auto& t : pool) t.join();
    };
    run([&] {
        auto local = std::make_unique<Table>();
        for (size_t j; (j = next++) < train.size();) {
            games += for_each_position(train[j], [&](const Yolah& y, Move played, const Yolah::MoveList& moves) {
                const int ph = phase(y), c = y.current_player();
                for (const Move& m : moves) local->avail[Table::at(ph, c, m)]++;
                local->chosen[Table::at(ph, c, played)]++;
            });
        }
        std::lock_guard lk(mtx);
        table.add(*local);
    });
    const uint64_t train_games = games;
    // The rates, and a table without phases (to see what the phase adds).
    vector<double> rate(table.avail.size()), rate_global(2 * 64 * 64);
    double mean[PH];
    for (int ph = 0; ph < PH; ph++) {
        uint64_t a = 0, c = 0;
        for (size_t k = Table::at(ph, 0, Move::none()); k < Table::at(ph + 1, 0, Move::none()); k++) {
            a += table.avail[k]; c += table.chosen[k];
        }
        mean[ph] = a ? double(c) / a : 0;
        for (size_t k = Table::at(ph, 0, Move::none()); k < Table::at(ph + 1, 0, Move::none()); k++)
            rate[k] = (table.chosen[k] + smooth * mean[ph]) / (table.avail[k] + smooth);
    }
    {
        vector<uint64_t> a(2 * 64 * 64), c(2 * 64 * 64);
        uint64_t sa = 0, sc = 0;
        for (size_t k = 0; k < table.avail.size(); k++) {
            a[k % (2 * 64 * 64)] += table.avail[k]; c[k % (2 * 64 * 64)] += table.chosen[k];
            sa += table.avail[k]; sc += table.chosen[k];
        }
        for (size_t k = 0; k < a.size(); k++) rate_global[k] = (c[k] + smooth * double(sc) / sa) / (a[k] + smooth);
    }

    if (!prior_out.empty()) {
        vector<int16_t> prior(rate.size());
        auto logit = [](double p) { p = std::clamp(p, 1e-6, 1 - 1e-6); return std::log(p / (1 - p)); };
        for (size_t k = 0; k < rate.size(); k++) {
            const int ph = int(k / (2 * 64 * 64));
            prior[k] = int16_t(std::clamp(1500 * (logit(rate[k]) - logit(mean[ph])), -8192.0, 8192.0));
        }
        std::ofstream(prior_out, std::ios::binary).write((const char*)prior.data(), prior.size() * sizeof(int16_t));
        std::printf("prior written to %s\n", prior_out.c_str());
    }
    // 2. The rank of the played move, on the odd files.
    Eval ev;
    next = 0;
    games = 0;
    run([&] {
        Eval local;
        for (size_t j; (j = next++) < test.size();) {
            games += for_each_position(test[j], [&](const Yolah& y, Move played, const Yolah::MoveList& moves) {
                const int ph = phase(y), c = y.current_player();
                const double r = rate[Table::at(ph, c, played)];
                const double rg = rate_global[(size_t(c) * 64 + played.from_sq()) * 64 + played.to_sq()];
                int above = 0, tied = 0, above_g = 0, tied_g = 0;
                for (const Move& m : moves) {
                    if (m == played) continue;
                    const double x = rate[Table::at(ph, c, m)];
                    above += x > r; tied += x == r;
                    const double xg = rate_global[(size_t(c) * 64 + m.from_sq()) * 64 + m.to_sq()];
                    above_g += xg > rg; tied_g += xg == rg;
                }
                // Ties: the played move is placed at random among them.
                local.positions[ph]++;
                local.moves[ph] += moves.size();
                local.top1[ph] += above == 0 ? 1.0 / (tied + 1) : 0;
                local.top3[ph] += above >= 3 ? 0 : std::min(1.0, double(3 - above) / (tied + 1));
                local.top1_global[ph] += above_g == 0 ? 1.0 / (tied_g + 1) : 0;
                local.chance[ph] += 1.0 / moves.size();
            });
        }
        std::lock_guard lk(mtx);
        ev.add(local);
    });

    std::printf("%zu files: table from %lu games (even files), measured on %lu games (odd files)\n",
                files.size(), (unsigned long)train_games, (unsigned long)uint64_t(games));
    std::printf("%5s %7s %11s %7s %8s %8s %8s %10s\n", "phase", "free", "positions", "moves", "chance", "top 1",
                "top 3", "top 1 (no");
    std::printf("%5s %7s %11s %7s %8s %8s %8s %10s\n", "", "", "", "mean", "top 1", "", "", "phase)");
    for (int ph = PH - 1; ph >= 0; ph--) {
        const double n = double(ev.positions[ph]);
        if (n == 0) continue;
        std::printf("%5d %3d–%-3d %11lu %7.1f %7.1f%% %7.1f%% %7.1f%% %9.1f%%\n", ph, (ph * 57 + PH - 1) / PH,
                    ((ph + 1) * 57 + PH - 1) / PH - 1, (unsigned long)ev.positions[ph], ev.moves[ph] / n,
                    100 * ev.chance[ph] / n, 100 * ev.top1[ph] / n, 100 * ev.top3[ph] / n, 100 * ev.top1_global[ph] / n);
    }
    // The moves that stand out most, per phase: chosen far more often than the phase's mean rate.
    std::printf("\nMoves chosen most often when available (at least 1000 times available), per phase:\n");
    for (int ph = PH - 1; ph >= 0; ph--) {
        vector<std::pair<double, size_t>> best;
        for (size_t k = Table::at(ph, 0, Move::none()); k < Table::at(ph + 1, 0, Move::none()); k++)
            if (table.avail[k] >= 1000) best.push_back({double(table.chosen[k]) / table.avail[k], k});
        std::sort(best.rbegin(), best.rend());
        std::printf("  phase %d (mean rate %.1f%%):", ph, 100 * mean[ph]);
        for (size_t i = 0; i < std::min<size_t>(6, best.size()); i++) {
            const size_t k = best[i].second;
            const int c = int(k / (64 * 64)) % 2;
            const Move m(Square((k / 64) % 64), Square(k % 64));
            std::cout << ' ' << (c ? "W " : "B ") << m;
            std::printf(" %.0f%%", 100 * best[i].first);
        }
        std::printf("\n");
    }
    return 0;
}
