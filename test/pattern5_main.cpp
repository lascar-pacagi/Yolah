// pattern5_main.cpp — the 5×5 patterns around a piece that go with won games
// and with lost ones, found by a beam search (the user's pattern study, after
// test/pattern_stats_main.cpp for 3×3).
//
//   ./pattern5 GAMES_DIR OUT_PREFIX --phase 0|1|2 [--target result|mobility]
//              [--stride 4] [--threads 24] [--beam 100] [--max-fixed 12]
//              [--min-share 0.0005] [--min-step 0.005] [--top 12]
//
// Data: every --stride-th position of every game of GAMES_DIR (after the
// random opening moves) whose number of free squares is in the phase
// (0: 40–56, 1: 20–39, 2: 0–19), and for each of its 8 pieces the 24 squares
// of the 5×5 block centred on it, from the piece owner's view:
//     1 free   2 hole   3 mine   4 his   5 off the board
// Target: the owner's final result (+1 / 0 / −1) minus his mean result in
// the same situation — colour, side to move or not, number of free squares
// (--target result), and also the number of free squares among the piece's
// 8 neighbours (--target mobility: what a pattern says beyond the piece's
// immediate mobility).
//
// A pattern fixes some of the 24 squares, the others are ? (anything). With
// 6^24 patterns, they cannot all be counted: a BEAM SEARCH builds them one
// square at a time (subgroup discovery):
//   level 1: every pattern with 1 fixed square;
//   level k: each of the `beam` best patterns of level k−1, in each direction
//            (winning, losing), extended by one more fixed square in every
//            possible way; the `beam` best extensions are kept.
// Quality of a pattern: gain × √occurrences (gain = mean target of its
// occurrences): a significance, so that an extra fixed square must pay for
// the occurrences it loses. An extension is only kept if it moves the gain
// by at least --min-step in its direction (each fixed square must matter:
// the squares that change nothing stay ?; at level 1, any square on the side
// of the direction), and if it still covers --min-share of the pieces.
// Patterns that cover exactly the same pieces (squares off the board imply
// each other) are kept once. Rotations and reflections of a pattern are the
// same pattern (a piece matches it if it matches one of its 8 versions).
//
// Output: OUT_PREFIX.txt (per level, the best patterns of each direction,
// drawn) and OUT_PREFIX.tsv (every kept pattern: level, direction, the 24
// squares, occurrences, share, gain, z, gain of its parent). z counts the
// pieces of one game as one sample (they share its result).
#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

constexpr int NC = 24, NS = 5, NR = NC * NS;   // squares, states (1..5), refinements (square, state)
static int DR[NC], DC[NC];                     // (rank, file) step of each square, rank +2 (north) first
static int PERM[8][NC];                        // square c goes to PERM[t][c] under symmetry t

struct Occ {
    uint32_t m[NS];       // m[s − 1]: the squares in state s
    float target;
    int8_t result;
    uint8_t key, free, mob;   // key = colour · 2 + side to move
};

using Pattern = std::array<uint8_t, NC>;       // 0 = ?, 1..5 = state

static void init_geometry() {
    int c = 0;
    for (int r = 2; r >= -2; r--)
        for (int f = -2; f <= 2; f++)
            if (r || f) { DR[c] = r; DC[c] = f; c++; }
    for (int k = 0, t = 0; k < 4; k++)
        for (int flip = 0; flip < 2; flip++, t++)
            for (int d = 0; d < NC; d++) {
                int r = DR[d], f = flip ? -DC[d] : DC[d];
                for (int q = 0; q < k; q++) { const int r0 = r; r = f; f = -r0; }
                for (int e = 0; e < NC; e++) if (DR[e] == r && DC[e] == f) PERM[t][d] = e;
            }
}

static Pattern transform(const Pattern& p, int t) {
    Pattern q{};
    for (int c = 0; c < NC; c++) q[PERM[t][c]] = p[c];
    return q;
}

static Pattern canonical(const Pattern& p) {
    Pattern best = p;
    for (int t = 1; t < 8; t++) best = std::min(best, transform(p, t));
    return best;
}

static string text(const Pattern& p) {
    static const char* SYM = "?.oma#";
    string s;
    for (uint8_t x : p) s += SYM[x];
    return s;
}

// The 5 lines of the drawing of p (P = the piece).
static vector<string> draw(const Pattern& p) {
    static const char* SYM = "?.oma#";
    vector<string> lines;
    int c = 0;
    for (int r = 2; r >= -2; r--) {
        string l;
        for (int f = -2; f <= 2; f++) {
            l += (r || f) ? SYM[p[c++]] : 'P';
            if (f < 2) l += ' ';
        }
        lines.push_back(l);
    }
    return lines;
}

// The occurrences of one game file in the phase.
static uint64_t read_file(const string& path, size_t stride, int phase, vector<Occ>& out) {
    std::ifstream in(path, std::ios::binary);
    vector<uint8_t> e((std::istreambuf_iterator<char>(in)), {});
    vector<Occ> occ;
    uint64_t games = 0;
    size_t game = 0;
    for (size_t i = 0; i + 1 < e.size(); game++) {
        const int n = e[i], nb_random = e[i + 1];
        if (i + 2 * n + 4 > e.size()) break;
        const int bs = e[i + 2 + 2 * n], ws = e[i + 3 + 2 * n];
        const int black_result = bs > ws ? 1 : bs < ws ? -1 : 0;
        Yolah y;
        Yolah::MoveList moves;
        occ.clear();
        bool ok = true;
        for (int k = 0; k <= n; k++) {
            const int nfree = std::popcount(y.free_squares());
            const int ph = nfree >= 40 ? 0 : nfree >= 20 ? 1 : 2;
            if (k >= nb_random && (k + game) % stride == 0 && ph == phase && !y.game_over()) {
                const uint64_t free = y.free_squares(), holes = y.empty_bitboard();
                for (int c = 0; c < 2; c++) {
                    const uint64_t mine = y.bitboard(c);
                    for (uint64_t b = mine; b; b &= b - 1) {
                        const int sq = std::countr_zero(b), r = sq / 8, f = sq % 8;
                        Occ o{};
                        o.result = int8_t(c == 0 ? black_result : -black_result);
                        o.key = uint8_t(c * 2 + (y.current_player() == c));
                        o.free = uint8_t(nfree);
                        for (int d = 0; d < NC; d++) {
                            const int rr = r + DR[d], ff = f + DC[d];
                            int s = 5;
                            if (rr >= 0 && rr <= 7 && ff >= 0 && ff <= 7) {
                                const uint64_t bit = uint64_t(1) << (rr * 8 + ff);
                                s = (free & bit) ? 1 : (holes & bit) ? 2 : (mine & bit) ? 3 : 4;
                            }
                            o.m[s - 1] |= 1u << d;
                            if (s == 1 && std::abs(DR[d]) <= 1 && std::abs(DC[d]) <= 1) o.mob++;
                        }
                        occ.push_back(o);
                    }
                }
            }
            if (k == n) break;
            const Move m(Square(e[i + 2 + 2 * k]), Square(e[i + 3 + 2 * k]));
            y.moves(moves);
            if (std::find(moves.begin(), moves.end(), m) == moves.end()) { ok = false; break; }
            y.play(m);
        }
        if (ok && !occ.empty()) { out.insert(out.end(), occ.begin(), occ.end()); games++; }
        i += 2 * n + 4;
    }
    return games;
}

struct Entry {
    Pattern p;
    int dir;                 // +1 winning, −1 losing
    double n, gain, parent_gain;
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "usage: pattern5 GAMES_DIR OUT_PREFIX --phase 0|1|2 [--target result|mobility] [--stride S] "
                     "[--threads T] [--beam B] [--max-fixed K] [--min-share F] [--min-step G] [--top N]\n";
        return 1;
    }
    magic::init();
    zobrist::init();
    init_geometry();
    const string games_dir = argv[1], prefix = argv[2];
    int phase = -1, threads = 24, beam = 100, max_fixed = 12, top = 12;
    size_t stride = 4;
    double min_share = 0.0005, min_step = 0.005;
    string target = "result";
    for (int i = 3; i + 1 < argc; i += 2) {
        const string k = argv[i], v = argv[i + 1];
        if (k == "--phase") phase = std::stoi(v);
        else if (k == "--target") target = v;
        else if (k == "--stride") stride = std::stoul(v);
        else if (k == "--threads") threads = std::stoi(v);
        else if (k == "--beam") beam = std::stoi(v);
        else if (k == "--max-fixed") max_fixed = std::stoi(v);
        else if (k == "--min-share") min_share = std::stod(v);
        else if (k == "--min-step") min_step = std::stod(v);
        else if (k == "--top") top = std::stoi(v);
    }
    if (phase < 0 || phase > 2 || (target != "result" && target != "mobility")) { std::cerr << "bad --phase / --target\n"; return 1; }
    const bool by_mob = target == "mobility";

    // 1. The occurrences.
    vector<string> files;
    for (const auto& d : fs::directory_iterator(games_dir)) if (d.is_regular_file()) files.push_back(d.path());
    std::sort(files.begin(), files.end());
    vector<vector<Occ>> parts(files.size());
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> games{0};
    {
        vector<std::thread> pool;
        for (int t = 0; t < threads; t++)
            pool.emplace_back([&] { for (size_t j; (j = next++) < files.size();) games += read_file(files[j], stride, phase, parts[j]); });
        for (auto& t : pool) t.join();
    }
    size_t total = 0;
    for (auto& p : parts) total += p.size();
    vector<Occ> occ;
    occ.reserve(total);
    for (auto& p : parts) { occ.insert(occ.end(), p.begin(), p.end()); vector<Occ>().swap(p); }
    // 2. The target: the result minus the mean result of the same situation.
    static double bsum[4][65][9], bn[4][65][9];
    for (const Occ& o : occ) { const int mob = by_mob ? o.mob : 0; bsum[o.key][o.free][mob] += o.result; bn[o.key][o.free][mob]++; }
    double sq = 0;
    for (Occ& o : occ) {
        const int mob = by_mob ? o.mob : 0;
        o.target = float(o.result - bsum[o.key][o.free][mob] / bn[o.key][o.free][mob]);
        sq += double(o.target) * o.target;
    }
    const double N = double(occ.size()), sd = std::sqrt(sq / N), per_game = N / double(games);
    const double min_n = min_share * N;
    auto z_of = [&](double n, double gain) { return gain * std::sqrt(n / per_game) / sd; };
    std::FILE* txt = std::fopen((prefix + ".txt").c_str(), "w");
    std::FILE* tsv = std::fopen((prefix + ".tsv").c_str(), "w");
    std::fprintf(tsv, "level\tdirection\tpattern\toccurrences\tshare\tgain\tz\tparent_gain\n");
    const char* PHASES[] = {"40–56 free squares", "20–39 free squares", "0–19 free squares"};
    std::fprintf(txt, "5×5 patterns, phase %s, target %s; %zu files, every %zu-th position, %.0f pieces from %lu games "
                      "(%.1f per game), target sd %.3f; beam %d, min share %.2f %%, min step %.3f\n"
                      "P the piece, m his own pieces, a the opponent's, o hole, . free, # off the board, ? anything\n"
                      "gain: mean (result − expected result), result in −1/0/+1 (+0.10 = 5 points of score %%); "
                      "z: a game counts as one sample\n",
                 PHASES[phase], target.c_str(), files.size(), stride, N, (unsigned long)games.load(), per_game, sd,
                 beam, 100 * min_share, min_step);
    std::fflush(txt);
    std::fprintf(stderr, "%.0f occurrences from %lu games\n", N, (unsigned long)games.load());

    // 3. The beam search.
    vector<Entry> current = {{Pattern{}, 0, N, 0.0, 0.0}};
    for (int level = 1; level <= max_fixed && !current.empty(); level++) {
        const size_t B = current.size();
        // The 8 versions of each pattern, as masks per state.
        vector<std::array<std::array<uint32_t, NS>, 8>> masks(B);
        for (size_t b = 0; b < B; b++)
            for (int t = 0; t < 8; t++) {
                const Pattern q = transform(current[b].p, t);
                masks[b][t] = {};
                for (int c = 0; c < NC; c++) if (q[c]) masks[b][t][q[c] - 1] |= 1u << c;
            }
        // Count, for each pattern of the beam and each (square, state) it leaves ?, the occurrences of the extension.
        vector<vector<double>> cnt(threads, vector<double>(B * NR)), sum(threads, vector<double>(B * NR));
        std::atomic<size_t> chunk{0};
        const size_t CH = 1 << 16;
        vector<std::thread> pool;
        for (int th = 0; th < threads; th++)
            pool.emplace_back([&, th] {
                double* cn = cnt[th].data();
                double* sm = sum[th].data();
                for (size_t start; (start = chunk++ * CH) < occ.size();) {
                    const size_t end = std::min(occ.size(), start + CH);
                    for (size_t i = start; i < end; i++) {
                        const Occ& o = occ[i];
                        for (size_t b = 0; b < B; b++) {
                            uint64_t bits[2] = {0, 0};
                            bool any = false;
                            for (int t = 0; t < 8; t++) {
                                const auto& mk = masks[b][t];
                                if ((mk[0] & ~o.m[0]) | (mk[1] & ~o.m[1]) | (mk[2] & ~o.m[2]) | (mk[3] & ~o.m[3]) | (mk[4] & ~o.m[4])) continue;
                                any = true;
                                // square c of the pattern is square PERM[t][c] of the occurrence
                                for (int c = 0; c < NC; c++) {
                                    if (current[b].p[c]) continue;
                                    const uint32_t bit = 1u << PERM[t][c];
                                    int s = 0;
                                    while (!(o.m[s] & bit)) s++;
                                    const int r = c * NS + s;
                                    bits[r >> 6] |= uint64_t(1) << (r & 63);
                                }
                            }
                            if (!any) continue;
                            for (int w = 0; w < 2; w++)
                                for (uint64_t x = bits[w]; x; x &= x - 1) {
                                    const size_t r = b * NR + w * 64 + std::countr_zero(x);
                                    cn[r] += 1;
                                    sm[r] += o.target;
                                }
                        }
                    }
                }
            });
        for (auto& t : pool) t.join();
        for (int th = 1; th < threads; th++)
            for (size_t r = 0; r < B * NR; r++) { cnt[0][r] += cnt[th][r]; sum[0][r] += sum[th][r]; }
        // The extensions: kept if frequent enough and if the square moves the gain in the direction.
        std::map<std::pair<int, Pattern>, Entry> cand;
        for (size_t b = 0; b < B; b++)
            for (int c = 0; c < NC; c++) {
                if (current[b].p[c]) continue;
                for (int s = 1; s <= NS; s++) {
                    const size_t r = b * NR + c * NS + (s - 1);
                    const double n = cnt[0][r];
                    if (n < min_n) continue;
                    const double g = sum[0][r] / n;
                    Pattern p = current[b].p;
                    p[c] = uint8_t(s);
                    for (int dir : {1, -1}) {
                        if (current[b].dir && current[b].dir != dir) continue;
                        // level 1: every square on the side of the direction (two squares that
                        // matter only together must be able to start); then a real step
                        if (level == 1 ? dir * g <= 0 : dir * (g - current[b].gain) < min_step) continue;
                        const auto key = std::make_pair(dir, canonical(p));
                        auto it = cand.find(key);
                        if (it == cand.end() || n > it->second.n) cand[key] = {key.second, dir, n, g, current[b].gain};
                    }
                }
            }
        vector<Entry> next_beam;
        for (int dir : {1, -1}) {
            // Patterns covering exactly the same pieces are one (e.g. squares off the board imply
            // each other: a corner piece always has the same ones); same count and same sum = same.
            std::map<std::pair<double, long long>, Entry> distinct;
            for (auto& [k, e] : cand)
                if (k.first == dir) distinct.try_emplace({e.n, std::llround(e.n * e.gain * 1e3)}, e);
            vector<Entry> v;
            for (auto& [k, e] : distinct) v.push_back(e);
            std::sort(v.begin(), v.end(), [&](const Entry& a, const Entry& b) {
                return dir * a.gain * std::sqrt(a.n) > dir * b.gain * std::sqrt(b.n);
            });
            if (int(v.size()) > beam) v.resize(beam);
            for (const Entry& e : v)
                std::fprintf(tsv, "%d\t%s\t%s\t%.0f\t%.6f\t%+.5f\t%+.2f\t%+.5f\n", level, dir > 0 ? "win" : "loss",
                             text(e.p).c_str(), e.n, e.n / N, e.gain, z_of(e.n, e.gain), e.parent_gain);
            // the drawing of the best ones
            const int shown = std::min<int>(top, int(v.size()));
            std::fprintf(txt, "\n── %d fixed square(s), %s patterns (%zu candidates) ──\n", level,
                         dir > 0 ? "WINNING" : "LOSING", v.size());
            for (int start = 0; start < shown; start += 6) {
                const int stop = std::min(shown, start + 6);
                vector<vector<string>> pics;
                for (int i = start; i < stop; i++) pics.push_back(draw(v[i].p));
                for (int row = 0; row < 5; row++) {
                    string l = "  ";
                    for (auto& pic : pics) { l += pic[row]; l += "    "; }
                    std::fprintf(txt, "%s\n", l.c_str());
                }
                string g = "  ", sh = "  ", z = "  ";
                char buf[64];
                for (int i = start; i < stop; i++) {
                    std::snprintf(buf, sizeof buf, "%+-13.3f", v[i].gain); g += buf;
                    std::snprintf(buf, sizeof buf, "%-13s", (std::to_string(v[i].n / N * 100).substr(0, 5) + " %").c_str()); sh += buf;
                    std::snprintf(buf, sizeof buf, "%+-13.1f", z_of(v[i].n, v[i].gain)); z += buf;
                }
                std::fprintf(txt, "%sgain\n%sshare\n%sz\n\n", g.c_str(), sh.c_str(), z.c_str());
            }
            next_beam.insert(next_beam.end(), v.begin(), v.end());
        }
        std::fflush(txt);
        std::fflush(tsv);
        std::fprintf(stderr, "level %d done: %zu patterns in the beam\n", level, next_beam.size());
        current = std::move(next_beam);
    }
    std::fclose(txt);
    std::fclose(tsv);
    return 0;
}
