// opening_book_main.cpp — a book of balanced openings for the matches between
// networks (alphazero_learn match --book, nnue/alphazero_match.sh BOOK=...).
//
// Why: after a random 4-ply opening, Black wins almost every game between two
// strong players, so a pair of games (one opening, each colour) nearly always
// ends 1-1 and says nothing about which network is stronger. A balanced
// opening is one where neither colour is clearly ahead: the colour no longer
// decides the result, the players do.
//
//   opening_book build --player ../config/search_study/nnue_k8_them_spacelite.cfg
//         [--plies 6] [--candidates 20000] [--pre-depth 6] [--pre-max 6000] [--depth 10] [--target 0] [--max-value 1500]
//         [--size 1000] [--threads 16] [--seed 1] [--out ../config/openings/balanced.txt]
//     1. Draws --candidates random openings of --plies plies (uniform random
//        legal moves; none ends the game), keeping one of each class of
//        equivalent positions: 4 of the 8 symmetries of the board map the
//        initial position to itself (identity, 180° rotation, the two diagonal
//        flips), so up to 4 openings lead to the same game up to symmetry (the
//        other 4 send the black pieces to the white ones' initial squares).
//     2. Searches each with the dev player at fixed depth --depth (no clock:
//        the book is reproducible), from an empty table. The value is turned
//        into Black's point of view (±30000 = tanh ±1, a value v ≈ the
//        expected result 2p − 1 of the side to move).
//     3. Keeps the --size openings whose value is closest to --target, within
//        ±--max-value, in a random order (so the first openings of the file are
//        not the most balanced ones only). --target: the value at which Black
//        scores ½ in actual games, which is not exactly 0 (check --csv).
//     Book lines: the moves ("a1:b2 a8:b7 ..."), then "# black value V", then
//     the value histogram of all the candidates in comments at the top.
//
//   opening_book check --book FILE --player CFG [--depth 8] [--games 400] [--threads 16]
//                      [--random-plies 4] [--seed 1] [--csv value_result.csv] [--opponent-depth D2]
//     The dev player against itself at fixed depth, one game from each of the
//     first --games openings of the book, then from as many random openings
//     of --random-plies plies: Black's score in each case. A balanced book
//     gives Black about 50 %; random openings give Black much more. --csv: the
//     value of each book opening and Black's result, to calibrate --target
//     (on a book built with a large --max-value; --random-plies 0 skips the
//     random openings). --opponent-depth D2: pairs instead, as in the matches
//     between networks — depth --depth against depth D2, each opening once with
//     each colour: how often the colour decides the pair, and the score.
#include "magic.h"
#include "zobrist.h"
#include "player.h"
#include "minmax_nnue_dev_player.h"
#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector, std::cerr, std::cout;

namespace {

std::map<string, string> parse_args(int argc, char* argv[], int from) {
    std::map<string, string> kv;
    for (int i = from; i < argc; i++) {
        string k = argv[i];
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) { cerr << "bad argument " << k << '\n'; std::exit(2); }
        kv[k.substr(2)] = argv[++i];
    }
    return kv;
}
string get(const std::map<string, string>& kv, const string& k, const string& def) {
    auto it = kv.find(k);
    return it == kv.end() ? def : it->second;
}

std::unique_ptr<Player> make_dev_player(const string& cfg) {
    std::ifstream f(cfg);
    if (!f) { cerr << "cannot open " << cfg << '\n'; std::exit(1); }
    json j = json::parse(f);
    j["verbose"] = false;
    j["nb threads"] = 1;
    j["tt size"] = 64;
    auto p = Player::create(j);
    if (!dynamic_cast<MinMaxNNUE_DevPlayer*>(p.get())) { cerr << cfg << ": not a MinMaxNNUE_DevPlayer\n"; std::exit(1); }
    return p;
}

string move_list(const vector<Move>& moves) {
    std::ostringstream os;
    for (size_t i = 0; i < moves.size(); i++) os << (i ? " " : "") << moves[i];
    return os.str();
}

Move parse_move(const string& s) {
    auto sq = [&](const string& t) {
        if (t.size() != 2 || t[0] < 'a' || t[0] > 'h' || t[1] < '1' || t[1] > '8') {
            cerr << "bad move " << s << '\n';
            std::exit(1);
        }
        return Square((t[1] - '1') * 8 + (t[0] - 'a'));
    };
    const auto colon = s.find(':');
    if (colon == string::npos) { cerr << "bad move " << s << " (expected a1:b2)\n"; std::exit(1); }
    return Move(sq(s.substr(0, colon)), sq(s.substr(colon + 1)));
}

// The 4 symmetries of the board that map the initial position to itself:
// square (file, rank) → itself, (7−f, 7−r), (r, f), (7−r, 7−f).
uint64_t transform(uint64_t bb, int sym) {
    uint64_t r = 0;
    for (; bb; bb &= bb - 1) {
        const int s = std::countr_zero(bb), f = s & 7, k = s >> 3;
        int f2 = f, k2 = k;
        if (sym == 1) { f2 = 7 - f; k2 = 7 - k; }
        else if (sym == 2) { f2 = k; k2 = f; }
        else if (sym == 3) { f2 = 7 - k; k2 = 7 - f; }
        r |= uint64_t(1) << (k2 * 8 + f2);
    }
    return r;
}
// The smallest (black, white, holes) of the 4 images: equal for equivalent positions
// (same side to move and scores: the openings all have the same number of plies).
std::tuple<uint64_t, uint64_t, uint64_t> canonical(const Yolah& y) {
    std::tuple<uint64_t, uint64_t, uint64_t> best{~0ull, ~0ull, ~0ull};
    for (int s = 0; s < 4; s++)
        best = std::min(best, {transform(y.bitboard(Yolah::BLACK), s), transform(y.bitboard(Yolah::WHITE), s),
                               transform(y.empty_bitboard(), s)});
    return best;
}

// A random opening of `plies` plies that does not end the game.
vector<Move> random_opening(std::mt19937_64& rng, int plies) {
    for (;;) {
        Yolah y;
        vector<Move> res;
        for (int p = 0; p < plies && !y.game_over(); p++) {
            Yolah::MoveList moves;
            y.moves(moves);
            const Move m = moves[rng() % moves.size()];
            res.push_back(m);
            y.play(m);
        }
        if (!y.game_over()) return res;
    }
}

Yolah after(const vector<Move>& moves) {
    Yolah y;
    for (Move m : moves) y.play(m);
    return y;
}

// Runs body(i, worker) for i in [0, n) on `threads` threads, each with its own worker index.
template <class F> void parallel_for(size_t n, int threads, F&& body) {
    std::atomic<size_t> next{0};
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] { for (size_t i; (i = next++) < n;) body(i, t); });
    for (auto& th : pool) th.join();
}

int build(const std::map<string, string>& kv) {
    const string cfg = get(kv, "player", "../config/search_study/nnue_k8_them_spacelite.cfg");
    const int plies = std::stoi(get(kv, "plies", "6"));
    const size_t nb_candidates = std::stoul(get(kv, "candidates", "20000"));
    const int depth = std::stoi(get(kv, "depth", "10"));
    const int pre_depth = std::stoi(get(kv, "pre-depth", "6"));
    const int pre_max = std::stoi(get(kv, "pre-max", "6000"));
    const int max_value = std::stoi(get(kv, "max-value", "1500"));
    const int target = std::stoi(get(kv, "target", "0"));
    const size_t size = std::stoul(get(kv, "size", "1000"));
    const int threads = std::stoi(get(kv, "threads", "16"));
    const uint64_t seed = std::stoull(get(kv, "seed", "1"));
    const string out = get(kv, "out", "../config/openings/balanced.txt");

    // 1. Candidates, one per symmetry class.
    std::mt19937_64 rng(seed);
    vector<vector<Move>> cands;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> seen;
    size_t draws = 0;
    while (cands.size() < nb_candidates && draws < 50 * nb_candidates) {
        draws++;
        vector<Move> o = random_opening(rng, plies);
        if (seen.insert(canonical(after(o))).second) cands.push_back(std::move(o));
    }
    cerr << cands.size() << " distinct openings of " << plies << " plies (" << draws << " drawn)\n";

    // 2. Their value for Black at fixed depth: first a cheap search at --pre-depth for
    // all of them, then --depth for those within ±--pre-max only (most random openings
    // are far from balanced: the shallow search is enough to discard them).
    vector<std::unique_ptr<Player>> players;
    for (int t = 0; t < threads; t++) players.push_back(make_dev_player(cfg));
    std::mutex mu;
    auto values_at = [&](const vector<vector<Move>>& openings, int d) {
        vector<int> v(openings.size());
        std::atomic<size_t> done{0};
        parallel_for(openings.size(), threads, [&](size_t i, int t) {
            auto* dev = static_cast<MinMaxNNUE_DevPlayer*>(players[t].get());
            dev->clear_table();
            const Yolah y = after(openings[i]);
            const auto r = dev->search(y, uint8_t(d), 1'000'000'000'000ull);
            v[i] = y.current_player() == Yolah::BLACK ? r.value : -r.value;
            const size_t n = ++done;
            if (n % 10000 == 0) { std::lock_guard lk(mu); cerr << n << "/" << openings.size() << " searched at depth " << d << "\n"; }
        });
        return v;
    };
    const vector<int> pre_value = values_at(cands, pre_depth);
    vector<vector<Move>> close;
    for (size_t i = 0; i < cands.size(); i++) if (std::abs(pre_value[i]) <= pre_max) close.push_back(cands[i]);
    cerr << close.size() << " within ±" << pre_max << " at depth " << pre_depth << "\n";
    const vector<int> value = values_at(close, depth);

    // 3. The most balanced ones, in a random order: the closest to --target, the value
    // at which Black's expected score is ½ (measured with check --csv: not exactly 0).
    vector<size_t> idx(close.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    auto dist = [&](size_t i) { return std::abs(value[i] - target); };
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return dist(a) < dist(b); });
    vector<size_t> keep;
    for (size_t i : idx) if (keep.size() < size && dist(i) <= max_value) keep.push_back(i);
    std::shuffle(keep.begin(), keep.end(), rng);

    // Histogram of the candidates' values (Black's point of view), for the record.
    std::map<int, int> hist;
    const int bin = 1500;
    double mean = 0;
    for (int v : pre_value) { hist[int(std::floor(double(v) / bin))]++; mean += v; }
    mean /= std::max<size_t>(1, pre_value.size());
    int worst = 0;
    for (size_t i : keep) worst = std::max(worst, dist(i));

    std::ofstream f(out);
    if (!f) { cerr << "cannot write " << out << '\n'; return 1; }
    f << "# Balanced openings for the matches between networks (test/opening_book_main.cpp).\n"
      << "# opening_book build --player " << cfg << " --plies " << plies << " --candidates " << nb_candidates
      << " --pre-depth " << pre_depth << " --pre-max " << pre_max << " --depth " << depth << " --target " << target << " --max-value " << max_value << " --size " << size << " --seed " << seed << "\n"
      << "# " << cands.size() << " distinct random openings (up to the 4 symmetries of the initial position), searched by the dev\n"
      << "# player at depth " << pre_depth << "; the " << close.size() << " within ±" << pre_max << " searched again at depth " << depth << ";\n"
      << "# kept: the " << keep.size() << " with the value (Black's point of view, ±30000 = tanh ±1) closest to " << target << ",\n"
      << "# all within " << target << " ± " << worst << ", in a random order. Mean depth-" << pre_depth << " value of the candidates: "
      << std::lround(mean) << ".\n# Depth-" << pre_depth << " values of the candidates (bins of " << bin << "):\n";
    for (auto [b, c] : hist)
        f << std::format("#   [{:+6}, {:+6}) {:6}  {:5.1f} %\n", b * bin, (b + 1) * bin, c, 100.0 * c / pre_value.size());
    for (size_t i : keep) f << move_list(close[i]) << "  # black value " << value[i] << "\n";
    cerr << "wrote " << keep.size() << " openings to " << out << " (|value − " << target << "| ≤ " << worst << ", candidates' mean "
         << std::lround(mean) << ")\n";
    return 0;
}

// The openings of a book and their values (the "# black value V" comments; 0 if none).
vector<vector<Move>> read_book(const string& path, vector<int>* values = nullptr) {
    std::ifstream f(path);
    if (!f) { cerr << "cannot open " << path << '\n'; std::exit(1); }
    vector<vector<Move>> res;
    for (string line; std::getline(f, line);) {
        const size_t hash = line.find('#');
        std::istringstream is(line.substr(0, hash));
        vector<Move> o;
        for (string t; is >> t;) o.push_back(parse_move(t));
        if (o.empty()) continue;
        res.push_back(std::move(o));
        if (values) {
            const size_t k = hash == string::npos ? string::npos : line.find("black value", hash);
            values->push_back(k == string::npos ? 0 : std::stoi(line.substr(k + 11)));
        }
    }
    return res;
}

int check(const std::map<string, string>& kv) {
    const string cfg = get(kv, "player", "../config/search_study/nnue_k8_them_spacelite.cfg");
    const int depth = std::stoi(get(kv, "depth", "8"));
    const size_t games = std::stoul(get(kv, "games", "400"));
    const int threads = std::stoi(get(kv, "threads", "16"));
    const int random_plies = std::stoi(get(kv, "random-plies", "4"));
    std::mt19937_64 rng(std::stoull(get(kv, "seed", "1")));
    vector<int> book_values;
    vector<vector<Move>> book = read_book(get(kv, "book", "../config/openings/balanced.txt"), &book_values);
    if (book.size() > games) book.resize(games);
    vector<vector<Move>> randoms;
    for (size_t i = 0; i < book.size(); i++) randoms.push_back(random_opening(rng, random_plies));

    vector<std::unique_ptr<Player>> players;
    for (int t = 0; t < threads; t++) players.push_back(make_dev_player(cfg));
    // One game of the dev player against itself from each opening: Black's result (1, ½, 0).
    auto play_all = [&](const vector<vector<Move>>& openings) {
        vector<double> res(openings.size());
        parallel_for(openings.size(), threads, [&](size_t i, int t) {
            auto* dev = static_cast<MinMaxNNUE_DevPlayer*>(players[t].get());
            dev->clear_table();
            Yolah y = after(openings[i]);
            while (!y.game_over()) y.play(dev->search(y, uint8_t(depth), 1'000'000'000'000ull).move);
            const auto [b, w] = y.score();
            res[i] = b > w ? 1.0 : (b < w ? 0.0 : 0.5);
        });
        return res;
    };
    auto report = [&](const char* name, const vector<double>& res) {
        const double n = double(res.size());
        double s = 0, s2 = 0;
        int w = 0, d = 0, l = 0;
        for (double x : res) { s += x; s2 += x * x; (x == 1 ? w : (x == 0 ? l : d))++; }
        const double m = s / n, se = std::sqrt(std::max(0.0, s2 / n - m * m) / n);
        cout << std::format("{:>22}: {} games, Black {}W {}D {}L, Black's score {:.3f} ± {:.3f}\n",
                            name, res.size(), w, d, l, m, 1.96 * se);
    };
    if (kv.contains("opponent-depth")) {
        // Pairs, as in the matches between networks: A (--depth) against B (--opponent-depth),
        // each opening once with each colour. How often the colour decides (the pair ends 1-1
        // with Black winning both games), and A's score, from the book and from random openings.
        const int depth_b = std::stoi(get(kv, "opponent-depth", "10"));
        auto pairs = [&](const vector<vector<Move>>& openings) {
            vector<std::array<double, 2>> res(openings.size());   // Black's result with A black, with A white
            parallel_for(2 * openings.size(), threads, [&](size_t k, int t) {
                auto* dev = static_cast<MinMaxNNUE_DevPlayer*>(players[t].get());
                const size_t i = k / 2;
                const bool a_black = k % 2 == 0;
                Yolah y = after(openings[i]);
                dev->clear_table();
                while (!y.game_over()) {
                    const bool a_to_move = (y.current_player() == Yolah::BLACK) == a_black;
                    dev->clear_table();   // one player, two depths: no table shared between them
                    y.play(dev->search(y, uint8_t(a_to_move ? depth : depth_b), 1'000'000'000'000ull).move);
                }
                const auto [b, w] = y.score();
                res[i][k % 2] = b > w ? 1.0 : (b < w ? 0.0 : 0.5);
            });
            return res;
        };
        auto report_pairs = [&](const string& name, const vector<std::array<double, 2>>& res) {
            int colour = 0, a_both = 0, b_both = 0;
            double sa = 0;
            for (auto [ab, aw] : res) {   // A's results: ab with Black, 1 − aw with White
                sa += ab + (1 - aw);
                colour += ab == 1 && aw == 1;
                colour += ab == 0 && aw == 0;
                a_both += ab == 1 && aw == 0;
                b_both += ab == 0 && aw == 1;
            }
            const double n = 2.0 * res.size(), s_a = sa / n;
            const double elo = 400 * std::log10(std::clamp(s_a, 1e-3, 1 - 1e-3) / (1 - std::clamp(s_a, 1e-3, 1 - 1e-3)));
            cout << std::format("{:>22}: {} pairs; colour decides (same colour wins both) {:.1f} %, A wins both {:.1f} %, "
                                "B wins both {:.1f} %; A's score {:.3f} (Elo {:+.0f})\n",
                                name, res.size(), 100.0 * colour / res.size(), 100.0 * a_both / res.size(),
                                100.0 * b_both / res.size(), s_a, elo);
        };
        cout << "A = dev player at depth " << depth << ", B = at depth " << depth_b << ", each opening with both colours\n";
        report_pairs("book", pairs(book));
        if (random_plies > 0) report_pairs(std::format("random {}-ply openings", random_plies), pairs(randoms));
        return 0;
    }
    cout << "dev player against itself at depth " << depth << ", one game per opening\n";
    const vector<double> res = play_all(book);
    report("book", res);
    if (kv.contains("csv")) {   // value of each opening and Black's result: the calibration of build --target
        std::ofstream csv(get(kv, "csv", ""));
        csv << "value,black_result\n";
        for (size_t i = 0; i < res.size(); i++) csv << book_values[i] << ',' << res[i] << '\n';
    }
    if (random_plies > 0) report(std::format("random {}-ply openings", random_plies).c_str(), play_all(randoms));
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    if (argc < 2) { cerr << "usage: opening_book build|check [--options]  (see test/opening_book_main.cpp)\n"; return 2; }
    const auto kv = parse_args(argc, argv, 2);
    const string cmd = argv[1];
    if (cmd == "build") return build(kv);
    if (cmd == "check") return check(kv);
    cerr << "unknown command " << cmd << '\n';
    return 2;
}
