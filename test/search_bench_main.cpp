// search_bench — measures the minimax search on a fixed set of positions.
//
//   search_bench positions --out bench_positions.txt [--games 25] [--every 6] [--seed 1]
//       Plays games (4 random plies, then the reference player at depth 5) and
//       keeps every `every`-th position: realistic positions, one JSON per line.
//
//   search_bench solve --positions bench_positions.txt --max-free 24 [--ordering none|tt|fastest]
//                      [--wld 1] [--brute 6] [--pass-rule 0|1] [--time MICROSECONDS] [--csv out.csv]
//       The exact endgame solver (player/endgame_solver.h) on the positions
//       with at most --max-free free squares: value, move, nodes, time.
//       --wld 1: win / draw / loss only. Positions not solved within --time
//       are reported as such.
//
//   search_bench blunders --player CFG --positions bench_positions.txt --max-free 26 --time US
//       For each position (with at most --max-free free squares): the player's
//       move in --time, and the solver's verdict (win / draw / loss) of the
//       position and of the position after that move. A blunder: the move
//       loses the best result (a won game drawn or lost, a drawn game lost).
//       Positions the solver cannot prove within 30 s are skipped.
//
//   search_bench run --player ../config/mm_nnue_dev_player.cfg --positions bench_positions.txt
//                    (--depth D | --time MICROSECONDS) [--csv out.csv] [--compare ref.csv] [--limit N]
//       Searches every position from an empty transposition table and prints,
//       per position, the depth reached, the value, the move, the nodes and the
//       time; then the totals. --depth: time to depth (the work to reach the
//       same depth); --time: depth reached in a given time. --compare: the same
//       positions searched by another player (its --csv): identical values
//       and moves, and the ratio of nodes and times.
//
// Players: MinMaxNNUE_BaselinePlayer and MinMaxNNUE_DevPlayer (they expose
// search() and clear_table()).
#include "magic.h"
#include "zobrist.h"
#include "player.h"
#include "minmax_nnue_baseline_player.h"
#include "minmax_nnue_dev_player.h"
#include "endgame_solver.h"
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <format>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using std::cout, std::cerr, std::string, std::vector;

namespace {
    struct Args {
        std::map<string, string> kv;
        string get(const string& k, const string& def = "") const {
            auto it = kv.find(k);
            return it == kv.end() ? def : it->second;
        }
        bool has(const string& k) const { return kv.contains(k); }
    };

    Args parse(int argc, char* argv[], int first) {
        Args a;
        for (int i = first; i < argc; i++) {
            string k = argv[i];
            if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
                cerr << "bad argument: " << k << '\n';
                std::exit(1);
            }
            a.kv[k.substr(2)] = argv[++i];
        }
        return a;
    }

    // One interface over the two players.
    struct Searcher {
        std::unique_ptr<Player> player;
        MinMaxNNUE_BaselinePlayer* baseline = nullptr;
        MinMaxNNUE_DevPlayer* dev = nullptr;

        explicit Searcher(const string& cfg) {
            std::ifstream f(cfg);
            if (!f) { cerr << "cannot open " << cfg << '\n'; std::exit(1); }
            json j = json::parse(f);
            j["verbose"] = false;
            player = Player::create(j);
            baseline = dynamic_cast<MinMaxNNUE_BaselinePlayer*>(player.get());
            dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(player.get());
            if (!baseline && !dev) {
                cerr << cfg << ": not a MinMaxNNUE_BaselinePlayer / MinMaxNNUE_DevPlayer\n";
                std::exit(1);
            }
        }
        void clear() { baseline ? baseline->clear_table() : dev->clear_table(); }
        // the same fields for both players
        MinMaxNNUE_BaselinePlayer::Result search(const Yolah& y, uint8_t depth, uint64_t us) {
            if (baseline) return baseline->search(y, depth, us);
            auto r = dev->search(y, depth, us);
            return {r.move, r.value, r.depth, r.nb_nodes, r.nb_hits, r.seconds};
        }
    };

    string move_str(Move m) {
        std::ostringstream os;
        os << m;
        return os.str();
    }

    // --max-free F: only the positions with at most F free squares; the
    // indices stay those of the file.
    vector<std::pair<size_t, Yolah>> select_positions(const vector<Yolah>& all, const Args& a) {
        const int max_free = std::stoi(a.get("max-free", "64"));
        vector<std::pair<size_t, Yolah>> res;
        for (size_t i = 0; i < all.size(); i++) {
            if (std::popcount(all[i].free_squares()) <= max_free) res.push_back({i, all[i]});
        }
        return res;
    }

    vector<Yolah> read_positions(const string& path) {
        std::ifstream f(path);
        if (!f) { cerr << "cannot open " << path << '\n'; std::exit(1); }
        vector<Yolah> res;
        string line;
        while (std::getline(f, line)) {
            if (!line.empty()) res.push_back(Yolah::from_json(line));
        }
        return res;
    }

    int make_positions(const Args& a) {
        const string out = a.get("out", "bench_positions.txt");
        const int games = std::stoi(a.get("games", "25"));
        const int every = std::stoi(a.get("every", "6"));
        std::mt19937_64 rng(std::stoull(a.get("seed", "1")));
        MinMaxNNUE_BaselinePlayer player(0, 64, 2, 3, a.get("weights", "../models/nnue_193x1024x64x32x1_distill.quantized.txt"));
        std::ofstream f(out);
        size_t n = 0;
        for (int g = 0; g < games; g++) {
            Yolah y;
            Yolah::MoveList moves;
            for (int p = 0; p < 4 && !y.game_over(); p++) {
                y.moves(moves);
                y.play(moves[rng() % moves.size()]);
            }
            while (!y.game_over()) {
                if (y.nb_plies() % every == 0) {
                    f << y.to_json() << '\n';
                    n++;
                }
                player.clear_table();
                y.play(player.search(y, 5, 0).move);
            }
        }
        cout << n << " positions written to " << out << '\n';
        return 0;
    }

    int solve(const Args& a) {
        auto positions = select_positions(read_positions(a.get("positions", "bench_positions.txt")), a);
        EndgameSolver::Options o;
        const string ord = a.get("ordering", "fastest");
        o.ordering = ord == "none" ? EndgameSolver::Ordering::None
                   : ord == "tt"   ? EndgameSolver::Ordering::TT : EndgameSolver::Ordering::Fastest;
        o.brute_force_free = std::stoi(a.get("brute", "6"));
        o.tt_bits = std::stoi(a.get("tt-bits", "21"));
        o.pass_rule = a.get("pass-rule", "1") == "1";
        const bool wld = a.get("wld", "0") == "1";
        const uint64_t us = std::stoull(a.get("time", "0"));
        EndgameSolver solver(o);
        std::ofstream csv;
        if (a.has("csv")) {
            csv.open(a.get("csv"));
            csv << "idx,ply,free,complete,value,move,nodes,seconds\n";
        }
        cout << std::format("{:>4} {:>4} {:>4} {:>6} {:>6} {:>12} {:>9} {:>10}\n",
                            "pos", "ply", "free", "value", "move", "nodes", "seconds", "knodes/s");
        uint64_t total_nodes = 0;
        double total_time = 0;
        int solved = 0;
        for (const auto& [i, position] : positions) {
            solver.clear();
            std::atomic_bool stop = false;
            std::jthread clock;
            std::mutex mtx;
            std::condition_variable_any cv;
            if (us) {
                clock = std::jthread([&](std::stop_token st) {
                    std::unique_lock lock(mtx);
                    if (!cv.wait_for(lock, st, std::chrono::microseconds(us), [] { return false; })) stop = true;
                });
            }
            const auto t0 = std::chrono::steady_clock::now();
            const auto r = solver.solve(position, wld, &stop);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (clock.joinable()) { clock.request_stop(); clock.join(); }
            total_nodes += r.nodes; total_time += sec; solved += r.complete;
            const int nb_free = std::popcount(position.free_squares());
            const string value = r.complete ? std::to_string(r.value) : "-";
            cout << std::format("{:>4} {:>4} {:>4} {:>6} {:>6} {:>12} {:>9.3f} {:>10.0f}\n", i, position.nb_plies(), nb_free,
                                value, r.complete ? move_str(r.move) : "-", r.nodes, sec,
                                sec > 0 ? r.nodes / sec / 1000 : 0.0) << std::flush;
            if (csv.is_open()) {
                csv << i << ',' << position.nb_plies() << ',' << nb_free << ',' << r.complete << ',' << r.value << ','
                    << move_str(r.move) << ',' << r.nodes << ',' << sec << '\n' << std::flush;
            }
        }
        cout << std::format("total: {} positions, {} solved, {} nodes, {:.2f} s, {:.0f} knodes/s\n", positions.size(), solved,
                            total_nodes, total_time, total_nodes / std::max(total_time, 1e-9) / 1000);
        return 0;
    }

    int blunders(const Args& a) {
        Searcher s(a.get("player"));
        auto positions = select_positions(read_positions(a.get("positions", "bench_positions.txt")), a);
        const uint64_t us = std::stoull(a.get("time", "200000"));
        EndgameSolver solver;
        const auto sign = [](int v) { return (v > 0) - (v < 0); };
        int checked = 0, nb_blunders = 0;
        std::map<int, std::pair<int, int>> by_free;      // free → (checked, blunders)
        for (const auto& [i, position] : positions) {
            std::atomic_bool stop = false;
            std::jthread clock([&](std::stop_token st) {
                std::mutex mtx; std::condition_variable_any cv;
                std::unique_lock lock(mtx);
                if (!cv.wait_for(lock, st, std::chrono::seconds(30), [] { return false; })) stop = true;
            });
            solver.clear();
            const auto best = solver.solve(position, true, &stop);
            if (!best.complete) continue;
            s.clear();
            const Move m = s.search(position, 63, us).move;
            Yolah after = position;
            after.play(m);
            const auto reply = solver.solve(after, true, &stop);
            if (!reply.complete) continue;
            // the opponent's result after m, seen from the side to move
            const int got = after.game_over() ? sign(position.score(position.current_player()) + (m != Move::none()))
                                              : -sign(reply.value);
            const bool blunder = got < sign(best.value);
            const int nb_free = std::popcount(position.free_squares());
            checked++; nb_blunders += blunder;
            by_free[nb_free].first++; by_free[nb_free].second += blunder;
            if (blunder) cout << std::format("pos {:>4} (ply {}, {} free): best {:+d}, played {} → {:+d}\n",
                                             i, position.nb_plies(), nb_free, sign(best.value), move_str(m), got);
        }
        for (auto [f, cb] : by_free) cout << std::format("{:>3} free: {} blunders / {}\n", f, cb.second, cb.first);
        cout << std::format("total: {} blunders / {} positions\n", nb_blunders, checked);
        return 0;
    }

    struct Row { int idx, ply, depth, value; string move; uint64_t nodes; double seconds; };

    std::map<int, Row> read_csv(const string& path) {
        std::ifstream f(path);
        if (!f) { cerr << "cannot open " << path << '\n'; std::exit(1); }
        std::map<int, Row> res;
        string line;
        std::getline(f, line);   // header
        while (std::getline(f, line)) {
            std::istringstream is(line);
            Row r; string tok;
            std::getline(is, tok, ','); r.idx = std::stoi(tok);
            std::getline(is, tok, ','); r.ply = std::stoi(tok);
            std::getline(is, tok, ','); r.depth = std::stoi(tok);
            std::getline(is, tok, ','); r.value = std::stoi(tok);
            std::getline(is, r.move, ',');
            std::getline(is, tok, ','); r.nodes = std::stoull(tok);
            std::getline(is, tok, ','); r.seconds = std::stod(tok);
            res[r.idx] = r;
        }
        return res;
    }

    int run(const Args& a) {
        Searcher s(a.get("player"));
        auto positions = select_positions(read_positions(a.get("positions", "bench_positions.txt")), a);
        if (a.has("limit")) positions.resize(std::min(positions.size(), size_t(std::stoul(a.get("limit")))));
        const uint8_t depth = uint8_t(std::stoi(a.get("depth", "63")));
        const uint64_t us = std::stoull(a.get("time", "0"));
        if (!a.has("depth") && !a.has("time")) { cerr << "--depth or --time expected\n"; return 1; }
        std::map<int, Row> ref;
        if (a.has("compare")) ref = read_csv(a.get("compare"));
        std::ofstream csv;
        if (a.has("csv")) {
            csv.open(a.get("csv"));
            csv << "idx,ply,depth,value,move,nodes,seconds\n";
        }
        uint64_t total_nodes = 0, ref_nodes = 0;
        double total_time = 0, ref_time = 0, total_depth = 0, ref_depth = 0;
        int same_value = 0, same_move = 0, compared = 0;
        cout << std::format("{:>4} {:>4} {:>5} {:>7} {:>6} {:>12} {:>9} {:>10}{}\n", "pos", "ply", "depth", "value", "move",
                            "nodes", "seconds", "knodes/s", ref.empty() ? "" : "   ref nodes  ref s  same");
        for (const auto& [i, position] : positions) {
            s.clear();
            auto r = s.search(position, depth, us);
            const string mv = move_str(r.move);
            total_nodes += r.nb_nodes; total_time += r.seconds; total_depth += r.depth;
            if (csv.is_open()) {
                csv << i << ',' << position.nb_plies() << ',' << int(r.depth) << ',' << r.value << ','
                    << mv << ',' << r.nb_nodes << ',' << r.seconds << '\n' << std::flush;
            }
            string cmp;
            if (auto it = ref.find(int(i)); it != ref.end()) {
                const Row& q = it->second;
                compared++;
                ref_nodes += q.nodes; ref_time += q.seconds; ref_depth += q.depth;
                const bool sv = q.depth == r.depth && q.value == r.value, sm = q.depth == r.depth && q.move == mv;
                same_value += sv; same_move += sm;
                cmp = std::format(" {:>12} {:>6.2f}  {}{}", q.nodes, q.seconds, sv ? "v" : "-", sm ? "m" : "-");
            }
            cout << std::format("{:>4} {:>4} {:>5} {:>7} {:>6} {:>12} {:>9.3f} {:>10.0f}{}\n", i, position.nb_plies(),
                                int(r.depth), r.value, mv, r.nb_nodes, r.seconds,
                                r.seconds > 0 ? r.nb_nodes / r.seconds / 1000 : 0.0, cmp) << std::flush;
        }
        const size_t n = positions.size();
        if (s.dev) {
            const auto [evals, hits] = s.dev->eval_stats();
            if (evals) cout << std::format("leaf evaluations: {}, from the evaluation cache: {} ({:.1f}%)\n",
                                           evals, hits, 100.0 * hits / evals);
        }
        cout << std::format("total: {} positions, {} nodes, {:.2f} s, {:.0f} knodes/s, mean depth {:.2f}\n",
                            n, total_nodes, total_time, total_nodes / std::max(total_time, 1e-9) / 1000, total_depth / n);
        if (compared) {
            cout << std::format("reference: {} nodes, {:.2f} s, mean depth {:.2f}  →  nodes ×{:.3f}, time ×{:.3f}; "
                                "same value {}/{}, same move {}/{}\n",
                                ref_nodes, ref_time, ref_depth / compared,
                                double(total_nodes) / std::max<uint64_t>(ref_nodes, 1), total_time / std::max(ref_time, 1e-9),
                                same_value, compared, same_move, compared);
        }
        return 0;
    }
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    if (argc < 2) {
        cerr << "usage: search_bench positions|run [--options] (see test/search_bench_main.cpp)\n";
        return 1;
    }
    const string mode = argv[1];
    Args a = parse(argc, argv, 2);
    if (mode == "positions") return make_positions(a);
    if (mode == "run") return run(a);
    if (mode == "solve") return solve(a);
    if (mode == "blunders") return blunders(a);
    cerr << "unknown mode " << mode << '\n';
    return 1;
}
