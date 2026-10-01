// search_bench — measures the minimax search on a fixed set of positions.
//
//   search_bench positions --out bench_positions.txt [--games 25] [--every 6] [--seed 1]
//       Plays games (4 random plies, then the reference player at depth 5) and
//       keeps every `every`-th position: realistic positions, one JSON per line.
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
        vector<Yolah> positions = read_positions(a.get("positions", "bench_positions.txt"));
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
        for (size_t i = 0; i < positions.size(); i++) {
            s.clear();
            auto r = s.search(positions[i], depth, us);
            const string mv = move_str(r.move);
            total_nodes += r.nb_nodes; total_time += r.seconds; total_depth += r.depth;
            if (csv.is_open()) {
                csv << i << ',' << positions[i].nb_plies() << ',' << int(r.depth) << ',' << r.value << ','
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
            cout << std::format("{:>4} {:>4} {:>5} {:>7} {:>6} {:>12} {:>9.3f} {:>10.0f}{}\n", i, positions[i].nb_plies(),
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
    cerr << "unknown mode " << mode << '\n';
    return 1;
}
