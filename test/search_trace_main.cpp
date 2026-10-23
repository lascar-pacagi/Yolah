// search_trace — records every step of the dev player's search, for the search
// debugger (ui/search_debugger.html: the code, the tree, the iterative
// deepening, the table, the network's values, the history, with breakpoints).
//
//   make search_trace
//   ./search_trace [--player ../config/search_study/nnue_k8_them_spacelite.cfg]
//                  (--positions ../test/bench_positions.txt --index K | --position JSON | --moves "a1:b2 h8:g7 ...")
//                  [--depth 7] [--warm N --warm-time US] [--child-evals-depth 1] [--max-mb 256]
//                  [--check 1] --out trace.ytrace
//
// The search is the player's own code (player/minmax_nnue_dev_player.cpp,
// compiled with its trace hooks, YOLAH_SEARCH_TRACE): iterative deepening up
// to --depth, one thread (lazy SMP threads would make every run different),
// no clock (the hooks slow the search down: a time limit would stop it
// elsewhere than in a game).
//
// --warm N: first, N untraced searches of --warm-time µs each, as in a game:
// the player's earlier moves (the positions of the --moves game where the
// same side is to move, or the N previous positions of --positions with the
// same side to move). The traced search then starts with the table, history
// and countermoves those searches left, as it would in a game. Without
// --warm: empty table and history (as search_bench).
//
// --check 1: the same search again on a fresh player without the recorder:
// same move, value, depth and nodes (the recording changes nothing).
//
// File: "YTRC", uint32 version, uint32 length of a JSON header, the header
// (position, configuration, constants, event names, the dev player's source),
// zero padding to a multiple of 4, then the events (int32, see search_trace.h).
#include "magic.h"
#include "zobrist.h"
#include "player.h"
#include "minmax_nnue_dev_player.h"
#include "search_trace.h"
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using std::cout, std::cerr, std::string, std::vector;

extern const char search_trace_source[];   // the dev player's source (generated, see CMakeLists.txt)

namespace {
    std::map<string, string> parse(int argc, char* argv[]) {
        std::map<string, string> kv;
        for (int i = 1; i < argc; i++) {
            string k = argv[i];
            if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
                cerr << "bad argument: " << k << '\n';
                std::exit(1);
            }
            kv[k.substr(2)] = argv[++i];
        }
        return kv;
    }

    string get(const std::map<string, string>& kv, const string& k, const string& def = "") {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }

    std::unique_ptr<Player> make_player(const string& cfg, json& j) {
        std::ifstream f(cfg);
        if (!f) { cerr << "cannot open " << cfg << '\n'; std::exit(1); }
        j = json::parse(f);
        j["verbose"] = false;
        j["nb threads"] = 1;
        auto p = Player::create(j);
        if (!dynamic_cast<MinMaxNNUE_DevPlayer*>(p.get())) {
            cerr << cfg << ": not a MinMaxNNUE_DevPlayer\n";
            std::exit(1);
        }
        return p;
    }

    string move_str(Move m) {
        std::ostringstream os;
        os << m;
        return os.str();
    }

    Move parse_move(const string& s) {
        auto sq = [&](const string& t) {
            if (t.size() != 2 || t[0] < 'a' || t[0] > 'h' || t[1] < '1' || t[1] > '8') {
                cerr << "bad square in move " << s << '\n';
                std::exit(1);
            }
            return Square((t[1] - '1') * 8 + (t[0] - 'a'));
        };
        const auto colon = s.find(':');
        if (colon == string::npos) { cerr << "bad move " << s << " (expected a1:b2)\n"; std::exit(1); }
        return Move(sq(s.substr(0, colon)), sq(s.substr(colon + 1)));
    }
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    const auto kv = parse(argc, argv);
    const string cfg = get(kv, "player", "../config/search_study/nnue_k8_them_spacelite.cfg");
    const string out = get(kv, "out", "trace.ytrace");
    const int depth = std::stoi(get(kv, "depth", "7"));
    const int warm = std::stoi(get(kv, "warm", "0"));
    const uint64_t warm_time = std::stoull(get(kv, "warm-time", "200000"));

    // The position, and the earlier positions of the same side for --warm.
    Yolah root;
    vector<Yolah> earlier;            // in game order
    string origin;
    if (kv.contains("moves")) {
        std::istringstream is(get(kv, "moves"));
        string t;
        vector<Yolah> line{root};
        while (is >> t) {
            const Move m = t == "pass" ? Move::none() : parse_move(t);
            Yolah::MoveList moves;
            root.moves(moves);
            bool legal = false;
            for (Move x : moves) legal |= x == m;
            if (!legal) { cerr << "illegal move " << t << " at ply " << root.nb_plies() << '\n'; return 1; }
            root.play(m);
            line.push_back(root);
        }
        for (size_t i = 0; i + 1 < line.size(); i++)
            if (line[i].current_player() == root.current_player()) earlier.push_back(line[i]);
        origin = "moves: " + get(kv, "moves");
    } else if (kv.contains("positions")) {
        std::ifstream f(get(kv, "positions"));
        if (!f) { cerr << "cannot open " << get(kv, "positions") << '\n'; return 1; }
        vector<Yolah> all;
        string l;
        while (std::getline(f, l)) if (!l.empty()) all.push_back(Yolah::from_json(l));
        const size_t index = std::stoul(get(kv, "index", "0"));
        if (index >= all.size()) { cerr << "--index beyond the " << all.size() << " positions\n"; return 1; }
        root = all[index];
        for (size_t i = 0; i < index; i++)
            if (all[i].current_player() == root.current_player()) earlier.push_back(all[i]);
        origin = get(kv, "positions") + " #" + std::to_string(index);
    } else if (kv.contains("position")) {
        root = Yolah::from_json(get(kv, "position"));
        origin = "position";
    }
    if (root.game_over()) { cerr << "the game is over in this position\n"; return 1; }
    if (int(earlier.size()) > warm) earlier.erase(earlier.begin(), earlier.end() - warm);
    if (warm > 0 && earlier.empty()) cerr << "--warm: no earlier position of the same side, not warmed\n";

    auto run = [&](bool record, SearchTracer* tracer, json& j) {
        auto p = make_player(cfg, j);
        auto* dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(p.get());
        for (const Yolah& y : earlier) dev->search(y, 63, warm_time);
        if (record) {
            tracer->stop = dev->stop_flag();
            dev->tracer = tracer;
        }
        const auto r = dev->search(root, uint8_t(depth), 0);
        dev->tracer = nullptr;
        return std::make_pair(std::move(p), r);
    };

    SearchTracer tracer;
    tracer.max_words = size_t(std::stoull(get(kv, "max-mb", "256"))) * 1024 * 1024 / 4;
    tracer.child_evals_depth = std::stoi(get(kv, "child-evals-depth", "1"));
    json j;
    const auto t0 = std::chrono::steady_clock::now();
    auto [player, r] = run(true, &tracer, j);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cout << origin << "\n" << root << "\n";
    cout << "depth " << int(r.depth) << ", value " << r.value << ", move " << move_str(r.move) << ", " << r.nb_nodes
         << " nodes, " << tracer.words.size() * 4 / 1e6 << " MB of events" << (tracer.full ? " (FULL: search stopped)" : "")
         << ", " << secs << " s\n";

    bool check_ok = true;
    if (get(kv, "check", "0") == "1") {
        json j2;
        auto [p2, r2] = run(false, nullptr, j2);
        check_ok = r2.move == r.move && r2.value == r.value && r2.depth == r.depth && r2.nb_nodes == r.nb_nodes;
        cout << "check without the recorder: depth " << int(r2.depth) << ", value " << r2.value << ", move "
             << move_str(r2.move) << ", " << r2.nb_nodes << " nodes → " << (check_ok ? "IDENTICAL" : "DIFFERENT") << '\n';
    }

    json h;
    h["version"] = 1;
    h["origin"] = origin;
    h["position"] = json::parse(root.to_json());
    h["config file"] = cfg;
    h["config"] = player->config();
    h["depth"] = depth;
    h["warm"] = int(earlier.size());
    h["warm time"] = warm_time;
    h["child evals depth"] = tracer.child_evals_depth;
    h["result"] = {{"move", r.move.raw()}, {"value", r.value}, {"depth", r.depth}, {"nodes", r.nb_nodes}};
    h["full"] = tracer.full;
    h["constants"] = {{"WIN", int(heuristic::MAX_VALUE)}, {"INFINITE", 32000}, {"SCORE_TT", 1 << 30},
                      {"SCORE_KILLER1", 1 << 29}, {"SCORE_KILLER2", (1 << 29) - 1}, {"SCORE_COUNTER", (1 << 29) - 2},
                      {"SCORE_ARTICULATION", 1 << 28}, {"HISTORY_MAX", 16384}};
    // The Zobrist keys as (low 32 bits, high 32 bits): the page hashes any
    // position to look it up in the table (and checks its hashes against the trace's).
    auto halves = [](uint64_t k) { return json::array({uint32_t(k), uint32_t(k >> 32)}); };
    json psq = json::array();
    for (int c = 0; c < 4; c++) {
        json row = json::array();
        for (int sq = 0; sq < SQUARE_NB; sq++) row.push_back(halves(zobrist::psq[c][sq]));
        psq.push_back(row);
    }
    h["zobrist"] = {{"psq", psq}, {"side", halves(zobrist::side)}};
    vector<string> names;
    for (int e = 0; e < SearchTracer::NB_EVENTS; e++) names.push_back(SearchTracer::EVENT_NAMES[e]);
    h["events"] = names;
    h["source name"] = "player/minmax_nnue_dev_player.cpp";
    h["source"] = string(search_trace_source);
    const string hs = h.dump();
    std::ofstream f(out, std::ios::binary);
    if (!f) { cerr << "cannot write " << out << '\n'; return 1; }
    const uint32_t version = 1, len = uint32_t(hs.size());
    f.write("YTRC", 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&len), 4);
    f.write(hs.data(), hs.size());
    const char zeros[4]{};
    f.write(zeros, (4 - hs.size() % 4) % 4);
    f.write(reinterpret_cast<const char*>(tracer.words.data()), std::streamsize(tracer.words.size() * 4));
    cout << "wrote " << out << '\n';
    return check_ok ? 0 : 2;
}
