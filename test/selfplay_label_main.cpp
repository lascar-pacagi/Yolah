// selfplay_label — training data from games of the dev player (the distillation
// loop, nnue/distill_loop.sh).
//
//   selfplay_label --player CFG --out FILE --first-game G --games N
//                  [--threads 8] [--seed 1] [--book ../config/openings/balanced.txt]
//                  [--random-plies 2] [--max-nodes 650000] [--record-depth 10]
//                  [--tree-per-search 8] [--exact-free 22] [--exact-ms 300] [--tt 64]
//
// Plays games G .. G + N − 1 (an opening of the book, then --random-plies
// random plies; the game number seeds both), every move a search of the dev
// player limited to --max-nodes nodes (the same depth on any machine; about
// 0.5 s on an i7-12700K), one thread per game. Each game writes, in one block:
//
//   • every position where the player searched (kind 0): the search's value
//     and depth, the move played;
//   • up to --tree-per-search positions of each search's tree (kind 1): PV
//     nodes with an exact value searched to at least --record-depth plies,
//     drawn at random, each position once per game, with their best move;
//
// and for all of them the game's result z for the side to move, and with at
// most --exact-free free squares the exact result (win / draw / loss) by the
// endgame solver, given --exact-ms per position (EXACT_UNKNOWN beyond).
//
// The file is a flat array of Record (40 bytes). Interrupted, the program
// starts again where it stopped: the last game of the file, maybe written in
// part, is dropped and played again (same game number, same game).

#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include "player.h"
#include "minmax_nnue_dev_player.h"
#include "endgame_solver.h"
#include "json.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

using std::string, std::cerr;
using json = nlohmann::json;

namespace {

constexpr int8_t EXACT_UNKNOWN = -128;   // not solved (too many free squares, or out of time)

struct Record {
    uint64_t black, white, empty;   // empty: the holes
    uint16_t ply;                   // its parity: the side to move
    uint8_t  black_score, white_score;
    int16_t  value;                 // the search's value for the side to move (±30000 = tanh ±1; beyond: a proven result)
    uint16_t move;                  // kind 0: the move played; kind 1: the node's best move (Move::raw)
    uint8_t  depth;                 // depth of the search that gave the value
    uint8_t  kind;                  // 0 = a position of the game, 1 = a node of a search tree
    int8_t   z;                     // the game's result for the side to move: 1, 0, −1
    int8_t   exact;                 // the exact result for the side to move: 1, 0, −1, or EXACT_UNKNOWN
    uint32_t game;
};
static_assert(sizeof(Record) == 40);

std::map<string, string> parse(int argc, char* argv[]) {
    std::map<string, string> kv;
    for (int i = 1; i < argc; i++) {
        const string k = argv[i];
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
            cerr << "bad argument: " << k << '\n';
            std::exit(1);
        }
        kv[k.substr(2)] = argv[++i];
    }
    return kv;
}

Square sq(const string& s) { return make_square(File(s[0] - 'a'), Rank(s[1] - '1')); }

// One opening per line, moves "a1:b2", '#' starts a comment.
std::vector<std::vector<Move>> read_book(const string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open the book " + path);
    std::vector<std::vector<Move>> book;
    for (string line; std::getline(f, line);) {
        std::istringstream is(line.substr(0, line.find('#')));
        std::vector<Move> o;
        for (string t; is >> t;) {
            const auto colon = t.find(':');
            o.push_back(Move(sq(t.substr(0, colon)), sq(t.substr(colon + 1))));
        }
        if (!o.empty()) book.push_back(std::move(o));
    }
    if (book.empty()) throw std::runtime_error("no opening in " + path);
    return book;
}

Record record_of(const Yolah& y) {
    Record r{};
    r.black = y.bitboard(Yolah::BLACK);
    r.white = y.bitboard(Yolah::WHITE);
    r.empty = y.empty_bitboard();
    r.ply = y.nb_plies();
    const auto [bs, ws] = y.score();
    r.black_score = uint8_t(bs);
    r.white_score = uint8_t(ws);
    return r;
}

Yolah position_of(const Record& r) {
    Yolah y;
    y.set_state(r.black, r.white, r.empty, r.black_score, r.white_score, r.ply);
    return y;
}

int sign(int v) { return (v > 0) - (v < 0); }

// The exact result (win / draw / loss) for the side to move, or EXACT_UNKNOWN
// if the solver needs more than `ms` milliseconds.
int8_t solve(EndgameSolver& solver, const Yolah& y, int ms) {
    std::atomic_bool stop{false};
    std::mutex m;
    std::condition_variable cv;
    bool finished = false;
    std::thread dog([&] {
        std::unique_lock lk(m);
        if (!cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return finished; })) stop = true;
    });
    const EndgameSolver::Result r = solver.solve(y, true, &stop);
    { std::lock_guard lk(m); finished = true; }
    cv.notify_one();
    dog.join();
    return r.complete ? int8_t(sign(r.value)) : EXACT_UNKNOWN;
}

// Drops the last game of the file (perhaps written in part) and returns the
// number of the next game to play.
uint32_t resume(const string& path, uint32_t first_game) {
    if (!std::filesystem::exists(path)) return first_game;
    const auto size = std::filesystem::file_size(path);
    size_t n = size / sizeof(Record);
    if (n == 0) { std::filesystem::resize_file(path, 0); return first_game; }
    std::vector<Record> recs(n);
    {
        std::ifstream f(path, std::ios::binary);
        f.read((char*)recs.data(), n * sizeof(Record));
    }
    // Games are written whole, one after the other: the last one is the only
    // one that can be incomplete.
    const uint32_t last = recs.back().game;
    while (n > 0 && recs[n - 1].game == last) n--;
    std::filesystem::resize_file(path, n * sizeof(Record));
    return last;
}

}  // namespace

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    auto a = parse(argc, argv);
    auto get = [&](const string& k, const string& def) { return a.contains(k) ? a[k] : def; };
    if (!a.contains("player") || !a.contains("out") || !a.contains("games")) {
        cerr << "usage: selfplay_label --player CFG --out FILE --first-game G --games N [options] (see test/selfplay_label_main.cpp)\n";
        return 1;
    }
    const string out_path = a["out"];
    const uint32_t first_game = uint32_t(std::stoul(get("first-game", "0")));
    const uint32_t nb_games = uint32_t(std::stoul(a["games"]));
    const int threads = std::stoi(get("threads", "8"));
    const uint64_t seed = std::stoull(get("seed", "1"));
    const int random_plies = std::stoi(get("random-plies", "2"));
    const uint64_t max_nodes = std::stoull(get("max-nodes", "650000"));
    const int record_depth = std::stoi(get("record-depth", "10"));
    const size_t tree_per_search = std::stoul(get("tree-per-search", "8"));
    const int exact_free = std::stoi(get("exact-free", "22"));
    const int exact_ms = std::stoi(get("exact-ms", "300"));
    const std::vector<std::vector<Move>> book = a.contains("book") ? read_book(a["book"]) : std::vector<std::vector<Move>>{};

    json cfg = json::parse(std::ifstream(a["player"]));
    cfg["microseconds"] = 0;                 // no clock: the node limit stops each search
    cfg["nb threads"] = 1;
    cfg["max nodes"] = max_nodes;
    cfg["record depth"] = record_depth;
    cfg["tt size"] = std::stoul(get("tt", "64"));
    cfg["verbose"] = false;

    const uint32_t start = resume(out_path, first_game);
    const uint32_t end = first_game + nb_games;
    std::FILE* out = std::fopen(out_path.c_str(), "ab");
    if (!out) { cerr << "cannot open " << out_path << '\n'; return 1; }
    cerr << "games " << start << " .. " << end - 1 << " (" << end - start << " to play) -> " << out_path << '\n';

    std::atomic<uint32_t> next{start};
    std::atomic<uint64_t> nb_done{0}, nb_root{0}, nb_tree{0}, nb_exact{0}, nb_unknown{0};
    std::mutex out_mutex;
    const auto t0 = std::chrono::steady_clock::now();

    auto worker = [&] {
        auto player = Player::create(cfg);
        auto* dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(player.get());
        if (!dev) { cerr << "the player must be a MinMaxNNUE_DevPlayer\n"; std::exit(1); }
        std::vector<MinMaxNNUE_DevPlayer::NodeRecord> nodes;
        dev->recorder = &nodes;
        EndgameSolver solver;
        std::vector<Record> recs;
        for (uint32_t g; (g = next++) < end;) {
            std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL ^ (uint64_t(g) + 1) * 0xD1B54A32D192ED03ULL);
            // The opening: a book line, then random plies (again if the game ended).
            Yolah y;
            for (;;) {
                y = Yolah();
                if (!book.empty()) for (Move m : book[rng() % book.size()]) y.play(m);
                for (int i = 0; i < random_plies && !y.game_over(); i++) {
                    Yolah::MoveList moves;
                    y.moves(moves);
                    y.play(moves[rng() % moves.size()]);
                }
                if (!y.game_over()) break;
            }
            dev->clear_table();
            solver.clear();
            recs.clear();
            std::unordered_set<uint64_t> seen;     // the tree positions already taken in this game
            while (!y.game_over()) {
                nodes.clear();
                const auto r = dev->search(y, 63, 0);
                Record root = record_of(y);
                root.value = r.value;
                root.depth = r.depth;
                root.move = r.move.raw();
                root.kind = 0;
                recs.push_back(root);
                seen.insert(zobrist::hash(y));
                // A random sample of the search tree's exact PV nodes, new to this game.
                std::shuffle(nodes.begin(), nodes.end(), rng);
                size_t taken = 0;
                for (const auto& nd : nodes) {
                    if (taken == tree_per_search) break;
                    Record t{};
                    t.black = nd.black; t.white = nd.white; t.empty = nd.empty; t.ply = nd.ply;
                    t.black_score = nd.black_score; t.white_score = nd.white_score;
                    t.value = nd.value;
                    t.depth = nd.depth;
                    t.move = nd.move;
                    t.kind = 1;
                    const Yolah ty = position_of(t);
                    if (ty.game_over() || !seen.insert(zobrist::hash(ty)).second) continue;
                    recs.push_back(t);
                    taken++;
                }
                y.play(r.move);
            }
            // The result, for each record's side to move; the exact one near the end.
            const auto [bs, ws] = y.score();
            const int black_result = sign(int(bs) - int(ws));
            for (Record& rc : recs) {
                rc.game = g;
                rc.z = int8_t((rc.ply & 1) == Yolah::BLACK ? black_result : -black_result);
                const Yolah ry = position_of(rc);
                if (std::popcount(ry.free_squares()) <= exact_free) {
                    rc.exact = solve(solver, ry, exact_ms);
                    (rc.exact == EXACT_UNKNOWN ? nb_unknown : nb_exact)++;
                } else {
                    rc.exact = EXACT_UNKNOWN;
                }
                (rc.kind == 0 ? nb_root : nb_tree)++;
            }
            {
                std::lock_guard lk(out_mutex);
                std::fwrite(recs.data(), sizeof(Record), recs.size(), out);
                std::fflush(out);
            }
            const uint64_t d = ++nb_done;
            if (d % 100 == 0) {
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "%llu games in %.0f s (%.2f games/s): %llu game positions, %llu tree positions, %llu solved, %llu not solved in time\n",
                             (unsigned long long)d, s, d / s, (unsigned long long)nb_root.load(), (unsigned long long)nb_tree.load(),
                             (unsigned long long)nb_exact.load(), (unsigned long long)nb_unknown.load());
            }
        }
    };
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; t++) pool.emplace_back(worker);
    }
    std::fclose(out);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "done: %llu games in %.0f s, %llu game positions, %llu tree positions, %llu solved, %llu not solved in time\n",
                 (unsigned long long)nb_done.load(), s, (unsigned long long)nb_root.load(), (unsigned long long)nb_tree.load(),
                 (unsigned long long)nb_exact.load(), (unsigned long long)nb_unknown.load());
    return 0;
}
