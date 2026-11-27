// puzzle_gen_main.cpp — candidate puzzles from real games, for the app's editor
// (a human then keeps the good ones).
//
//   ./puzzle_gen GAMES... [--out DIR] [--goals points,collect,first] [--free 10-20]
//                [--positions 600] [--tries 6] [--n 20] [--threads T] [--nodes 30000000]
//                [--min-rank 2] [--seed 1]
//
// GAMES: game files of data/generate_games (or directories of them). From each of
// --positions games picked at random, one position with a number of free squares
// in --free, the side to move being the puzzle's player. Every legal move is
// solved EXACTLY (player/endgame_solver.h, the app's options: no pass rule,
// private regions), which gives each move's value for the goal:
//   points   the final score difference (the scores so far become the bonuses,
//            the smaller one set to 0);
//   collect  targets taken, player − opponent (--tries random sets of 2 to 5
//            target squares per position);
//   first    +1 the player lands on a target first, −1 the opponent, 0 nobody
//            (--tries random sets of 1 to 3 targets; none reachable at once).
// A candidate keeps the position only if exactly ONE move reaches the best value
// V ≥ 1 (the puzzle: "win by V", "take V more targets", "be first on a target").
//
// What makes it interesting, measured:
//   rank      the key move's rank for a naive player (one ply, mobility; for
//             the targets: taking one now, threatening one next): 1 = the
//             obvious move, larger = a surprise; every move ranked before it fails
//             (the traps). Positions below --min-rank are dropped.
//   precise   along the main line (key move, the solver's best reply, ...), the
//             player's turns (up to 4) where the key move is again the only one
//             to keep V: the puzzle asks for that many exact moves.
//   in        first: the fewest player moves to force a landing (a small search
//             up to 5 moves; 0 = not found within 5).
//   score     3·precise + min(rank − 1, 8) (+ 2·in for first): the sort key.
// The best --n of each goal are written to DIR/gen_<goal>_<k>.json, the format of
// app/assets/puzzles (opponent: the exact solver; the text lists the metrics and
// the key move, to be rewritten), and a summary line each on stdout.
#include "game.h"
#include "endgame_solver.h"
#include "magic.h"
#include "zobrist.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::string, std::vector;
namespace fs = std::filesystem;

enum Goal { POINTS, COLLECT, FIRST };
static const char* GOAL_NAME[] = {"points", "collect", "first"};
static const char* GOAL_FR[] = {"points", "collecte", "premi\xC3\xA8re cible"};   // in the puzzle's name

struct Candidate {
    Yolah pos;          // the puzzle's position: the bonuses as its scores
    int goal = POINTS;
    uint64_t targets = 0;
    int value = 0;      // V: the best value, reached by the key move only
    int second = 0;     // the best value of the other moves
    Move key = Move::none();
    int nb_moves = 0, rank = 0, precise = 0, in = 0, score = 0;
    size_t game = 0;    // the game it comes from (one puzzle per game and goal)
};

static string sq_name(int sq) { return {char('a' + sq % 8), char('1' + sq / 8)}; }
static string move_name(Move m) { return m == Move::none() ? "passe" : sq_name(m.from_sq()) + "-" + sq_name(m.to_sq()); }
static bool on(uint64_t bb, int sq) { return bb >> sq & 1; }
static bool hits(Move m, uint64_t targets) { return m != Move::none() && on(targets, m.to_sq()); }

static int nb_moves_of(const Yolah& y, uint8_t player) {
    Yolah::MoveList l;
    y.moves(player, l);
    return l.size() == 1 && l[0] == Move::none() ? 0 : int(l.size());
}
static bool can_hit(const Yolah& y, uint8_t player, uint64_t targets) {
    Yolah::MoveList l;
    y.moves(player, l);
    for (const Move m : l)
        if (hits(m, targets)) return true;
    return false;
}

// Every move of the side to move and its exact value for the goal, from the side
// to move's view; acc = what it has already gained (collect: targets taken,
// its own − the opponent's). ok = false: a solve hit the node budget.
struct MoveValues {
    vector<Move> moves, replies;
    vector<int> values;
    bool ok = true;
};
static MoveValues move_values(EndgameSolver& s, const Yolah& y, int goal, uint64_t targets, int acc) {
    MoveValues r;
    Yolah::MoveList l;
    y.moves(l);
    for (const Move m : l) {
        const bool hit = goal != POINTS && hits(m, targets);
        r.moves.push_back(m);
        if (goal == FIRST && hit) { r.values.push_back(1); r.replies.push_back(Move::none()); continue; }
        Yolah c = y;
        c.play(m);
        const int gained = acc + (goal == COLLECT && hit);
        if (c.game_over()) {
            const auto [b, w] = c.score();
            const int me = y.current_player();
            r.values.push_back(goal == POINTS ? (me == Yolah::BLACK ? b - w : w - b) : goal == COLLECT ? gained : 0);
            r.replies.push_back(Move::none());
            continue;
        }
        const int current = goal == COLLECT ? -gained : 0;   // for the opponent, who moves in c
        const EndgameSolver::Result res = goal == POINTS ? s.solve(c, false) : s.solve(c, false, nullptr, &current);
        r.ok &= res.complete;
        r.values.push_back(-res.value);
        r.replies.push_back(res.move);
    }
    return r;
}

// first: can the side to move force a landing on a target within d of its moves?
static bool first_within(const Yolah& y, uint64_t targets, int d, uint64_t& budget) {
    if (budget == 0 || y.game_over()) return false;
    budget--;
    const uint8_t me = y.current_player(), opp = Yolah::other_player(me);
    Yolah::MoveList l;
    y.moves(l);
    for (const Move m : l)
        if (hits(m, targets)) return true;
    if (d <= 1) return false;
    for (const Move m : l) {
        Yolah c = y;
        c.play(m);
        if (c.game_over()) continue;
        if (can_hit(c, opp, targets)) continue;               // the opponent lands first
        Yolah::MoveList rl;
        c.moves(rl);
        bool all = true;
        for (const Move rm : rl) {
            Yolah g = c;
            g.play(rm);
            if (!first_within(g, targets, d - 1, budget)) { all = false; break; }
        }
        if (all) return true;
    }
    return false;
}

// The naive player's preference for a move (larger = played first).
static int naive(const Yolah& y, Move m, int goal, uint64_t targets) {
    const uint8_t me = y.current_player(), opp = Yolah::other_player(me);
    Yolah c = y;
    c.play(m);
    int h = nb_moves_of(c, me) - nb_moves_of(c, opp);
    if (goal == COLLECT) h += 100 * hits(m, targets);
    if (goal == FIRST) h += 100 * can_hit(c, me, targets) - 100 * can_hit(c, opp, targets);
    return h;
}

// The candidate of position y for a goal and its targets, if it has exactly one key move.
static bool examine(EndgameSolver& s, const Yolah& y, int goal, uint64_t targets, int min_rank, Candidate& out) {
    const MoveValues mv = move_values(s, y, goal, targets, 0);
    if (!mv.ok || mv.moves.size() < 2) return false;
    const int n = int(mv.moves.size());
    const int best = *std::max_element(mv.values.begin(), mv.values.end());
    if (best < 1 || std::count(mv.values.begin(), mv.values.end(), best) != 1) return false;
    const int k = int(std::find(mv.values.begin(), mv.values.end(), best) - mv.values.begin());
    int second = -1000;
    for (int i = 0; i < n; i++)
        if (i != k) second = std::max(second, mv.values[i]);
    // The naive rank: 1 + the moves it prefers to the key move (ties count against it).
    const int hk = naive(y, mv.moves[k], goal, targets);
    int rank = 1;
    for (int i = 0; i < n; i++)
        if (i != k && naive(y, mv.moves[i], goal, targets) >= hk) rank++;
    if (rank < min_rank) return false;
    // Precise moves along the main line.
    int precise = 1;
    {
        const uint8_t me = y.current_player();
        Yolah p = y;
        int acc = 0;
        Move m = mv.moves[k], reply = mv.replies[k];
        for (int turn = 1; turn < 4; turn++) {
            if (goal == FIRST && hits(m, targets)) break;
            acc += goal == COLLECT && hits(m, targets);
            p.play(m);
            if (p.game_over()) break;
            if (goal == FIRST && hits(reply, targets)) break;
            acc -= goal == COLLECT && hits(reply, targets);
            p.play(reply);
            if (p.game_over() || p.current_player() != me) break;
            const MoveValues next = move_values(s, p, goal, targets, acc);
            if (!next.ok || next.moves.size() < 2) break;
            int keep = 0, idx = 0;
            for (size_t i = 0; i < next.moves.size(); i++)
                if (next.values[i] >= best) keep++, idx = int(i);
            if (keep != 1) break;
            precise++;
            m = next.moves[idx];
            reply = next.replies[idx];
        }
    }
    int in = 0;
    if (goal == FIRST)
        for (int d = 1; d <= 5 && !in; d++) {
            uint64_t budget = 20'000'000;
            if (first_within(y, targets, d, budget)) in = d;
        }
    out.pos = y;
    out.goal = goal;
    out.targets = targets;
    out.value = best;
    out.second = second;
    out.key = mv.moves[k];
    out.nb_moves = n;
    out.rank = rank;
    out.precise = precise;
    out.in = in;
    out.score = 3 * precise + std::min(rank - 1, 8) + (goal == FIRST ? 2 * in : 0);
    return true;
}

static string json_escape(const string& s) {
    string o;
    for (char c : s) o += c == '"' ? string("\\\"") : c == '\\' ? string("\\\\") : string(1, c);
    return o;
}

static void write_puzzle(const Candidate& c, const string& path, int k) {
    const Yolah& y = c.pos;
    const auto [b, w] = y.score();
    const int me = y.current_player();
    const string side = me == Yolah::BLACK ? "black" : "white";
    std::ostringstream text;
    text << "Coup cl\xC3\xA9 : " << move_name(c.key) << " \xC2\xB7 rang na\xC3\xAF" "f " << c.rank << "/" << c.nb_moves
         << " \xC2\xB7 " << c.precise << " coup(s) pr\xC3\xA9" "cis";
    if (c.goal == FIRST) text << " \xC2\xB7 " << (c.in ? "cible forc\xC3\xA9" "e en " + std::to_string(c.in) + " coup(s)" : string("cible forc\xC3\xA9" "e en plus de 5 coups"));
    else text << " \xC2\xB7 meilleur autre coup : " << c.second;
    std::ofstream f(path);
    f << "{\n  \"name\": \"G\xC3\xA9n\xC3\xA9r\xC3\xA9 " << GOAL_FR[c.goal] << " " << k << "\",\n";
    f << "  \"text\": \"" << json_escape(text.str()) << "\",\n  \"board\": [\n";
    for (int r = 7; r >= 0; r--) {
        f << "    \"";
        for (int file = 0; file < 8; file++) {
            const int sq = r * 8 + file;
            f << (on(y.bitboard(Yolah::BLACK), sq) ? 'B' : on(y.bitboard(Yolah::WHITE), sq) ? 'W' : on(y.empty_bitboard(), sq) ? '#' : '.');
        }
        f << "\"" << (r ? "," : "") << "\n";
    }
    f << "  ],\n  \"to move\": \"" << side << "\",\n  \"player\": \"" << side << "\",\n  \"opponent\": \"solver\",\n";
    f << "  \"margin\": " << c.value << ",\n  \"bonus black\": " << b << ",\n  \"bonus white\": " << w << ",\n";
    f << "  \"goal\": \"" << GOAL_NAME[c.goal] << "\",\n  \"targets\": [";
    bool first = true;
    for (int sq = 0; sq < 64; sq++)
        if (on(c.targets, sq)) f << (first ? "" : ", ") << "\"" << sq_name(sq) << "\"", first = false;
    f << "],\n  \"time index\": 1,\n  \"end on win\": -1,\n  \"end on loss\": -1,\n"
         "  \"stop when blocked\": true,\n  \"stop when ai blocked\": true,\n  \"goal text\": \"\",\n  \"hide goal\": false,\n"
         "  \"theme\": \"\",\n  \"hole style\": \"pit\",\n  \"sprite black\": \"\",\n  \"sprite white\": \"\",\n"
         "  \"background\": \"\",\n  \"tiles\": {},\n  \"decor\": []\n}\n";
}

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    vector<string> inputs;
    string out_dir = "app/assets/puzzles";
    vector<int> goals = {POINTS, COLLECT, FIRST};
    int free_lo = 10, free_hi = 20, tries = 6, n_out = 20, min_rank = 2;
    size_t nb_positions = 600;
    uint64_t node_limit = 30'000'000, seed = 1;
    int threads = std::max(1u, std::thread::hardware_concurrency() - 2);
    for (int i = 1; i < argc; i++) {
        const string a = argv[i];
        auto val = [&] { if (i + 1 >= argc) { std::cerr << a << ": value missing\n"; std::exit(1); } return string(argv[++i]); };
        if (a == "--out") out_dir = val();
        else if (a == "--goals") {
            goals.clear();
            std::stringstream ss(val());
            for (string t; std::getline(ss, t, ',');)
                for (int g = 0; g < 3; g++)
                    if (t == GOAL_NAME[g]) goals.push_back(g);
        } else if (a == "--free") { const string v = val(); const auto d = v.find('-'); free_lo = std::stoi(v.substr(0, d)); free_hi = std::stoi(v.substr(d + 1)); }
        else if (a == "--positions") nb_positions = std::stoul(val());
        else if (a == "--tries") tries = std::stoi(val());
        else if (a == "--n") n_out = std::stoi(val());
        else if (a == "--threads") threads = std::stoi(val());
        else if (a == "--nodes") node_limit = std::stoull(val());
        else if (a == "--min-rank") min_rank = std::stoi(val());
        else if (a == "--seed") seed = std::stoull(val());
        else if (a.rfind("--", 0) == 0) { std::cerr << "unknown option " << a << "\n"; return 1; }
        else inputs.push_back(a);
    }
    if (inputs.empty()) {
        std::cerr << "usage: puzzle_gen GAMES... [--out DIR] [--goals points,collect,first] [--free 10-20] [--positions N]\n"
                     "       [--tries T] [--n N] [--threads T] [--nodes N] [--min-rank R] [--seed S]\n";
        return 1;
    }
    // The games (record: nb moves, nb random moves, (from, to) per move, black score, white score).
    vector<string> files;
    for (const string& in : inputs)
        if (fs::is_directory(in)) { for (const auto& e : fs::directory_iterator(in)) if (e.is_regular_file()) files.push_back(e.path()); }
        else files.push_back(in);
    std::sort(files.begin(), files.end());
    std::mt19937_64 rng(seed);
    std::shuffle(files.begin(), files.end(), rng);
    vector<std::pair<size_t, Yolah>> jobs;   // (game number, position)
    size_t game_no = 0;
    for (const string& path : files) {
        std::ifstream f(path, std::ios::binary);
        vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (size_t i = 0; i + 1 < data.size() && jobs.size() < nb_positions;) {
            const int nb = data[i];
            if (i + 2 * nb + 4 > data.size()) break;
            // The plies whose position has a number of free squares in range.
            vector<Yolah> cands;
            Yolah y;
            for (int k = 0; k < nb && !y.game_over(); k++) {
                const int fr = std::popcount(y.free_squares());
                if (fr >= free_lo && fr <= free_hi) cands.push_back(y);
                y.play(Move(Square(data[i + 2 + 2 * k]), Square(data[i + 3 + 2 * k])));
            }
            if (!cands.empty()) {
                Yolah p = cands[rng() % cands.size()];
                const auto [b, w] = p.score();
                const uint16_t m = std::min(b, w);
                p.set_state(p.bitboard(Yolah::BLACK), p.bitboard(Yolah::WHITE), p.empty_bitboard(), uint16_t(b - m), uint16_t(w - m), p.nb_plies());
                jobs.push_back({game_no, p});
            }
            game_no++;
            i += 2 * nb + 4;
        }
        if (jobs.size() >= nb_positions) break;
    }
    std::printf("%zu positions (%d to %d free squares) from %zu games\n", jobs.size(), free_lo, free_hi, game_no);

    vector<vector<Candidate>> found(3);
    std::mutex mu;
    std::atomic<size_t> next{0}, done{0};
    auto worker = [&](int t) {
        std::mt19937_64 r(seed * 1000003 + t);
        for (size_t j; (j = next++) < jobs.size();) {
            const auto& [g, y] = jobs[j];
            for (int goal : goals) {
                const int nb_tries = goal == POINTS ? 1 : tries;
                for (int tr = 0; tr < nb_tries; tr++) {
                    uint64_t targets = 0;
                    if (goal != POINTS) {
                        vector<int> free;
                        for (int sq = 0; sq < 64; sq++)
                            if (on(y.free_squares(), sq)) free.push_back(sq);
                        std::shuffle(free.begin(), free.end(), r);
                        const int k = goal == COLLECT ? 2 + int(r() % 4) : 1 + int(r() % 3);
                        for (int i = 0; i < k && i < int(free.size()); i++) targets |= uint64_t(1) << free[i];
                        // first: no target reachable at once, by either side
                        if (goal == FIRST && (can_hit(y, y.current_player(), targets) || can_hit(y, Yolah::other_player(y.current_player()), targets))) continue;
                    }
                    EndgameSolver::Options opt;
                    opt.pass_rule = false;
                    opt.regions = true;
                    opt.tt_bits = 20;
                    opt.goal = goal == COLLECT ? EndgameGoal::Collect : goal == FIRST ? EndgameGoal::First : EndgameGoal::Points;
                    opt.targets = targets;
                    EndgameSolver s(opt);
                    s.set_node_limit(node_limit);
                    Candidate c;
                    if (examine(s, y, goal, targets, min_rank, c)) {
                        c.game = g;
                        std::lock_guard lk(mu);
                        found[goal].push_back(c);
                        break;   // one candidate per position and goal
                    }
                }
            }
            const size_t d = ++done;
            if (d % 50 == 0) {
                std::lock_guard lk(mu);
                std::printf("  %zu / %zu positions: %zu points, %zu collect, %zu first\n", d, jobs.size(), found[0].size(), found[1].size(), found[2].size());
                std::fflush(stdout);
            }
        }
    };
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
    for (auto& t : pool) t.join();

    fs::create_directories(out_dir);
    for (int goal : goals) {
        auto& v = found[goal];
        std::stable_sort(v.begin(), v.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        std::printf("%s: %zu candidates, best %d written\n", GOAL_NAME[goal], v.size(), std::min<int>(n_out, int(v.size())));
        for (int k = 0; k < n_out && k < int(v.size()); k++) {
            const Candidate& c = v[k];
            const string path = out_dir + "/gen_" + GOAL_NAME[goal] + "_" + (k < 9 ? "0" : "") + std::to_string(k + 1) + ".json";
            write_puzzle(c, path, k + 1);
            std::printf("  %s  score %2d  V %d (other %d)  key %s  rank %d/%d  precise %d%s  %d free\n", fs::path(path).filename().c_str(),
                        c.score, c.value, c.second, move_name(c.key).c_str(), c.rank, c.nb_moves, c.precise,
                        goal == FIRST ? (" in " + std::to_string(c.in)).c_str() : "", std::popcount(c.pos.free_squares()));
        }
    }
}
