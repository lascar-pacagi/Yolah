// alphazero_learn_main.cpp — the C++ side of the AlphaZero learning loop
// (driven by nnue/alphazero_learn.py; see player/README_ALPHAZERO_LEARN.md).
//
//   alphazero_learn selfplay --config ../config/alphazero_mcts_selfplay_player.cfg --work <dir>
//         [--games 128] [--max-games 0] [--flush-samples 2000] [--flush-seconds 600]
//         [--poll 30] [--report 300] [--max-stale-hours 0] [--tag name] [--set 'key=value' ...]
//     Plays self-play games with the network named in <dir>/latest.json (hot-
//     swapped when it changes) and writes training samples to <dir>/selfplay/.
//     Stops cleanly on SIGINT/SIGTERM (finished games are written).
//
//   alphazero_learn match --config ../config/alphazero_mcts_eval_player.cfg
//         --a new.ts --b old.ts [--games 10] [--time 2000000] [--opening-plies 4]
//         [--seed 1] [--out result.json] [--set 'key=value' ...]
//     Plays A against B, alternating colours, and reports A's result. Games
//     2k and 2k+1 start from the same random opening of --opening-plies plies
//     (A black in one, white in the other): with deterministic players and no
//     opening, the 5 games of each colour would be the same game 5 times.
//
// --set overrides any key of the player JSON config ('key=value', value parsed as JSON).
#include "alphazero_mcts_player.h"
#include "alphazero_selfplay.h"
#include "magic.h"
#include "zobrist.h"
#include <atomic>
#include <csignal>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using std::string;

namespace {

std::atomic<bool> stop_requested{false};
void on_signal(int) { stop_requested.store(true); }

struct Args {
    std::vector<string> argv;
    size_t i = 0;
    bool more() const { return i < argv.size(); }
    string next(const string& opt) {
        if (i >= argv.size()) { std::cerr << "missing value for " << opt << "\n"; std::exit(2); }
        return argv[i++];
    }
};

void apply_set(json& cfg, const string& kv) {
    const size_t eq = kv.find('=');
    if (eq == string::npos) { std::cerr << "--set expects key=value\n"; std::exit(2); }
    cfg[kv.substr(0, eq)] = json::parse(kv.substr(eq + 1));
}

json load_config(const string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    return json::parse(in);
}

int selfplay_main(Args& a) {
    az::SelfPlayOptions opt;
    string config;
    std::vector<string> sets;
    while (a.more()) {
        const string o = a.next("");
        if (o == "--config") config = a.next(o);
        else if (o == "--work") opt.work_dir = a.next(o);
        else if (o == "--games") opt.nb_games = std::stoul(a.next(o));
        else if (o == "--max-games") opt.max_games = std::stoull(a.next(o));
        else if (o == "--flush-samples") opt.flush_samples = std::stoul(a.next(o));
        else if (o == "--flush-seconds") opt.flush_seconds = std::stod(a.next(o));
        else if (o == "--poll") opt.poll_seconds = std::stod(a.next(o));
        else if (o == "--report") opt.report_seconds = std::stod(a.next(o));
        else if (o == "--max-stale-hours") opt.max_stale_hours = std::stod(a.next(o));
        else if (o == "--tag") opt.tag = a.next(o);
        else if (o == "--set") sets.push_back(a.next(o));
        else { std::cerr << "unknown option " << o << "\n"; return 2; }
    }
    if (config.empty() || opt.work_dir.empty()) { std::cerr << "selfplay needs --config and --work\n"; return 2; }
    opt.player = load_config(config);
    for (const string& kv : sets) apply_set(opt.player, kv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    az::run_selfplay(opt, stop_requested);
    return 0;
}

string move_list(const std::vector<Move>& moves) {
    std::ostringstream os;
    for (size_t i = 0; i < moves.size(); i++) os << (i ? " " : "") << moves[i];
    return os.str();
}

int match_main(Args& a) {
    string config, model_a, model_b, out;
    size_t games = 10, opening_plies = 4;
    uint64_t time_us = 2'000'000, seed = 1;
    std::vector<string> sets;
    while (a.more()) {
        const string o = a.next("");
        if (o == "--config") config = a.next(o);
        else if (o == "--a") model_a = a.next(o);
        else if (o == "--b") model_b = a.next(o);
        else if (o == "--games") games = std::stoul(a.next(o));
        else if (o == "--time") time_us = std::stoull(a.next(o));
        else if (o == "--opening-plies") opening_plies = std::stoul(a.next(o));
        else if (o == "--seed") seed = std::stoull(a.next(o));
        else if (o == "--out") out = a.next(o);
        else if (o == "--set") sets.push_back(a.next(o));
        else { std::cerr << "unknown option " << o << "\n"; return 2; }
    }
    if (config.empty() || model_a.empty() || model_b.empty()) {
        std::cerr << "match needs --config, --a and --b\n";
        return 2;
    }
    json cfg = load_config(config);
    for (const string& kv : sets) apply_set(cfg, kv);
    cfg["microseconds"] = time_us;
    cfg["nb simulations"] = 0;
    json cfg_a = cfg, cfg_b = cfg;
    cfg_a["weights"] = model_a;
    cfg_b["weights"] = model_b;
    AlphaZeroMCTSPlayer player_a(cfg_a), player_b(cfg_b);

    std::mt19937_64 rng(seed);
    int wins = 0, draws = 0, losses = 0;
    json records = json::array();
    std::vector<Move> opening;
    for (size_t g = 0; g < games; g++) {
        const bool a_black = (g % 2 == 0);
        if (g % 2 == 0) {
            // A fresh random opening for the pair (not a finished game).
            for (;;) {
                Yolah y;
                opening.clear();
                for (size_t p = 0; p < opening_plies && !y.game_over(); p++) {
                    Yolah::MoveList moves;
                    y.moves(moves);
                    const Move m = moves[rng() % moves.size()];
                    opening.push_back(m);
                    y.play(m);
                }
                if (!y.game_over()) break;
            }
        }
        Yolah y;
        for (Move m : opening) y.play(m);
        std::vector<Move> played;
        while (!y.game_over()) {
            const bool black_to_move = y.current_player() == Yolah::BLACK;
            AlphaZeroMCTSPlayer& p = (black_to_move == a_black) ? player_a : player_b;
            const Move m = p.play(y);
            played.push_back(m);
            y.play(m);
        }
        player_a.game_over(y);
        player_b.game_over(y);
        const auto [bs, ws] = y.score();
        const int a_diff = (a_black ? 1 : -1) * (int(bs) - int(ws));
        const char* result = a_diff > 0 ? "win" : (a_diff < 0 ? "loss" : "draw");
        (a_diff > 0 ? wins : (a_diff < 0 ? losses : draws))++;
        records.push_back({{"a_color", a_black ? "black" : "white"},
                           {"opening", move_list(opening)},
                           {"moves", move_list(played)},
                           {"black_score", bs}, {"white_score", ws},
                           {"plies", y.nb_plies()}, {"result_for_a", result}});
        std::cout << std::format("game {}/{}: A {} score {}-{} ({} plies) -> A {}  [A {}W {}D {}L]\n",
                                 g + 1, games, a_black ? "black" : "white", bs, ws, y.nb_plies(),
                                 result, wins, draws, losses) << std::flush;
    }
    const double score = games ? (wins + 0.5 * draws) / games : 0.0;
    json res = {{"a", model_a}, {"b", model_b}, {"games", games}, {"time_us", time_us},
                {"opening_plies", opening_plies}, {"seed", seed},
                {"wins", wins}, {"draws", draws}, {"losses", losses}, {"score", score},
                {"records", records}};
    if (!out.empty()) {
        std::ofstream f(out);
        f << res.dump(2) << "\n";
    }
    std::cout << std::format("match: A {}W {}D {}L, score {:.3f}\n", wins, draws, losses, score) << std::flush;
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    if (argc < 2) {
        std::cerr << "usage: alphazero_learn selfplay|match [options]  (see alphazero_learn_main.cpp)\n";
        return 2;
    }
    Args a;
    for (int i = 2; i < argc; i++) a.argv.emplace_back(argv[i]);
    const string cmd = argv[1];
    try {
        if (cmd == "selfplay") return selfplay_main(a);
        if (cmd == "match") return match_main(a);
    } catch (const std::exception& e) {
        std::cerr << "alphazero_learn " << cmd << ": " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
}
