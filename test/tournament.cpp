#include "tournament.h"
#include <iostream>
#include <iomanip>
#include "misc.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <execution>
#include <filesystem>
#include <format>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include "player.h"
#include <fstream>
#include <random>

namespace test {
    void tournament(const std::vector<std::string>& players_configs, size_t nb_random_moves, size_t nb_games) {
        using namespace std;
        auto first_n_moves_random = [](Yolah& yolah, uint64_t seed, size_t n) {
            PRNG prng(seed);
            Yolah::MoveList moves;
            size_t i = 0;
            while (!yolah.game_over()) {
                yolah.moves(moves);
                Move m = moves[prng.rand<size_t>() % moves.size()];
                yolah.play(m);
                if (++i >= n) break;
            }
        };
        vector<unique_ptr<Player>> players;
        for (const string& cfg : players_configs) {
            players.push_back(Player::create(nlohmann::json::parse(ifstream(cfg))));
        }
        std::mutex mutex;
        {
            vector<jthread> threads;
            for (size_t p1 = 0; p1 < players.size(); p1++) {
                for (size_t p2 = p1 + 1; p2 < players.size(); p2++) {
                    threads.emplace_back([&, p1, p2]{
                        double p1_black_victories = 0;
                        double p1_white_victories = 0;
                        double p2_black_victories = 0;
                        double p2_white_victories = 0;
                        double draws = 0;
                        vector<json> configs{players[p1]->config(), players[p2]->config()};
                        for (size_t side = 0; side < 2; side++) {
                            for (size_t j = 0; j < nb_games; j++) {
                                auto black = Player::create(configs[0]);
                                auto white = Player::create(configs[1]);
                                Yolah yolah;
                                if (nb_random_moves) {
                                    first_n_moves_random(yolah, j, nb_random_moves);
                                }
                                while (!yolah.game_over()) {
                                    Move m = (yolah.current_player() == Yolah::BLACK ? black : white)->play(yolah);
                                    yolah.play(m);
                                }
                                black->game_over(yolah);
                                white->game_over(yolah);
                                const auto [black_score, white_score] = yolah.score();
                                if (black_score > white_score) {
                                    (side == 0 ? p1_black_victories : p2_black_victories) += 1;
                                } else if (white_score > black_score) {
                                    (side == 0 ? p2_white_victories : p1_white_victories) += 1;
                                } else {
                                    draws += 1;
                                }
                            }
                            swap(configs[0], configs[1]);
                        }
                        {
                            size_t n = nb_games * 2;
                            std::lock_guard lock(mutex);
                            cout << "player 1:\n";
                            cout << players[p1]->info() << '\n';
                            cout << "player 2:\n";
                            cout << players[p2]->info() << '\n';
                            cout << "[ player 1 % of black victories ]: " << (p1_black_victories / n * 100) << '\n';
                            cout << "[ player 2 % of black victories ]: " << (p2_black_victories / n * 100) << '\n';
                            cout << "[ player 1 % of white victories ]: " << (p1_white_victories / n * 100) << '\n';
                            cout << "[ player 2 % of white victories ]: " << (p2_white_victories / n * 100) << '\n';
                            cout << "[          % of draws           ]: " << (draws / n * 100) << '\n';
                            cout << "[   player 1 % of victories     ]: " << ((p1_black_victories + p1_white_victories + draws / 2) / n * 100) << '\n';
                            cout << "[   player 2 % of victories     ]: " << ((p2_black_victories + p2_white_victories + draws / 2) / n * 100) << '\n';
                        }
                    });
                }
            }
        }
    }

    // ── The long, resumable round robin ─────────────────────────────────────
    //
    // Rounds: in each round every pair of players plays one random opening of
    // `opening_plies` plies twice, once with each colour — the opening's bias
    // cancels within the pair, and deterministic players do not replay the
    // same game. The openings depend only on (seed, round, pair), so a
    // resumed tournament plays exactly the games that are missing.
    //
    // Fairness: every player gets the same time per move and the same number
    // of search threads (overridden in its config). Players that use the GPU
    // (AlphaZeroMCTSPlayer with "backend": "torch") only play in games of the
    // GPU queue, `gpu_parallel` at a time (1 by default): two instances
    // sharing the GPU would each get less than their time's worth.
    //
    // Results: one CSV line per finished game, appended and flushed at once,
    //   round,pair,game,black,white,black_score,white_score,result,plies,seconds,opening
    // with result 1 (black wins), 0.5 (draw) or 0 (white wins). The file is
    // the resume state; tournament_elo.py turns it into ratings.
    namespace {
        std::atomic<bool> stop_requested{false};
        void on_signal(int) { stop_requested.store(true); }

        struct Game {
            size_t round, pair, index;      // index 0: first player black, 1: colours swapped
            size_t black, white;            // indices in the player list
            std::vector<Move> opening;
            bool gpu;
        };

        std::string player_name(const std::string& config_path) {
            return std::filesystem::path(config_path).stem().string();
        }

        bool uses_gpu(const json& cfg) {
            return cfg.value("name", std::string()) == "AlphaZeroMCTSPlayer" &&
                   cfg.value("backend", std::string("cpu")) == "torch" &&
                   cfg.value("device", std::string("cuda")) != "cpu";
        }

        std::vector<Move> random_opening(uint64_t seed, size_t plies) {
            std::mt19937_64 rng(seed);
            for (;;) {
                Yolah y;
                std::vector<Move> moves;
                for (size_t p = 0; p < plies && !y.game_over(); p++) {
                    Yolah::MoveList ml;
                    y.moves(ml);
                    const Move m = ml[rng() % ml.size()];
                    moves.push_back(m);
                    y.play(m);
                }
                if (!y.game_over()) return moves;
            }
        }

        std::string moves_string(const std::vector<Move>& moves) {
            std::ostringstream os;
            for (size_t i = 0; i < moves.size(); i++) os << (i ? " " : "") << moves[i];
            return os.str();
        }
    }

    std::vector<std::string> read_players_list(const std::string& path) {
        const std::filesystem::path list = path;
        std::ifstream in(list);
        if (!in) throw std::invalid_argument("cannot open " + path);
        std::vector<std::string> configs;
        for (std::string line; std::getline(in, line); ) {
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t\r") + 1);
            if (line.empty() || line[0] == '#') continue;
            std::filesystem::path cfg = line;
            if (cfg.is_relative()) cfg = list.parent_path() / cfg;
            configs.push_back(cfg.string());
        }
        return configs;
    }

    void tournament(const TournamentOptions& opt) {
        using namespace std;
        // The players print their searches on cout (depth, nodes, PV...): over
        // two weeks that would be gigabytes. cout is muted for the whole
        // tournament; its own messages go to `log`, on the real stdout.
        struct NullBuffer : streambuf { int overflow(int c) override { return c; } } null_buffer;
        streambuf* const stdout_buffer = cout.rdbuf(&null_buffer);
        struct Restore { streambuf* b; ~Restore() { cout.rdbuf(b); } } restore{stdout_buffer};
        ostream log(stdout_buffer);
        const size_t n = opt.configs.size();
        if (n < 2) throw invalid_argument("tournament: at least two players are needed");

        // Configs, with the common time control and thread count.
        vector<json> cfgs;
        vector<string> names;
        for (const string& path : opt.configs) {
            ifstream in(path);
            if (!in) throw invalid_argument("tournament: cannot open " + path);
            json j = json::parse(in);
            if (opt.microseconds && j.contains("microseconds")) {
                j["microseconds"] = opt.microseconds;
                if (j.contains("nb simulations")) j["nb simulations"] = 0;   // time-limited search
            }
            if (opt.threads && j.contains("nb threads")) j["nb threads"] = opt.threads;
            if (j.contains("verbose")) j["verbose"] = false;
            Player::create(j);                  // fail now, not in the middle of the night
            cfgs.push_back(j);
            names.push_back(player_name(path));
        }
        if (set<string>(names.begin(), names.end()).size() != names.size())
            throw invalid_argument("tournament: two config files have the same name");

        // Games already played (resume).
        set<tuple<size_t, size_t, size_t>> done;          // (round, pair, index)
        {
            ifstream in(opt.results);
            string line;
            while (getline(in, line)) {
                if (line.empty() || line.starts_with("round")) continue;
                size_t r, p, g;
                char c1, c2;
                istringstream is(line);
                if (is >> r >> c1 >> p >> c2 >> g) done.insert({r, p, g});
            }
        }
        const bool new_file = !filesystem::exists(opt.results);
        ofstream results(opt.results, ios::app);
        if (new_file) results << "round,pair,game,black,white,black_score,white_score,result,plies,seconds,opening\n" << flush;

        vector<pair<size_t, size_t>> pairs;
        for (size_t a = 0; a < n; a++)
            for (size_t b = a + 1; b < n; b++) pairs.emplace_back(a, b);

        log << format("tournament: {} players, {} pairs, {} games per round, {} µs per move, {} threads per player, "
                       "{} CPU + {} GPU games at a time, {} games already in {}\n",
                       n, pairs.size(), 2 * pairs.size(), opt.microseconds, opt.threads, opt.parallel,
                       opt.gpu_parallel, done.size(), opt.results) << flush;

        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        const auto start = chrono::steady_clock::now();
        auto out_of_time = [&] {
            return opt.hours > 0 && chrono::duration<double>(chrono::steady_clock::now() - start).count() > opt.hours * 3600;
        };

        // Running totals per player, for the periodic summary.
        vector<double> points(n, 0), games(n, 0);
        mutex mtx;
        size_t played = 0;

        auto play_game = [&](const Game& g) {
            auto black = Player::create(cfgs[g.black]);
            auto white = Player::create(cfgs[g.white]);
            Yolah y;
            for (Move m : g.opening) y.play(m);
            const auto t0 = chrono::steady_clock::now();
            while (!y.game_over()) {
                if (stop_requested.load()) return;
                const Move m = (y.current_player() == Yolah::BLACK ? black : white)->play(y);
                y.play(m);
            }
            black->game_over(y);
            white->game_over(y);
            const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            const auto [bs, ws] = y.score();
            const double result = bs > ws ? 1.0 : (bs < ws ? 0.0 : 0.5);
            lock_guard lock(mtx);
            results << format("{},{},{},{},{},{},{},{},{},{:.1f},{}\n", g.round, g.pair, g.index, names[g.black],
                              names[g.white], bs, ws, result, y.nb_plies(), secs, moves_string(g.opening)) << flush;
            points[g.black] += result;       games[g.black] += 1;
            points[g.white] += 1 - result;   games[g.white] += 1;
            if (++played % 50 == 0) {
                vector<size_t> order(n);
                for (size_t i = 0; i < n; i++) order[i] = i;
                sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return points[a] / max(1.0, games[a]) > points[b] / max(1.0, games[b]);
                });
                log << format("── {} games played in this run ──\n", played);
                for (size_t i : order)
                    log << format("  {:40s} {:5.1f}%  ({} games)\n", names[i], 100 * points[i] / max(1.0, games[i]),
                                   size_t(games[i]));
                log << flush;
            }
        };

        // Two independent queues, each advancing round after round at its own
        // pace: the CPU games (every pair without a GPU player) and the GPU
        // games (the pairs with one). A round of the GPU queue — the GPU
        // player against everybody, `gpu_parallel` games at a time — can take
        // much longer than a round of the CPU queue on a node with many cores:
        // with a common barrier per round, most of the cores would wait.
        // The consequence: pairs of the CPU queue get more rounds than the GPU
        // player's pairs, which the ratings handle (they use every game).
        struct Queue {
            bool gpu;
            size_t round = 0;               // round being dealt
            deque<Game> games;              // what is left of it (shuffled)
        };
        Queue queues[2] = {{false}, {true}};
        // Next game of a queue (under mtx), dealing the next rounds as needed.
        auto next_game = [&](Queue& q, Game& out) -> bool {
            while (q.games.empty()) {
                if (opt.rounds && q.round >= opt.rounds) return false;
                const size_t round = q.round++;
                vector<Game> all;
                for (size_t p = 0; p < pairs.size(); p++) {
                    const auto [a, b] = pairs[p];
                    if ((uses_gpu(cfgs[a]) || uses_gpu(cfgs[b])) != q.gpu) continue;
                    const auto opening = random_opening(opt.seed * 1000003 + round * 7919 + p, opt.opening_plies);
                    for (size_t idx = 0; idx < 2; idx++) {
                        if (done.count({round, p, idx})) continue;
                        all.push_back({round, p, idx, idx == 0 ? a : b, idx == 0 ? b : a, opening, q.gpu});
                    }
                }
                // Shuffled, so that a stopped round is not biased towards the first pairs.
                shuffle(all.begin(), all.end(), mt19937_64(opt.seed + round * 2 + q.gpu));
                q.games.assign(all.begin(), all.end());
                if (!all.empty())
                    log << format("{} queue: round {}, {} games\n", q.gpu ? "GPU" : "CPU", round, all.size()) << flush;
                // A queue with no pair at all (no GPU player): nothing to deal, ever.
                bool any = false;
                for (auto [a, b] : pairs) any |= (uses_gpu(cfgs[a]) || uses_gpu(cfgs[b])) == q.gpu;
                if (!any) return false;
            }
            out = q.games.front();
            q.games.pop_front();
            return true;
        };
        auto worker = [&](Queue& q) {
            for (;;) {
                if (stop_requested.load() || out_of_time()) return;
                Game g;
                {
                    lock_guard lock(mtx);
                    if (!next_game(q, g)) return;
                }
                try {
                    play_game(g);
                } catch (const exception& e) {
                    lock_guard lock(mtx);
                    cerr << format("game {}/{}/{} ({} vs {}) failed: {}\n", g.round, g.pair, g.index,
                                   names[g.black], names[g.white], e.what());
                }
            }
        };
        {
            vector<jthread> threads;
            for (size_t t = 0; t < opt.parallel; t++) threads.emplace_back(worker, ref(queues[0]));
            for (size_t t = 0; t < opt.gpu_parallel; t++) threads.emplace_back(worker, ref(queues[1]));
        }
        log << format("tournament stopped: {} games played in this run; results in {}\n", played, opt.results) << flush;
    }
}
