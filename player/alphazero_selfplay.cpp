#include "alphazero_selfplay.h"
#include "alphazero_mcts_player.h"
#include "batched_evaluator.h"
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <mutex>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using std::string;

namespace az {

bool read_latest_model(const string& work_dir, ModelRef& out) {
    std::ifstream in(fs::path(work_dir) / "latest.json");
    if (!in) return false;
    try {
        const json j = json::parse(in);
        ModelRef r;
        r.step = j.at("step").get<uint64_t>();
        fs::path p = j.at("ts").get<string>();
        if (p.is_relative()) p = fs::path(work_dir) / p;
        r.path = p.string();
        r.aux = j.value("aux", false);
        out = r;
        return true;
    } catch (const std::exception&) {
        return false;   // being rewritten (the trainer renames atomically, but be safe)
    }
}

namespace {

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

string default_tag() {
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    return std::format("{}_{}", host, getpid());
}

// Collects finished games and writes them in files of ~flush_samples samples.
// Files are written under a temporary name and renamed, so the trainer never
// sees a partial file. Names start with the wall-clock time in milliseconds:
// sorting them by name is sorting them by age, across hosts and restarts.
//
// Rows are raw bytes of one fixed size per file: TrainingSample (version 1)
// or TrainingSampleAux (version 2). A file never mixes the two: when the
// format of the incoming games changes (the trainer switched --aux), what is
// pending is written first.
class SampleWriter {
public:
    SampleWriter(fs::path dir, string tag, size_t flush_samples, double flush_seconds)
        : dir(std::move(dir)), tag(std::move(tag)), flush_samples(flush_samples),
          flush_seconds(flush_seconds), last_flush(now_seconds()) {
        fs::create_directories(this->dir);
    }

    template <class Row>
    void add_game(const std::vector<Row>& game, uint32_t version) {
        std::lock_guard lock(mutex);
        if (nb_pending && (version != pending_version || sizeof(Row) != row_size)) write_locked();
        pending_version = version;
        row_size = sizeof(Row);
        const char* bytes = reinterpret_cast<const char*>(game.data());
        pending.insert(pending.end(), bytes, bytes + game.size() * sizeof(Row));
        nb_pending += game.size();
        ++pending_games;
        if (nb_pending >= flush_samples) write_locked();
    }

    void flush_if_old() {
        std::lock_guard lock(mutex);
        if (nb_pending && now_seconds() - last_flush >= flush_seconds) write_locked();
    }

    void flush() {
        std::lock_guard lock(mutex);
        if (nb_pending) write_locked();
    }

private:
    void write_locked() {
        using namespace std::chrono;
        const auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        const string name = std::format("{:013d}_{}_{:05d}.bin", ms, tag, seq++);
        const fs::path tmp = dir / ("." + name + ".tmp");
        {
            std::ofstream out(tmp, std::ios::binary);
            SampleFileHeader h{};
            std::memcpy(h.magic, "YOLAHSP1", 8);
            h.version = pending_version;
            h.sample_size = static_cast<uint32_t>(row_size);
            h.nb_samples = static_cast<uint32_t>(nb_pending);
            h.nb_games = static_cast<uint32_t>(pending_games);
            out.write(reinterpret_cast<const char*>(&h), sizeof(h));
            out.write(pending.data(), static_cast<std::streamsize>(pending.size()));
            if (!out) {
                std::cerr << "selfplay: cannot write " << tmp << ", keeping the samples in memory\n";
                return;
            }
        }
        fs::rename(tmp, dir / name);
        pending.clear();
        nb_pending = 0;
        pending_games = 0;
        last_flush = now_seconds();
    }

    const fs::path dir;
    const string tag;
    const size_t flush_samples;
    const double flush_seconds;
    std::mutex mutex;
    std::vector<char> pending;           // raw rows
    size_t nb_pending = 0;               // rows in `pending`
    size_t row_size = sizeof(TrainingSample);
    uint32_t pending_version = 1;
    size_t pending_games = 0;
    size_t seq = 0;
    double last_flush;
};

TrainingSample make_sample(const Yolah& y, const SearchResult& r, uint32_t model_step) {
    TrainingSample s{};
    s.black = y.bitboard(Yolah::BLACK);
    s.white = y.bitboard(Yolah::WHITE);
    s.empty = y.empty_bitboard();
    s.turn = y.current_player();
    s.root_q = r.root_value;
    s.model_step = model_step;
    const size_t n = std::min<size_t>(r.children.size(), Yolah::MAX_NB_MOVES);
    s.nb_moves = static_cast<uint8_t>(n);
    for (size_t i = 0; i < n; i++) {
        s.action[i] = static_cast<uint16_t>(nn::action_index(r.children[i].move));
        const float p = std::clamp(r.policy[i], 0.0f, 1.0f);
        s.prob[i] = static_cast<uint16_t>(std::lround(p * 65535.0f));
    }
    return s;
}

// The future ownership map of one sample, from `leaver[q]`: the colour that
// left square q during the game (0xFF = nobody). Squares that are already
// holes in the sample's position were left BEFORE it — by whom is not visible
// in the position, so they carry no target (OWN_PAST, masked in the loss).
// Every other square that someone leaves later was left after the sample:
// the map counts exactly the moves still to be played, so
//     #OWN_MINE − #OWN_OPP = (final margin) − (margin at the sample),
// the rest of the score, for the side to move.
TrainingSampleAux with_ownership(const TrainingSample& s, const uint8_t leaver[64]) {
    TrainingSampleAux a{};
    a.base = s;
    for (int q = 0; q < 64; q++) {
        uint8_t code;
        if ((s.empty >> q) & 1)      code = OWN_PAST;
        else if (leaver[q] == 0xFF)  code = OWN_NONE;
        else                         code = leaver[q] == s.turn ? OWN_MINE : OWN_OPP;
        a.own[q / 4] |= static_cast<uint8_t>(code << (2 * (q % 4)));
    }
    return a;
}

} // namespace

void run_selfplay(const SelfPlayOptions& opt, const std::atomic<bool>& stop) {
    ModelRef model;
    if (!read_latest_model(opt.work_dir, model))
        throw std::runtime_error("selfplay: no readable " + opt.work_dir + "/latest.json");

    // One backend for all the games, behind the batching wrapper.
    json cfg = opt.player;
    cfg["weights"] = model.path;
    SearchParams params = AlphaZeroMCTSPlayer::search_params(cfg);
    params.nb_threads = 1;                      // parallelism comes from the games
    std::unique_ptr<nn::Evaluator> backend = nn::make_evaluator(cfg);
    if (params.batch_size == 0) params.batch_size = 8;
    const size_t nb_games = std::max<size_t>(1, opt.nb_games);
    const size_t merged = cfg.value("merged batch size", nb_games * params.batch_size);
    nn::BatchingEvaluator evaluator(std::move(backend), nb_games, merged);

    SampleWriter writer(fs::path(opt.work_dir) / "selfplay",
                        opt.tag.empty() ? default_tag() : opt.tag,
                        opt.flush_samples, opt.flush_seconds);

    std::atomic<uint64_t> generation{0};         // bumped at every weight swap
    std::atomic<uint32_t> model_step{static_cast<uint32_t>(model.step)};
    std::atomic<uint64_t> games_started{0}, games_done{0}, samples_done{0}, plies_done{0};
    std::atomic<uint64_t> evaluations{0}, black_wins{0}, white_wins{0};
    std::atomic<size_t>   active{nb_games};
    // The games stop on `halt`: set by the caller's `stop` or by staleness.
    std::atomic<bool>     halt{false};
    // Record the auxiliary targets (latest.json "aux"; re-read at every poll).
    std::atomic<bool>     record_aux{model.aux};

    std::cout << std::format("selfplay: {} concurrent games, merged batch {}, {} sims ({} fast, p={}), model step {}\n"
                             "selfplay: network {}\n"
                             "selfplay: auxiliary targets (ownership) {}\n",
                             nb_games, merged, params.nb_simulations, params.nb_simulations_fast,
                             params.playout_cap_fast_prob, model.step, evaluator.info(),
                             model.aux ? "on" : "off") << std::flush;

    auto game_loop = [&] {
        std::pmr::synchronized_pool_resource memory;
        {
            Search search(evaluator, params, &memory);
            uint64_t seen = generation.load();
            while (!halt.load()) {
                if (opt.max_games && games_started.fetch_add(1) >= opt.max_games) break;
                Yolah y;
                std::vector<TrainingSample> samples;
                // Who left each square during this game (auxiliary targets).
                uint8_t leaver[64];
                std::fill(std::begin(leaver), std::end(leaver), uint8_t(0xFF));
                search.reset();
                while (!y.game_over() && !halt.load()) {
                    // New weights: the graph's priors/values and the cached
                    // evaluations belong to the old network.
                    const uint64_t g = generation.load(std::memory_order_acquire);
                    if (g != seen) {
                        search.reset();
                        search.clear_cache();
                        seen = g;
                    }
                    const SearchResult r = search.search(y);
                    evaluations.fetch_add(r.nb_evaluations, std::memory_order_relaxed);
                    if (r.full_search && !r.children.empty())
                        samples.push_back(make_sample(y, r, model_step.load()));
                    if (r.best_move != Move::none())   // a pass leaves no hole
                        leaver[static_cast<int>(r.best_move.from_sq())] = y.current_player();
                    y.play(r.best_move);
                }
                if (!y.game_over()) break;           // stopped mid-game: drop it
                // Outcome for the side to move at each recorded position.
                const auto [bs, ws] = y.score();
                const int black_diff = int(bs) - int(ws);
                for (TrainingSample& s : samples) {
                    const int d = s.turn == Yolah::BLACK ? black_diff : -black_diff;
                    s.z = static_cast<int8_t>((d > 0) - (d < 0));
                }
                if (black_diff > 0) black_wins.fetch_add(1);
                if (black_diff < 0) white_wins.fetch_add(1);
                plies_done.fetch_add(y.nb_plies());
                samples_done.fetch_add(samples.size());
                games_done.fetch_add(1);
                if (record_aux.load()) {
                    std::vector<TrainingSampleAux> rows;
                    rows.reserve(samples.size());
                    for (const TrainingSample& s : samples) rows.push_back(with_ownership(s, leaver));
                    writer.add_game(rows, 2);
                } else {
                    writer.add_game(samples, 1);
                }
            }
        }
        // Fewer clients: the batcher must stop waiting for this one.
        evaluator.set_nb_clients(active.fetch_sub(1) - 1);
    };

    std::vector<std::jthread> threads;
    for (size_t i = 0; i < nb_games; i++) threads.emplace_back(game_loop);

    // Supervisor: hot-swap the network, flush old samples, report.
    const double start = now_seconds();
    double last_poll = start, last_report = start;
    uint64_t last_games = 0, last_samples = 0, last_evals = 0;
    double last_model_change = start;
    while (!halt.load() && active.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double t = now_seconds();
        if (stop.load()) halt.store(true);
        if (opt.max_stale_hours > 0 && t - last_model_change > opt.max_stale_hours * 3600) {
            std::cout << std::format("selfplay: no new network for {:.2f} h, stopping\n", opt.max_stale_hours) << std::flush;
            halt.store(true);
        }
        if (t - last_poll >= opt.poll_seconds) {
            last_poll = t;
            ModelRef latest;
            const bool readable = read_latest_model(opt.work_dir, latest);
            if (readable && latest.aux != record_aux.load()) {
                record_aux.store(latest.aux);
                std::cout << std::format("selfplay: auxiliary targets (ownership) {}\n", latest.aux ? "on" : "off") << std::flush;
            }
            if (readable && latest.step != model.step) {
                try {
                    evaluator.reload(latest.path);
                    model = latest;
                    last_model_change = t;
                    model_step.store(static_cast<uint32_t>(model.step));
                    generation.fetch_add(1, std::memory_order_release);
                    std::cout << std::format("selfplay: now playing with step {} ({})\n", model.step, model.path) << std::flush;
                } catch (const std::exception& e) {
                    // A half-written file or a transient FS error: retry at the next poll.
                    std::cerr << "selfplay: reload of " << latest.path << " failed: " << e.what() << "\n";
                }
            }
            writer.flush_if_old();
        }
        if (t - last_report >= opt.report_seconds) {
            const uint64_t g = games_done.load(), s = samples_done.load(), e = evaluations.load();
            const double dt = t - last_report;
            std::cout << std::format("selfplay: {:.1f}h  games {} (+{:.2f}/s)  samples {} (+{:.1f}/s)  "
                                     "evals/s {:.0f}  avg plies {:.1f}  black wins {:.1f}%  white wins {:.1f}%\n",
                                     (t - start) / 3600, g, (g - last_games) / dt, s, (s - last_samples) / dt,
                                     (e - last_evals) / dt, g ? double(plies_done.load()) / g : 0.0,
                                     g ? 100.0 * black_wins.load() / g : 0.0, g ? 100.0 * white_wins.load() / g : 0.0)
                      << std::flush;
            last_report = t;
            last_games = g; last_samples = s; last_evals = e;
        }
    }
    threads.clear();   // join (the game threads watch `halt` / max_games)
    writer.flush();
    std::cout << std::format("selfplay: done, {} games, {} samples\n", games_done.load(), samples_done.load()) << std::flush;
}

} // namespace az
