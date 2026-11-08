// ════════════════════════════════════════════════════════════════════════════
//  YOLAH — the graphical application (SDL3)
// ════════════════════════════════════════════════════════════════════════════
//
//  One source file on purpose: everything the application does is here, in
//  sections. Each section starts with a tag: search for it (e.g. "@@AI") to
//  jump there. Resources (fonts, themes, sprites, the AI roster) live in
//  app/assets/; the engine and the AI players are the repository's sources
//  (game/, player/, nnue/), compiled by app/CMakeLists.txt.
//
//  Table of contents
//  ─────────────────
//   @@INCLUDES      SDL3, stb (images, fonts), the engine, the standard library
//   @@CONFIG        constants and paths (assets, repository, preferences)
//   @@UTIL          small tools: vectors, boxes, colours, easing, noise, files
//   @@ENGINE        a game in progress (Match): moves, passes, undo, scores
//   @@AI            the AI roster (assets/ai_roster.json), AIs thinking in a
//                   background thread (AiAgent), the analysis engine that feeds
//                   the evaluation bar, the hint and the principal variation
//   @@DRAW          anti-aliased vector drawing with SDL_RenderGeometry
//                   (rounded boxes, circles, rings, gradients, soft shadows)
//   @@TEXT          text with stb_truetype (glyph atlases, UTF-8, wrapping)
//   @@THEME         board themes (assets/themes/*.json) and the procedural
//                   board textures (wood, marble, stone, neon, flat)
//   @@SPRITE        piece sprites (assets/sprites/sprites.json): stones,
//                   rings, emblems drawn by code, optional PNG replacements,
//                   PNG export
//   @@FX            animations and effects (particles, floating text, ripples,
//                   confetti); the Effect interface is the hook for future
//                   effects
//   @@UI            immediate-mode widgets (buttons, toggles, segmented
//                   choices, panels, scrolling lists)
//   @@BOARD         the board: layout, drawing, piece animation, mouse input
//   @@APP           the application state, settings, scene switching
//   @@SCENE_MENU    main menu
//   @@SCENE_SETUP   choice of the players (human or AI) and of the options
//   @@SCENE_GAME    a game: board, evaluation bar, players, moves, analysis
//   @@SCENE_THEMES  themes and sprites (preview, selection, PNG export)
//   @@SCENE_RULES   the rules (placeholder until the interactive tutorial)
//   @@SCENE_PUZZLES puzzles and quests (placeholder)
//   @@SCENE_EDITOR  puzzle editor (placeholder)
//   @@MAIN          the SDL callbacks: SDL_AppInit, SDL_AppEvent,
//                   SDL_AppIterate, SDL_AppQuit; screenshot mode
//
//  Command line (all optional):
//   --size WxH            window size
//   --screenshot FILE     render, save FILE (.bmp) and quit (tests, docs)
//   --frames N            frames rendered before the screenshot (default 30)
//   --scene NAME          menu, setup, game, themes, rules, puzzles, editor
//   --black ID --white ID players of the game scene (human or an AI id)
//   --plies N             random moves played before the game scene starts
//   --hint                show the hint (game scene)
//   --tab N               tab of the game scene (0 moves, 1 analysis, 2 AI)
//   --time N              time per move: index in 0,5 s, 1 s, 2 s, 5 s
//   --theme ID            board theme
//   --verbose             let the AI players print to the terminal
// ════════════════════════════════════════════════════════════════════════════

// ─── @@INCLUDES ─────────────────────────────────────────────────────────────
// SDL_MAIN_USE_CALLBACKS: instead of writing main() and a loop, the program
// gives SDL four functions (see @@MAIN); SDL owns the loop. This is the
// recommended structure in SDL3 and works the same on every platform.
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "game.h"
#include "magic.h"
#include "zobrist.h"
#include "player.h"
#include "minmax_nnue_dev_player.h"
#include "heuristic.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace yapp {

// ─── @@CONFIG ───────────────────────────────────────────────────────────────
// The layout is designed for a 1600 × 900 window and scaled to the real size
// (App::s, the scale factor): every length below is in "design pixels".
constexpr float DESIGN_W = 1600.0f;
constexpr float DESIGN_H = 900.0f;
constexpr const char* APP_NAME = "Yolah";
constexpr const char* APP_VERSION = "0.1";

#ifndef YOLAH_APP_ASSETS
#define YOLAH_APP_ASSETS "assets"
#endif
#ifndef YOLAH_APP_ROOT
#define YOLAH_APP_ROOT ".."
#endif

struct Paths {
    fs::path assets;   // app/assets: fonts, themes, sprites, ai_roster.json
    fs::path root;     // the repository: config/, models/, nnue/ (the AI weights)
    fs::path prefs;    // per-user folder (SDL_GetPrefPath): settings.json
};
Paths g_paths;

// The assets folder: next to the executable first (an installed or moved
// application), then the one given at compile time (app/assets).
void init_paths() {
    const char* base = SDL_GetBasePath();
    fs::path exe = base ? fs::path(base) : fs::path(".");
    if (fs::exists(exe / "assets" / "ai_roster.json")) g_paths.assets = exe / "assets";
    else g_paths.assets = YOLAH_APP_ASSETS;
    g_paths.root = fs::weakly_canonical(fs::path(YOLAH_APP_ROOT));
    if (char* pref = SDL_GetPrefPath("Yolah", "YolahApp")) {
        g_paths.prefs = pref;
        SDL_free(pref);
    } else {
        g_paths.prefs = ".";
    }
}

// ─── @@UTIL ─────────────────────────────────────────────────────────────────
struct V2 {
    float x = 0, y = 0;
};
inline V2 operator+(V2 a, V2 b) { return {a.x + b.x, a.y + b.y}; }
inline V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
inline V2 operator*(V2 a, float k) { return {a.x * k, a.y * k}; }
inline float dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
inline float length(V2 a) { return std::sqrt(dot(a, a)); }
inline V2 normalized(V2 a) {
    float l = length(a);
    return l > 1e-6f ? a * (1.0f / l) : V2{0, 0};
}
inline V2 perp(V2 a) { return {-a.y, a.x}; }

// An axis-aligned box (x, y: top-left corner).
struct Box {
    float x = 0, y = 0, w = 0, h = 0;
    bool contains(V2 p) const { return p.x >= x && p.x < x + w && p.y >= y && p.y < y + h; }
    V2 center() const { return {x + w * 0.5f, y + h * 0.5f}; }
    Box inset(float d) const { return {x + d, y + d, w - 2 * d, h - 2 * d}; }
    Box inset(float dx, float dy) const { return {x + dx, y + dy, w - 2 * dx, h - 2 * dy}; }
    float right() const { return x + w; }
    float bottom() const { return y + h; }
    SDL_FRect frect() const { return {x, y, w, h}; }
};

struct Rgba {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};
constexpr Rgba hex(uint32_t rgb, uint8_t a = 255) {
    return {uint8_t(rgb >> 16), uint8_t(rgb >> 8), uint8_t(rgb), a};
}
inline float clamp01(float t) { return t < 0 ? 0 : (t > 1 ? 1 : t); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline Rgba mix(Rgba a, Rgba b, float t) {
    t = clamp01(t);
    return {uint8_t(lerpf(a.r, b.r, t) + 0.5f), uint8_t(lerpf(a.g, b.g, t) + 0.5f),
            uint8_t(lerpf(a.b, b.b, t) + 0.5f), uint8_t(lerpf(a.a, b.a, t) + 0.5f)};
}
inline Rgba with_alpha(Rgba c, float k) { return {c.r, c.g, c.b, uint8_t(clamp01(k * c.a / 255.0f) * 255.0f + 0.5f)}; }
// k > 0: towards white, k < 0: towards black.
inline Rgba shade(Rgba c, float k) {
    return k >= 0 ? mix(c, Rgba{255, 255, 255, c.a}, k) : mix(c, Rgba{0, 0, 0, c.a}, -k);
}
inline SDL_FColor fcolor(Rgba c) { return {c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f}; }

// "#rrggbb" or "#rrggbbaa" (themes and sprites files).
Rgba parse_color(const std::string& s, Rgba fallback = {255, 0, 255, 255}) {
    if (s.size() != 7 && s.size() != 9) return fallback;
    if (s[0] != '#') return fallback;
    uint32_t v = 0;
    for (size_t i = 1; i < s.size(); i++) {
        char c = s[i];
        int d = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (d < 0) return fallback;
        v = v * 16 + d;
    }
    if (s.size() == 7) return hex(v);
    return {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
}
std::string color_string(Rgba c) {
    char buf[16];
    if (c.a == 255) std::snprintf(buf, sizeof buf, "#%02x%02x%02x", c.r, c.g, c.b);
    else std::snprintf(buf, sizeof buf, "#%02x%02x%02x%02x", c.r, c.g, c.b, c.a);
    return buf;
}

// Easing curves: t in [0, 1] → [0, 1], for animations.
inline float ease_out_cubic(float t) { t = clamp01(t); return 1 - std::pow(1 - t, 3.0f); }
inline float ease_in_out_cubic(float t) {
    t = clamp01(t);
    return t < 0.5f ? 4 * t * t * t : 1 - std::pow(-2 * t + 2, 3.0f) / 2;
}
inline float ease_out_back(float t) {
    t = clamp01(t);
    const float c1 = 1.70158f, c3 = c1 + 1;
    return 1 + c3 * std::pow(t - 1, 3.0f) + c1 * std::pow(t - 1, 2.0f);
}
inline float smoothstep(float a, float b, float x) {
    float t = clamp01((x - a) / (b - a));
    return t * t * (3 - 2 * t);
}

// Seconds since SDL started (SDL_GetTicksNS: nanoseconds).
inline double now_seconds() { return SDL_GetTicksNS() * 1e-9; }

std::string strf(const char* format, ...) {
    char buf[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof buf, format, args);
    va_end(args);
    return buf;
}

std::optional<std::string> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
bool write_file(const fs::path& p, const std::string& data) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f << data;
    return bool(f);
}
json load_json(const fs::path& p) {
    auto text = read_file(p);
    if (!text) return json();
    try {
        return json::parse(*text, nullptr, true, true);   // comments allowed
    } catch (const std::exception& e) {
        SDL_Log("JSON error in %s: %s", p.string().c_str(), e.what());
        return json();
    }
}
std::string jstr(const json& j, const char* key, const std::string& def = "") {
    return j.is_object() && j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : def;
}
float jnum(const json& j, const char* key, float def) {
    return j.is_object() && j.contains(key) && j[key].is_number() ? j[key].get<float>() : def;
}
Rgba jcolor(const json& j, const char* key, Rgba def) {
    return j.is_object() && j.contains(key) && j[key].is_string() ? parse_color(j[key].get<std::string>(), def) : def;
}

// Value noise and fractal noise (fbm), for the procedural board textures.
inline float hash2(int x, int y, uint32_t seed) {
    uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return float((h ^ (h >> 16)) & 0xffffff) / float(0xffffff);
}
inline float value_noise(float x, float y, uint32_t seed) {
    int xi = int(std::floor(x)), yi = int(std::floor(y));
    float fx = x - xi, fy = y - yi;
    float u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    float a = hash2(xi, yi, seed), b = hash2(xi + 1, yi, seed);
    float c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
    return lerpf(lerpf(a, b, u), lerpf(c, d, u), v);
}
inline float fbm(float x, float y, uint32_t seed, int octaves = 4) {
    float sum = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < octaves; i++) {
        sum += amp * value_noise(x, y, seed + i * 17);
        norm += amp;
        x *= 2.03f;
        y *= 2.03f;
        amp *= 0.5f;
    }
    return sum / norm;
}

// ─── @@ENGINE ───────────────────────────────────────────────────────────────
// The rules are those of the engine (game/game.h): 8 × 8 board, 4 pieces per
// side, a piece moves like a chess queen over free squares, the square it
// leaves becomes a hole, every move scores 1 point; a side that cannot move
// passes (Move::none); the game ends when neither side can move; the higher
// score wins. Squares: 0 = a1 … 63 = h8 (file = s % 8, rank = s / 8).
constexpr int BLACK = Yolah::BLACK, WHITE = Yolah::WHITE;

inline int sq_file(int s) { return s & 7; }
inline int sq_rank(int s) { return s >> 3; }
std::string sq_name(int s) { return {char('a' + sq_file(s)), char('1' + sq_rank(s))}; }
std::string move_name(Move m) {
    if (m == Move::none()) return "passe";
    return sq_name(m.from_sq()) + "\xE2\x86\x92" + sq_name(m.to_sq());   // "e4→f3"
}
const char* side_name(int side) { return side == BLACK ? "Noirs" : "Blancs"; }

// Legal moves of the side to move ({Move::none()} when it must pass).
std::vector<Move> legal_moves(const Yolah& y) {
    Yolah::MoveList ml;
    y.moves(ml);
    return {ml.begin(), ml.end()};
}
bool must_pass(const Yolah& y) {
    auto v = legal_moves(y);
    return v.size() == 1 && v[0] == Move::none();
}
bool is_legal(const Yolah& y, Move m) {
    for (Move x : legal_moves(y))
        if (x == m) return true;
    return false;
}

// A game in progress: the start position (any position, for the puzzles),
// the moves played, the current position.
class Match {
public:
    Yolah start, pos;
    std::vector<Move> history;

    void reset(const Yolah& s = Yolah()) {
        start = pos = s;
        history.clear();
    }
    void play(Move m) {
        pos.play(m);
        history.push_back(m);
    }
    bool undo() {
        if (history.empty()) return false;
        pos.undo(history.back());
        history.pop_back();
        return true;
    }
    bool over() const { return pos.game_over(); }
    int to_move() const { return pos.current_player(); }
    int score(int side) const {
        auto [b, w] = pos.score();
        return side == BLACK ? b : w;
    }
    // BLACK, WHITE, or -1 for a draw.
    int winner() const {
        int b = score(BLACK), w = score(WHITE);
        return b > w ? BLACK : (w > b ? WHITE : -1);
    }
    // The side that played history[i].
    int side_of(size_t i) const { return (start.nb_plies() + i) % 2 == 0 ? BLACK : WHITE; }
};

// ─── @@AI ───────────────────────────────────────────────────────────────────
// assets/ai_roster.json lists the AIs, weakest first: an id, a name, an
// indicative Elo, a sprite, the engine configuration (a .cfg of config/, the
// same files as the tournaments) and optional overrides of that
// configuration ("overrides"; "torch overrides" with YOLAH_APP_TORCH).
// Relative weight paths: "../x" is relative to config/ (the convention of the
// .cfg files), anything else to the repository.
struct AiSpec {
    std::string id, name, family, sprite, description, config_file;
    int elo = 0;
    json config;          // the engine configuration, overrides applied, paths absolute
    bool available = false;
    std::string reason;   // why it is not available
};

fs::path resolve_engine_path(const std::string& p) {
    fs::path path(p);
    if (path.is_absolute()) return path;
    if (p.rfind("../", 0) == 0) return fs::weakly_canonical(g_paths.root / "config" / path);
    return fs::weakly_canonical(g_paths.root / path);
}

std::vector<AiSpec> load_roster(const fs::path& file) {
    std::vector<AiSpec> list;
    json j = load_json(file);
    if (!j.is_object() || !j.contains("ais")) {
        SDL_Log("no AI roster in %s", file.string().c_str());
        return list;
    }
    for (const json& e : j["ais"]) {
        AiSpec s;
        s.id = jstr(e, "id");
        s.name = jstr(e, "name", s.id);
        s.family = jstr(e, "family");
        s.sprite = jstr(e, "sprite", "default");
        s.config_file = jstr(e, "config");
        s.elo = int(jnum(e, "elo", 0));
        s.config = load_json(g_paths.root / "config" / s.config_file);
        if (!s.config.is_object()) {
            s.reason = "configuration introuvable : config/" + s.config_file;
            list.push_back(s);
            continue;
        }
        auto apply = [&](const char* key) {
            if (e.contains(key) && e[key].is_object())
                for (auto& [k, v] : e[key].items()) s.config[k] = v;
        };
        apply("overrides");
#ifdef YOLAH_APP_TORCH
        apply("torch overrides");
#endif
        s.description = jstr(e, "description", jstr(s.config, "description"));
        s.available = true;
        if (s.config.contains("weights") && s.config["weights"].is_string()) {
            fs::path w = resolve_engine_path(s.config["weights"].get<std::string>());
            s.config["weights"] = w.string();
            if (!fs::exists(w)) {
                s.available = false;
                s.reason = "réseau introuvable : " + w.string() + (e.contains("missing hint") ? "\n" + jstr(e, "missing hint") : "");
            }
        }
        list.push_back(s);
    }
    return list;
}

// The configuration for one game: time per move, threads capped to leave
// cores to the interface and to the analysis.
json game_config(const AiSpec& s, double seconds) {
    json c = s.config;
    if (c.contains("microseconds")) c["microseconds"] = uint64_t(seconds * 1e6);
    const int hw = std::max(2u, std::thread::hardware_concurrency());
    if (c.contains("nb threads") && c["nb threads"].is_number())
        c["nb threads"] = std::max(1, std::min(c["nb threads"].get<int>(), hw - 2));
    c["verbose"] = false;
    return c;
}

// Threads that are no longer wanted (a game abandoned while an AI thinks):
// they are not killed (a search cannot be interrupted safely), they finish in
// the background and are joined here, at the latest when the application quits.
struct Graveyard {
    std::vector<std::pair<std::shared_ptr<std::atomic<bool>>, std::thread>> threads;
    void bury(std::shared_ptr<std::atomic<bool>> done, std::thread t) {
        if (t.joinable()) threads.emplace_back(std::move(done), std::move(t));
    }
    void reap() {
        for (size_t i = 0; i < threads.size();) {
            if (threads[i].first->load()) {
                threads[i].second.join();
                threads.erase(threads.begin() + i);
            } else {
                i++;
            }
        }
    }
    void join_all() {
        for (auto& [d, t] : threads) t.join();
        threads.clear();
    }
};
Graveyard g_graveyard;

// An AI that thinks in a background thread, so that the window stays
// responsive. The engine player (Player::create, player/player.cpp) is created
// by the first search (loading a network takes time) and is only used by one
// thread at a time (the mutex of Shared).
class AiAgent {
public:
    explicit AiAgent(json cfg) : sh_(std::make_shared<Shared>()) { sh_->cfg = std::move(cfg); }
    ~AiAgent() { cancel(); }

    void think(const Yolah& pos, uint64_t token) {
        cancel();
        auto job = std::make_shared<Job>();
        job->token = token;
        job_ = job;
        auto sh = sh_;
        thread_ = std::thread([sh, job, pos] {
            std::lock_guard lock(sh->m);
            try {
                if (!sh->player) {
                    sh->player = Player::create(sh->cfg);
                    sh->info = sh->player->info();
                    sh->dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(sh->player.get());
                }
                const double t0 = now_seconds();
                job->move = sh->player->play(pos);
                job->seconds = now_seconds() - t0;
            } catch (const std::exception& e) {
                job->error = e.what();
            }
            job->done->store(true);
        });
    }
    // The move of the search started with `token`, once it is finished.
    bool poll(uint64_t token, Move& move, double& seconds, std::string& error) {
        if (!job_ || job_->token != token || !job_->done->load()) return false;
        if (thread_.joinable()) thread_.join();
        move = job_->move;
        seconds = job_->seconds;
        error = job_->error;
        job_.reset();
        return true;
    }
    bool thinking(uint64_t token) const { return job_ && job_->token == token; }
    bool thinking() const { return job_ != nullptr; }
    // Abandons the current search (the dev player can be told to stop).
    void cancel() {
        if (!job_) return;
        if (auto* dev = sh_->dev.load()) dev->stop_flag()->store(true);
        g_graveyard.bury(job_->done, std::move(thread_));
        job_.reset();
    }
    std::string info() const { return sh_->info; }

private:
    struct Shared {
        std::mutex m;
        json cfg;
        std::unique_ptr<Player> player;
        std::string info;
        std::atomic<MinMaxNNUE_DevPlayer*> dev{nullptr};
    };
    struct Job {
        std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);
        uint64_t token = 0;
        Move move = Move::none();
        double seconds = 0;
        std::string error;
    };
    std::shared_ptr<Shared> sh_;
    std::shared_ptr<Job> job_;
    std::thread thread_;
};

// The analysis engine: the dev minimax player searches the current position
// in the background, in short slices (each slice restarts the iterative
// deepening, the transposition table makes it cheap), and publishes after each
// slice its value, its depth, its best move and its principal variation. It
// feeds the evaluation bar, the hint and the analysis tab, for every game,
// whatever the players are.
// Values: from the side to move in the engine, here from Black's point of view;
// |v| < WIN: tanh of the network × WIN (an expected result in [−1, 1]);
// |v| > WIN: a proven result, WIN + the final score difference.
constexpr int EVAL_WIN = heuristic::MAX_VALUE;   // = MinMaxNNUE_DevPlayer::WIN

class Analyzer {
public:
    struct Snapshot {
        uint64_t token = 0;
        bool valid = false;      // at least one slice finished
        bool running = false;
        int value = 0;           // Black's point of view
        int depth = 0;
        Move best = Move::none();
        std::vector<Move> pv;
        uint64_t nodes = 0;
        double seconds = 0;
        std::string error;
    };

    ~Analyzer() { stop(); }
    void configure(json cfg) {
        stop();
        st_ = std::make_shared<State>();
        st_->cfg = std::move(cfg);
    }
    bool configured() const { return st_ != nullptr; }

    void analyse(const Yolah& pos, uint64_t token, double budget) {
        if (!st_) return;
        stop();
        st_->stop.store(false);
        {
            std::lock_guard lock(st_->snap_m);
            st_->snap = Snapshot{};
            st_->snap.token = token;
            st_->snap.running = true;
        }
        auto st = st_;
        thread_ = std::thread([st, pos, token, budget] {
            std::lock_guard lock(st->m);
            try {
                if (!st->dev.load()) {
                    st->player = Player::create(st->cfg);
                    st->dev = dynamic_cast<MinMaxNNUE_DevPlayer*>(st->player.get());
                    if (!st->dev.load()) throw std::runtime_error("the analysis engine must be a MinMaxNNUE_DevPlayer");
                }
                const double t0 = now_seconds();
                const int sign = pos.current_player() == BLACK ? 1 : -1;
                uint64_t nodes = 0;
                while (!st->stop.load() && now_seconds() - t0 < budget && !pos.game_over()) {
                    auto r = st->dev.load()->search(pos, 64, 120000);
                    if (st->stop.load()) break;
                    nodes += r.nb_nodes;
                    Snapshot s;
                    s.token = token;
                    s.valid = true;
                    s.running = true;
                    s.value = sign * int(r.value);
                    s.depth = r.depth;
                    s.best = r.move;
                    s.pv = st->dev.load()->principal_variation(pos, 16);
                    if (s.pv.empty() && r.move != Move::none()) s.pv.push_back(r.move);
                    s.nodes = nodes;
                    s.seconds = now_seconds() - t0;
                    {
                        std::lock_guard l2(st->snap_m);
                        st->snap = s;
                    }
                    if (std::abs(int(r.value)) > EVAL_WIN && r.depth >= 8) break;   // proven
                }
            } catch (const std::exception& e) {
                std::lock_guard l2(st->snap_m);
                st->snap.error = e.what();
            }
            std::lock_guard l2(st->snap_m);
            st->snap.running = false;
        });
    }
    void stop() {
        if (!st_) return;
        st_->stop.store(true);
        if (auto* dev = st_->dev.load()) dev->stop_flag()->store(true);
        if (thread_.joinable()) thread_.join();
    }
    Snapshot snapshot() const {
        if (!st_) return {};
        std::lock_guard lock(st_->snap_m);
        return st_->snap;
    }

private:
    struct State {
        std::mutex m, snap_m;
        json cfg;
        std::unique_ptr<Player> player;
        std::atomic<MinMaxNNUE_DevPlayer*> dev{nullptr};
        std::atomic<bool> stop{false};
        Snapshot snap;
    };
    std::shared_ptr<State> st_;
    std::thread thread_;
};

// Black's share of the evaluation bar, in [0, 1].
float eval_share(int value) {
    if (value > EVAL_WIN) return 1;
    if (value < -EVAL_WIN) return 0;
    return clamp01(0.5f + 0.5f * float(value) / EVAL_WIN);
}
// "62 %", "Gain +3"...: the text of a value for the side it favours.
std::string eval_text(int value) {
    if (std::abs(value) > EVAL_WIN) return strf("Gain +%d", std::abs(value) - EVAL_WIN);
    float p = eval_share(value);
    return strf("%d %%", int(std::round(100 * std::max(p, 1 - p))));
}

// ─── @@DRAW ─────────────────────────────────────────────────────────────────
// SDL_RenderGeometry draws triangles whose vertices carry a colour: enough
// for every shape of the interface, at any size, with gradients for free.
// Anti-aliasing: around each filled shape, a thin "fringe" of triangles goes
// from the shape's colour to transparent (1 pixel); a wide fringe makes a soft
// shadow. Shapes are "fans": a centre and a closed outline (convex or
// star-shaped around the centre).
namespace draw {

SDL_Renderer* R = nullptr;   // set by the App at start

void fan(V2 c, Rgba cc, const std::vector<V2>& ring, const std::vector<Rgba>& cols, float feather = 1.0f) {
    const int n = int(ring.size());
    if (n < 3) return;
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    v.reserve(2 * n + 1);
    idx.reserve(9 * n);
    v.push_back({{c.x, c.y}, fcolor(cc), {0, 0}});
    for (int i = 0; i < n; i++) v.push_back({{ring[i].x, ring[i].y}, fcolor(cols[i]), {0, 0}});
    for (int i = 0; i < n; i++) {
        idx.push_back(0);
        idx.push_back(1 + i);
        idx.push_back(1 + (i + 1) % n);
    }
    if (feather > 0) {
        // Outward normal of each outline vertex (mean of its two edges).
        for (int i = 0; i < n; i++) {
            V2 a = ring[(i + n - 1) % n], b = ring[i], d = ring[(i + 1) % n];
            V2 n1 = normalized(perp(b - a)), n2 = normalized(perp(d - b));
            V2 nm = normalized(n1 + n2);
            if (dot(nm, b - c) < 0) nm = nm * -1.0f;
            V2 o = b + nm * feather;
            Rgba t = cols[i];
            t.a = 0;
            v.push_back({{o.x, o.y}, fcolor(t), {0, 0}});
        }
        for (int i = 0; i < n; i++) {
            int a = 1 + i, b = 1 + (i + 1) % n, ao = 1 + n + i, bo = 1 + n + (i + 1) % n;
            idx.insert(idx.end(), {a, b, bo, a, bo, ao});
        }
    }
    SDL_RenderGeometry(R, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}

std::vector<V2> round_box_outline(Box b, float r) {
    r = std::max(0.0f, std::min(r, std::min(b.w, b.h) * 0.5f));
    std::vector<V2> pts;
    const int seg = std::clamp(int(r * 0.6f), 2, 18);
    const V2 corners[4] = {{b.x + b.w - r, b.y + r}, {b.x + b.w - r, b.y + b.h - r}, {b.x + r, b.y + b.h - r}, {b.x + r, b.y + r}};
    for (int c = 0; c < 4; c++) {
        float a0 = -float(M_PI) / 2 + c * float(M_PI) / 2;
        for (int i = 0; i <= seg; i++) {
            float a = a0 + float(M_PI) / 2 * i / seg;
            pts.push_back({corners[c].x + r * std::cos(a), corners[c].y + r * std::sin(a)});
        }
    }
    return pts;
}

// A rounded box, with a vertical gradient (top → bottom).
void round_box(Box b, float r, Rgba top, Rgba bottom, float feather = 1.0f) {
    if (b.w <= 0 || b.h <= 0) return;
    auto pts = round_box_outline(b, r);
    std::vector<Rgba> cols(pts.size());
    for (size_t i = 0; i < pts.size(); i++) cols[i] = mix(top, bottom, (pts[i].y - b.y) / b.h);
    fan(b.center(), mix(top, bottom, 0.5f), pts, cols, feather);
}
void round_box(Box b, float r, Rgba c) { round_box(b, r, c, c); }

// A soft shadow under a rounded box: the box shrunk by blur / 2, with a fringe
// of `blur` pixels.
void shadow(Box b, float r, float blur, Rgba c) {
    Box in = b.inset(blur * 0.5f);
    auto pts = round_box_outline(in, std::max(0.0f, r - blur * 0.5f));
    std::vector<Rgba> cols(pts.size(), c);
    fan(in.center(), c, pts, cols, blur);
}

// A rounded outline (frame) of thickness t, drawn as a ring of quads.
void round_frame(Box b, float r, float t, Rgba c) {
    auto outer = round_box_outline(b, r);
    auto inner = round_box_outline(b.inset(t), std::max(0.0f, r - t));
    if (outer.size() != inner.size()) return;
    const int n = int(outer.size());
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    for (int i = 0; i < n; i++) v.push_back({{outer[i].x, outer[i].y}, fcolor(c), {0, 0}});
    for (int i = 0; i < n; i++) v.push_back({{inner[i].x, inner[i].y}, fcolor(c), {0, 0}});
    for (int i = 0; i < n; i++) {
        int a = i, b2 = (i + 1) % n, ai = n + i, bi = n + (i + 1) % n;
        idx.insert(idx.end(), {a, b2, bi, a, bi, ai});
    }
    SDL_RenderGeometry(R, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}

std::vector<V2> circle_outline(V2 c, float r, int seg = 0) {
    if (seg <= 0) seg = std::clamp(int(r * 0.9f), 12, 96);
    std::vector<V2> pts(seg);
    for (int i = 0; i < seg; i++) {
        float a = 2 * float(M_PI) * i / seg;
        pts[i] = {c.x + r * std::cos(a), c.y + r * std::sin(a)};
    }
    return pts;
}
// A disc; inner != outer: radial gradient from the centre.
void circle(V2 c, float r, Rgba inner, Rgba outer, float feather = 1.0f) {
    if (r <= 0) return;
    auto pts = circle_outline(c, r);
    std::vector<Rgba> cols(pts.size(), outer);
    fan(c, inner, pts, cols, feather);
}
void circle(V2 c, float r, Rgba col, float feather = 1.0f) { circle(c, r, col, col, feather); }
// A radial gradient centred elsewhere than the disc (light coming from a side).
void circle_lit(V2 c, float r, V2 light, Rgba inner, Rgba outer) {
    auto pts = circle_outline(c, r);
    std::vector<Rgba> cols(pts.size());
    for (size_t i = 0; i < pts.size(); i++) cols[i] = mix(inner, outer, 0.55f + 0.45f * length(pts[i] - light) / (2 * r));
    fan(light, inner, pts, cols, 1.0f);
}

// A ring (annulus) of outer radius r and thickness t, with an AA fringe on
// both sides; top/bottom: vertical gradient.
void ring(V2 c, float r, float t, Rgba top, Rgba bottom) {
    const int seg = std::clamp(int(r * 0.9f), 16, 128);
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    auto col = [&](float y, float a_mul) {
        Rgba k = mix(top, bottom, (y - (c.y - r)) / (2 * r));
        return fcolor(with_alpha(k, a_mul));
    };
    const float rs[4] = {r + 0.8f, r, r - t, r - t - 0.8f};
    const float al[4] = {0, 1, 1, 0};
    for (int i = 0; i < seg; i++) {
        float a = 2 * float(M_PI) * i / seg;
        for (int k = 0; k < 4; k++) {
            float x = c.x + rs[k] * std::cos(a), y = c.y + rs[k] * std::sin(a);
            v.push_back({{x, y}, col(y, al[k]), {0, 0}});
        }
    }
    for (int i = 0; i < seg; i++) {
        int j = (i + 1) % seg;
        for (int k = 0; k < 3; k++) {
            int a = i * 4 + k, b = j * 4 + k, a2 = i * 4 + k + 1, b2 = j * 4 + k + 1;
            idx.insert(idx.end(), {a, b, b2, a, b2, a2});
        }
    }
    SDL_RenderGeometry(R, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}
void ring(V2 c, float r, float t, Rgba col) { ring(c, r, t, col, col); }

// A convex polygon (or star-shaped around its centroid).
void poly(const std::vector<V2>& pts, Rgba col, float feather = 1.0f) {
    if (pts.size() < 3) return;
    V2 c{0, 0};
    for (V2 p : pts) c = c + p;
    c = c * (1.0f / pts.size());
    std::vector<Rgba> cols(pts.size(), col);
    fan(c, col, pts, cols, feather);
}
void poly_around(V2 c, const std::vector<V2>& pts, Rgba col, float feather = 1.0f) {
    std::vector<Rgba> cols(pts.size(), col);
    fan(c, col, pts, cols, feather);
}
// A thick segment with round-ish ends (a quad with an AA fringe).
void line(V2 a, V2 b, float w, Rgba col) {
    V2 d = normalized(b - a), n = perp(d) * (w * 0.5f);
    poly({a + n, b + n, b - n, a - n}, col);
}
// An arrow from a to b (hint, principal variation).
void arrow(V2 a, V2 b, float w, Rgba col) {
    V2 d = normalized(b - a), n = perp(d);
    float head = w * 2.6f;
    V2 base = b - d * head;
    poly({a + n * (w * 0.5f), base + n * (w * 0.5f), base - n * (w * 0.5f), a - n * (w * 0.5f)}, col);
    poly({base + n * (head * 0.75f), b, base - n * (head * 0.75f)}, col);
}
void rect(Box b, Rgba c) {
    SDL_SetRenderDrawColor(R, c.r, c.g, c.b, c.a);
    SDL_FRect f = b.frect();
    SDL_RenderFillRect(R, &f);
}
// The whole window, vertical gradient.
void gradient_rect(Box b, Rgba top, Rgba bottom) {
    SDL_Vertex v[4] = {{{b.x, b.y}, fcolor(top), {0, 0}}, {{b.right(), b.y}, fcolor(top), {0, 0}},
                       {{b.right(), b.bottom()}, fcolor(bottom), {0, 0}}, {{b.x, b.bottom()}, fcolor(bottom), {0, 0}}};
    int idx[6] = {0, 1, 2, 0, 2, 3};
    SDL_RenderGeometry(R, nullptr, v, 4, idx, 6);
}
// Darkens the borders of b softly (a ring of quads from transparent inside to
// `alpha` outside).
void vignette(Box b, float s, uint8_t alpha) {
    auto inner = round_box_outline(b.inset(std::min(b.w, b.h) * 0.22f), 260 * s);
    auto outer = round_box_outline(b.inset(-40 * s), 300 * s);
    if (inner.size() != outer.size()) return;
    const int n = int(inner.size());
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    for (int i = 0; i < n; i++) v.push_back({{inner[i].x, inner[i].y}, fcolor(hex(0x000000, 0)), {0, 0}});
    for (int i = 0; i < n; i++) v.push_back({{outer[i].x, outer[i].y}, fcolor(hex(0x000000, alpha)), {0, 0}});
    for (int i = 0; i < n; i++) {
        int a = i, b2 = (i + 1) % n, ao = n + i, bo = n + (i + 1) % n;
        idx.insert(idx.end(), {a, b2, bo, a, bo, ao});
    }
    SDL_RenderGeometry(R, nullptr, v.data(), int(v.size()), idx.data(), int(idx.size()));
}
void clip(const Box* b) {
    if (!b) {
        SDL_SetRenderClipRect(R, nullptr);
        return;
    }
    SDL_Rect r{int(b->x), int(b->y), int(std::ceil(b->w)), int(std::ceil(b->h))};
    SDL_SetRenderClipRect(R, &r);
}

}   // namespace draw

// ─── @@TEXT ─────────────────────────────────────────────────────────────────
// stb_truetype rasterises the glyphs of a font at a given pixel size into one
// texture (an "atlas"); a string is then drawn glyph by glyph, each glyph a
// rectangle of the atlas (SDL_RenderTexture), tinted by the texture's colour
// modulation. One atlas per (font, size), created on first use.
enum class Font { Regular, Bold, Black, Title };
enum class Align { Left, Center, Right };

class TextEngine {
public:
    bool load(const fs::path& dir) {
        const char* files[4] = {"Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Black.ttf", "Comfortaa-Bold.ttf"};
        for (int i = 0; i < 4; i++) {
            auto data = read_file(dir / files[i]);
            if (!data) {
                SDL_Log("font missing: %s", (dir / files[i]).string().c_str());
                return false;
            }
            faces_[i].data.assign(data->begin(), data->end());
            if (!stbtt_InitFont(&faces_[i].info, faces_[i].data.data(), stbtt_GetFontOffsetForIndex(faces_[i].data.data(), 0)))
                return false;
        }
        // The characters packed: ASCII, Latin-1 (French accents), a few symbols.
        for (int c = 32; c < 127; c++) codepoints_.push_back(c);
        for (int c = 160; c < 256; c++) codepoints_.push_back(c);
        for (int c : {0x152, 0x153, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2026, 0x2039, 0x203A, 0x20AC,
                      0x2190, 0x2191, 0x2192, 0x2193, 0x21C4, 0x2248, 0x2212, 0x2605, 0x2606, 0x2713, 0x2715, 0x25B6, 0x25C0})
            codepoints_.push_back(c);
        return true;
    }
    void clear() {
        for (auto& f : faces_)
            for (auto& [k, a] : f.atlases)
                if (a.tex) SDL_DestroyTexture(a.tex);
        for (auto& f : faces_) f.atlases.clear();
    }

    // Draws `s` with its top-left (Align::Left) / top-centre / top-right at p;
    // returns the width.
    float draw(Font f, float size, const std::string& s, V2 p, Rgba col, Align al = Align::Left) {
        Atlas& a = atlas(f, size);
        float w = width(f, size, s);
        float x = p.x - (al == Align::Center ? w * 0.5f : al == Align::Right ? w : 0.0f);
        float y = p.y + a.ascent;
        SDL_SetTextureColorMod(a.tex, col.r, col.g, col.b);
        SDL_SetTextureAlphaMod(a.tex, col.a);
        for (size_t i = 0; i < s.size();) {
            uint32_t cp = next_codepoint(s, i);
            auto it = a.index.find(cp);
            if (it == a.index.end()) it = a.index.find('?');
            stbtt_aligned_quad q;
            float xx = x, yy = y;
            stbtt_GetPackedQuad(a.chars.data(), a.w, a.h, it->second, &xx, &yy, &q, 0);
            SDL_FRect src{q.s0 * a.w, q.t0 * a.h, (q.s1 - q.s0) * a.w, (q.t1 - q.t0) * a.h};
            SDL_FRect dst{q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0};
            if (dst.w > 0) SDL_RenderTexture(draw::R, a.tex, &src, &dst);
            x = xx;
        }
        return w;
    }
    // Text with a soft shadow below (titles).
    float draw_shadowed(Font f, float size, const std::string& s, V2 p, Rgba col, Align al, Rgba sh, float off) {
        draw(f, size, s, p + V2{0, off}, sh, al);
        return draw(f, size, s, p, col, al);
    }
    float width(Font f, float size, const std::string& s) {
        Atlas& a = atlas(f, size);
        float w = 0;
        for (size_t i = 0; i < s.size();) {
            uint32_t cp = next_codepoint(s, i);
            auto it = a.index.find(cp);
            if (it == a.index.end()) it = a.index.find('?');
            w += a.chars[it->second].xadvance;
        }
        return w;
    }
    float line_height(Font f, float size) {
        Atlas& a = atlas(f, size);
        return a.ascent - a.descent + a.gap;
    }
    // Splits `s` into lines of at most `max_w` pixels (on spaces; "\n" breaks).
    std::vector<std::string> wrap(Font f, float size, const std::string& s, float max_w) {
        std::vector<std::string> lines;
        std::string para;
        std::stringstream ss(s);
        while (std::getline(ss, para, '\n')) {
            std::string cur;
            std::stringstream ws(para);
            std::string word;
            while (ws >> word) {
                std::string trial = cur.empty() ? word : cur + " " + word;
                if (!cur.empty() && width(f, size, trial) > max_w) {
                    lines.push_back(cur);
                    cur = word;
                } else {
                    cur = trial;
                }
            }
            lines.push_back(cur);
        }
        return lines;
    }
    // Draws wrapped text; returns the height used.
    float paragraph(Font f, float size, const std::string& s, Box b, Rgba col, float spacing = 1.25f) {
        float lh = line_height(f, size) * spacing / 1.2f, y = b.y;
        for (auto& l : wrap(f, size, s, b.w)) {
            if (y + lh > b.bottom() + 1) break;
            draw(f, size, l, {b.x, y}, col);
            y += lh;
        }
        return y - b.y;
    }

private:
    struct Atlas {
        SDL_Texture* tex = nullptr;
        int w = 0, h = 0;
        std::vector<stbtt_packedchar> chars;
        std::unordered_map<uint32_t, int> index;
        float ascent = 0, descent = 0, gap = 0;
    };
    struct Face {
        std::vector<unsigned char> data;
        stbtt_fontinfo info{};
        std::map<int, Atlas> atlases;
    };
    Face faces_[4];
    std::vector<int> codepoints_;

    static uint32_t next_codepoint(const std::string& s, size_t& i) {
        unsigned char c = s[i];
        uint32_t cp;
        int extra;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c >> 5) == 6) { cp = c & 0x1f; extra = 1; }
        else if ((c >> 4) == 14) { cp = c & 0x0f; extra = 2; }
        else { cp = c & 0x07; extra = 3; }
        i++;
        for (int k = 0; k < extra && i < s.size(); k++, i++) cp = (cp << 6) | (s[i] & 0x3f);
        return cp;
    }

    Atlas& atlas(Font f, float size) {
        Face& face = faces_[int(f)];
        int px = std::max(6, int(std::round(size)));
        auto it = face.atlases.find(px);
        if (it != face.atlases.end()) return it->second;
        Atlas& a = face.atlases[px];
        const int over = px <= 36 ? 2 : 1;   // horizontal oversampling: sharper small text
        a.w = px <= 20 ? 512 : px <= 44 ? 1024 : px <= 100 ? 2048 : 4096;
        a.h = a.w;
        std::vector<unsigned char> alpha(size_t(a.w) * a.h);
        a.chars.resize(codepoints_.size());
        stbtt_pack_context pc;
        stbtt_PackBegin(&pc, alpha.data(), a.w, a.h, 0, 2, nullptr);
        stbtt_PackSetOversampling(&pc, over, 1);
        stbtt_pack_range range{};
        range.font_size = float(px);
        range.array_of_unicode_codepoints = codepoints_.data();
        range.num_chars = int(codepoints_.size());
        range.chardata_for_range = a.chars.data();
        stbtt_PackFontRanges(&pc, face.data.data(), 0, &range, 1);
        stbtt_PackEnd(&pc);
        for (size_t i = 0; i < codepoints_.size(); i++) a.index[uint32_t(codepoints_[i])] = int(i);
        int asc, desc, gap;
        stbtt_GetFontVMetrics(&face.info, &asc, &desc, &gap);
        float scale = stbtt_ScaleForPixelHeight(&face.info, float(px));
        a.ascent = asc * scale;
        a.descent = desc * scale;
        a.gap = gap * scale;
        // White pixels whose alpha is the glyph coverage, tinted when drawn.
        std::vector<uint32_t> rgba(alpha.size());
        for (size_t i = 0; i < alpha.size(); i++) rgba[i] = 0x00ffffffu | (uint32_t(alpha[i]) << 24);
        a.tex = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, a.w, a.h);
        SDL_UpdateTexture(a.tex, nullptr, rgba.data(), a.w * 4);
        SDL_SetTextureBlendMode(a.tex, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(a.tex, SDL_SCALEMODE_LINEAR);
        return a;
    }
};

// ─── @@THEME ────────────────────────────────────────────────────────────────
// A theme (assets/themes/<id>.json) gives the colours of the board, of the
// holes, of the highlights and of the window background, and the pattern of
// the squares: "wood", "marble", "stone", "neon" or "flat". To create a theme,
// copy a file, change its "id", "name" and colours: it appears in the list.
struct Theme {
    std::string id = "default", name = "Défaut", pattern = "flat";
    Rgba light = hex(0xe8d3b0), dark = hex(0xb58863);
    Rgba frame_top = hex(0x5a3a22), frame_bottom = hex(0x3a2414), frame_line = hex(0xd9b382);
    Rgba coords = hex(0xf3e2c4);
    Rgba hole = hex(0x1b120c), hole_rim = hex(0x6b4a2e);
    Rgba bg_top = hex(0x2a2f3a), bg_bottom = hex(0x12151c);
    Rgba last_move = hex(0xf6d55c, 110), select = hex(0x7fd1ff, 150), legal = hex(0x1d2733, 110), hint = hex(0x4ade80, 220);
    Rgba accent = hex(0xf2b45a);
    float strength = 0.5f;   // how visible the pattern is
    float seed = 1;
};

Theme parse_theme(const json& j) {
    Theme t;
    t.id = jstr(j, "id", t.id);
    t.name = jstr(j, "name", t.name);
    t.pattern = jstr(j, "pattern", t.pattern);
    t.light = jcolor(j, "light", t.light);
    t.dark = jcolor(j, "dark", t.dark);
    t.frame_top = jcolor(j, "frame top", t.frame_top);
    t.frame_bottom = jcolor(j, "frame bottom", t.frame_bottom);
    t.frame_line = jcolor(j, "frame line", t.frame_line);
    t.coords = jcolor(j, "coordinates", t.coords);
    t.hole = jcolor(j, "hole", t.hole);
    t.hole_rim = jcolor(j, "hole rim", t.hole_rim);
    t.bg_top = jcolor(j, "background top", t.bg_top);
    t.bg_bottom = jcolor(j, "background bottom", t.bg_bottom);
    t.last_move = jcolor(j, "last move", t.last_move);
    t.select = jcolor(j, "selection", t.select);
    t.legal = jcolor(j, "legal move", t.legal);
    t.hint = jcolor(j, "hint", t.hint);
    t.accent = jcolor(j, "accent", t.accent);
    t.strength = jnum(j, "pattern strength", t.strength);
    t.seed = jnum(j, "seed", t.seed);
    return t;
}

std::vector<Theme> load_themes(const fs::path& dir) {
    std::vector<Theme> themes;
    std::error_code ec;
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        json j = load_json(f);
        if (j.is_object()) themes.push_back(parse_theme(j));
    }
    // "order" in the files would be nicer; the file names start with a number.
    if (themes.empty()) themes.push_back(Theme{});
    return themes;
}

// The 8 × 8 squares of a theme as one texture of px × px pixels, computed
// pixel by pixel (procedural patterns), with a small bevel on each square.
SDL_Texture* make_board_texture(const Theme& t, int px) {
    std::vector<uint32_t> pix(size_t(px) * px);
    const float sq = px / 8.0f;
    const uint32_t seed = uint32_t(t.seed * 7919);
    for (int y = 0; y < px; y++) {
        for (int x = 0; x < px; x++) {
            int f = std::min(7, int(x / sq)), r = std::min(7, int(y / sq));
            bool light = (f + r) % 2 == 0;
            Rgba base = light ? t.light : t.dark;
            float u = x / sq, v = y / sq;          // in squares
            float k = 0;                            // brightness change in [−1, 1]
            if (t.pattern == "wood") {
                // Grain along the square, horizontal or vertical in turn.
                // a: across the grain, b: along it; the rings bend slowly.
                bool horiz = (f + r) % 2 == 0;
                float a = horiz ? v : u, b = horiz ? u : v;
                float bend = fbm(a * 1.2f + f * 3.1f, b * 0.35f + r * 1.7f, seed);
                float rings = std::sin((a * 5.5f + bend * 2.2f + f * 0.37f + r * 0.61f) * float(M_PI) * 2);
                float fibres = fbm(a * 40.0f, b * 1.5f, seed + 9, 2) - 0.5f;
                k = 0.45f * std::pow(0.5f + 0.5f * rings, 3.0f) - 0.2f + fibres * 0.55f + (bend - 0.5f) * 0.4f;
            } else if (t.pattern == "marble") {
                // Soft veins: a sine bent by turbulence, sharpened a little.
                float n = fbm(u * 0.45f, v * 0.45f, seed, 6);
                float vein = std::abs(std::sin((u * 0.35f + v * 0.55f + n * 3.2f) * float(M_PI)));
                float cloud = fbm(u * 1.1f + 7, v * 1.1f, seed + 3, 4);
                k = -std::pow(1.0f - vein, 3.5f) * 1.1f + (cloud - 0.5f) * 0.7f;
            } else if (t.pattern == "stone") {
                float n = fbm(u * 3.0f, v * 3.0f, seed, 5);
                float speck = hash2(x, y, seed) > 0.985f ? 0.6f : 0.0f;
                k = (n - 0.5f) * 1.2f + speck;
            } else if (t.pattern == "neon") {
                float fu = u - std::floor(u), fv = v - std::floor(v);
                float edge = std::min(std::min(fu, 1 - fu), std::min(fv, 1 - fv));
                k = std::exp(-edge * 28.0f) * 1.6f + (fbm(u * 2, v * 2, seed, 3) - 0.5f) * 0.25f;
            } else {
                k = (fbm(u * 2, v * 2, seed, 3) - 0.5f) * 0.3f;
            }
            k *= t.strength;
            // Bevel: light on the top-left edges of each square, dark bottom-right.
            float fu = u - std::floor(u), fv = v - std::floor(v);
            float bev = 0.06f;
            if (fu < bev || fv < bev) k += 0.10f * (1 - std::min(fu, fv) / bev);
            if (fu > 1 - bev || fv > 1 - bev) k -= 0.12f * (1 - std::min(1 - fu, 1 - fv) / bev);
            Rgba c = shade(base, std::clamp(k, -1.0f, 1.0f) * 0.35f);
            pix[size_t(y) * px + x] = (0xffu << 24) | (uint32_t(c.r) << 16) | (uint32_t(c.g) << 8) | c.b;
        }
    }
    SDL_Texture* tex = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, px, px);
    SDL_UpdateTexture(tex, nullptr, pix.data(), px * 4);
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
    return tex;
}

// Board textures, one per (theme, size): computed once, reused every frame.
class BoardTextures {
public:
    SDL_Texture* get(const Theme& t, int px) {
        std::string key = t.id + "@" + std::to_string(px);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
        if (cache_.size() > 24) clear();
        return cache_[key] = make_board_texture(t, px);
    }
    void clear() {
        for (auto& [k, t] : cache_) SDL_DestroyTexture(t);
        cache_.clear();
    }

private:
    std::unordered_map<std::string, SDL_Texture*> cache_;
};

// ─── @@SPRITE ───────────────────────────────────────────────────────────────
// A piece is a stone (black or white, a lit gradient), a coloured ring whose
// metal tells the strength of its player, small pips for the tier, an emblem
// drawn by code, and a halo for the human player. assets/sprites/sprites.json
// describes them; "image": "x.png" replaces the drawing by a picture (with the
// same size as the stone). The editor will change these files.
struct SpriteDef {
    std::string id = "default", name = "Pion", emblem = "none", image;
    Rgba ring_top = hex(0xcfd6df), ring_bottom = hex(0x7d8794), emblem_color = hex(0xffffff);
    int tier = 0;            // 0..8: pips under the emblem
    bool halo = false;       // pulsing halo (the human player)
    Rgba halo_color = hex(0x5ee7ff);
    SDL_Texture* texture = nullptr;   // the picture, if any
};

std::vector<SpriteDef> load_sprites(const fs::path& file) {
    std::vector<SpriteDef> list;
    json j = load_json(file);
    if (j.is_object() && j.contains("sprites")) {
        for (const json& e : j["sprites"]) {
            SpriteDef s;
            s.id = jstr(e, "id", s.id);
            s.name = jstr(e, "name", s.id);
            s.emblem = jstr(e, "emblem", s.emblem);
            s.image = jstr(e, "image");
            s.ring_top = jcolor(e, "ring top", s.ring_top);
            s.ring_bottom = jcolor(e, "ring bottom", s.ring_bottom);
            s.emblem_color = jcolor(e, "emblem color", s.emblem_color);
            s.tier = int(jnum(e, "tier", 0));
            s.halo = e.value("halo", false);
            s.halo_color = jcolor(e, "halo color", s.halo_color);
            if (!s.image.empty()) {
                int w, h, n;
                fs::path p = file.parent_path() / s.image;
                if (unsigned char* px = stbi_load(p.string().c_str(), &w, &h, &n, 4)) {
                    s.texture = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
                    SDL_UpdateTexture(s.texture, nullptr, px, w * 4);
                    SDL_SetTextureBlendMode(s.texture, SDL_BLENDMODE_BLEND);
                    SDL_SetTextureScaleMode(s.texture, SDL_SCALEMODE_LINEAR);
                    stbi_image_free(px);
                } else {
                    SDL_Log("sprite image not loaded: %s", p.string().c_str());
                }
            }
            list.push_back(s);
        }
    }
    if (list.empty()) list.push_back(SpriteDef{});
    return list;
}

// The emblems: each one built from discs, segments and convex polygons, in a
// box of "radius" s around c.
void draw_emblem(const std::string& e, V2 c, float s, Rgba col) {
    using namespace draw;
    auto P = [&](float x, float y) { return V2{c.x + x * s, c.y + y * s}; };
    if (e == "star") {
        std::vector<V2> pts;
        for (int i = 0; i < 10; i++) {
            float a = -float(M_PI) / 2 + i * float(M_PI) / 5;
            float r = (i % 2 == 0) ? 0.95f : 0.42f;
            pts.push_back(P(r * std::cos(a), r * std::sin(a) + 0.05f));
        }
        poly_around(P(0, 0.05f), pts, col);
    } else if (e == "dice") {
        round_box({c.x - 0.72f * s, c.y - 0.72f * s, 1.44f * s, 1.44f * s}, 0.3f * s, with_alpha(col, 0.22f));
        round_frame({c.x - 0.72f * s, c.y - 0.72f * s, 1.44f * s, 1.44f * s}, 0.3f * s, std::max(1.5f, 0.12f * s), col);
        for (V2 d : {V2{-0.35f, -0.35f}, V2{0, 0}, V2{0.35f, 0.35f}}) circle(P(d.x, d.y), 0.15f * s, col);
    } else if (e == "branch") {   // a minimax tree: a root, two levels
        float w = std::max(1.5f, 0.12f * s);
        V2 root = P(0, -0.6f), l = P(-0.5f, 0.05f), r = P(0.5f, 0.05f);
        line(root, l, w, col);
        line(root, r, w, col);
        for (V2 leaf : {P(-0.75f, 0.65f), P(-0.25f, 0.65f)}) line(l, leaf, w, col);
        for (V2 leaf : {P(0.25f, 0.65f), P(0.75f, 0.65f)}) line(r, leaf, w, col);
        for (V2 p : {root, l, r}) circle(p, 0.16f * s, col);
        for (V2 p : {P(-0.75f, 0.65f), P(-0.25f, 0.65f), P(0.25f, 0.65f), P(0.75f, 0.65f)}) circle(p, 0.11f * s, col);
    } else if (e == "tree") {     // a search tree that grows unevenly (MCTS)
        float w = std::max(1.2f, 0.09f * s);
        V2 a = P(0, -0.7f), b = P(-0.45f, -0.1f), d = P(0.4f, -0.2f), e2 = P(0.6f, 0.4f), f = P(0.15f, 0.35f), g = P(0.35f, 0.8f), h = P(-0.6f, 0.5f);
        for (auto [p, q] : std::vector<std::pair<V2, V2>>{{a, b}, {a, d}, {d, e2}, {d, f}, {f, g}, {b, h}}) line(p, q, w, col);
        for (V2 p : {a, d, f}) circle(p, 0.15f * s, col);
        for (V2 p : {b, e2, g, h}) circle(p, 0.1f * s, col);
    } else if (e == "hex") {      // a neuron cell (NNUE)
        std::vector<V2> pts;
        for (int i = 0; i < 6; i++) {
            float a = float(M_PI) / 6 + i * float(M_PI) / 3;
            pts.push_back(P(0.8f * std::cos(a), 0.8f * std::sin(a)));
        }
        poly(pts, with_alpha(col, 0.3f));
        for (int i = 0; i < 6; i++) line(pts[i], pts[(i + 1) % 6], std::max(1.5f, 0.1f * s), col);
        circle(c, 0.25f * s, col);
        for (int i = 0; i < 6; i += 2) line(c, pts[i], std::max(1.0f, 0.07f * s), col);
    } else if (e == "prism") {    // hand-crafted features: a cut gem
        std::vector<V2> outer = {P(0, -0.85f), P(0.7f, -0.2f), P(0, 0.9f), P(-0.7f, -0.2f)};
        poly(outer, col);
        poly({P(0, -0.85f), P(0.7f, -0.2f), P(0, -0.2f)}, shade(col, 0.45f));
        poly({P(-0.7f, -0.2f), P(0, -0.2f), P(0, 0.9f)}, shade(col, -0.25f));
    } else if (e == "bolt") {     // the fast, deep minimax
        poly({P(0.15f, -0.95f), P(-0.55f, 0.12f), P(-0.02f, 0.12f), P(0.08f, -0.2f)}, col);
        poly({P(-0.05f, 0.95f), P(0.55f, -0.12f), P(0.02f, -0.12f), P(-0.08f, 0.2f)}, col);
        poly({P(0.08f, -0.2f), P(-0.02f, 0.12f), P(-0.08f, 0.2f), P(0.02f, -0.12f)}, col);
    } else if (e == "eye") {      // the convolutional network "sees" the board
        std::vector<V2> pts;
        for (int i = 0; i <= 20; i++) {
            float t = float(i) / 20, x = -0.9f + 1.8f * t;
            pts.push_back(P(x, -0.55f * std::sin(float(M_PI) * t)));
        }
        for (int i = 1; i < 20; i++) {
            float t = 1 - float(i) / 20, x = -0.9f + 1.8f * t;
            pts.push_back(P(x, 0.55f * std::sin(float(M_PI) * t)));
        }
        poly(pts, col);
        circle(c, 0.36f * s, shade(col, -0.75f));
        circle(c, 0.16f * s, col);
        circle(P(0.1f, -0.12f), 0.07f * s, hex(0xffffff, 220));
    } else if (e == "crown") {    // the champion
        poly({P(-0.8f, 0.55f), P(0.8f, 0.55f), P(0.75f, 0.2f), P(-0.75f, 0.2f)}, col);
        poly({P(-0.8f, 0.3f), P(-0.85f, -0.55f), P(-0.35f, 0.25f)}, col);
        poly({P(-0.35f, 0.3f), P(0, -0.75f), P(0.35f, 0.3f)}, col);
        poly({P(0.35f, 0.25f), P(0.85f, -0.55f), P(0.8f, 0.3f)}, col);
        for (V2 p : {P(-0.85f, -0.6f), P(0, -0.82f), P(0.85f, -0.6f)}) circle(p, 0.12f * s, col);
        circle(P(0, 0.38f), 0.1f * s, shade(col, -0.5f));
    } else if (e == "heart") {
        circle(P(-0.35f, -0.2f), 0.42f * s, col);
        circle(P(0.35f, -0.2f), 0.42f * s, col);
        poly({P(-0.76f, -0.05f), P(0.76f, -0.05f), P(0, 0.85f)}, col);
    } else if (e == "shield") {
        poly({P(-0.7f, -0.7f), P(0.7f, -0.7f), P(0.65f, 0.15f), P(0, 0.9f), P(-0.65f, 0.15f)}, col);
        poly({P(0, -0.7f), P(0.7f, -0.7f), P(0.65f, 0.15f), P(0, 0.9f)}, shade(col, -0.2f));
    } else if (e == "dot") {
        circle(c, 0.4f * s, col);
    }
}

// Draws a piece of `side` with sprite `sp`, centred at c, radius r.
// lift: 0 on the board, up to 1 when picked up (bigger, higher shadow).
// time: seconds, for the halo's pulse.
void draw_piece(const SpriteDef& sp, int side, V2 c, float r, float lift = 0, double time = 0, float alpha = 1) {
    using namespace draw;
    r *= 1 + 0.08f * lift;
    // Shadow.
    float soff = r * (0.10f + 0.25f * lift);
    circle(c + V2{soff * 0.4f, soff}, r * (1.0f + 0.05f * lift), hex(0x000000, uint8_t(110 * alpha)), hex(0x000000, 0), r * 0.35f);
    if (sp.halo) {
        float p = 0.5f + 0.5f * float(std::sin(time * 3.0));
        ring(c, r * (1.18f + 0.05f * p), r * 0.10f, with_alpha(sp.halo_color, (0.35f + 0.35f * p) * alpha));
    }
    if (sp.texture) {
        SDL_FRect dst{c.x - r, c.y - r, 2 * r, 2 * r};
        SDL_SetTextureAlphaModFloat(sp.texture, alpha);
        SDL_RenderTexture(R, sp.texture, nullptr, &dst);
        return;
    }
    // The stone: lit from the top left.
    V2 light = c + V2{-r * 0.35f, -r * 0.4f};
    if (side == BLACK)
        circle_lit(c, r, light, with_alpha(hex(0x5b6170), alpha), with_alpha(hex(0x0a0c11), alpha));
    else
        circle_lit(c, r, light, with_alpha(hex(0xffffff), alpha), with_alpha(hex(0xb9bec8), alpha));
    // The ring (the player's metal).
    ring(c, r * 0.97f, r * 0.17f, with_alpha(sp.ring_top, alpha), with_alpha(sp.ring_bottom, alpha));
    ring(c, r * 0.80f, r * 0.025f, with_alpha(hex(0x000000, 70), alpha));
    // Specular highlight.
    circle(c + V2{-r * 0.32f, -r * 0.38f}, r * 0.22f, with_alpha(hex(0xffffff, side == BLACK ? 70 : 150), alpha), with_alpha(hex(0xffffff, 0), alpha), r * 0.12f);
    // The emblem, with a small shadow; darker on the white stone.
    Rgba ec = side == BLACK ? sp.emblem_color : shade(sp.emblem_color, -0.45f);
    draw_emblem(sp.emblem, c + V2{0, r * 0.04f}, r * 0.42f, with_alpha(hex(0x000000, 90), alpha));
    draw_emblem(sp.emblem, c, r * 0.42f, with_alpha(ec, alpha));
    // Tier pips along the bottom of the ring.
    for (int i = 0; i < sp.tier; i++) {
        float a = float(M_PI) / 2 + (i - (sp.tier - 1) * 0.5f) * 0.2f;
        V2 p = c + V2{std::cos(a), std::sin(a)} * (r * 0.885f);
        circle(p, r * 0.05f, with_alpha(hex(0xffffff, 230), alpha));
    }
}

// ─── @@FX ───────────────────────────────────────────────────────────────────
// Short-lived visual effects. An Effect is updated every frame (dt in seconds)
// and drawn; update returns false when it is over. Future effects (puzzle
// effects, weather, special squares...) only have to implement this interface
// and be added to an FxLayer.
struct Effect {
    virtual ~Effect() = default;
    virtual bool update(float dt) = 0;
    virtual void draw(TextEngine& text) = 0;
};

// Debris thrown up when a hole opens.
struct Debris : Effect {
    struct P { V2 p, v; float life, max, size; Rgba c; };
    std::vector<P> ps;
    Debris(V2 c, float sq, Rgba col, int n, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0, 1);
        for (int i = 0; i < n; i++) {
            float a = U(rng) * 2 * float(M_PI), sp = sq * (0.6f + 1.6f * U(rng));
            P p{c + V2{(U(rng) - 0.5f) * sq * 0.5f, (U(rng) - 0.5f) * sq * 0.5f},
                V2{std::cos(a) * sp, std::sin(a) * sp - sq * 1.2f}, 0, 0.45f + 0.4f * U(rng), sq * (0.03f + 0.05f * U(rng)),
                shade(col, (U(rng) - 0.5f) * 0.5f)};
            ps.push_back(p);
        }
    }
    bool update(float dt) override {
        bool alive = false;
        for (auto& p : ps) {
            p.life += dt;
            p.v.y += 900 * dt * (p.size > 0 ? 1 : 0) * 0.004f * p.size * 60;
            p.p = p.p + p.v * dt;
            alive |= p.life < p.max;
        }
        return alive;
    }
    void draw(TextEngine&) override {
        for (auto& p : ps)
            if (p.life < p.max) draw::circle(p.p, p.size, with_alpha(p.c, 1 - p.life / p.max));
    }
};

// "+1" that floats up and fades.
struct FloatText : Effect {
    V2 p; std::string s; Rgba c; float size, t = 0, dur = 1.0f;
    FloatText(V2 p, std::string s, Rgba c, float size) : p(p), s(std::move(s)), c(c), size(size) {}
    bool update(float dt) override { t += dt; return t < dur; }
    void draw(TextEngine& text) override {
        float k = t / dur;
        V2 q = p + V2{0, -size * 1.6f * ease_out_cubic(k)};
        text.draw_shadowed(Font::Black, size, s, q, with_alpha(c, 1 - k * k), Align::Center, hex(0x000000, uint8_t(120 * (1 - k))), size * 0.06f);
    }
};

// A ring that grows where a piece lands.
struct Ripple : Effect {
    V2 c; float r0, r1, t = 0, dur = 0.45f; Rgba col;
    Ripple(V2 c, float r0, float r1, Rgba col) : c(c), r0(r0), r1(r1), col(col) {}
    bool update(float dt) override { t += dt; return t < dur; }
    void draw(TextEngine&) override {
        float k = ease_out_cubic(t / dur);
        draw::ring(c, lerpf(r0, r1, k), std::max(1.5f, r0 * 0.12f * (1 - k)), with_alpha(col, 1 - k));
    }
};

// Confetti over the window (a won game).
struct Confetti : Effect {
    struct P { V2 p, v; float a, va, size; Rgba c; };
    std::vector<P> ps; float t = 0, dur = 4.0f, floor_y;
    Confetti(Box area, std::mt19937& rng) : floor_y(area.bottom()) {
        std::uniform_real_distribution<float> U(0, 1);
        const Rgba cols[] = {hex(0xf2b45a), hex(0x5ee7ff), hex(0xff6b8b), hex(0x7cf29a), hex(0xb28dff), hex(0xffffff)};
        for (int i = 0; i < 160; i++)
            ps.push_back({{area.x + U(rng) * area.w, area.y - U(rng) * area.h * 0.5f},
                          {(U(rng) - 0.5f) * 120, 120 + U(rng) * 220}, U(rng) * 6.28f, (U(rng) - 0.5f) * 10,
                          area.h * (0.006f + 0.006f * U(rng)), cols[i % 6]});
    }
    bool update(float dt) override {
        t += dt;
        for (auto& p : ps) {
            p.p = p.p + p.v * dt;
            p.a += p.va * dt;
            p.v.x += std::sin(t * 2 + p.a) * 40 * dt;
        }
        return t < dur;
    }
    void draw(TextEngine&) override {
        float fade = clamp01((dur - t) / 0.8f);
        for (auto& p : ps) {
            V2 d{std::cos(p.a) * p.size, std::sin(p.a) * p.size * 0.5f};
            V2 n = perp(normalized(d)) * (p.size * 0.6f);
            draw::poly({p.p + d + n, p.p - d + n, p.p - d - n, p.p + d - n}, with_alpha(p.c, fade), 0.8f);
        }
    }
};

class FxLayer {
public:
    void add(std::unique_ptr<Effect> e) { list_.push_back(std::move(e)); }
    void update(float dt) {
        for (size_t i = 0; i < list_.size();)
            if (!list_[i]->update(dt)) list_.erase(list_.begin() + i);
            else i++;
    }
    void draw(TextEngine& text) {
        for (auto& e : list_) e->draw(text);
    }
    void clear() { list_.clear(); }
    bool empty() const { return list_.empty(); }

private:
    std::vector<std::unique_ptr<Effect>> list_;
};

// ─── @@UI ───────────────────────────────────────────────────────────────────
// Immediate-mode widgets: there is no widget object; every frame, the scene
// calls ui.button(...) where the button must be, and the call both draws it and
// tells whether it was clicked. The only memory is a small per-id animation
// value (hover, press), kept in a map.
struct Input {
    V2 mouse{-1, -1};
    bool down = false;       // left button held
    bool pressed = false;    // went down this frame
    bool released = false;   // went up this frame
    float wheel = 0;
    std::vector<SDL_Keycode> keys;   // keys pressed this frame
    bool key(SDL_Keycode k) const { return std::find(keys.begin(), keys.end(), k) != keys.end(); }
};

// The colours of the interface (independent of the board theme).
namespace pal {
const Rgba panel = hex(0x1a202b, 235), panel_border = hex(0xffffff, 22), text = hex(0xeef2f7), dim = hex(0x9aa6b6),
           faint = hex(0x5c6878), accent = hex(0xf2b45a), accent2 = hex(0x5ee7ff), good = hex(0x4ade80), bad = hex(0xff6b6b),
           button = hex(0x2a3342), button_hi = hex(0x3a4659);
}

enum class Style { Normal, Primary, Ghost, Danger };

class Ui {
public:
    Ui(TextEngine& t, Input& in) : text(t), in(in) {}
    TextEngine& text;
    Input& in;
    float s = 1;              // scale (design → pixels)
    float dt = 0;
    bool hovering = false;    // a clickable widget is under the mouse (hand cursor)
    bool blocked = false;     // a modal box is open: widgets outside it ignore the mouse

    float anim(const std::string& id, bool target, float speed = 14) {
        float& v = anims_[id];
        v += ((target ? 1.0f : 0.0f) - v) * std::min(1.0f, speed * dt);
        return v;
    }
    bool hover(Box b) const { return !blocked && b.contains(in.mouse); }

    bool button(const std::string& id, Box b, const std::string& label, Style st = Style::Normal, bool enabled = true, float font = 20) {
        bool h = enabled && hover(b);
        if (h) hovering = true;
        float k = anim(id, h), pr = anim(id + "#p", h && in.down, 30);
        Rgba top, bottom, fg = pal::text;
        switch (st) {
        case Style::Primary: top = shade(pal::accent, 0.15f + 0.1f * k); bottom = shade(pal::accent, -0.2f + 0.1f * k); fg = hex(0x1a1206); break;
        case Style::Ghost: top = with_alpha(hex(0xffffff), 0.04f + 0.08f * k); bottom = top; break;
        case Style::Danger: top = shade(pal::bad, -0.2f + 0.1f * k); bottom = shade(pal::bad, -0.45f + 0.1f * k); break;
        default: top = mix(pal::button, pal::button_hi, k); bottom = shade(mix(pal::button, pal::button_hi, k), -0.15f); break;
        }
        if (!enabled) { top = with_alpha(top, 0.4f); bottom = with_alpha(bottom, 0.4f); fg = with_alpha(fg, 0.45f); }
        Box bb = b;
        bb.y += 2 * s * pr - s * k;
        draw::shadow({bb.x, bb.y + 3 * s, bb.w, bb.h}, 10 * s, 10 * s, hex(0x000000, 70));
        draw::round_box(bb, 10 * s, top, bottom);
        if (st != Style::Primary) draw::round_frame(bb, 10 * s, 1, with_alpha(hex(0xffffff), 0.06f + 0.1f * k));
        float fs = font * s;
        text.draw(Font::Bold, fs, label, {bb.center().x, bb.center().y - text.line_height(Font::Bold, fs) * 0.42f}, fg, Align::Center);
        return h && in.released;
    }
    bool toggle(const std::string& id, Box b, const std::string& label, bool& v, float font = 18) {
        bool h = hover(b);
        if (h) hovering = true;
        float k = anim(id, v), hv = anim(id + "#h", h);
        float th = b.h * 0.62f, tw = th * 1.8f;
        Box track{b.right() - tw, b.center().y - th / 2, tw, th};
        draw::round_box(track, th / 2, mix(hex(0x39424f), pal::accent, k));
        draw::circle({track.x + th / 2 + (tw - th) * k, track.center().y}, th * 0.4f, hex(0xffffff));
        text.draw(Font::Regular, font * s, label, {b.x, b.center().y - text.line_height(Font::Regular, font * s) * 0.45f},
                  mix(pal::dim, pal::text, std::max(k, hv)));
        if (h && in.released) { v = !v; return true; }
        return false;
    }
    // A row of choices; returns the selected index.
    int segmented(const std::string& id, Box b, const std::vector<std::string>& opts, int sel, float font = 17) {
        draw::round_box(b, b.h / 2, hex(0x0f141c, 200));
        float w = b.w / opts.size();
        float target = sel * w;
        float& x = anims_[id + "#x"];
        if (x == 0 && sel != 0) x = target;
        x += (target - x) * std::min(1.0f, 16 * dt);
        draw::round_box({b.x + x + 3, b.y + 3, w - 6, b.h - 6}, (b.h - 6) / 2, shade(pal::accent, 0.1f), shade(pal::accent, -0.15f));
        for (size_t i = 0; i < opts.size(); i++) {
            Box c{b.x + i * w, b.y, w, b.h};
            bool h = hover(c);
            if (h) hovering = true;
            Rgba fg = int(i) == sel ? hex(0x1a1206) : (h ? pal::text : pal::dim);
            text.draw(Font::Bold, font * s, opts[i], {c.center().x, c.center().y - text.line_height(Font::Bold, font * s) * 0.42f}, fg, Align::Center);
            if (h && in.released) sel = int(i);
        }
        return sel;
    }
    void panel(Box b, float radius = 16, Rgba col = pal::panel) {
        draw::shadow({b.x, b.y + 8 * s, b.w, b.h}, radius * s, 28 * s, hex(0x000000, 90));
        draw::round_box(b, radius * s, col, shade(col, -0.12f));
        draw::round_frame(b, radius * s, 1, pal::panel_border);
    }
    // Tabs; returns the selected index.
    int tabs(const std::string& id, Box b, const std::vector<std::string>& names, int sel) {
        float w = b.w / names.size();
        for (size_t i = 0; i < names.size(); i++) {
            Box c{b.x + i * w, b.y, w, b.h};
            bool h = hover(c);
            if (h) hovering = true;
            float k = anim(id + std::to_string(i), int(i) == sel);
            text.draw(Font::Bold, 17 * s, names[i], {c.center().x, c.y + (b.h - text.line_height(Font::Bold, 17 * s)) * 0.5f},
                      mix(h ? pal::text : pal::dim, pal::accent, k), Align::Center);
            draw::round_box({c.x + w * 0.2f, c.bottom() - 3 * s, w * 0.6f, 3 * s}, 1.5f * s, with_alpha(pal::accent, k));
            if (h && in.released) sel = int(i);
        }
        draw::rect({b.x, b.bottom(), b.w, 1}, pal::panel_border);
        return sel;
    }
    void forget(const std::string& prefix) {
        for (auto it = anims_.begin(); it != anims_.end();)
            if (it->first.rfind(prefix, 0) == 0) it = anims_.erase(it);
            else ++it;
    }

private:
    std::unordered_map<std::string, float> anims_;
};

// ─── @@BOARD ────────────────────────────────────────────────────────────────
// The board on screen: where the squares are (rank 8 at the top, or at the
// bottom when flipped), how the pieces, holes and highlights are drawn, the
// animation of the last move, and the mouse (click a piece then a square, or
// drag it).
class BoardView {
public:
    Box area;                 // the 8 × 8 squares
    bool flipped = false;
    int selected = -1;        // the human's selected piece
    int drag_from = -1;       // a piece being dragged
    bool dragging = false;
    V2 press_pos;
    struct MoveAnim { bool active = false; Move m = Move::none(); int side = BLACK; double t0 = 0, dur = 0.34; } anim;
    std::array<double, 64> hole_since{};   // time each hole appeared (for its animation)
    Move last = Move::none();
    int hover_sq = -1;

    float sq() const { return area.w / 8; }
    Box square(int s) const {
        int f = sq_file(s), r = sq_rank(s);
        int col = flipped ? 7 - f : f, row = flipped ? r : 7 - r;
        return {area.x + col * sq(), area.y + row * sq(), sq(), sq()};
    }
    V2 center(int s) const { return square(s).center(); }
    int at(V2 p) const {
        if (!area.contains(p)) return -1;
        int col = std::clamp(int((p.x - area.x) / sq()), 0, 7), row = std::clamp(int((p.y - area.y) / sq()), 0, 7);
        int f = flipped ? 7 - col : col, r = flipped ? row : 7 - row;
        return r * 8 + f;
    }
    bool animating(double now) const { return anim.active && now < anim.t0 + anim.dur; }
    void reset() {
        anim = {};
        hole_since.fill(-100);
        last = Move::none();
        selected = drag_from = -1;
        dragging = false;
    }
    // A move was played (match already updated): animate it.
    void on_move(Move m, int side, double now, FxLayer& fx, const Theme& theme, std::mt19937& rng) {
        last = m;
        selected = -1;
        if (m == Move::none()) return;
        anim = {true, m, side, now, 0.34};
        hole_since[m.from_sq()] = now + 0.08;
        fx.add(std::make_unique<Debris>(center(m.from_sq()), sq(), theme.dark, 18, rng));
    }

    // The human's input; returns a move when one is made.
    std::optional<Move> input(const Input& in, const Yolah& pos, bool human_turn, bool blocked) {
        hover_sq = blocked ? -1 : at(in.mouse);
        if (!human_turn || blocked) {
            if (!human_turn) selected = drag_from = -1, dragging = false;
            return std::nullopt;
        }
        const int me = pos.current_player();
        auto own = [&](int s) { return s >= 0 && (pos.bitboard(me) >> s & 1); };
        std::optional<Move> result;
        if (in.pressed && hover_sq >= 0) {
            if (own(hover_sq)) {
                selected = drag_from = hover_sq;
                press_pos = in.mouse;
            } else if (selected >= 0) {
                Move m{Square(selected), Square(hover_sq)};
                if (is_legal(pos, m)) result = m;
                else selected = -1;
            }
        }
        if (in.down && drag_from >= 0 && length(in.mouse - press_pos) > sq() * 0.15f) dragging = true;
        if (in.released && drag_from >= 0) {
            if (dragging && hover_sq >= 0 && hover_sq != drag_from) {
                Move m{Square(drag_from), Square(hover_sq)};
                if (is_legal(pos, m)) result = m;
            }
            dragging = false;
            drag_from = -1;
        }
        return result;
    }

    struct Look {
        const SpriteDef* sprite[2] = {nullptr, nullptr};
        bool coords = true;
        std::optional<Move> hint;
        std::string hint_label;
        std::vector<Move> pv;     // drawn as faint numbered arrows (analysis tab)
        bool human_turn = false;
    };

    void draw(TextEngine& text, BoardTextures& textures, const Theme& t, const Yolah& pos, const Look& look, double now, float s) {
        using namespace draw;
        const float q = sq();
        // Frame and shadow.
        Box frame = area.inset(-26 * s);
        shadow({frame.x, frame.y + 14 * s, frame.w, frame.h}, 22 * s, 46 * s, hex(0x000000, 150));
        round_box(frame, 22 * s, t.frame_top, t.frame_bottom);
        round_frame(frame.inset(6 * s), 17 * s, std::max(1.0f, 1.5f * s), with_alpha(t.frame_line, 0.55f));
        round_frame(area.inset(-3 * s), 4 * s, std::max(1.0f, 2 * s), with_alpha(hex(0x000000), 0.35f));
        // The squares (flipped boards rotate the texture by 180°).
        SDL_Texture* tex = textures.get(t, int(area.w));
        SDL_FRect dst = area.frect();
        SDL_RenderTextureRotated(R, tex, nullptr, &dst, flipped ? 180.0 : 0.0, nullptr, SDL_FLIP_NONE);
        // Coordinates in the frame.
        if (look.coords) {
            float fs = 15 * s;
            for (int i = 0; i < 8; i++) {
                int f = flipped ? 7 - i : i;
                text.draw(Font::Bold, fs, std::string(1, char('a' + f)), {area.x + (i + 0.5f) * q, area.bottom() + 5 * s}, with_alpha(t.coords, 0.85f), Align::Center);
                int r = flipped ? i : 7 - i;
                text.draw(Font::Bold, fs, std::string(1, char('1' + r)), {area.x - 14 * s, area.y + (i + 0.5f) * q - fs * 0.6f}, with_alpha(t.coords, 0.85f), Align::Center);
            }
        }
        // Last move.
        if (last != Move::none()) {
            for (int s2 : {int(last.from_sq()), int(last.to_sq())}) rect(square(s2), t.last_move);
        }
        // Holes.
        uint64_t holes = pos.empty_bitboard();
        for (int s2 = 0; s2 < 64; s2++) {
            if (!(holes >> s2 & 1)) continue;
            float k = ease_out_back(float((now - hole_since[s2]) / 0.35));
            if (now < hole_since[s2]) continue;
            draw_hole(square(s2), t, k, s);
        }
        // Selection and legal moves.
        const int me = pos.current_player();
        int sel = drag_from >= 0 ? drag_from : selected;
        if (sel >= 0 && look.human_turn) {
            rect(square(sel), t.select);
            for (Move m : legal_moves(pos)) {
                if (m == Move::none() || m.from_sq() != sel) continue;
                V2 c = center(m.to_sq());
                if (hover_sq == m.to_sq()) ring(c, q * 0.44f, q * 0.07f, with_alpha(t.legal, 1.6f));
                else circle(c, q * 0.15f, t.legal);
            }
        } else if (look.human_turn && hover_sq >= 0 && (pos.bitboard(me) >> hover_sq & 1)) {
            rect(square(hover_sq), with_alpha(t.select, 0.45f));
        }
        // Pieces (the one moving or dragged is drawn last, above everything).
        for (int side : {BLACK, WHITE}) {
            uint64_t bb = pos.bitboard(side);
            for (int s2 = 0; s2 < 64; s2++) {
                if (!(bb >> s2 & 1)) continue;
                if (anim.active && now < anim.t0 + anim.dur && s2 == anim.m.to_sq()) continue;
                if (dragging && s2 == drag_from) continue;
                draw_piece(*look.sprite[side], side, center(s2), q * 0.4f, 0, now);
            }
        }
        // Principal variation: numbered faint arrows.
        for (size_t i = 0; i < look.pv.size() && i < 6; i++) {
            Move m = look.pv[i];
            if (m == Move::none()) continue;
            Rgba col = i % 2 == 0 ? hex(0x5ee7ff, uint8_t(170 - i * 22)) : hex(0xff9ec0, uint8_t(170 - i * 22));
            V2 a = center(m.from_sq()), b = center(m.to_sq());
            arrow(a, b - normalized(b - a) * (q * 0.25f), q * 0.07f, col);
            circle(b, q * 0.14f, with_alpha(col, 1.2f));
            text.draw(Font::Black, q * 0.18f, std::to_string(i + 1), {b.x, b.y - q * 0.12f}, hex(0x10141c), Align::Center);
        }
        // Hint arrow.
        if (look.hint && *look.hint != Move::none()) {
            V2 a = center(look.hint->from_sq()), b = center(look.hint->to_sq());
            float p = 0.75f + 0.25f * float(std::sin(now * 4));
            arrow(a, b - normalized(b - a) * (q * 0.18f), q * 0.11f, with_alpha(t.hint, p));
            if (!look.hint_label.empty()) {
                float fs = q * 0.2f, w = text.width(Font::Black, fs, look.hint_label) + q * 0.2f;
                V2 mid = a + (b - a) * 0.5f;
                Box tag{mid.x - w / 2, mid.y - q * 0.42f, w, fs * 1.45f};
                round_box(tag, tag.h / 2, with_alpha(t.hint, 0.95f));
                text.draw(Font::Black, fs, look.hint_label, {tag.center().x, tag.y + fs * 0.12f}, hex(0x0b1a10), Align::Center);
            }
        }
        // The moving piece.
        if (anim.active && now < anim.t0 + anim.dur) {
            float k = ease_in_out_cubic(float((now - anim.t0) / anim.dur));
            V2 a = center(anim.m.from_sq()), b = center(anim.m.to_sq());
            V2 p = a + (b - a) * k;
            float lift = std::sin(k * float(M_PI));
            draw_piece(*look.sprite[anim.side], anim.side, p + V2{0, -lift * q * 0.12f}, q * 0.4f, lift, now);
        }
        // The dragged piece follows the mouse.
        if (dragging && drag_from >= 0 && (pos.bitboard(me) >> drag_from & 1))
            draw_piece(*look.sprite[me], me, drag_pos, q * 0.4f, 1, now);
    }
    V2 drag_pos;

private:
    void draw_hole(Box b, const Theme& t, float k, float s) {
        using namespace draw;
        Box h = b.inset(b.w * 0.07f);
        V2 c = h.center();
        h = {c.x - h.w * 0.5f * k, c.y - h.h * 0.5f * k, h.w * k, h.h * k};
        float r = h.w * 0.22f;
        round_box(h.inset(-1.5f * s), r + 1.5f * s, with_alpha(t.hole_rim, 0.9f), with_alpha(shade(t.hole_rim, 0.25f), 0.9f));
        round_box(h, r, shade(t.hole, -0.3f), shade(t.hole, 0.12f));
        // Inner shadow at the top: the hole is deep.
        round_box({h.x, h.y, h.w, h.h * 0.45f}, r, hex(0x000000, 110), hex(0x000000, 0));
    }
};

// ─── @@APP ──────────────────────────────────────────────────────────────────
enum class Scene { Menu, Setup, Game, Themes, Rules, Puzzles, Editor };

// Saved in the preferences folder (settings.json).
struct Settings {
    std::string theme = "bois";
    bool show_eval = true, show_hint = false, show_coords = true, show_pv = false;
    int time_index = 1;                       // index in TIMES
    std::string black = "human", white = "dev";
    json to_json() const {
        return {{"theme", theme}, {"show eval", show_eval}, {"show hint", show_hint}, {"show coordinates", show_coords},
                {"show pv", show_pv}, {"time index", time_index}, {"black", black}, {"white", white}};
    }
    void from_json(const json& j) {
        if (!j.is_object()) return;
        theme = j.value("theme", theme);
        show_eval = j.value("show eval", show_eval);
        show_hint = j.value("show hint", show_hint);
        show_coords = j.value("show coordinates", show_coords);
        show_pv = j.value("show pv", show_pv);
        time_index = j.value("time index", time_index);
        black = j.value("black", black);
        white = j.value("white", white);
    }
};
const std::vector<double> TIMES = {0.5, 1.0, 2.0, 5.0};
const std::vector<std::string> TIME_LABELS = {"0,5 s", "1 s", "2 s", "5 s"};

// One side of a game: a human or an AI of the roster.
struct Slot {
    std::string who = "human";            // "human" or an AI id
    const AiSpec* spec = nullptr;
    std::unique_ptr<AiAgent> agent;
    double last_seconds = 0;
    std::string error;
    bool human() const { return spec == nullptr; }
};

struct App {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    int w = 1600, h = 900;
    float s = 1;                          // design → pixels
    double time = 0, dt = 0;
    std::mt19937 rng{12345};

    TextEngine text;
    Input in;
    Ui ui{text, in};
    BoardTextures board_textures;
    FxLayer fx;
    std::vector<Theme> themes;
    std::vector<SpriteDef> sprites;
    std::vector<AiSpec> roster;
    Settings settings;

    Scene scene = Scene::Menu;
    double scene_t0 = 0;                  // for the fade-in
    SDL_Cursor* cursor_arrow = nullptr;
    SDL_Cursor* cursor_hand = nullptr;
    std::string toast;
    double toast_t = -10;

    // Screenshot mode.
    std::string screenshot;
    int frames_left = -1;

    // The game scene.
    struct GameState {
        Match match;
        Slot slot[2];
        BoardView board;
        Analyzer analyzer;
        uint64_t token = 1;               // changes with every position
        uint64_t analysed = 0;
        double pass_at = -1;
        bool paused = false;
        bool over = false;
        double over_t = 0;
        int tab = 0;
        float move_scroll = 0;
        float eval_shown = 0.5f;          // the bar's animated share
        double think_t0 = 0;
    } game;

    // The setup scene.
    struct SetupState {
        bool ai_vs_ai = false;
        float scroll[2] = {0, 0};
    } setup;

    // ── helpers ──
    const Theme& theme() const {
        for (auto& t : themes)
            if (t.id == settings.theme) return t;
        return themes.front();
    }
    const SpriteDef& sprite(const std::string& id) const {
        for (auto& sp : sprites)
            if (sp.id == id) return sp;
        return sprites.front();
    }
    const AiSpec* ai(const std::string& id) const {
        for (auto& a : roster)
            if (a.id == id) return &a;
        return nullptr;
    }
    const SpriteDef& slot_sprite(const Slot& sl) const { return sl.human() ? sprite("human") : sprite(sl.spec->sprite); }
    std::string slot_name(const Slot& sl) const { return sl.human() ? "Vous" : sl.spec->name; }
    void notify(const std::string& msg) {
        toast = msg;
        toast_t = time;
    }
    void go(Scene sc) {
        if (scene == Scene::Game && sc != Scene::Game) leave_game();
        scene = sc;
        scene_t0 = time;
        fx.clear();
    }
    void save_settings() {
        if (screenshot.empty()) write_file(g_paths.prefs / "settings.json", settings.to_json().dump(2));
    }

    // ── game management (@@SCENE_GAME uses these) ──
    void start_game(const Yolah& start = Yolah()) {
        leave_game();
        auto& g = game;
        g.match.reset(start);
        g.board.reset();
        g.board.flipped = false;
        g.token++;
        g.analysed = 0;
        g.over = false;
        g.paused = false;
        g.pass_at = -1;
        g.move_scroll = 0;
        g.eval_shown = 0.5f;
        const std::string who[2] = {settings.black, settings.white};
        for (int side : {BLACK, WHITE}) {
            Slot& sl = g.slot[side];
            sl = Slot{};
            sl.who = who[side];
            sl.spec = ai(who[side]);
            if (sl.spec && sl.spec->available)
                sl.agent = std::make_unique<AiAgent>(game_config(*sl.spec, TIMES[settings.time_index]));
            else
                sl.spec = nullptr;   // unknown or unavailable AI: a human plays it
        }
        // A human playing White only: White at the bottom.
        g.board.flipped = g.slot[WHITE].human() && !g.slot[BLACK].human();
        go(Scene::Game);
    }
    void leave_game() {
        for (auto& sl : game.slot)
            if (sl.agent) sl.agent->cancel();
        game.analyzer.stop();
    }
    void play_move(Move m) {
        auto& g = game;
        const int side = g.match.to_move();
        g.match.play(m);
        g.board.on_move(m, side, time, fx, theme(), rng);
        if (m != Move::none()) {
            float q = g.board.sq();
            V2 c = g.board.center(m.to_sq());
            fx.add(std::make_unique<FloatText>(c + V2{0, -q * 0.3f}, "+1", side == BLACK ? hex(0xb6c2d6) : hex(0xffffff), q * 0.32f));
            fx.add(std::make_unique<Ripple>(c, q * 0.35f, q * 0.62f, theme().accent));
        } else {
            notify(std::string(side_name(side)) + " ne peuvent plus bouger : ils passent.");
        }
        g.token++;
        g.move_scroll = 1e9f;   // scroll the move list to its end
    }
    void undo_move() {
        auto& g = game;
        if (g.match.history.empty()) return;
        for (auto& sl : g.slot)
            if (sl.agent) sl.agent->cancel();
        // Against an AI: back to the human's previous turn.
        bool any_human = g.slot[0].human() || g.slot[1].human();
        do {
            g.match.undo();
        } while (any_human && !g.match.history.empty() && !g.slot[g.match.to_move()].human());
        g.board.last = g.match.history.empty() ? Move::none() : g.match.history.back();
        g.board.anim = {};
        g.board.selected = -1;
        g.over = false;
        g.token++;
    }
};
App* g_app = nullptr;

// Drawn behind every scene: the theme's gradient, a vignette, faint shapes.
void draw_background(App& a, bool decorated = true) {
    const Theme& t = a.theme();
    draw::gradient_rect({0, 0, float(a.w), float(a.h)}, t.bg_top, t.bg_bottom);
    if (decorated) {
        for (int i = 0; i < 5; i++) {
            float x = a.w * (0.12f + 0.2f * i) + std::sin(float(a.time) * 0.11f + i) * 40 * a.s;
            float y = a.h * (0.3f + 0.35f * ((i * 37) % 3) / 2.0f) + std::cos(float(a.time) * 0.09f + i * 2) * 30 * a.s;
            draw::circle({x, y}, 260 * a.s, with_alpha(t.accent, 0.035f), with_alpha(t.accent, 0), 0);
        }
    }
    draw::vignette({0, 0, float(a.w), float(a.h)}, a.s, 80);
}

void draw_title(App& a, const std::string& title, const std::string& subtitle = "") {
    float s = a.s;
    a.text.draw_shadowed(Font::Title, 44 * s, title, {60 * s, 34 * s}, pal::text, Align::Left, hex(0x000000, 120), 3 * s);
    if (!subtitle.empty()) a.text.draw(Font::Regular, 18 * s, subtitle, {62 * s, 92 * s}, pal::dim);
}
bool back_button(App& a) {
    float s = a.s;
    return a.ui.button("back", {a.w - 200 * s, 40 * s, 150 * s, 50 * s}, "\xE2\x86\x90  Retour", Style::Ghost) || a.in.key(SDLK_ESCAPE);
}

// ─── @@SCENE_MENU ───────────────────────────────────────────────────────────
void scene_menu(App& a) {
    const float s = a.s;
    draw_background(a);
    // Pieces drifting around the title: the sprites of the roster.
    const int n = int(a.roster.size()) + 1;
    for (int i = 0; i < n; i++) {
        float ang = float(a.time) * 0.12f + i * 2 * float(M_PI) / n;
        float rx = a.w * 0.40f, ry = a.h * 0.36f;
        V2 c{a.w * 0.5f + std::cos(ang) * rx, a.h * 0.55f + std::sin(ang) * ry};
        // Faded out when passing behind the title and the buttons.
        float fade = smoothstep(260 * s, 420 * s, std::abs(c.x - a.w * 0.5f));
        if (fade <= 0.01f) continue;
        const SpriteDef& sp = i == 0 ? a.sprite("human") : a.sprite(a.roster[i - 1].sprite);
        draw_piece(sp, i % 2 ? WHITE : BLACK, c, (36 + 6 * std::sin(float(a.time) + i)) * s, 0.4f, a.time, 0.9f * fade);
    }
    // Title on a soft light.
    V2 tp{a.w * 0.5f, 75 * s};
    draw::circle({a.w * 0.5f, 150 * s}, 330 * s, with_alpha(a.theme().accent, 0.16f), with_alpha(a.theme().accent, 0), 0);
    a.text.draw_shadowed(Font::Title, 140 * s, "YOLAH", tp, pal::text, Align::Center, hex(0x000000, 150), 6 * s);
    a.text.draw(Font::Regular, 21 * s, "Quatre pions, soixante-quatre cases : chaque pas creuse un trou.", {a.w * 0.5f, 250 * s}, pal::dim, Align::Center);

    struct Item { const char* label; std::function<void()> act; Style st; };
    std::vector<Item> items = {
        {"Jouer contre une IA", [&] { a.setup.ai_vs_ai = false; a.go(Scene::Setup); }, Style::Primary},
        {"Regarder deux IA s'affronter", [&] { a.setup.ai_vs_ai = true; a.go(Scene::Setup); }, Style::Normal},
        {"R\xC3\xA8gles du jeu", [&] { a.go(Scene::Rules); }, Style::Normal},
        {"Puzzles et qu\xC3\xAA" "tes", [&] { a.go(Scene::Puzzles); }, Style::Normal},
        {"\xC3\x89" "diteur de puzzles", [&] { a.go(Scene::Editor); }, Style::Normal},
        {"Th\xC3\xA8mes et sprites", [&] { a.go(Scene::Themes); }, Style::Normal},
        {"Quitter", [&] { SDL_Event e{}; e.type = SDL_EVENT_QUIT; SDL_PushEvent(&e); }, Style::Ghost},
    };
    float bw = 440 * s, bh = 58 * s, gap = 14 * s, y = 310 * s;
    for (size_t i = 0; i < items.size(); i++) {
        float appear = ease_out_cubic(float((a.time - a.scene_t0) * 3 - i * 0.12f));
        Box b{a.w * 0.5f - bw / 2, y + i * (bh + gap) + (1 - appear) * 30 * s, bw, bh};
        if (a.ui.button(std::string("menu") + std::to_string(i), b, items[i].label, items[i].st, true, 21)) items[i].act();
    }
    a.text.draw(Font::Regular, 14 * s, strf("Yolah %s \xE2\x80\xA2 %zu IA \xE2\x80\xA2 th\xC3\xA8me %s", APP_VERSION, a.roster.size(), a.theme().name.c_str()),
                {a.w * 0.5f, a.h - 34 * s}, pal::faint, Align::Center);
}

// ─── @@SCENE_SETUP ──────────────────────────────────────────────────────────
void scene_setup(App& a) {
    const float s = a.s;
    draw_background(a);
    draw_title(a, a.setup.ai_vs_ai ? "Deux IA face \xC3\xA0 face" : "Nouvelle partie",
               "Choisissez qui joue les Noirs (ils commencent) et qui joue les Blancs. Les IA sont class\xC3\xA9" "es de la plus faible \xC3\xA0 la plus forte.");
    if (back_button(a)) { a.go(Scene::Menu); return; }

    struct Entry { std::string id, name, sub, sprite; bool available; std::string reason, description; };
    std::vector<Entry> entries;
    entries.push_back({"human", "Humain", "Vous jouez \xC3\xA0 la souris", "human", true, "", "Cliquez sur un de vos pions puis sur la case d'arriv\xC3\xA9" "e, ou faites-le glisser."});
    for (auto& ai : a.roster)
        entries.push_back({ai.id, ai.name, strf("Elo \xE2\x89\x88 %d \xE2\x80\xA2 %s", ai.elo, ai.family.c_str()), ai.sprite, ai.available, ai.reason, ai.description});

    const float top = 130 * s, bottom_bar = 150 * s, desc_h = 70 * s;
    const float col_w = std::min(560 * s, (a.w - 180 * s) / 2), gap = 40 * s;
    const float x0 = a.w * 0.5f - col_w - gap / 2;
    const float avail = a.h - top - bottom_bar - desc_h - 60 * s;
    const float row_h = std::min(74 * s, avail / entries.size());
    std::string* choice[2] = {&a.settings.black, &a.settings.white};
    std::string hovered_desc;
    for (int side : {BLACK, WHITE}) {
        Box col{x0 + side * (col_w + gap), top, col_w, row_h * entries.size() + 64 * s};
        a.ui.panel(col);
        a.text.draw(Font::Black, 22 * s, side == BLACK ? "Noirs" : "Blancs", {col.x + 22 * s, col.y + 16 * s}, pal::text);
        a.text.draw(Font::Regular, 15 * s, side == BLACK ? "jouent en premier" : "jouent en second", {col.x + 110 * s, col.y + 22 * s}, pal::dim);
        for (size_t i = 0; i < entries.size(); i++) {
            const Entry& e = entries[i];
            Box r{col.x + 10 * s, col.y + 56 * s + i * row_h, col.w - 20 * s, row_h - 4 * s};
            bool sel = *choice[side] == e.id;
            bool h = a.ui.hover(r) && e.available;
            if (a.ui.hover(r)) hovered_desc = e.available ? e.description : e.reason;
            if (h) a.ui.hovering = true;
            float k = a.ui.anim("setup" + std::to_string(side) + e.id, h), ks = a.ui.anim("setsel" + std::to_string(side) + e.id, sel);
            draw::round_box(r, 12 * s, with_alpha(mix(hex(0xffffff), a.theme().accent, ks), 0.03f + 0.06f * k + 0.12f * ks));
            if (ks > 0.01f) draw::round_frame(r, 12 * s, 2 * s, with_alpha(a.theme().accent, ks));
            float alpha = e.available ? 1.0f : 0.35f;
            float pr = std::min(row_h * 0.36f, 27 * s);
            draw_piece(a.sprite(e.sprite), side, {r.x + 16 * s + pr, r.center().y}, pr, 0.2f * k, a.time, alpha);
            float tx = r.x + 30 * s + 2 * pr;
            a.text.draw(Font::Bold, 19 * s, e.name, {tx, r.center().y - 22 * s}, with_alpha(pal::text, alpha));
            a.text.draw(Font::Regular, 14 * s, e.available ? e.sub : "indisponible \xE2\x80\x94 survolez pour savoir pourquoi",
                        {tx, r.center().y + 2 * s}, with_alpha(e.available ? pal::dim : pal::bad, std::max(alpha, 0.7f)));
            if (e.id != "human" && e.available) {   // strength gauge
                float frac = float(i) / (entries.size() - 1);
                Box gb{r.right() - 96 * s, r.center().y - 4 * s, 80 * s, 8 * s};
                draw::round_box(gb, 4 * s, hex(0xffffff, 25));
                draw::round_box({gb.x, gb.y, gb.w * frac, gb.h}, 4 * s, shade(a.theme().accent, -0.2f), a.theme().accent);
            }
            if (h && a.in.released) *choice[side] = e.id;
        }
    }
    // Description of the hovered player.
    Box db{x0, top + row_h * entries.size() + 80 * s, 2 * col_w + gap, desc_h};
    if (!hovered_desc.empty()) a.text.paragraph(Font::Regular, 16 * s, hovered_desc, db, pal::dim);

    // Bottom bar: options and start.
    Box bar{x0, a.h - bottom_bar + 10 * s, 2 * col_w + gap, bottom_bar - 40 * s};
    a.ui.panel(bar);
    a.text.draw(Font::Bold, 16 * s, "Temps de r\xC3\xA9" "flexion des IA", {bar.x + 24 * s, bar.y + 14 * s}, pal::dim);
    a.settings.time_index = a.ui.segmented("time", {bar.x + 24 * s, bar.y + 42 * s, 320 * s, 40 * s}, TIME_LABELS, a.settings.time_index);
    a.ui.toggle("t_eval", {bar.x + 380 * s, bar.y + 18 * s, 270 * s, 30 * s}, "Barre d'\xC3\xA9valuation", a.settings.show_eval);
    a.ui.toggle("t_hint", {bar.x + 380 * s, bar.y + 56 * s, 270 * s, 30 * s}, "Indices (meilleur coup)", a.settings.show_hint);
    if (a.ui.button("swap", {bar.right() - 400 * s, bar.y + 26 * s, 160 * s, 50 * s}, "\xC3\x89" "changer"))
        std::swap(a.settings.black, a.settings.white);
    if (a.ui.button("start", {bar.right() - 224 * s, bar.y + 26 * s, 200 * s, 50 * s}, "Commencer", Style::Primary) || a.in.key(SDLK_RETURN)) {
        a.save_settings();
        a.start_game();
    }
}

// ─── @@SCENE_GAME ───────────────────────────────────────────────────────────
// Each frame: (1) the logic — whose turn, AI searches, passes, end of game,
// analysis; (2) the layout; (3) the drawing: evaluation bar, board, side
// panel (players, tabs, buttons), end-of-game box.
void game_logic(App& a) {
    auto& g = a.game;
    const bool anim = g.board.animating(a.time);
    if (g.match.over()) {
        if (!g.over && !anim) {
            g.over = true;
            g.over_t = a.time;
            int win = g.match.winner();
            bool human_won = win >= 0 && g.slot[win].human() && !g.slot[1 - win].human();
            if (human_won || (win >= 0 && g.slot[0].human() && g.slot[1].human()))
                a.fx.add(std::make_unique<Confetti>(Box{0, 0, float(a.w), float(a.h)}, a.rng));
        }
        return;
    }
    if (anim) return;
    const int side = g.match.to_move();
    Slot& sl = g.slot[side];
    if (must_pass(g.match.pos)) {
        if (g.pass_at < 0) g.pass_at = a.time + 0.8;
        if (a.time >= g.pass_at) {
            g.pass_at = -1;
            a.play_move(Move::none());
        }
        return;
    }
    if (!sl.human() && !g.paused) {
        if (!sl.agent->thinking(g.token)) {
            sl.agent->think(g.match.pos, g.token);
            g.think_t0 = a.time;
        }
        Move m = Move::none();
        double secs = 0;
        std::string err;
        if (sl.agent->poll(g.token, m, secs, err)) {
            sl.last_seconds = secs;
            if (!err.empty()) {
                sl.error = err;
                g.paused = true;
                a.notify(sl.spec->name + " : erreur (" + err + ")");
            } else if (!is_legal(g.match.pos, m)) {
                g.paused = true;
                a.notify(sl.spec->name + " a propos un coup ill\xC3\xA9gal : " + move_name(m));
            } else {
                a.play_move(m);
            }
        }
    }
}

void draw_eval_bar(App& a, Box bar, const Analyzer::Snapshot& snap) {
    auto& g = a.game;
    const float s = a.s;
    float target = snap.valid && snap.token == g.token ? eval_share(snap.value) : g.eval_shown;
    if (g.match.over()) {
        int win = g.match.winner();
        target = win == BLACK ? 1.0f : win == WHITE ? 0.0f : 0.5f;
    }
    g.eval_shown += (target - g.eval_shown) * std::min(1.0f, 5 * a.ui.dt);
    // The side at the bottom of the board fills the bar from the bottom.
    const int bottom_side = g.board.flipped ? WHITE : BLACK;
    float bottom_share = bottom_side == BLACK ? g.eval_shown : 1 - g.eval_shown;
    Rgba col_black = hex(0x20242c), col_white = hex(0xf1f3f6);
    Rgba col_bottom = bottom_side == BLACK ? col_black : col_white, col_top = bottom_side == BLACK ? col_white : col_black;
    draw::shadow({bar.x, bar.y + 4 * s, bar.w, bar.h}, bar.w / 2, 14 * s, hex(0x000000, 120));
    draw::round_box(bar, bar.w / 2, col_top);
    float hb = bar.h * bottom_share;
    draw::clip(&bar);
    draw::round_box({bar.x, bar.bottom() - hb, bar.w, hb + bar.w}, bar.w / 2, col_bottom);
    draw::clip(nullptr);
    draw::rect({bar.x, bar.y + bar.h / 2, bar.w, 1}, hex(0xff0000, 120));
    draw::round_frame(bar, bar.w / 2, 1, hex(0xffffff, 40));
    if (snap.valid && snap.token == g.token) {
        bool black_ahead = snap.value >= 0;
        std::string lbl = eval_text(snap.value);
        bool lead_bottom = (black_ahead == (bottom_side == BLACK));
        float fs = 12 * s;
        V2 p = lead_bottom ? V2{bar.center().x, bar.bottom() - 22 * s} : V2{bar.center().x, bar.y + 8 * s};
        Rgba fg = (lead_bottom ? col_bottom : col_top).r > 128 ? hex(0x20242c) : hex(0xf1f3f6);
        a.text.draw(Font::Black, fs, lbl, p, fg, Align::Center);
    }
}

void draw_player_card(App& a, Box b, int side, bool active) {
    auto& g = a.game;
    const float s = a.s;
    const Slot& sl = g.slot[side];
    float k = a.ui.anim(strf("card%d", side), active);
    draw::round_box(b, 14 * s, with_alpha(mix(hex(0xffffff), a.theme().accent, k), 0.04f + 0.1f * k));
    draw::round_frame(b, 14 * s, 1.5f * s, with_alpha(a.theme().accent, 0.15f + 0.85f * k));
    float pr = b.h * 0.33f;
    draw_piece(a.slot_sprite(sl), side, {b.x + 18 * s + pr, b.center().y}, pr, 0.3f * k, a.time);
    float tx = b.x + 34 * s + 2 * pr;
    a.text.draw(Font::Bold, 19 * s, a.slot_name(sl), {tx, b.y + 14 * s}, pal::text);
    std::string sub = sl.human() ? std::string("Humain \xE2\x80\xA2 ") + side_name(side) : strf("Elo \xE2\x89\x88 %d \xE2\x80\xA2 %s", sl.spec->elo, side_name(side));
    a.text.draw(Font::Regular, 14 * s, sub, {tx, b.y + 40 * s}, pal::dim);
    std::string status;
    if (active && !g.match.over()) {
        if (sl.human()) status = "\xC3\x80 vous de jouer";
        else if (g.paused) status = "en pause";
        else {
            int dots = int(a.time * 3) % 4;
            status = strf("r\xC3\xA9" "fl\xC3\xA9" "chit%s  %.1f s", std::string(dots, '.').c_str(), a.time - g.think_t0);
        }
    } else if (!sl.human() && sl.last_seconds > 0) {
        status = strf("dernier coup : %.2f s", sl.last_seconds);
    }
    a.text.draw(Font::Regular, 14 * s, status, {tx, b.y + 60 * s}, active ? a.theme().accent : pal::faint);
    a.text.draw(Font::Black, 40 * s, std::to_string(g.match.score(side)), {b.right() - 20 * s, b.center().y - 26 * s}, pal::text, Align::Right);
}

void tab_moves(App& a, Box b) {
    auto& g = a.game;
    const float s = a.s;
    const auto& hist = g.match.history;
    const int first = g.match.start.current_player();   // the side of the first move
    const int n_rows = int((hist.size() + first + 1) / 2);
    const float rh = 30 * s;
    float content = n_rows * rh;
    if (a.ui.hover(b)) g.move_scroll -= a.in.wheel * rh * 3;
    g.move_scroll = std::clamp(g.move_scroll, 0.0f, std::max(0.0f, content - b.h + 10 * s));
    a.text.draw(Font::Bold, 14 * s, "Noirs", {b.x + 70 * s, b.y}, pal::faint);
    a.text.draw(Font::Bold, 14 * s, "Blancs", {b.x + 230 * s, b.y}, pal::faint);
    Box list{b.x, b.y + 26 * s, b.w, b.h - 26 * s};
    draw::clip(&list);
    for (int row = 0; row < n_rows; row++) {
        float y = list.y + row * rh - g.move_scroll;
        if (y < list.y - rh || y > list.bottom()) continue;
        if (row % 2 == 0) draw::round_box({list.x, y, list.w, rh}, 6 * s, hex(0xffffff, 8));
        a.text.draw(Font::Regular, 15 * s, strf("%d.", row + 1), {list.x + 12 * s, y + 6 * s}, pal::faint);
        for (int col = 0; col < 2; col++) {
            int i = row * 2 + col - first;
            if (i < 0 || i >= int(hist.size())) continue;
            bool last = i == int(hist.size()) - 1;
            V2 p{list.x + (70 + 160 * col) * s, y + 5 * s};
            if (last) draw::round_box({p.x - 8 * s, y + 3 * s, 130 * s, rh - 6 * s}, 6 * s, with_alpha(a.theme().accent, 0.25f));
            a.text.draw(last ? Font::Bold : Font::Regular, 16 * s, move_name(hist[i]), p, hist[i] == Move::none() ? pal::faint : pal::text);
        }
    }
    draw::clip(nullptr);
    if (hist.empty()) a.text.draw(Font::Regular, 15 * s, "Aucun coup jou\xC3\xA9 pour l'instant.", {list.x + 12 * s, list.y + 8 * s}, pal::faint);
}

void tab_analysis(App& a, Box b, const Analyzer::Snapshot& snap) {
    auto& g = a.game;
    const float s = a.s;
    float y = b.y;
    bool fresh = snap.valid && snap.token == g.token;
    if (g.match.over()) {
        a.text.draw(Font::Bold, 18 * s, "Partie termin\xC3\xA9" "e.", {b.x, y}, pal::text);
        return;
    }
    if (!fresh) {
        a.text.draw(Font::Regular, 16 * s, snap.error.empty() ? "Analyse en cours\xE2\x80\xA6" : "Analyse impossible : " + snap.error, {b.x, y}, pal::dim);
        return;
    }
    bool black = snap.value >= 0;
    std::string who = std::abs(snap.value) > EVAL_WIN ? strf("%s gagnent \xC3\xA0 coup s\xC3\xBBr", black ? "Les Noirs" : "Les Blancs")
                                                      : (std::abs(snap.value) < 600 ? std::string("Position \xC3\xA9quilibr\xC3\xA9" "e") : strf("Avantage %s", black ? "Noirs" : "Blancs"));
    a.text.draw(Font::Black, 26 * s, eval_text(snap.value), {b.x, y}, pal::text);
    a.text.draw(Font::Bold, 17 * s, who, {b.x + 120 * s, y + 6 * s}, pal::dim);
    y += 44 * s;
    // A horizontal version of the bar.
    Box hb{b.x, y, b.w, 10 * s};
    draw::round_box(hb, 5 * s, hex(0xf1f3f6));
    draw::round_box({hb.x, hb.y, hb.w * eval_share(snap.value), hb.h}, 5 * s, hex(0x20242c));
    draw::round_frame(hb, 5 * s, 1, hex(0xffffff, 50));
    y += 24 * s;
    a.text.draw(Font::Regular, 15 * s, strf("Profondeur %d \xE2\x80\xA2 %.1f M n\xC5\x93uds \xE2\x80\xA2 %.1f s%s", snap.depth, snap.nodes / 1e6, snap.seconds, snap.running ? " \xE2\x80\xA2 en cours" : ""),
                {b.x, y}, pal::dim);
    y += 30 * s;
    a.text.draw(Font::Bold, 16 * s, "Meilleur coup", {b.x, y}, pal::faint);
    a.text.draw(Font::Black, 18 * s, move_name(snap.best) + strf("  (%s)", side_name(g.match.to_move())), {b.x + 140 * s, y - 1 * s}, a.theme().hint);
    y += 34 * s;
    a.text.draw(Font::Bold, 16 * s, "Variation principale", {b.x, y}, pal::faint);
    y += 26 * s;
    // The variation as chips, wrapped.
    float x = b.x;
    int side = g.match.to_move();
    for (size_t i = 0; i < snap.pv.size(); i++) {
        std::string lbl = move_name(snap.pv[i]);
        float w = a.text.width(Font::Bold, 15 * s, lbl) + 18 * s;
        if (x + w > b.right()) { x = b.x; y += 32 * s; }
        if (y > b.bottom() - 110 * s) break;
        draw::round_box({x, y, w, 26 * s}, 13 * s, side == BLACK ? hex(0x20242c) : hex(0xe9ecf1));
        a.text.draw(Font::Bold, 15 * s, lbl, {x + w / 2, y + 4 * s}, side == BLACK ? hex(0xf1f3f6) : hex(0x20242c), Align::Center);
        x += w + 6 * s;
        side = 1 - side;
    }
    y = std::max(y + 44 * s, b.bottom() - 112 * s);
    a.ui.toggle("g_hint", {b.x, y, b.w, 30 * s}, "Indice sur le plateau (H)", a.settings.show_hint);
    a.ui.toggle("g_pv", {b.x, y + 36 * s, b.w, 30 * s}, "Variation sur le plateau", a.settings.show_pv);
    a.ui.toggle("g_eval", {b.x, y + 72 * s, b.w, 30 * s}, "Barre d'\xC3\xA9valuation (E)", a.settings.show_eval);
}

void tab_ai(App& a, Box b) {
    auto& g = a.game;
    const float s = a.s;
    float y = b.y;
    bool any = false;
    for (int side : {BLACK, WHITE}) {
        const Slot& sl = g.slot[side];
        if (sl.human()) continue;
        any = true;
        draw_piece(a.slot_sprite(sl), side, {b.x + 24 * s, y + 24 * s}, 22 * s, 0, a.time);
        a.text.draw(Font::Bold, 18 * s, sl.spec->name, {b.x + 58 * s, y + 2 * s}, pal::text);
        a.text.draw(Font::Regular, 14 * s, strf("Elo \xE2\x89\x88 %d \xE2\x80\xA2 %s \xE2\x80\xA2 joue les %s", sl.spec->elo, sl.spec->family.c_str(), side_name(side)),
                    {b.x + 58 * s, y + 26 * s}, pal::dim);
        y += 56 * s;
        float hgt = a.text.paragraph(Font::Regular, 14 * s, sl.spec->description, {b.x, y, b.w, 120 * s}, pal::text);
        y += hgt + 6 * s;
        std::string info = sl.agent ? sl.agent->info() : "";
        if (!info.empty()) {
            hgt = a.text.paragraph(Font::Regular, 12 * s, "Moteur : " + info, {b.x, y, b.w, 70 * s}, pal::faint);
            y += hgt + 4 * s;
        }
        y += 16 * s;
    }
    if (!any) a.text.paragraph(Font::Regular, 15 * s, "Deux humains s'affrontent : pas d'IA dans cette partie. L'onglet Analyse reste disponible.", b, pal::dim);
    a.text.paragraph(Font::Regular, 12 * s, "Analyse : Minimax NNUE dev (1 thread), ind\xC3\xA9pendant des joueurs.", {b.x, b.bottom() - 18 * s, b.w, 20 * s}, pal::faint);
}

void draw_game_over(App& a, Box area) {
    auto& g = a.game;
    const float s = a.s;
    float k = ease_out_cubic(float((a.time - g.over_t) / 0.5));
    a.ui.blocked = false;
    draw::round_box(area, 6 * s, hex(0x000000, uint8_t(110 * k)));
    Box card{area.center().x - 230 * s, area.center().y - 150 * s + (1 - k) * 40 * s, 460 * s, 300 * s};
    a.ui.panel(card, 22, with_alpha(pal::panel, k));
    int win = g.match.winner();
    std::string title = win < 0 ? "\xC3\x89" "galit\xC3\xA9 !" : strf("Victoire des %s !", side_name(win));
    if (win >= 0 && g.slot[win].human() && !g.slot[1 - win].human()) title = "Vous avez gagn\xC3\xA9 !";
    if (win >= 0 && !g.slot[win].human() && g.slot[1 - win].human()) title = g.slot[win].spec->name + " gagne";
    a.text.draw_shadowed(Font::Title, 34 * s, title, {card.center().x, card.y + 30 * s}, with_alpha(pal::text, k), Align::Center, hex(0x000000, 100), 2 * s);
    a.text.draw(Font::Black, 54 * s, strf("%d \xE2\x80\x93 %d", g.match.score(BLACK), g.match.score(WHITE)), {card.center().x, card.y + 88 * s}, with_alpha(a.theme().accent, k), Align::Center);
    a.text.draw(Font::Regular, 16 * s, strf("Noirs \xE2\x80\x94 Blancs \xE2\x80\xA2 %zu coups", g.match.history.size()), {card.center().x, card.y + 158 * s}, with_alpha(pal::dim, k), Align::Center);
    float bw = 130 * s, by = card.bottom() - 74 * s;
    if (a.ui.button("go_again", {card.x + 24 * s, by, bw, 50 * s}, "Rejouer", Style::Primary)) a.start_game(g.match.start);
    if (a.ui.button("go_setup", {card.center().x - bw / 2, by, bw, 50 * s}, "Joueurs")) a.go(Scene::Setup);
    if (a.ui.button("go_menu", {card.right() - 24 * s - bw, by, bw, 50 * s}, "Menu", Style::Ghost)) a.go(Scene::Menu);
}

void scene_game(App& a) {
    auto& g = a.game;
    const float s = a.s;
    // Keyboard shortcuts.
    if (a.in.key(SDLK_ESCAPE)) { a.go(Scene::Menu); return; }
    if (a.in.key(SDLK_H)) a.settings.show_hint = !a.settings.show_hint;
    if (a.in.key(SDLK_E)) a.settings.show_eval = !a.settings.show_eval;
    if (a.in.key(SDLK_F)) g.board.flipped = !g.board.flipped;
    if (a.in.key(SDLK_U) || a.in.key(SDLK_BACKSPACE)) a.undo_move();
    if (a.in.key(SDLK_SPACE) && !g.slot[0].human() && !g.slot[1].human()) g.paused = !g.paused;

    game_logic(a);

    // Analysis: restarted whenever the position changes and something shows it.
    bool need = a.settings.show_eval || a.settings.show_hint || a.settings.show_pv || g.tab == 1;
    if (need && g.analysed != g.token && !g.board.animating(a.time) && !g.match.over()) {
        g.analyzer.analyse(g.match.pos, g.token, 20.0);
        g.analysed = g.token;
    }
    if (!need && g.analysed == g.token) { g.analyzer.stop(); g.analysed = 0; }
    Analyzer::Snapshot snap = g.analyzer.snapshot();
    bool fresh = snap.valid && snap.token == g.token;

    // Layout.
    const float margin = 34 * s, panel_w = 450 * s, bar_w = 26 * s, frame = 26 * s, gap = 22 * s;
    const float left_w = a.w - panel_w - 3 * margin;
    float B = std::min(a.h - 2 * margin - 2 * frame - 10 * s, left_w - bar_w - gap - 2 * frame);
    B = std::floor(B / 8) * 8;
    float block_w = (a.settings.show_eval ? bar_w + gap : 0) + 2 * frame + B;
    float bx = margin + (left_w - block_w) / 2 + (a.settings.show_eval ? bar_w + gap : 0) + frame;
    float by = (a.h - B) / 2;
    g.board.area = {std::floor(bx), std::floor(by), B, B};
    Box panel{a.w - margin - panel_w, margin, panel_w, a.h - 2 * margin};

    draw_background(a, false);
    if (a.settings.show_eval) draw_eval_bar(a, {g.board.area.x - frame - gap - bar_w, g.board.area.y, bar_w, B}, snap);

    // Board.
    const int side = g.match.to_move();
    bool human_turn = !g.match.over() && g.slot[side].human() && !must_pass(g.match.pos);
    bool modal = g.over && a.time > g.over_t + 0.3;
    auto mv = g.board.input(a.in, g.match.pos, human_turn && !g.board.animating(a.time), modal);
    g.board.drag_pos = a.in.mouse;
    if (mv) a.play_move(*mv);
    BoardView::Look look;
    look.sprite[BLACK] = &a.slot_sprite(g.slot[BLACK]);
    look.sprite[WHITE] = &a.slot_sprite(g.slot[WHITE]);
    look.coords = a.settings.show_coords;
    look.human_turn = human_turn;
    if (fresh && !g.board.animating(a.time) && !g.match.over()) {
        if (a.settings.show_hint && snap.best != Move::none()) {
            look.hint = snap.best;
            int v = side == BLACK ? snap.value : -snap.value;
            look.hint_label = std::abs(v) > EVAL_WIN ? (v > 0 ? strf("gagne +%d", v - EVAL_WIN) : "perd") : strf("%d %%", int(std::round(100 * eval_share(v))));
        }
        if (a.settings.show_pv) look.pv = snap.pv;
    }
    if (human_turn && g.board.hover_sq >= 0 && (g.match.pos.bitboard(side) >> g.board.hover_sq & 1)) a.ui.hovering = true;
    g.board.draw(a.text, a.board_textures, a.theme(), g.match.pos, look, a.time, s);
    a.fx.update(float(a.ui.dt));
    a.fx.draw(a.text);

    // Side panel.
    a.ui.blocked = modal;
    a.ui.panel(panel);
    const int top_side = g.board.flipped ? BLACK : WHITE;
    Box card_top{panel.x + 16 * s, panel.y + 16 * s, panel.w - 32 * s, 92 * s};
    Box card_bottom{panel.x + 16 * s, panel.bottom() - 108 * s, panel.w - 32 * s, 92 * s};
    draw_player_card(a, card_top, top_side, side == top_side && !g.match.over());
    draw_player_card(a, card_bottom, 1 - top_side, side != top_side && !g.match.over());
    Box tabs{panel.x + 20 * s, card_top.bottom() + 14 * s, panel.w - 40 * s, 38 * s};
    g.tab = a.ui.tabs("gtabs", tabs, {"Coups", "Analyse", "IA"}, g.tab);
    Box buttons{panel.x + 16 * s, card_bottom.y - 66 * s, panel.w - 32 * s, 50 * s};
    Box content{panel.x + 24 * s, tabs.bottom() + 16 * s, panel.w - 48 * s, buttons.y - tabs.bottom() - 30 * s};
    if (g.tab == 0) tab_moves(a, content);
    else if (g.tab == 1) tab_analysis(a, content, snap);
    else tab_ai(a, content);
    bool ai_vs_ai = !g.slot[0].human() && !g.slot[1].human();
    float bw = (buttons.w - 3 * 8 * s) / 4;
    auto bbox = [&](int i) { return Box{buttons.x + i * (bw + 8 * s), buttons.y, bw, buttons.h}; };
    if (ai_vs_ai) {
        if (a.ui.button("g_pause", bbox(0), g.paused ? "Reprendre" : "Pause", g.paused ? Style::Primary : Style::Normal, true, 17)) g.paused = !g.paused;
    } else if (a.ui.button("g_hintb", bbox(0), "Indice", a.settings.show_hint ? Style::Primary : Style::Normal, true, 17)) {
        a.settings.show_hint = !a.settings.show_hint;
    }
    if (a.ui.button("g_undo", bbox(1), "Annuler", Style::Normal, !g.match.history.empty(), 17)) a.undo_move();
    if (a.ui.button("g_flip", bbox(2), "Retourner", Style::Normal, true, 17)) g.board.flipped = !g.board.flipped;
    if (a.ui.button("g_menu", bbox(3), "Menu", Style::Ghost, true, 17)) { a.go(Scene::Menu); return; }

    if (g.over && a.time > g.over_t + 0.3) draw_game_over(a, g.board.area.inset(-26 * s));
    a.ui.blocked = false;
}

// ─── @@SCENE_THEMES ─────────────────────────────────────────────────────────
// Renders every sprite (both colours) into a 256 × 256 texture and saves it as
// a PNG in assets/sprites/export/: a starting point to draw one's own sprites
// ("image" in sprites.json).
std::string export_sprites(App& a) {
    fs::path dir = g_paths.assets / "sprites" / "export";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const int N = 256;
    int count = 0;
    SDL_Texture* target = SDL_CreateTexture(a.renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, N, N);
    if (!target) return std::string("export impossible : ") + SDL_GetError();
    SDL_SetTextureBlendMode(target, SDL_BLENDMODE_BLEND);
    for (const SpriteDef& sp : a.sprites) {
        for (int side : {BLACK, WHITE}) {
            SDL_SetRenderTarget(a.renderer, target);
            SDL_SetRenderDrawColor(a.renderer, 0, 0, 0, 0);
            SDL_RenderClear(a.renderer);
            SpriteDef plain = sp;
            plain.texture = nullptr;
            draw_piece(plain, side, {N / 2.0f, N / 2.0f}, N * 0.36f, 0, 0);
            SDL_Surface* surf = SDL_RenderReadPixels(a.renderer, nullptr);
            SDL_SetRenderTarget(a.renderer, nullptr);
            if (!surf) continue;
            SDL_Surface* rgba = SDL_ConvertSurface(surf, SDL_PIXELFORMAT_ABGR8888);
            SDL_DestroySurface(surf);
            if (!rgba) continue;
            // The blending stored premultiplied colours: undo it.
            auto* px = static_cast<uint8_t*>(rgba->pixels);
            for (int y = 0; y < rgba->h; y++)
                for (int x = 0; x < rgba->w; x++) {
                    uint8_t* p = px + y * rgba->pitch + x * 4;
                    if (p[3] > 0 && p[3] < 255)
                        for (int k = 0; k < 3; k++) p[k] = uint8_t(std::min(255, p[k] * 255 / p[3]));
                }
            fs::path file = dir / (sp.id + (side == BLACK ? "_noir.png" : "_blanc.png"));
            if (stbi_write_png(file.string().c_str(), rgba->w, rgba->h, 4, rgba->pixels, rgba->pitch)) count++;
            SDL_DestroySurface(rgba);
        }
    }
    SDL_DestroyTexture(target);
    return strf("%d images export\xC3\xA9" "es dans %s", count, dir.string().c_str());
}

void scene_themes(App& a) {
    const float s = a.s;
    draw_background(a);
    draw_title(a, "Th\xC3\xA8mes et sprites", "Choisissez l'apparence du plateau. Les th\xC3\xA8mes sont des fichiers JSON de assets/themes : copiez-en un pour cr\xC3\xA9" "er le v\xC3\xB4tre.");
    if (back_button(a)) { a.save_settings(); a.go(Scene::Menu); return; }
    // Themes: cards with a small live board.
    const float card_w = 250 * s, card_h = 300 * s, gap = 24 * s;
    const int cols = 3;
    Box area{60 * s, 140 * s, cols * card_w + (cols - 1) * gap, a.h - 180 * s};
    for (size_t i = 0; i < a.themes.size(); i++) {
        const Theme& t = a.themes[i];
        Box c{area.x + (i % cols) * (card_w + gap), area.y + (i / cols) * (card_h + gap), card_w, card_h};
        bool sel = t.id == a.settings.theme, h = a.ui.hover(c);
        if (h) a.ui.hovering = true;
        float k = a.ui.anim("th" + t.id, h), ks = a.ui.anim("ths" + t.id, sel);
        Box cc{c.x, c.y - 6 * s * k, c.w, c.h};
        draw::shadow({cc.x, cc.y + 10 * s, cc.w, cc.h}, 18 * s, 30 * s, hex(0x000000, 110));
        draw::round_box(cc, 18 * s, t.bg_top, t.bg_bottom);
        if (ks > 0.01f) draw::round_frame(cc, 18 * s, 3 * s, with_alpha(t.accent, ks));
        // Mini board: frame, squares, a hole, two pieces.
        Box b{cc.x + 25 * s, cc.y + 22 * s, 200 * s, 200 * s};
        draw::round_box(b.inset(-10 * s), 10 * s, t.frame_top, t.frame_bottom);
        SDL_FRect dst = b.frect();
        SDL_RenderTexture(a.renderer, a.board_textures.get(t, int(b.w)), nullptr, &dst);
        float q = b.w / 8;
        for (V2 hsq : {V2{3, 3}, V2{4, 2}, V2{2, 5}}) {
            Box hb{b.x + hsq.x * q + q * 0.08f, b.y + hsq.y * q + q * 0.08f, q * 0.84f, q * 0.84f};
            draw::round_box(hb, q * 0.2f, shade(t.hole, -0.3f), shade(t.hole, 0.12f));
        }
        draw_piece(a.sprite("human"), BLACK, {b.x + 2.5f * q, b.y + 2.5f * q}, q * 0.4f, 0, a.time);
        draw_piece(a.sprite("dev"), WHITE, {b.x + 5.5f * q, b.y + 4.5f * q}, q * 0.4f, 0, a.time);
        a.text.draw(Font::Bold, 20 * s, t.name, {cc.center().x, cc.bottom() - 62 * s}, pal::text, Align::Center);
        a.text.draw(Font::Regular, 14 * s, sel ? "th\xC3\xA8me actuel" : "cliquer pour choisir", {cc.center().x, cc.bottom() - 34 * s}, sel ? t.accent : pal::dim, Align::Center);
        if (h && a.in.released) a.settings.theme = t.id;
    }
    // Sprites: the gallery and the export.
    Box gal{area.right() + 50 * s, 140 * s, a.w - area.right() - 110 * s, a.h - 180 * s};
    a.ui.panel(gal);
    a.text.draw(Font::Black, 22 * s, "Sprites des joueurs", {gal.x + 22 * s, gal.y + 18 * s}, pal::text);
    a.text.paragraph(Font::Regular, 14 * s, "Chaque IA a son embl\xC3\xA8me ; l'anneau et les points disent sa force. Le halo bleu signale le joueur humain. D\xC3\xA9" "crits dans assets/sprites/sprites.json.",
                     {gal.x + 22 * s, gal.y + 52 * s, gal.w - 44 * s, 60 * s}, pal::dim);
    float rh = std::min(64.0f * s, (gal.h - 220 * s) / std::max<size_t>(1, a.sprites.size()));
    for (size_t i = 0; i < a.sprites.size(); i++) {
        const SpriteDef& sp = a.sprites[i];
        float y = gal.y + 120 * s + i * rh;
        float pr = rh * 0.38f;
        draw_piece(sp, BLACK, {gal.x + 30 * s + pr, y + rh / 2}, pr, 0, a.time);
        draw_piece(sp, WHITE, {gal.x + 40 * s + 3 * pr, y + rh / 2}, pr, 0, a.time);
        a.text.draw(Font::Bold, 17 * s, sp.name, {gal.x + 60 * s + 4 * pr, y + rh / 2 - 20 * s}, pal::text);
        a.text.draw(Font::Regular, 13 * s, strf("embl\xC3\xA8me \xC2\xAB %s \xC2\xBB \xE2\x80\xA2 niveau %d%s", sp.emblem.c_str(), sp.tier, sp.texture ? " \xE2\x80\xA2 image PNG" : ""),
                    {gal.x + 60 * s + 4 * pr, y + rh / 2 + 2 * s}, pal::dim);
    }
    if (a.ui.button("export", {gal.x + 22 * s, gal.bottom() - 74 * s, gal.w - 44 * s, 52 * s}, "Exporter les sprites en PNG", Style::Primary))
        a.notify(export_sprites(a));
}

// ─── @@SCENE_RULES ──────────────────────────────────────────────────────────
void placeholder(App& a, const std::string& title, const std::string& subtitle, const std::string& body, const std::string& next) {
    const float s = a.s;
    draw_background(a);
    draw_title(a, title, subtitle);
    if (back_button(a)) { a.go(Scene::Menu); return; }
    Box p{a.w * 0.5f - 460 * s, 150 * s, 920 * s, a.h - 220 * s};
    a.ui.panel(p, 22);
    float y = p.y + 30 * s;
    y += a.text.paragraph(Font::Regular, 19 * s, body, {p.x + 40 * s, y, p.w - 80 * s, p.h - 160 * s}, pal::text, 1.45f);
    a.text.paragraph(Font::Regular, 16 * s, next, {p.x + 40 * s, p.bottom() - 90 * s, p.w - 80 * s, 70 * s}, a.theme().accent);
}

void scene_rules(App& a) {
    placeholder(a, "R\xC3\xA8gles du jeu", "Yolah en une minute",
                "\xE2\x80\xA2 Le plateau a 8 \xC3\x97 8 cases. Chaque camp a 4 pions ; les Noirs commencent.\n"
                "\xE2\x80\xA2 Un pion se d\xC3\xA9place comme la dame aux \xC3\xA9" "checs : en ligne droite ou en diagonale, d'autant de cases libres qu'il veut, sans sauter par-dessus un pion ni un trou.\n"
                "\xE2\x80\xA2 La case qu'il quitte devient un trou : plus personne ne pourra y aller.\n"
                "\xE2\x80\xA2 Chaque coup rapporte 1 point.\n"
                "\xE2\x80\xA2 Un camp qui ne peut plus bouger passe son tour ; la partie s'arr\xC3\xAA" "te quand plus personne ne peut bouger.\n"
                "\xE2\x80\xA2 Le camp qui a jou\xC3\xA9 le plus de coups gagne : il faut garder de la place pour soi et en \xC3\xB4ter \xC3\xA0 l'adversaire.",
                "Bient\xC3\xB4t : un tutoriel interactif, \xC3\xA9tape par \xC3\xA9tape, avec des exemples anim\xC3\xA9s sur le plateau.");
}

// ─── @@SCENE_PUZZLES ────────────────────────────────────────────────────────
void scene_puzzles(App& a) {
    placeholder(a, "Puzzles et qu\xC3\xAA" "tes", "Des positions \xC3\xA0 r\xC3\xA9soudre, regroup\xC3\xA9" "es en niveaux",
                "Chaque puzzle part d'une position choisie (pions plac\xC3\xA9s librement, trous d\xC3\xA9j\xC3\xA0 creus\xC3\xA9s), avec le camp qui commence et l'IA adverse. "
                "Une image de fond accompagne chaque puzzle et entoure le plateau. Les puzzles se suivent dans des niveaux que chacun peut composer \xC3\xA0 partir de ses propres cr\xC3\xA9" "ations.",
                "Cette partie arrive avec l'\xC3\xA9" "diteur, dans une prochaine \xC3\xA9tape.");
}

// ─── @@SCENE_EDITOR ─────────────────────────────────────────────────────────
void scene_editor(App& a) {
    placeholder(a, "\xC3\x89" "diteur de puzzles", "Cr\xC3\xA9" "er positions, sprites et niveaux",
                "L'\xC3\xA9" "diteur permettra de noircir les cases de son choix, de placer les 4 pions de chaque camp, de choisir les sprites des joueurs, des trous et des cases vides "
                "(une collection est fournie, modifiable), le camp qui commence et l'IA adverse, l'image de fond, puis d'encha\xC3\xAEner les puzzles en niveaux.",
                "Prochaine \xC3\xA9tape du d\xC3\xA9veloppement.");
}

}   // namespace yapp

// ─── @@MAIN ─────────────────────────────────────────────────────────────────
// SDL calls SDL_AppInit once, then, until one of them returns something else
// than SDL_APP_CONTINUE: SDL_AppEvent for every event (mouse, keyboard,
// window), SDL_AppIterate once per frame (logic + drawing); finally SDL_AppQuit.
using namespace yapp;

namespace {
std::ofstream g_null_stream;   // the AI players print a lot: silenced unless --verbose
std::streambuf* g_cout_buf = nullptr;

std::string arg_value(int argc, char** argv, const char* name, const char* def = "") {
    for (int i = 1; i + 1 < argc; i++)
        if (std::string(argv[i]) == name) return argv[i + 1];
    return def;
}
bool arg_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == name) return true;
    return false;
}
}   // namespace

SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    App* a = new App();
    *appstate = a;
    g_app = a;
    a->screenshot = arg_value(argc, argv, "--screenshot");
    std::string size = arg_value(argc, argv, "--size", "1600x900");
    std::sscanf(size.c_str(), "%dx%d", &a->w, &a->h);
    if (!a->screenshot.empty()) {
        // Tests and documentation: no window on screen, software rendering.
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
        a->frames_left = std::stoi(arg_value(argc, argv, "--frames", "30"));
    }
    SDL_SetAppMetadata("Yolah", APP_VERSION, "fr.yolah.app");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    if (!SDL_CreateWindowAndRenderer("Yolah", a->w, a->h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY, &a->window, &a->renderer)) {
        SDL_Log("SDL_CreateWindowAndRenderer: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    SDL_SetWindowMinimumSize(a->window, 1024, 600);
    SDL_SetRenderVSync(a->renderer, 1);
    SDL_SetRenderDrawBlendMode(a->renderer, SDL_BLENDMODE_BLEND);
    draw::R = a->renderer;

    init_paths();
    if (!arg_flag(argc, argv, "--verbose")) {
        g_cout_buf = std::cout.rdbuf();
        std::cout.rdbuf(g_null_stream.rdbuf());
    }
    magic::init();
    zobrist::init();
    if (!a->text.load(g_paths.assets / "fonts")) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Yolah", ("Polices introuvables dans " + (g_paths.assets / "fonts").string()).c_str(), a->window);
        return SDL_APP_FAILURE;
    }
    a->themes = load_themes(g_paths.assets / "themes");
    a->sprites = load_sprites(g_paths.assets / "sprites" / "sprites.json");
    a->roster = load_roster(g_paths.assets / "ai_roster.json");
    {   // The analysis engine.
        json r = load_json(g_paths.assets / "ai_roster.json");
        json an = r.is_object() && r.contains("analysis") ? r["analysis"] : json::object();
        json cfg = load_json(g_paths.root / "config" / jstr(an, "config", "search_study/dev_final.cfg"));
        if (cfg.is_object()) {
            if (an.contains("overrides"))
                for (auto& [k, v] : an["overrides"].items()) cfg[k] = v;
            if (cfg.contains("weights")) cfg["weights"] = resolve_engine_path(cfg["weights"].get<std::string>()).string();
            cfg["verbose"] = false;
            a->game.analyzer.configure(cfg);
        }
    }
    a->settings.from_json(load_json(g_paths.prefs / "settings.json"));
    a->cursor_arrow = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_DEFAULT);
    a->cursor_hand = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_POINTER);

    // Start scene (command line).
    std::string scene = arg_value(argc, argv, "--scene", "menu");
    std::string black = arg_value(argc, argv, "--black"), white = arg_value(argc, argv, "--white");
    if (!black.empty()) a->settings.black = black;
    if (!white.empty()) a->settings.white = white;
    if (arg_flag(argc, argv, "--hint")) a->settings.show_hint = true;
    if (!arg_value(argc, argv, "--time").empty()) a->settings.time_index = std::stoi(arg_value(argc, argv, "--time"));
    if (!arg_value(argc, argv, "--theme").empty()) a->settings.theme = arg_value(argc, argv, "--theme");
    if (scene == "game") {
        int plies = std::stoi(arg_value(argc, argv, "--plies", "0"));
        Yolah y;
        std::mt19937 rng(7);
        for (int i = 0; i < plies && !y.game_over(); i++) {
            auto mv = legal_moves(y);
            y.play(mv[rng() % mv.size()]);
        }
        a->start_game(y);
        // The holes of the random moves are not animated.
        a->game.board.hole_since.fill(-100);
        a->game.tab = std::stoi(arg_value(argc, argv, "--tab", "0"));
    } else if (scene == "setup") a->go(Scene::Setup);
    else if (scene == "themes") a->go(Scene::Themes);
    else if (scene == "rules") a->go(Scene::Rules);
    else if (scene == "puzzles") a->go(Scene::Puzzles);
    else if (scene == "editor") a->go(Scene::Editor);
    a->time = now_seconds();
    a->scene_t0 = a->time - 10;
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* e) {
    App* a = static_cast<App*>(appstate);
    // Mouse coordinates in rendering pixels (they differ from window
    // coordinates on high-density screens).
    SDL_ConvertEventToRenderCoordinates(a->renderer, e);
    switch (e->type) {
    case SDL_EVENT_QUIT: return SDL_APP_SUCCESS;
    case SDL_EVENT_MOUSE_MOTION: a->in.mouse = {e->motion.x, e->motion.y}; break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (e->button.button == SDL_BUTTON_LEFT) { a->in.down = true; a->in.pressed = true; a->in.mouse = {e->button.x, e->button.y}; }
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (e->button.button == SDL_BUTTON_LEFT) { a->in.down = false; a->in.released = true; a->in.mouse = {e->button.x, e->button.y}; }
        break;
    case SDL_EVENT_MOUSE_WHEEL: a->in.wheel += e->wheel.y; break;
    case SDL_EVENT_KEY_DOWN:
        if (!e->key.repeat) a->in.keys.push_back(e->key.key);
        break;
    default: break;
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    App* a = static_cast<App*>(appstate);
    double t = now_seconds();
    a->dt = std::min(0.1, t - a->time);
    a->time = t;
    SDL_GetRenderOutputSize(a->renderer, &a->w, &a->h);
    a->s = std::min(a->w / DESIGN_W, a->h / DESIGN_H);
    a->ui.s = a->s;
    a->ui.dt = float(a->dt);
    a->ui.hovering = false;
    g_graveyard.reap();

    SDL_SetRenderDrawColor(a->renderer, 0, 0, 0, 255);
    SDL_RenderClear(a->renderer);
    switch (a->scene) {
    case Scene::Menu: scene_menu(*a); break;
    case Scene::Setup: scene_setup(*a); break;
    case Scene::Game: scene_game(*a); break;
    case Scene::Themes: scene_themes(*a); break;
    case Scene::Rules: scene_rules(*a); break;
    case Scene::Puzzles: scene_puzzles(*a); break;
    case Scene::Editor: scene_editor(*a); break;
    }
    if (a->scene != Scene::Game) {
        a->fx.update(float(a->dt));
        a->fx.draw(a->text);
    }
    // Toast message at the bottom.
    float tk = float(a->time - a->toast_t);
    if (tk < 4.0f && !a->toast.empty()) {
        float al = clamp01(std::min(tk * 5, (4.0f - tk) * 2));
        float s = a->s, w = a->text.width(Font::Bold, 17 * s, a->toast) + 48 * s;
        Box b{a->w * 0.5f - w / 2, a->h - 90 * s + (1 - al) * 20 * s, w, 46 * s};
        draw::shadow({b.x, b.y + 6 * s, b.w, b.h}, 23 * s, 20 * s, hex(0x000000, uint8_t(120 * al)));
        draw::round_box(b, 23 * s, with_alpha(hex(0x10151d), 0.95f * al));
        draw::round_frame(b, 23 * s, 1, with_alpha(pal::accent, 0.6f * al));
        a->text.draw(Font::Bold, 17 * s, a->toast, {b.center().x, b.y + 12 * s}, with_alpha(pal::text, al), Align::Center);
    }
    // Fade-in after a scene change.
    float fade = 1 - clamp01(float(a->time - a->scene_t0) / 0.25f);
    if (fade > 0) draw::rect({0, 0, float(a->w), float(a->h)}, hex(0x000000, uint8_t(255 * fade)));
    SDL_SetCursor(a->ui.hovering ? a->cursor_hand : a->cursor_arrow);

    if (a->frames_left >= 0) {
        if (a->frames_left-- == 0) {
            SDL_Surface* surf = SDL_RenderReadPixels(a->renderer, nullptr);
            if (surf) {
                SDL_SaveBMP(surf, a->screenshot.c_str());
                SDL_DestroySurface(surf);
            }
            return SDL_APP_SUCCESS;
        }
    }
    SDL_RenderPresent(a->renderer);
    a->in.pressed = a->in.released = false;
    a->in.wheel = 0;
    a->in.keys.clear();
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult) {
    App* a = static_cast<App*>(appstate);
    if (!a) return;
    if (a->screenshot.empty()) a->save_settings();   // test runs leave the user's settings alone
    a->leave_game();
    for (auto& sl : a->game.slot) sl.agent.reset();
    g_graveyard.join_all();
    a->text.clear();
    a->board_textures.clear();
    for (auto& sp : a->sprites)
        if (sp.texture) SDL_DestroyTexture(sp.texture);
    if (g_cout_buf) std::cout.rdbuf(g_cout_buf);
    delete a;
}
