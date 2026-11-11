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
//   @@ART           pixel-art drawings of the workshop (assets/art): frames,
//                   animations, the nearest-pixel texture, 9-grid frames
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
//   @@SCENE_THEMES  themes, the sprite of each player (picker over the sprite
//                   library), PNG export
//   @@SCENE_TUTORIAL learn to play: lessons on the board, each with a goal
//                   checked by code (reach a square, shut a piece in, win an
//                   endgame...), a small perfect solver for the endgames
//   @@PUZZLE        puzzles and levels: data, JSON files, validity, hole styles
//   @@SCENE_PUZZLES levels and puzzles to play (each solved puzzle unlocks the
//                   next), the level editor, the introduction card, the goal
//   @@SCENE_EDITOR  puzzle editor: paint the board, who starts, the player's
//                   side, the AI, the goal, theme, holes, sprites, background
//   @@SCENE_STUDIO  the workshop: pixel editor (tools, colours, frames, undo,
//                   import) and board editor (theme colours and drawings)
//   @@MAIN          the SDL callbacks: SDL_AppInit, SDL_AppEvent,
//                   SDL_AppIterate, SDL_AppQuit; screenshot mode
//
//  Command line (all optional):
//   --size WxH            window size
//   --screenshot FILE     render, save FILE (.bmp) and quit (tests, docs)
//   --frames N            frames rendered before the screenshot (default 30)
//   --scene NAME          menu, setup, game, themes, tutorial, puzzles, editor, studio
//   --scene setup         (--params ID: the settings box of that AI open)
//   --lesson N            lesson of the tutorial scene (0 = the first)
//   --pick ID             themes scene: open the sprite picker of player ID
//   --level ID            puzzles scene: the puzzles of a level
//   --ptab N              puzzles scene: 0 play, 1 edit the levels
//   --puzzle ID           play a puzzle (--no-intro: skip its introduction)
//   --scene studio        the workshop (--tab 1: the boards; --art ID: open a drawing)
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
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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

// A file name from a title: lower case letters, digits and '_'.
std::string slug(const std::string& name) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c)) out += char(std::tolower(c));
        else if ((c == ' ' || c == '-' || c == '_') && !out.empty() && out.back() != '_') out += '_';
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "puzzle" : out;
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
// An AI setting the player can change before a game (the "Régler" box of the
// new-game screen): a key of the engine configuration ("nb threads", "tt
// size", ...) or of the application ("time factor", "random moves"), and the
// values offered. def: the value without a setting (the roster's).
struct AiParam {
    std::string key, label, help, unit;
    bool app = false;                 // "time factor" / "random moves": not an engine key
    std::vector<json> values;
    json def;
    std::string value_text(const json& v) const {
        if (key == "time factor") return strf("\xC3\x97 %g", v.get<double>());
        if (key == "random moves") return strf("%d %%", int(std::lround(v.get<double>() * 100)));
        if (v.is_boolean()) return v.get<bool>() ? "oui" : "non";
        if (v.is_number()) return strf("%g", v.get<double>()) + (unit.empty() ? "" : " " + unit);
        return v.is_string() ? v.get<std::string>() : v.dump();
    }
};

struct AiSpec {
    std::string id, name, family, sprite, description, config_file;
    int elo = 0;
    double time_factor = 1;   // × the time per move chosen in the application
    double random_moves = 0;  // probability of a random legal move instead of a search
    json config;          // the engine configuration, overrides applied, paths absolute
    bool available = false;
    std::string reason;   // why it is not available
    std::vector<AiParam> params;   // what the player may set (AiParam)
    bool threads_set = false;      // "nb threads" chosen by the player: not capped
};

int hardware_threads() { return std::max(2u, std::thread::hardware_concurrency()); }

// The roster's "params" of an AI, then the two settings every AI but the random
// one has (time factor, random moves). A default missing from the values is
// added to them, in order.
void load_params(AiSpec& s, const json& e) {
    const int hw = hardware_threads();
    auto add = [&](AiParam p) {
        if (p.def.is_null()) return;
        bool found = false;
        for (auto& v : p.values)   // numbers read as float (jnum): the nearest value
            if (v == p.def || (v.is_number() && p.def.is_number() && std::abs(v.get<double>() - p.def.get<double>()) < 1e-6 * std::max(1.0, std::abs(v.get<double>())))) {
                p.def = v;
                found = true;
            }
        if (!found) {
            p.values.push_back(p.def);
            if (p.def.is_number())
                std::sort(p.values.begin(), p.values.end(), [](const json& x, const json& y) { return x.get<double>() < y.get<double>(); });
        }
        s.params.push_back(p);
    };
    if (e.contains("params") && e["params"].is_array())
        for (const json& q : e["params"]) {
            AiParam p;
            p.key = jstr(q, "key");
            p.label = jstr(q, "label", p.key);
            p.help = jstr(q, "help");
            p.unit = jstr(q, "unit");
            if (p.key.empty() || !s.config.contains(p.key)) continue;   // not a key of this engine
            p.def = s.config[p.key];
            if (q.contains("values") && q["values"].is_array()) p.values = q["values"].get<std::vector<json>>();
            else if (jstr(q, "values") == "threads") {   // 1 .. the cores of this computer
                for (int n : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128})
                    if (n <= hw) p.values.push_back(n);
                if (p.values.back() != hw) p.values.push_back(hw);
                if (p.def.is_number()) p.def = std::max(1, std::min(p.def.get<int>(), hw - 2));   // what game_config gives
            } else if (p.def.is_boolean()) p.values = {false, true};
            if (p.def.is_number() && p.key == "nb threads" && p.def.get<int>() > hw) p.def = hw;
            add(p);
        }
    if (s.id == "random") return;
    AiParam t;
    t.key = "time factor";
    t.label = "Temps de r\xC3\xA9" "flexion";
    t.help = "multiplie le temps par coup choisi pour la partie";
    t.app = true;
    t.values = {0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0};
    t.def = s.time_factor;
    add(t);
    AiParam r;
    r.key = "random moves";
    r.label = "Coups au hasard";
    r.help = "part des coups jou\xC3\xA9s au hasard au lieu de r\xC3\xA9" "fl\xC3\xA9" "chir";
    r.app = true;
    r.values = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5};
    r.def = s.random_moves;
    add(r);
}

// An AI with the player's settings (Settings::ai_options[id]): values that are
// not among those offered are ignored.
AiSpec with_options(const AiSpec& base, const json& opts) {
    AiSpec s = base;
    if (!opts.is_object()) return s;
    for (const AiParam& p : base.params) {
        if (!opts.contains(p.key)) continue;
        const json& v = opts[p.key];
        if (std::find(p.values.begin(), p.values.end(), v) == p.values.end()) continue;
        if (p.key == "time factor") s.time_factor = v.get<double>();
        else if (p.key == "random moves") s.random_moves = v.get<double>();
        else {
            s.config[p.key] = v;
            if (p.key == "nb threads") s.threads_set = true;
        }
    }
    return s;
}

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
        s.time_factor = jnum(e, "time factor", 1);
        s.random_moves = jnum(e, "random moves", 0);
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
        load_params(s, e);
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
    if (c.contains("microseconds")) c["microseconds"] = std::max<uint64_t>(1000, uint64_t(seconds * s.time_factor * 1e6));
    const int hw = hardware_threads();
    if (c.contains("nb threads") && c["nb threads"].is_number() && !s.threads_set)
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
    // random_moves: probability of playing a random legal move (weaker AIs).
    explicit AiAgent(json cfg, double random_moves = 0) : sh_(std::make_shared<Shared>()), random_moves_(random_moves) {
        sh_->cfg = std::move(cfg);
    }
    ~AiAgent() { cancel(); }

    void think(const Yolah& pos, uint64_t token) {
        cancel();
        auto job = std::make_shared<Job>();
        job->token = token;
        job_ = job;
        auto sh = sh_;
        const bool random = random_moves_ > 0 && std::uniform_real_distribution<double>(0, 1)(rng_) < random_moves_;
        const uint32_t seed = rng_();
        thread_ = std::thread([sh, job, pos, random, seed] {
            std::lock_guard lock(sh->m);
            if (random) {   // a random move, after a short pause (as if it thought)
                auto mv = legal_moves(pos);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                job->move = mv[seed % mv.size()];
                job->seconds = 0.3;
                job->done->store(true);
                return;
            }
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
    double random_moves_ = 0;
    std::mt19937 rng_{std::random_device{}()};
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
// A thick polyline: segments joined by discs.
void polyline(const std::vector<V2>& pts, float w, Rgba col, bool closed = false) {
    const size_t n = pts.size();
    if (n < 2) return;
    const size_t segs = closed ? n : n - 1;
    for (size_t i = 0; i < segs; i++) line(pts[i], pts[(i + 1) % n], w, col);
    for (size_t i = closed ? 0 : 1; i < (closed ? n : n - 1); i++) circle(pts[i], w * 0.5f, col);
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
    float paragraph(Font f, float size, const std::string& s, Box b, Rgba col, float spacing = 1.25f, Align al = Align::Left) {
        float lh = line_height(f, size) * spacing / 1.2f, y = b.y;
        for (auto& l : wrap(f, size, s, b.w)) {
            if (y + lh > b.bottom() + 1) break;
            if (al == Align::Center) draw(f, size, l, {b.center().x, y}, col, Align::Center);
            else draw(f, size, l, {b.x, y}, col);
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
    // Drawings of the workshop (@@ART ids, "" = none): the light and dark
    // squares, the holes and the frame, instead of the procedural ones.
    std::string tile_light, tile_dark, hole_art, frame_art;
    // A scenery (assets/backgrounds): a picture behind every scene instead of
    // the gradient, a planet under the board (its diameter in board widths,
    // its top in board heights from the board's top), twinkling stars and
    // shooting stars drawn over the picture.
    std::string bg_image, planet_image;
    float planet_size = 2.4f, planet_top = 0.6f;
    bool stars = false;
    bool embers = false;     // embers rising over the picture
    bool petals = false;     // cherry petals falling over the picture
    bool smoke = false;      // smoke drifting along the bottom of the picture
    // How a square breaks when a piece leaves it (Shatter): "debris" (the
    // default), "stone", "space", "ink", "cannon"; stronger players break harder.
    std::string break_fx = "debris";
    // The set of pieces that goes with the board (assets/sprites/sets.json):
    // used when the settings say "Selon le plateau"; "" = the classic stones.
    std::string piece_set;
    std::string file;        // its file in assets/themes (the board editor rewrites it)
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
    t.tile_light = jstr(j, "tile light");
    t.tile_dark = jstr(j, "tile dark");
    t.hole_art = jstr(j, "hole art");
    t.frame_art = jstr(j, "frame art");
    t.bg_image = jstr(j, "background image");
    t.planet_image = jstr(j, "planet image");
    t.planet_size = jnum(j, "planet size", t.planet_size);
    t.planet_top = jnum(j, "planet top", t.planet_top);
    t.stars = j.value("stars", false);
    t.embers = j.value("embers", false);
    t.petals = j.value("petals", false);
    t.smoke = j.value("smoke", false);
    t.break_fx = jstr(j, "break fx", t.break_fx);
    t.piece_set = jstr(j, "piece set");
    return t;
}
json theme_to_json(const Theme& t) {
    json j = {{"id", t.id}, {"name", t.name}, {"pattern", t.pattern}, {"pattern strength", t.strength}, {"seed", t.seed},
              {"light", color_string(t.light)}, {"dark", color_string(t.dark)}, {"frame top", color_string(t.frame_top)},
              {"frame bottom", color_string(t.frame_bottom)}, {"frame line", color_string(t.frame_line)},
              {"coordinates", color_string(t.coords)}, {"hole", color_string(t.hole)}, {"hole rim", color_string(t.hole_rim)},
              {"background top", color_string(t.bg_top)}, {"background bottom", color_string(t.bg_bottom)},
              {"last move", color_string(t.last_move)}, {"selection", color_string(t.select)},
              {"legal move", color_string(t.legal)}, {"hint", color_string(t.hint)}, {"accent", color_string(t.accent)}};
    if (!t.tile_light.empty()) j["tile light"] = t.tile_light;
    if (!t.tile_dark.empty()) j["tile dark"] = t.tile_dark;
    if (!t.hole_art.empty()) j["hole art"] = t.hole_art;
    if (!t.frame_art.empty()) j["frame art"] = t.frame_art;
    if (!t.bg_image.empty()) j["background image"] = t.bg_image;
    if (!t.planet_image.empty()) {
        j["planet image"] = t.planet_image;
        j["planet size"] = t.planet_size;
        j["planet top"] = t.planet_top;
    }
    if (t.stars) j["stars"] = true;
    if (t.embers) j["embers"] = true;
    if (t.petals) j["petals"] = true;
    if (t.smoke) j["smoke"] = true;
    if (t.break_fx != "debris") j["break fx"] = t.break_fx;
    if (!t.piece_set.empty()) j["piece set"] = t.piece_set;
    return j;
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
        if (j.is_object()) { themes.push_back(parse_theme(j)); themes.back().file = f.filename().string(); }
    }
    // "order" in the files would be nicer; the file names start with a number.
    if (themes.empty()) themes.push_back(Theme{});
    return themes;
}

// The pictures of the themes' sceneries (assets/backgrounds), loaded once.
SDL_Texture* scenery_texture(const std::string& file) {
    static std::map<std::string, SDL_Texture*> cache;
    if (file.empty()) return nullptr;
    auto it = cache.find(file);
    if (it != cache.end()) return it->second;
    SDL_Texture* tex = nullptr;
    int w, h, n;
    fs::path p = g_paths.assets / "backgrounds" / file;
    if (unsigned char* px = stbi_load(p.string().c_str(), &w, &h, &n, 4)) {
        tex = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
        SDL_UpdateTexture(tex, nullptr, px, w * 4);
        SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
        stbi_image_free(px);
    } else {
        SDL_Log("scenery picture not loaded: %s", p.string().c_str());
    }
    return cache[file] = tex;
}

// Stars over a scenery: fixed places, each twinkling at its own pace, a few
// coloured; now and then a shooting star crosses the top of the window.
void draw_star_field(Box b, double time, float s) {
    using namespace draw;
    for (int i = 0; i < 160; i++) {
        float x = b.x + b.w * hash2(i, 1, 91), y = b.y + b.h * hash2(i, 2, 91);
        float size = (0.6f + 1.6f * std::pow(hash2(i, 3, 91), 4.0f)) * s;
        float tw = 0.55f + 0.45f * float(std::sin(time * (0.7 + 2.3 * hash2(i, 4, 91)) + 6.28 * hash2(i, 5, 91)));
        float hue = hash2(i, 6, 91);
        Rgba c = hue < 0.15f ? hex(0xffd9a8) : hue < 0.3f ? hex(0xa8c8ff) : hex(0xffffff);
        circle({x, y}, size * 2.5f, with_alpha(c, 0.25f * tw), with_alpha(c, 0), 0);
        circle({x, y}, size, with_alpha(c, 0.9f * tw));
    }
    // A shooting star every 9 s, for 0.9 s, along a line that changes each time.
    const double period = 9.0, life = 0.9;
    int k = int(time / period);
    double u = (time - k * period) / life;
    if (u < 1) {
        float x0 = b.x + b.w * (0.15f + 0.6f * hash2(k, 7, 17)), y0 = b.y + b.h * (0.05f + 0.25f * hash2(k, 8, 17));
        V2 dir = normalized(V2{0.8f + 0.4f * hash2(k, 9, 17), 0.35f + 0.3f * hash2(k, 10, 17)});
        float len = 260 * s, e = float(ease_out_cubic(float(u)));
        V2 head = V2{x0, y0} + dir * (e * 520 * s);
        float fade = float(std::sin(u * M_PI));
        for (int j = 0; j < 12; j++) {
            float a0 = j / 12.0f, a1 = (j + 1) / 12.0f;
            line(head - dir * (len * a0), head - dir * (len * a1), (2.2f - 1.8f * a0) * s, hex(0xffffff, uint8_t(220 * fade * (1 - a0))));
        }
        circle(head, 3.5f * s, hex(0xdff0ff, uint8_t(230 * fade)), hex(0x9fd0ff, 0), 0);
    }
}

// Embers rising over a scenery: each one starts at a hashed place along the
// bottom, rises at its own speed while swaying, glows and fades out.
void draw_embers(Box b, double time, float s) {
    using namespace draw;
    for (int i = 0; i < 70; i++) {
        float speed = 0.04f + 0.08f * hash2(i, 1, 33);           // window heights per second
        double life = 1.0 / speed;
        double u = std::fmod(time + life * hash2(i, 2, 33), life) / life;   // 0 → 1 along its rise
        float x = b.x + b.w * hash2(i, 3, 33) + std::sin(float(time) * (0.6f + hash2(i, 4, 33)) + i) * 30 * s;
        float y = b.bottom() - float(u) * b.h * 1.1f;
        float fade = float(std::sin(u * M_PI));
        float size = (1.0f + 2.0f * hash2(i, 5, 33)) * s;
        Rgba c = hash2(i, 6, 33) < 0.5f ? hex(0xffb040) : hex(0xff6a20);
        circle({x, y}, size * 4, with_alpha(c, 0.25f * fade), with_alpha(c, 0), 0);
        circle({x, y}, size, with_alpha(hex(0xffe0a0), 0.9f * fade));
    }
}

// Cherry petals falling: each one drifts down at its own speed, sways and
// turns (an ellipse that narrows as it turns edge-on).
void draw_petals(Box b, double time, float s) {
    using namespace draw;
    for (int i = 0; i < 46; i++) {
        float speed = 0.05f + 0.06f * hash2(i, 1, 57);
        double life = 1.0 / speed;
        double u = std::fmod(time + life * hash2(i, 2, 57), life) / life;
        float x = b.x + b.w * hash2(i, 3, 57) + std::sin(float(time) * (0.8f + hash2(i, 4, 57)) + i) * 60 * s - float(u) * 120 * s;
        float y = b.y - 20 * s + float(u) * (b.h + 40 * s);
        float turn = float(std::sin(time * (1.5 + 2 * hash2(i, 5, 57)) + i * 1.3));
        float w = (5 + 4 * hash2(i, 6, 57)) * s, h = w * 0.55f * std::abs(turn) + 0.6f * s;
        float ang = float(time) * (0.5f + hash2(i, 7, 57)) + i;
        std::vector<V2> pts;
        for (int k = 0; k < 10; k++) {
            float a = k * 0.6283f;
            V2 q{std::cos(a) * w, std::sin(a) * h};
            pts.push_back(V2{x, y} + V2{q.x * std::cos(ang) - q.y * std::sin(ang), q.x * std::sin(ang) + q.y * std::cos(ang)});
        }
        poly_around({x, y}, pts, with_alpha(hex(0xffc2d4), 0.85f));
    }
}

// Smoke drifting along the bottom of a scenery: soft puffs moving to the
// right at their own speed, swelling and fading as they go.
void draw_smoke(Box b, double time, float s) {
    using namespace draw;
    for (int i = 0; i < 18; i++) {
        float speed = 0.012f + 0.02f * hash2(i, 1, 71);
        double life = 1.0 / speed;
        double u = std::fmod(time + life * hash2(i, 2, 71), life) / life;
        float x = b.x - 200 * s + float(u) * (b.w + 400 * s);
        float y = b.y + b.h * (0.62f + 0.3f * hash2(i, 3, 71)) + std::sin(float(time) * 0.3f + i) * 10 * s;
        float r = (90 + 120 * hash2(i, 4, 71)) * s * (0.7f + 0.6f * float(u));
        float a = 0.18f * float(std::sin(u * M_PI));
        circle({x, y}, r, with_alpha(hex(0xd8d0c4), a), with_alpha(hex(0xd8d0c4), 0), r * 0.9f);
    }
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
    std::string id = "default", name = "Pion", emblem = "none", image, category;
    std::string art;         // a pixel-art sprite of the workshop (@@ART): drawn instead
    std::string style;       // "" (stone + ring + emblem) or a plain stone: go, flat, checkers, glass, marble
    Rgba ring_top = hex(0xcfd6df), ring_bottom = hex(0x7d8794), emblem_color = hex(0xffffff);
    int tier = 0;            // 0..8: pips under the emblem
    bool halo = false;       // pulsing halo (the human player)
    Rgba halo_color = hex(0x5ee7ff);
    SDL_Texture* texture = nullptr;   // the picture, if any
    // A sprite drawn as another one for White (a set of pieces: its black
    // and its white unit): the id, and the sprite once linked (link_sprites).
    std::string white_id;
    const SpriteDef* white_alt = nullptr;
};

// A set of pieces (assets/sprites/sets.json): what the pieces look like when
// the set is chosen. "players": each player keeps its own sprite (the robots
// and the human); otherwise "black" and "white" list the sprites of the four
// pieces of each side (the units of a faction; one sprite = all alike).
struct PieceSet {
    std::string id, name, description;
    bool players = false;
    std::vector<std::string> black, white;
    std::string human;       // the human player's pieces in this set (the user's figurine, dressed for it)
    std::map<std::string, std::string> cast;   // "cast": a figurine per player (AI id or "human"), whatever the side
};

std::vector<PieceSet> load_piece_sets(const fs::path& file) {
    std::vector<PieceSet> v;
    json j = load_json(file);
    if (j.is_object() && j.contains("sets") && j["sets"].is_array())
        for (auto& e : j["sets"]) {
            PieceSet ps;
            ps.id = jstr(e, "id");
            ps.name = jstr(e, "name", ps.id);
            ps.description = jstr(e, "description");
            ps.players = e.value("players", false);
            ps.human = jstr(e, "human");
            if (e.contains("cast") && e["cast"].is_object())
                for (auto& [k, v] : e["cast"].items())
                    if (v.is_string()) ps.cast[k] = v.get<std::string>();
            for (const char* k : {"black", "white"})
                if (e.contains(k) && e[k].is_array())
                    for (auto& x : e[k])
                        if (x.is_string()) (k[0] == 'b' ? ps.black : ps.white).push_back(x.get<std::string>());
            if (!ps.id.empty()) v.push_back(ps);
        }
    if (v.empty()) v.push_back({"classique", "Pions classiques", "", false, {"classic_go"}, {"classic_go"}, "", {}});
    return v;
}

// Each set is also a sprite of the library ("set:<id>", category "Jeux de
// pièces"), so that a player or a side of a puzzle can take a whole set: it
// is drawn as its first black unit, or its first white unit for White.
void add_set_sprites(std::vector<SpriteDef>& sprites, const std::vector<PieceSet>& sets) {
    sprites.erase(std::remove_if(sprites.begin(), sprites.end(), [](const SpriteDef& sp) { return sp.id.rfind("set:", 0) == 0; }), sprites.end());
    for (const PieceSet& ps : sets) {
        SpriteDef sp;
        sp.id = "set:" + ps.id;
        sp.name = ps.name;
        sp.category = "Jeux de pi\xC3\xA8" "ces";
        std::string b = ps.players ? "fig_human" : ps.black.empty() ? "classic_go" : ps.black[0];
        std::string w = ps.players ? "fig_sentinel" : ps.white.empty() ? b : ps.white[0];
        sp.emblem = "none";
        sp.style = "go";
        for (const SpriteDef& o : sprites)
            if (o.id == b) { sp.art = o.art; sp.style = o.style; sp.emblem = o.emblem; }
        sp.white_id = w;
        sprites.push_back(sp);
    }
}
// Resolves the white alternatives (after any change to the list).
void link_sprites(std::vector<SpriteDef>& sprites) {
    for (SpriteDef& sp : sprites) {
        sp.white_alt = nullptr;
        if (!sp.white_id.empty())
            for (const SpriteDef& o : sprites)
                if (o.id == sp.white_id && &o != &sp) sp.white_alt = &o;
    }
}

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
            s.category = jstr(e, "category");
            s.ring_top = jcolor(e, "ring top", s.ring_top);
            s.ring_bottom = jcolor(e, "ring bottom", s.ring_bottom);
            s.emblem_color = jcolor(e, "emblem color", s.emblem_color);
            s.tier = int(jnum(e, "tier", 0));
            s.halo = e.value("halo", false);
            s.halo_color = jcolor(e, "halo color", s.halo_color);
            s.style = jstr(e, "style");
            s.art = jstr(e, "art");      // a drawing of assets/art (a rendered figurine)
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
// box of "radius" s around c. `under`: the colour of the stone below, used to
// cut holes (moon, gear, eyes); transparent for the emblem's shadow.
const std::vector<std::string> EMBLEMS = {
    "none", "star", "dice", "branch", "tree", "hex", "prism", "bolt", "eye", "crown", "heart", "shield", "dot",
    "moon", "sun", "flame", "drop", "leaf", "snowflake", "clover", "target", "atom", "planet", "gear", "triangle",
    "square", "plus", "cross", "diamond", "hexagram", "star4", "rook", "pawn", "note", "spade", "mountain", "check",
    "infinity", "paw", "flower", "ring", "arrow", "ghost", "cat", "anchor", "key"};

void draw_emblem(const std::string& e, V2 c, float s, Rgba col, Rgba under = {0, 0, 0, 0}) {
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
    } else if (e == "moon") {
        circle(c, 0.8f * s, col);
        if (under.a) circle(P(0.36f, -0.26f), 0.66f * s, under);
    } else if (e == "sun") {
        circle(c, 0.44f * s, col);
        for (int i = 0; i < 8; i++) {
            float a = i * float(M_PI) / 4, d = 0.13f;
            poly({P(0.58f * std::cos(a - d), 0.58f * std::sin(a - d)), P(0.95f * std::cos(a), 0.95f * std::sin(a)),
                  P(0.58f * std::cos(a + d), 0.58f * std::sin(a + d))}, col);
        }
    } else if (e == "flame" || e == "drop") {
        // A round bottom and a pointed top (the flame's tip is bent).
        auto shape = [&](float ox, float oy, float k, Rgba kc) {
            auto Q = [&](float x, float y) { return P(ox + k * x, oy + k * y); };
            std::vector<V2> pts;
            for (int i = 0; i <= 16; i++) {
                float a = float(M_PI) * i / 16;
                pts.push_back(Q(0.55f * std::cos(a), 0.3f + 0.55f * std::sin(a)));
            }
            V2 L{-0.55f, 0.3f}, T{e == "flame" ? 0.15f : 0.0f, -0.95f}, Rr{0.55f, 0.3f};
            V2 cl{-0.6f, -0.25f}, cr{e == "flame" ? 0.75f : 0.6f, -0.25f};
            for (int i = 1; i <= 12; i++) {
                float t = i / 12.0f;
                V2 q = L * ((1 - t) * (1 - t)) + cl * (2 * t * (1 - t)) + T * (t * t);
                pts.push_back(Q(q.x, q.y));
            }
            for (int i = 1; i < 12; i++) {
                float t = i / 12.0f;
                V2 q = T * ((1 - t) * (1 - t)) + cr * (2 * t * (1 - t)) + Rr * (t * t);
                pts.push_back(Q(q.x, q.y));
            }
            poly_around(Q(0, 0.3f), pts, kc);
        };
        shape(0, 0, 1, col);
        if (e == "flame") shape(0.0f, 0.3f, 0.5f, shade(col, 0.55f));
        else circle(P(-0.2f, 0.38f), 0.12f * s, shade(col, 0.6f));
    } else if (e == "leaf") {
        auto rot = [&](float x, float y) { const float k = 0.7071f; return P((x - y) * k, (x + y) * k); };
        std::vector<V2> pts;
        for (int i = 0; i <= 16; i++) { float t = i / 16.0f; pts.push_back(rot(-0.95f + 1.9f * t, -0.42f * std::sin(float(M_PI) * t))); }
        for (int i = 15; i >= 1; i--) { float t = i / 16.0f; pts.push_back(rot(-0.95f + 1.9f * t, 0.42f * std::sin(float(M_PI) * t))); }
        poly(pts, col);
        line(rot(-0.95f, 0), rot(0.75f, 0), std::max(1.0f, 0.07f * s), shade(col, -0.35f));
    } else if (e == "snowflake") {
        float w = std::max(1.5f, 0.11f * s);
        for (int i = 0; i < 3; i++) {
            float a = float(M_PI) / 2 + i * float(M_PI) / 3;
            V2 d{std::cos(a), std::sin(a)};
            line(c - d * (0.9f * s), c + d * (0.9f * s), w, col);
            for (float sg : {1.0f, -1.0f}) {
                V2 u = d * sg, b = c + u * (0.55f * s);
                for (float tw : {0.8f, -0.8f}) {
                    V2 r{u.x * std::cos(tw) - u.y * std::sin(tw), u.x * std::sin(tw) + u.y * std::cos(tw)};
                    line(b, b + r * (0.3f * s), w * 0.8f, col);
                }
            }
        }
    } else if (e == "clover") {
        for (int i = 0; i < 3; i++) {
            float a = -float(M_PI) / 2 + i * 2 * float(M_PI) / 3;
            circle(P(0.38f * std::cos(a), -0.1f + 0.38f * std::sin(a)), 0.34f * s, col);
        }
        poly({P(-0.08f, 0.1f), P(0.08f, 0.1f), P(0.18f, 0.9f), P(-0.18f, 0.9f)}, col);
    } else if (e == "target") {
        ring(c, 0.88f * s, 0.17f * s, col);
        ring(c, 0.55f * s, 0.15f * s, col);
        circle(c, 0.2f * s, col);
    } else if (e == "atom" || e == "planet") {
        auto ellipse = [&](float rx, float ry, float rt) {
            std::vector<V2> pts;
            for (int i = 0; i < 48; i++) {
                float a = 2 * float(M_PI) * i / 48, x = rx * std::cos(a), y = ry * std::sin(a);
                pts.push_back(P(x * std::cos(rt) - y * std::sin(rt), x * std::sin(rt) + y * std::cos(rt)));
            }
            return pts;
        };
        float w = std::max(1.2f, 0.08f * s);
        if (e == "atom") {
            for (int i = 0; i < 3; i++) polyline(ellipse(0.92f, 0.32f, i * float(M_PI) / 3), w, col, true);
            circle(c, 0.17f * s, col);
        } else {
            circle(c, 0.48f * s, col);
            polyline(ellipse(0.95f, 0.28f, -0.4f), w * 1.3f, shade(col, 0.4f), true);
        }
    } else if (e == "gear") {
        std::vector<V2> pts;
        for (int i = 0; i < 8; i++) {
            float a = i * float(M_PI) / 4;
            for (auto [da, r] : std::vector<std::pair<float, float>>{{-0.3f, 0.66f}, {-0.17f, 0.92f}, {0.17f, 0.92f}, {0.3f, 0.66f}})
                pts.push_back(P(r * std::cos(a + da), r * std::sin(a + da)));
        }
        poly_around(c, pts, col);
        if (under.a) circle(c, 0.26f * s, under);
    } else if (e == "triangle") {
        poly({P(0, -0.85f), P(0.82f, 0.6f), P(-0.82f, 0.6f)}, col);
    } else if (e == "square") {
        round_box({c.x - 0.65f * s, c.y - 0.65f * s, 1.3f * s, 1.3f * s}, 0.22f * s, col);
    } else if (e == "plus" || e == "cross") {
        float rt = e == "cross" ? float(M_PI) / 4 : 0;
        for (int k = 0; k < 2; k++) {
            float a = rt + k * float(M_PI) / 2;
            V2 d{std::cos(a), std::sin(a)};
            line(c - d * (0.8f * s), c + d * (0.8f * s), 0.36f * s, col);
        }
    } else if (e == "diamond") {
        poly({P(0, -0.9f), P(0.65f, 0), P(0, 0.9f), P(-0.65f, 0)}, col);
    } else if (e == "hexagram") {
        poly({P(0, -0.9f), P(0.78f, 0.45f), P(-0.78f, 0.45f)}, col);
        poly({P(0, 0.9f), P(0.78f, -0.45f), P(-0.78f, -0.45f)}, col);
    } else if (e == "star4") {
        poly_around(c, {P(0, -0.95f), P(0.2f, -0.2f), P(0.95f, 0), P(0.2f, 0.2f), P(0, 0.95f), P(-0.2f, 0.2f), P(-0.95f, 0), P(-0.2f, -0.2f)}, col);
    } else if (e == "rook") {
        round_box({c.x - 0.62f * s, c.y + 0.5f * s, 1.24f * s, 0.3f * s}, 0.06f * s, col);
        poly({P(-0.42f, 0.52f), P(0.42f, 0.52f), P(0.34f, -0.35f), P(-0.34f, -0.35f)}, col);
        round_box({c.x - 0.55f * s, c.y - 0.55f * s, 1.1f * s, 0.25f * s}, 0.04f * s, col);
        for (float x : {-0.55f, -0.12f, 0.31f}) round_box({c.x + x * s, c.y - 0.85f * s, 0.24f * s, 0.32f * s}, 0.04f * s, col);
    } else if (e == "pawn") {
        circle(P(0, -0.5f), 0.3f * s, col);
        poly({P(-0.18f, -0.25f), P(0.18f, -0.25f), P(0.42f, 0.55f), P(-0.42f, 0.55f)}, col);
        round_box({c.x - 0.6f * s, c.y + 0.5f * s, 1.2f * s, 0.3f * s}, 0.08f * s, col);
    } else if (e == "note") {
        circle(P(-0.22f, 0.52f), 0.3f * s, col);
        line(P(0.05f, 0.5f), P(0.05f, -0.8f), std::max(1.5f, 0.12f * s), col);
        poly({P(0.05f, -0.8f), P(0.55f, -0.45f), P(0.55f, -0.2f), P(0.05f, -0.5f)}, col);
    } else if (e == "spade") {
        circle(P(-0.35f, 0.12f), 0.4f * s, col);
        circle(P(0.35f, 0.12f), 0.4f * s, col);
        poly({P(-0.73f, 0.0f), P(0.73f, 0.0f), P(0, -0.88f)}, col);
        poly({P(0, 0.2f), P(0.3f, 0.85f), P(-0.3f, 0.85f)}, col);
    } else if (e == "mountain") {
        poly({P(-0.95f, 0.65f), P(-0.25f, -0.6f), P(0.45f, 0.65f)}, shade(col, -0.2f));
        poly({P(-0.3f, 0.65f), P(0.35f, -0.25f), P(0.95f, 0.65f)}, col);
        poly({P(-0.25f, -0.6f), P(-0.06f, -0.26f), P(-0.44f, -0.26f)}, shade(col, 0.6f));
    } else if (e == "check") {
        polyline({P(-0.7f, 0.05f), P(-0.2f, 0.55f), P(0.75f, -0.55f)}, 0.24f * s, col);
    } else if (e == "infinity") {
        std::vector<V2> pts;
        for (int i = 0; i < 60; i++) {
            float t = 2 * float(M_PI) * i / 60;
            pts.push_back(P(0.9f * std::cos(t), 0.8f * std::sin(t) * std::cos(t)));
        }
        polyline(pts, std::max(1.5f, 0.15f * s), col, true);
    } else if (e == "paw") {
        circle(P(0, 0.3f), 0.38f * s, col);
        for (V2 t : {V2{-0.58f, -0.12f}, V2{-0.22f, -0.5f}, V2{0.22f, -0.5f}, V2{0.58f, -0.12f}}) circle(P(t.x, t.y), 0.17f * s, col);
    } else if (e == "flower") {
        for (int i = 0; i < 6; i++) {
            float a = i * float(M_PI) / 3;
            circle(P(0.5f * std::cos(a), 0.5f * std::sin(a)), 0.3f * s, col);
        }
        circle(c, 0.3f * s, shade(col, 0.5f));
    } else if (e == "ring") {
        ring(c, 0.78f * s, 0.24f * s, col);
    } else if (e == "arrow") {
        poly({P(-0.18f, 0.85f), P(0.18f, 0.85f), P(0.18f, -0.1f), P(-0.18f, -0.1f)}, col);
        poly({P(-0.62f, -0.05f), P(0.62f, -0.05f), P(0, -0.9f)}, col);
    } else if (e == "ghost") {
        circle(P(0, -0.2f), 0.6f * s, col);
        poly({P(-0.6f, -0.2f), P(0.6f, -0.2f), P(0.6f, 0.6f), P(-0.6f, 0.6f)}, col);
        for (float x : {-0.4f, 0.0f, 0.4f}) circle(P(x, 0.62f), 0.2f * s, col);
        if (under.a) {
            circle(P(-0.22f, -0.22f), 0.13f * s, under);
            circle(P(0.22f, -0.22f), 0.13f * s, under);
        }
    } else if (e == "cat") {
        circle(P(0, 0.12f), 0.62f * s, col);
        poly({P(-0.6f, -0.05f), P(-0.55f, -0.85f), P(-0.15f, -0.45f)}, col);
        poly({P(0.6f, -0.05f), P(0.55f, -0.85f), P(0.15f, -0.45f)}, col);
        if (under.a) {
            circle(P(-0.24f, 0.05f), 0.11f * s, under);
            circle(P(0.24f, 0.05f), 0.11f * s, under);
            poly({P(-0.09f, 0.27f), P(0.09f, 0.27f), P(0, 0.38f)}, under);
        }
    } else if (e == "anchor") {
        float w = std::max(1.5f, 0.12f * s);
        ring(P(0, -0.68f), 0.2f * s, 0.09f * s, col);
        line(P(0, -0.48f), P(0, 0.8f), w, col);
        line(P(-0.4f, -0.25f), P(0.4f, -0.25f), w, col);
        std::vector<V2> arc;
        for (int i = 0; i <= 16; i++) {
            float a = float(M_PI) * (0.1f + 0.8f * i / 16);
            arc.push_back(P(0.7f * std::cos(a), 0.15f + 0.65f * std::sin(a)));
        }
        polyline(arc, w, col);
    } else if (e == "key") {
        float w = std::max(1.5f, 0.14f * s);
        ring(P(-0.45f, -0.45f), 0.38f * s, 0.13f * s, col);
        line(P(-0.18f, -0.18f), P(0.75f, 0.75f), w, col);
        line(P(0.45f, 0.45f), P(0.62f, 0.28f), w, col);
        line(P(0.62f, 0.62f), P(0.8f, 0.44f), w, col);
    }
}


// ─── @@ART ──────────────────────────────────────────────────────────────────
// Pixel art made in the workshop (@@SCENE_STUDIO): pieces, decorations, board
// squares, holes and board frames, each with its animations. A drawing is a
// square of size × size pixels; an animation is a list of such frames played
// at `fps` frames per second, in a loop or once.
//
//   assets/art/<id>.json   {"name", "kind", "size", "on stone",
//                           "animations": [{"name", "fps", "loop", "frames"}]}
//   assets/art/<id>.png    the frames: one row per animation (in the order of
//                           the JSON), one column per frame
//
// On screen the frames are drawn from one texture with SDL_SCALEMODE_NEAREST:
// each pixel becomes a crisp square, whatever the zoom (pixel art blurred by
// a linear filter looks wrong).
//
// Rendered drawings ("smooth": true, made by app/tools/sprites, up to 512
// pixels): drawn with a linear filter from a chain of halved textures (the
// level closest to the size on screen, so no shimmering when they are much
// smaller than their pixels). They are not edited in the workshop. A piece
// of this kind is a figurine: "box" = its size in squares, "anchor" = the
// point of the frame placed at the centre of its square (the figure stands
// above it), it carries its own shadow; "white" = the drawing used for the
// white side (the black one is this one), "variant of" marks such a drawing
// (not offered on its own).
struct ArtKind {
    const char* id;
    const char* label;
    std::vector<std::pair<const char*, const char*>> anims;   // (name, label); the first one is the default
};
const std::vector<ArtKind> ART_KINDS = {
    {"piece", "Pion", {{"idle", "Repos"}, {"move", "D\xC3\xA9placement"}, {"blocked", "Bloqu\xC3\xA9"}, {"win", "Victoire"}}},
    {"decor", "D\xC3\xA9" "cor", {{"idle", "Animation"}}},
    {"tile", "Case", {{"idle", "Case"}}},
    {"hole", "Trou", {{"idle", "Trou"}, {"open", "Apparition"}}},
    {"frame", "Cadre", {{"idle", "Cadre"}}},
};
const ArtKind& art_kind(const std::string& id) {
    for (auto& k : ART_KINDS)
        if (id == k.id) return k;
    return ART_KINDS[0];
}

// A pixel: 0xAABBGGRR, i.e. the bytes R, G, B, A in memory (stb_image's
// order, SDL_PIXELFORMAT_ABGR8888 on little-endian machines).
using Pixel = uint32_t;
inline Pixel pack(Rgba c) { return Pixel(c.r) | Pixel(c.g) << 8 | Pixel(c.b) << 16 | Pixel(c.a) << 24; }
inline Rgba unpack(Pixel p) { return {uint8_t(p), uint8_t(p >> 8), uint8_t(p >> 16), uint8_t(p >> 24)}; }

struct ArtAnim {
    std::string name;
    float fps = 8;
    bool loop = true;
    std::vector<std::vector<Pixel>> frames;   // size × size each
    double duration() const { return frames.empty() ? 0 : frames.size() / std::max(0.1f, fps); }
};

struct Art {
    std::string id, name = "Nouveau dessin", kind = "piece";
    int size = 32;
    bool on_stone = true;          // a piece: drawn on the stone of its side (black or white)
    bool smooth = false;           // rendered: linear filter, halved levels (see above)
    float box = 1.0f, ax = 0.5f, ay = 0.5f;   // a figurine: size in squares, anchor in the frame
    std::string white, variant_of; // the white side's drawing; the drawing this one is a variant of
    std::vector<ArtAnim> anims;
    SDL_Texture* tex = nullptr;    // the frames, rebuilt when `dirty`
    std::vector<SDL_Texture*> levels;   // smooth: the sheet halved 1, 2, ... times
    bool dirty = true;
    int cols = 1;                  // frames per row in the texture

    ArtAnim* anim(const std::string& n) {
        for (auto& a : anims)
            if (a.name == n) return &a;
        return nullptr;
    }
    const ArtAnim* anim(const std::string& n) const { return const_cast<Art*>(this)->anim(n); }
    // Every animation of its kind, with at least one frame for the first one.
    void complete() {
        for (auto& [n, label] : art_kind(kind).anims)
            if (!anim(n)) anims.push_back(ArtAnim{n});
        auto& first = *anim(art_kind(kind).anims[0].first);
        if (first.frames.empty()) first.frames.push_back(std::vector<Pixel>(size_t(size) * size, 0));
        // Animations of another kind (the kind changed) are kept but unused.
    }
    void update_texture() {
        if (!dirty && tex) return;
        if (tex) SDL_DestroyTexture(tex);
        cols = 1;
        for (auto& a : anims) cols = std::max(cols, int(a.frames.size()));
        int rows = std::max(1, int(anims.size()));
        std::vector<Pixel> sheet(size_t(cols) * size * rows * size, 0);
        for (int r = 0; r < int(anims.size()); r++)
            for (int c = 0; c < int(anims[r].frames.size()); c++)
                for (int y = 0; y < size; y++)
                    std::memcpy(&sheet[(size_t(r) * size + y) * cols * size + size_t(c) * size], &anims[r].frames[c][size_t(y) * size], size * sizeof(Pixel));
        tex = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, cols * size, rows * size);
        SDL_UpdateTexture(tex, nullptr, sheet.data(), cols * size * int(sizeof(Pixel)));
        SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(tex, smooth ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST);
        for (SDL_Texture* l : levels) SDL_DestroyTexture(l);
        levels.clear();
        if (smooth) {   // halve while the frames stay even and larger than 16 pixels
            int w = cols * size, h = rows * size, fs = size;
            std::vector<Pixel> cur = std::move(sheet);
            while (fs % 2 == 0 && fs / 2 >= 16) {
                int w2 = w / 2, h2 = h / 2;
                std::vector<Pixel> nx(size_t(w2) * h2);
                for (int y = 0; y < h2; y++)
                    for (int x = 0; x < w2; x++) {
                        // average of 4 pixels with premultiplied alpha (no dark fringes)
                        float r = 0, g = 0, b = 0, al = 0;
                        for (int k = 0; k < 4; k++) {
                            Rgba c = unpack(cur[size_t(2 * y + k / 2) * w + 2 * x + k % 2]);
                            float aa = c.a / 255.0f;
                            r += c.r * aa; g += c.g * aa; b += c.b * aa; al += aa;
                        }
                        Rgba o{0, 0, 0, 0};
                        if (al > 0) o = {uint8_t(r / al + 0.5f), uint8_t(g / al + 0.5f), uint8_t(b / al + 0.5f), uint8_t(al / 4 * 255 + 0.5f)};
                        nx[size_t(y) * w2 + x] = pack(o);
                    }
                SDL_Texture* lt = SDL_CreateTexture(draw::R, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w2, h2);
                SDL_UpdateTexture(lt, nullptr, nx.data(), w2 * int(sizeof(Pixel)));
                SDL_SetTextureBlendMode(lt, SDL_BLENDMODE_BLEND);
                SDL_SetTextureScaleMode(lt, SDL_SCALEMODE_LINEAR);
                levels.push_back(lt);
                cur = std::move(nx);
                w = w2; h = h2; fs /= 2;
            }
        }
        dirty = false;
    }
    // The frame of animation `n` shown `t` seconds after it started (its row
    // and column in the texture), falling back to the first animation when
    // `n` has no frame; nullopt when there is nothing to draw.
    std::optional<SDL_FRect> frame_rect(const std::string& n, double t) {
        const ArtAnim* a = anim(n);
        if (!a || a->frames.empty()) a = anim(art_kind(kind).anims[0].first);
        if (!a || a->frames.empty()) return std::nullopt;
        int k = int(std::floor(std::max(0.0, t) * a->fps));
        int nf = int(a->frames.size());
        k = a->loop ? k % nf : std::min(k, nf - 1);
        update_texture();
        int row = int(a - anims.data());
        return SDL_FRect{float(k * size), float(row * size), float(size), float(size)};
    }
};

// A decoration placed on a puzzle's board.
struct PlacedDecor {
    std::string art;                  // a "decor" drawing
    float x = 4, y = 4, scale = 1;    // its centre, in squares from the top-left corner of the board (a8 side); its size in squares
};

// All the drawings, by id (loaded at start, changed by the workshop).
std::map<std::string, std::unique_ptr<Art>> g_arts;
Art* find_art(const std::string& id) {
    auto it = g_arts.find(id);
    return it == g_arts.end() ? nullptr : it->second.get();
}

// Draws frame `t` of animation `n` of `art` in box `b`.
void draw_art(Art& art, const std::string& n, double t, Box b, float alpha = 1, bool flip_h = false) {
    auto src = art.frame_rect(n, t);
    if (!src) return;
    SDL_FRect dst = b.frect();
    SDL_Texture* tex = art.tex;
    if (art.smooth) {   // the smallest level still at least as large as the box
        float k = 1;
        for (SDL_Texture* l : art.levels) {
            if (art.size * k * 0.5f < b.w * 0.95f) break;
            tex = l;
            k *= 0.5f;
        }
        src = SDL_FRect{src->x * k, src->y * k, src->w * k, src->h * k};
    }
    SDL_SetTextureAlphaModFloat(tex, alpha);
    SDL_RenderTextureRotated(draw::R, tex, &*src, &dst, 0, nullptr, flip_h ? SDL_FLIP_HORIZONTAL : SDL_FLIP_NONE);
}
// A board frame: the drawing cut in 3 × 3 (SDL_RenderTexture9Grid): the
// corners keep their shape, the edges stretch along the frame, the centre
// (under the board) fills the rest; `border` is the frame's thickness on screen.
void draw_art_frame(Art& art, double t, Box b, float border) {
    auto src = art.frame_rect("idle", t);
    if (!src) return;
    float third = art.size / 3.0f;
    SDL_FRect dst = b.frect();
    SDL_SetTextureAlphaModFloat(art.tex, 1);
    SDL_RenderTexture9Grid(draw::R, art.tex, &*src, third, third, third, third, border / third, &dst);
}

const fs::path& art_dir() {
    static fs::path d;
    d = g_paths.assets / "art";
    return d;
}

bool load_art(const std::string& id, Art& out) {
    json j = load_json(art_dir() / (id + ".json"));
    if (!j.is_object()) return false;
    out.id = id;
    out.name = jstr(j, "name", id);
    out.kind = jstr(j, "kind", "piece");
    out.smooth = j.value("smooth", false);
    out.size = std::clamp(int(jnum(j, "size", 32)), 4, out.smooth ? 512 : 128);
    out.on_stone = j.value("on stone", true);
    out.box = jnum(j, "box", 1.0f);
    if (j.contains("anchor") && j["anchor"].is_array() && j["anchor"].size() == 2) {
        out.ax = j["anchor"][0].get<float>();
        out.ay = j["anchor"][1].get<float>();
    }
    out.white = jstr(j, "white");
    out.variant_of = jstr(j, "variant of");
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load((art_dir() / (id + ".png")).string().c_str(), &w, &h, &n, 4);
    if (j.contains("animations") && j["animations"].is_array()) {
        int row = 0;
        for (auto& e : j["animations"]) {
            ArtAnim a;
            a.name = jstr(e, "name", "idle");
            a.fps = std::clamp(jnum(e, "fps", 8), 0.5f, 60.0f);
            a.loop = e.value("loop", true);
            int nf = int(jnum(e, "frames", 0));
            for (int c = 0; c < nf; c++) {
                std::vector<Pixel> f(size_t(out.size) * out.size, 0);
                if (px)
                    for (int y = 0; y < out.size; y++)
                        for (int x = 0; x < out.size; x++) {
                            int sx = c * out.size + x, sy = row * out.size + y;
                            if (sx < w && sy < h) std::memcpy(&f[size_t(y) * out.size + x], px + (size_t(sy) * w + sx) * 4, 4);
                        }
                a.frames.push_back(std::move(f));
            }
            out.anims.push_back(std::move(a));
            row++;
        }
    }
    if (px) stbi_image_free(px);
    out.complete();
    out.dirty = true;
    return true;
}

bool save_art(Art& art) {
    if (art.id.empty()) {
        std::string base = slug(art.name), id = base;
        for (int k = 2; fs::exists(art_dir() / (id + ".json")); k++) id = base + "_" + std::to_string(k);
        art.id = id;
    }
    int cols = 1, rows = std::max(1, int(art.anims.size()));
    for (auto& a : art.anims) cols = std::max(cols, int(a.frames.size()));
    std::vector<Pixel> sheet(size_t(cols) * art.size * rows * art.size, 0);
    json anims = json::array();
    for (int r = 0; r < int(art.anims.size()); r++) {
        auto& a = art.anims[r];
        anims.push_back({{"name", a.name}, {"fps", a.fps}, {"loop", a.loop}, {"frames", a.frames.size()}});
        for (int c = 0; c < int(a.frames.size()); c++)
            for (int y = 0; y < art.size; y++)
                std::memcpy(&sheet[(size_t(r) * art.size + y) * cols * art.size + size_t(c) * art.size], &a.frames[c][size_t(y) * art.size], art.size * sizeof(Pixel));
    }
    std::error_code ec;
    fs::create_directories(art_dir(), ec);
    bool ok = stbi_write_png((art_dir() / (art.id + ".png")).string().c_str(), cols * art.size, rows * art.size, 4, sheet.data(), cols * art.size * 4) != 0;
    json j = {{"name", art.name}, {"kind", art.kind}, {"size", art.size}, {"on stone", art.on_stone}, {"animations", anims}};
    return ok && write_file(art_dir() / (art.id + ".json"), j.dump(2));
}

void load_all_arts() {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(art_dir(), ec)) {
        if (e.path().extension() != ".json") continue;
        auto art = std::make_unique<Art>();
        std::string id = e.path().stem().string();
        if (load_art(id, *art)) g_arts[id] = std::move(art);
    }
}
// The drawings of a kind, by name (smooth_too: also the rendered ones, which
// the workshop does not edit; never the white variants).
std::vector<Art*> arts_of(const std::string& kind, bool smooth_too = true) {
    std::vector<Art*> v;
    for (auto& [id, a] : g_arts)
        if (a->kind == kind && (smooth_too || !a->smooth) && a->variant_of.empty()) v.push_back(a.get());
    std::sort(v.begin(), v.end(), [](Art* x, Art* y) { return x->name < y->name; });
    return v;
}
// The pieces of the workshop are sprites too ("art:<id>", category
// "Mes dessins"): the sprite picker, the themes scene and the puzzles use
// them like the others.
void register_art_sprites(std::vector<SpriteDef>& sprites) {
    sprites.erase(std::remove_if(sprites.begin(), sprites.end(), [](const SpriteDef& sp) { return sp.id.rfind("art:", 0) == 0; }), sprites.end());
    for (Art* a : arts_of("piece", false)) {
        SpriteDef sp;
        sp.id = "art:" + a->id;
        sp.name = a->name;
        sp.category = "Mes dessins";
        sp.art = a->id;
        sp.emblem = "none";
        sprites.push_back(sp);
    }
    link_sprites(sprites);
}

// Classic stones without any sign: black or white by the side.
//   go       slate and shell: biconvex, a broad soft light, a darker rim
//   flat     a flat counter with a thin edge
//   checkers a draughts piece: flat, two grooves
//   glass    a glass bead: deep colour, a sharp highlight, light through it
//   marble   polished stone with a few soft veins
void draw_plain_stone(const std::string& style, int side, V2 c, float r, float alpha) {
    using namespace draw;
    const bool B = side == BLACK;
    auto A = [&](Rgba k) { return with_alpha(k, alpha); };
    V2 light = c + V2{-r * 0.35f, -r * 0.4f};
    if (style == "go") {
        if (B) circle_lit(c, r, light, A(hex(0x4a4e55)), A(hex(0x050608)));
        else circle_lit(c, r, light, A(hex(0xffffff)), A(hex(0xc9c4b6)));
        ring(c, r, r * 0.10f, A(hex(0x000000, 0)), A(hex(0x000000, B ? 90 : 45)));     // the rim turns away
        circle(c + V2{-r * 0.28f, -r * 0.34f}, r * 0.42f, A(hex(0xffffff, B ? 45 : 110)), A(hex(0xffffff, 0)), r * 0.3f);
    } else if (style == "flat") {
        circle(c, r, A(B ? hex(0x111318) : hex(0xb8bcc4)));
        circle(c, r * 0.9f, A(B ? hex(0x262a31) : hex(0xf4f5f7)));
        circle(c + V2{-r * 0.2f, -r * 0.25f}, r * 0.55f, A(hex(0xffffff, B ? 18 : 40)), A(hex(0xffffff, 0)), r * 0.4f);
    } else if (style == "checkers") {
        Rgba face = B ? hex(0x23262c) : hex(0xeeeae0), edge = B ? hex(0x0b0c0f) : hex(0xb3ad9e);
        circle(c, r, A(edge));
        circle_lit(c, r * 0.93f, light, A(shade(face, 0.10f)), A(face));
        for (float k : {0.72f, 0.48f}) {   // grooves: a dark line with a light line under it
            ring(c, r * k, r * 0.035f, A(with_alpha(edge, 0.9f)));
            ring(c, r * k - r * 0.035f, r * 0.025f, A(hex(0xffffff, B ? 25 : 120)));
        }
    } else if (style == "glass") {
        if (B) circle_lit(c, r, c + V2{r * 0.3f, r * 0.35f}, A(hex(0x3a4150)), A(hex(0x020306)));   // light comes out at the bottom
        else circle_lit(c, r, c + V2{r * 0.3f, r * 0.35f}, A(hex(0xffffff)), A(hex(0xd5dbe4)));
        ring(c, r, r * 0.06f, A(hex(0xffffff, B ? 40 : 70)), A(hex(0x000000, B ? 60 : 30)));
        circle(c + V2{-r * 0.36f, -r * 0.42f}, r * 0.20f, A(hex(0xffffff, 230)), A(hex(0xffffff, 0)), r * 0.10f);
        circle(c + V2{r * 0.30f, r * 0.42f}, r * 0.16f, A(hex(0xffffff, B ? 40 : 90)), A(hex(0xffffff, 0)), r * 0.12f);
    } else {   // marble
        if (B) circle_lit(c, r, light, A(hex(0x3d3f46)), A(hex(0x0c0d10)));
        else circle_lit(c, r, light, A(hex(0xfbfaf7)), A(hex(0xcfcac0)));
        Rgba vein = B ? hex(0x8a8f99, 70) : hex(0x8c8576, 80);
        for (int i = 0; i < 3; i++) {   // veins: a few gentle curves inside the stone
            std::vector<V2> pts;
            float y0 = (-0.45f + 0.45f * i) * r, ph = 1.3f * i;
            for (int k = 0; k <= 40; k++) {
                float x = (-0.85f + 1.7f * k / 40) * r, y = y0 + std::sin(x / r * 3.1f + ph) * r * 0.14f + x * 0.25f;
                if (x * x + y * y < r * r * 0.80f) pts.push_back(c + V2{x, y});
            }
            if (pts.size() > 1) polyline(pts, r * 0.035f, A(vein));
        }
        circle(c + V2{-r * 0.3f, -r * 0.36f}, r * 0.3f, A(hex(0xffffff, B ? 60 : 130)), A(hex(0xffffff, 0)), r * 0.2f);
    }
}

// Draws a piece of `side` with sprite `sp`, centred at c, radius r.
// lift: 0 on the board, up to 1 when picked up (bigger, higher shadow).
// time: seconds, for the halo's pulse.
// halo: no longer used (the sprites tell who is who; a sprite can still ask
// for a halo in sprites.json).
// state, state_t: a drawing of the workshop plays its animation `state`
// ("idle", "move", "blocked", "win"), started `state_t` seconds ago (the
// other sprites ignore them).
void draw_piece(const SpriteDef& sp, int side, V2 c, float r, float lift = 0, double time = 0, float alpha = 1, bool halo = false,
                const char* state = "idle", double state_t = -1, bool flip = false) {
    using namespace draw;
    if (side == WHITE && sp.white_alt) {   // a set of pieces: its white unit
        draw_piece(*sp.white_alt, side, c, r, lift, time, alpha, halo, state, state_t, flip);
        return;
    }
    r *= 1 + 0.08f * lift;
    Art* art = sp.art.empty() ? nullptr : find_art(sp.art);
    if (art && art->smooth) {   // a rendered figurine: its own base and shadow, the white side's drawing
        Art* drawn = art;
        if (side == WHITE && !art->white.empty())
            if (Art* w = find_art(art->white)) drawn = w;
        float size = 2.5f * r * art->box;      // r = 0.4 square
        float ay = art->ay;
        if (state_t < 0) {   // not on a board (panels, lists, the picker): the whole figurine around c
            size *= 0.72f;
            ay = 0.6f;
        }
        Box b{c.x - size * (flip ? 1 - art->ax : art->ax), c.y - size * ay, size, size};
        draw_art(*drawn, state, state_t >= 0 ? state_t : time, b, alpha, flip);
        return;
    }
    // Shadow.
    float soff = r * (0.10f + 0.25f * lift);
    circle(c + V2{soff * 0.4f, soff}, r * (1.0f + 0.05f * lift), hex(0x000000, uint8_t(110 * alpha)), hex(0x000000, 0), r * 0.35f);
    (void)halo;
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
    if (art) {
        double t = state_t >= 0 ? state_t : time;
        if (art->on_stone) {   // the stone tells the side; the drawing sits on it
            V2 light = c + V2{-r * 0.35f, -r * 0.4f};
            if (side == BLACK) circle_lit(c, r, light, with_alpha(hex(0x5b6170), alpha), with_alpha(hex(0x0a0c11), alpha));
            else circle_lit(c, r, light, with_alpha(hex(0xffffff), alpha), with_alpha(hex(0xb9bec8), alpha));
            draw_art(*art, state, t, {c.x - r * 0.78f, c.y - r * 0.78f, r * 1.56f, r * 1.56f}, alpha);
        } else {
            draw_art(*art, state, t, {c.x - r * 1.1f, c.y - r * 1.1f, r * 2.2f, r * 2.2f}, alpha);
        }
        return;
    }
    if (!sp.style.empty()) { draw_plain_stone(sp.style, side, c, r, alpha); return; }
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
    Rgba under = side == BLACK ? hex(0x25282f) : hex(0xd0d3d9);   // the stone's colour at its centre
    draw_emblem(sp.emblem, c, r * 0.42f, with_alpha(ec, alpha), with_alpha(under, alpha));
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

// A square breaking when a piece leaves it, in the style of the theme, as
// strong as the player who moved (power 0..1: its Elo on the roster's scale).
//   debris  chips thrown up (the classic boards)
//   stone   rock chunks under gravity, a dust cloud, sparks for the strong
//   space   matter sucked in a spiral to the centre, a flash, a blue shock
//           ring; star rays for the strongest
//   ink     manga impact: ink speed lines, a white starburst, ink drops, a ring
//   cannon  a muzzle flash, smoke puffs, grapeshot, earth thrown up
struct Shatter : Effect {
    struct P { V2 p, v; float life, max, size, spin, ang; Rgba c; int kind; };   // kind: 0 dot, 1 chunk, 2 puff, 3 spark
    std::vector<P> ps;
    std::string style;
    V2 c;
    float sq, power, t = 0, dur;
    std::vector<float> rays;      // angles of the speed lines / star rays
    Shatter(V2 c, float sq, const Theme& th, float power, std::mt19937& rng)
        : style(th.break_fx), c(c), sq(sq), power(std::clamp(power, 0.0f, 1.0f)) {
        std::uniform_real_distribution<float> U(0, 1);
        const float P = this->power;
        dur = 0.7f + 0.5f * P;
        auto add = [&](V2 p, V2 v, float max, float size, Rgba col, int kind) {
            ps.push_back({p, v, 0, max, size, (U(rng) - 0.5f) * 12, U(rng) * 6.3f, col, kind});
        };
        if (style == "stone") {
            int n = 10 + int(22 * P);
            for (int i = 0; i < n; i++) {
                float a = -float(M_PI) * (0.1f + 0.8f * U(rng)), sp = sq * (0.8f + 2.2f * U(rng)) * (0.7f + 0.6f * P);
                add(c + V2{(U(rng) - 0.5f) * sq * 0.6f, (U(rng) - 0.5f) * sq * 0.6f}, V2{std::cos(a) * sp, std::sin(a) * sp},
                    0.7f + 0.5f * U(rng), sq * (0.07f + 0.11f * U(rng)) * (0.8f + 0.5f * P), shade(th.light, (U(rng) - 0.6f) * 0.6f), 1);
            }
            for (int i = 0; i < 6 + int(6 * P); i++)
                add(c + V2{(U(rng) - 0.5f) * sq * 0.5f, (U(rng) - 0.5f) * sq * 0.3f}, V2{(U(rng) - 0.5f) * sq * 0.8f, -sq * (0.2f + 0.4f * U(rng))},
                    0.9f + 0.6f * U(rng), sq * (0.22f + 0.18f * U(rng)) * (0.8f + 0.5f * P), hex(0xc8b8a0, 210), 2);
            if (P > 0.5f)
                for (int i = 0; i < int(30 * P); i++) {
                    float a = U(rng) * 6.283f, sp = sq * (1.5f + 3.0f * U(rng));
                    add(c, V2{std::cos(a) * sp, std::sin(a) * sp - sq}, 0.4f + 0.4f * U(rng), sq * 0.02f, hex(0xffb040), 3);
                }
        } else if (style == "space") {
            int n = 24 + int(40 * P);
            for (int i = 0; i < n; i++) {     // they start around and spiral in
                float a = U(rng) * 6.283f, rr = sq * (0.5f + 0.9f * U(rng));
                add(c + V2{std::cos(a), std::sin(a)} * rr, V2{a, rr}, 0.5f + 0.4f * U(rng), sq * (0.015f + 0.03f * U(rng)),
                    U(rng) < 0.5f ? hex(0x9fdcff) : hex(0xffd2a0), 0);
            }
            for (int i = 0; i < (P > 0.7f ? 6 : 0); i++) rays.push_back(i * 3.14159f / 3 + U(rng) * 0.2f);
        } else if (style == "ink") {
            int nl = 10 + int(18 * P);
            for (int i = 0; i < nl; i++) rays.push_back(U(rng) * 6.283f);
            for (int i = 0; i < 8 + int(16 * P); i++) {
                float a = U(rng) * 6.283f, sp = sq * (1.0f + 2.5f * U(rng)) * (0.6f + 0.7f * P);
                add(c, V2{std::cos(a) * sp, std::sin(a) * sp - sq * 0.5f}, 0.45f + 0.35f * U(rng), sq * (0.025f + 0.05f * U(rng)), hex(0x1b1420), 0);
            }
        } else if (style == "cannon") {
            for (int i = 0; i < 6 + int(8 * P); i++)
                add(c + V2{(U(rng) - 0.5f) * sq * 0.4f, 0}, V2{(U(rng) - 0.5f) * sq * 0.9f, -sq * (0.4f + 0.8f * U(rng))},
                    1.0f + 0.8f * U(rng), sq * (0.22f + 0.2f * U(rng)), hex(0xe0dad0, 220), 2);
            for (int i = 0; i < 10 + int(20 * P); i++) {
                float a = -float(M_PI) * (0.05f + 0.9f * U(rng)), sp = sq * (1.0f + 2.5f * U(rng)) * (0.7f + 0.6f * P);
                add(c, V2{std::cos(a) * sp, std::sin(a) * sp}, 0.5f + 0.4f * U(rng), sq * (0.03f + 0.05f * U(rng)), shade(th.dark, -0.2f), 1);
            }
            for (int i = 0; i < int(20 * P) + 6; i++) {
                float a = U(rng) * 6.283f, sp = sq * (2.0f + 3.0f * U(rng));
                add(c, V2{std::cos(a) * sp, std::sin(a) * sp}, 0.25f + 0.25f * U(rng), sq * 0.018f, hex(0xffd070), 3);
            }
        } else {   // debris
            int n = 12 + int(18 * P);
            for (int i = 0; i < n; i++) {
                float a = U(rng) * 2 * float(M_PI), sp = sq * (0.6f + 1.6f * U(rng)) * (0.8f + 0.5f * P);
                add(c + V2{(U(rng) - 0.5f) * sq * 0.5f, (U(rng) - 0.5f) * sq * 0.5f}, V2{std::cos(a) * sp, std::sin(a) * sp - sq * 1.2f},
                    0.45f + 0.4f * U(rng), sq * (0.03f + 0.05f * U(rng)), shade(th.dark, (U(rng) - 0.5f) * 0.5f), 0);
            }
        }
    }
    bool update(float dt) override {
        t += dt;
        for (auto& p : ps) {
            p.life += dt;
            if (style == "space") continue;              // computed from the time in draw
            float g = p.kind == 2 ? -sq * 0.3f : p.kind == 3 ? sq * 1.5f : sq * 5.0f;   // puffs rise slowly
            p.v.y += g * dt;
            if (p.kind == 2) p.v = p.v * (1 - 1.5f * dt);
            p.p = p.p + p.v * dt;
            p.ang += p.spin * dt;
        }
        return t < dur + 0.6f;
    }
    void draw(TextEngine&) override {
        using namespace draw;
        const float P = power;
        const float k = std::clamp(t / dur, 0.0f, 1.0f);
        if (style == "space") {
            // The black hole swallows the light: the whole window goes dark at
            // once and comes back slowly, the hole lit by its photon ring on top.
            float dark = std::min(1.0f, t / 0.06f) * std::exp(-std::max(0.0f, t - 0.25f) * (2.2f - 0.8f * P));
            if (dark > 0.01f) {
                int ww = 0, wh = 0;
                SDL_GetCurrentRenderOutputSize(R, &ww, &wh);
                rect({0, 0, float(ww), float(wh)}, hex(0x000000, uint8_t(255 * (0.62f + 0.25f * P) * dark)));
                circle(c, sq * (1.6f + 1.2f * P), hex(0x000000, 0), hex(0x000000, uint8_t(120 * dark)), 0);
                circle(c, sq * 0.42f, hex(0x000000, uint8_t(255 * dark)));
                ring(c, sq * 0.46f, sq * 0.05f, hex(0xffd8a0, uint8_t(255 * dark)));
                ring(c, sq * 0.52f, sq * 0.08f, hex(0xff9a40, uint8_t(120 * dark)));
            }
            // flash and shock ring
            float fl = std::exp(-t * 5);
            circle(c, sq * (0.5f + 0.6f * P), hex(0xe8f4ff, uint8_t(240 * fl)), hex(0x9fdcff, 0), 0);
            ring(c, sq * (0.3f + (1.2f + 1.3f * P) * ease_out_cubic(k)), sq * (0.05f + 0.04f * P) * (1 - k), hex(0x7fd0ff, uint8_t(240 * (1 - k))));
            for (float a : rays) {
                float l = sq * (0.8f + 1.4f * P) * (1 - k);
                V2 d{std::cos(a), std::sin(a)};
                line(c - d * l, c + d * l, sq * 0.03f * (1 - k), hex(0xffffff, uint8_t(200 * (1 - k))));
            }
            for (auto& p : ps) {
                float u = std::clamp(p.life / p.max, 0.0f, 1.0f);
                float a = p.v.x + u * (4 + 6 * P), rr = p.v.y * (1 - u) * (1 - u);     // spiral in
                V2 q = c + V2{std::cos(a), std::sin(a) * 0.6f} * rr;
                if (u < 1) {
                    circle(q, p.size * 3 * (1 - u * 0.5f), with_alpha(p.c, 0.3f * (1 - u)), with_alpha(p.c, 0), 0);
                    circle(q, p.size * 1.4f * (1 - u * 0.5f), with_alpha(p.c, 1 - u));
                }
            }
            return;
        }
        if (style == "ink") {
            float fl = std::exp(-t * 4);
            if (fl > 0.02f) {   // a white starburst with an ink edge
                std::vector<V2> star;
                int n = 14;
                float R = sq * (0.55f + 0.5f * P) * (0.8f + 0.4f * ease_out_cubic(std::min(1.0f, t * 6)));
                for (int i = 0; i < 2 * n; i++) {
                    float a = i * 3.14159f / n, rr = (i % 2 ? 0.45f : 1.0f) * R;
                    star.push_back(c + V2{std::cos(a), std::sin(a)} * rr);
                }
                for (size_t i = 0; i < star.size(); i++) line(star[i], star[(i + 1) % star.size()], sq * 0.035f, hex(0x1b1420, uint8_t(255 * fl)));
                poly_around(c, star, hex(0xffffff, uint8_t(255 * fl)));
            }
            for (float a : rays) {   // speed lines rushing out
                V2 d{std::cos(a), std::sin(a)};
                float r0 = sq * (0.5f + 1.6f * ease_out_cubic(k)), r1 = r0 + sq * (0.4f + 0.6f * P) * (1 - k);
                line(c + d * r0, c + d * r1, sq * 0.03f * (1 - k) + 1, hex(0x1b1420, uint8_t(230 * (1 - k))));
            }
            ring(c, sq * (0.4f + (1.0f + P) * ease_out_cubic(k)), sq * 0.03f * (1 - k), hex(0x1b1420, uint8_t(200 * (1 - k))));
        }
        if (style == "cannon") {
            float fl = std::exp(-t * 6);
            circle(c, sq * (0.55f + 0.45f * P), hex(0xfff0b0, uint8_t(250 * fl)), hex(0xff7a20, 0), 0);
        }
        if (style == "stone" && P > 0.6f) {   // a shock ring of dust for the strong
            ring(c, sq * (0.3f + 1.4f * ease_out_cubic(k)), sq * 0.05f * (1 - k), hex(0xd8c8a8, uint8_t(150 * (1 - k))));
        }
        for (auto& p : ps) {
            if (p.life >= p.max) continue;
            float u = p.life / p.max, a = 1 - u;
            if (p.kind == 1) {   // a chunk: a small turning quad
                std::vector<V2> q;
                for (int i = 0; i < 4; i++) {
                    float an = p.ang + i * 1.5708f + (i % 2) * 0.3f;
                    q.push_back(p.p + V2{std::cos(an), std::sin(an)} * p.size * (i % 2 ? 0.8f : 1.1f));
                }
                poly_around(p.p, q, with_alpha(p.c, a));
            } else if (p.kind == 2) {   // a puff: grows and fades
                circle(p.p, p.size * (0.6f + 1.2f * u), with_alpha(p.c, 0.8f * a), with_alpha(p.c, 0), p.size);
            } else if (p.kind == 3) {   // a spark: a short streak along its speed
                line(p.p, p.p - p.v * 0.03f, std::max(1.0f, p.size), with_alpha(p.c, a));
            } else {
                circle(p.p, p.size, with_alpha(p.c, a));
            }
        }
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
    bool rdown = false, rpressed = false;   // right button: held, went down this frame
    float wheel = 0;
    // Files given this frame: dropped on the window ("drop") or chosen in a
    // file dialog (the purpose given to App::choose_file).
    std::vector<std::pair<std::string, std::string>> files;   // (purpose, path)
    std::vector<SDL_Keycode> keys;   // keys pressed this frame
    std::string text;                // characters typed this frame (UTF-8, SDL_EVENT_TEXT_INPUT)
    int backspaces = 0;              // Backspace presses this frame, key repeats included
    bool key(SDL_Keycode k) const { return std::find(keys.begin(), keys.end(), k) != keys.end(); }
    void eat(SDL_Keycode k) { keys.erase(std::remove(keys.begin(), keys.end(), k), keys.end()); }
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
    std::string focus;        // the text field being typed in ("" = none)
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
    // A one-line text field: click to type; Enter, Escape or a click elsewhere
    // to stop. SDL only sends typed text between SDL_StartTextInput and
    // SDL_StopTextInput (see SDL_AppIterate, which follows `focus`).
    bool text_field(const std::string& id, Box b, std::string& value, const std::string& placeholder, float font = 17) {
        bool h = hover(b);
        if (h) hovering = true;
        if (in.pressed && !blocked) {
            if (h) focus = id;
            else if (focus == id) focus.clear();
        }
        const bool f = focus == id;
        bool changed = false;
        if (f) {
            if (!in.text.empty()) { value += in.text; changed = true; }
            for (int i = 0; i < in.backspaces && !value.empty(); i++) {
                size_t k = value.size() - 1;
                while (k > 0 && (uint8_t(value[k]) & 0xC0) == 0x80) k--;   // a whole UTF-8 character
                value.erase(k);
                changed = true;
            }
            if (in.key(SDLK_RETURN) || in.key(SDLK_ESCAPE)) { focus.clear(); in.eat(SDLK_RETURN); in.eat(SDLK_ESCAPE); }
        }
        draw::round_box(b, 8 * s, hex(0x0f141c, 230));
        draw::round_frame(b, 8 * s, f ? 2 * s : 1, f ? pal::accent : pal::panel_border);
        const float fs = font * s, ty = b.center().y - text.line_height(Font::Regular, fs) * 0.45f;
        std::string shown = value;   // the end of the text when it is too long
        while (!shown.empty() && text.width(Font::Regular, fs, shown) > b.w - 26 * s) {
            shown.erase(0, 1);
            while (!shown.empty() && (uint8_t(shown[0]) & 0xC0) == 0x80) shown.erase(0, 1);
        }
        if (value.empty() && !f) {
            text.draw(Font::Regular, fs, placeholder, {b.x + 12 * s, ty}, pal::faint);
        } else {
            float w = text.draw(Font::Regular, fs, shown, {b.x + 12 * s, ty}, pal::text);
            if (f && std::fmod(now_seconds(), 1.0) < 0.55) draw::rect({b.x + 13 * s + w, ty + 2 * s, 2 * s, fs * 1.05f}, pal::accent);
        }
        return changed;
    }
    // A label and "‹ value ›": returns the new index in names.
    int cycler(const std::string& id, Box b, const std::string& label, const std::vector<std::string>& names, int idx, float font = 16) {
        if (names.empty()) return 0;
        text.draw(Font::Regular, font * s, label, {b.x, b.center().y - text.line_height(Font::Regular, font * s) * 0.45f}, pal::dim);
        float vw = b.w * 0.62f;
        Box v{b.right() - vw, b.y, vw, b.h};
        draw::round_box(v, b.h / 2, hex(0x0f141c, 220));
        Box l{v.x, v.y, v.h, v.h}, r{v.right() - v.h, v.y, v.h, v.h};
        for (int k = 0; k < 2; k++) {
            Box a = k ? r : l;
            bool h = hover(a);
            if (h) hovering = true;
            float an = anim(id + (k ? ">" : "<"), h);
            draw::circle(a.center(), a.h * 0.4f, with_alpha(pal::accent, 0.15f + 0.5f * an));
            text.draw(Font::Black, font * 1.2f * s, k ? "\xE2\x80\xBA" : "\xE2\x80\xB9", {a.center().x, a.center().y - text.line_height(Font::Black, font * 1.2f * s) * 0.5f}, pal::text, Align::Center);
            if (h && in.released) idx = (idx + (k ? 1 : int(names.size()) - 1)) % int(names.size());
        }
        std::string shown = names[std::clamp(idx, 0, int(names.size()) - 1)];
        while (shown.size() > 3 && text.width(Font::Bold, font * s, shown) > vw - 2.4f * v.h) shown = shown.substr(0, shown.size() - 4) + "\xE2\x80\xA6";
        text.draw(Font::Bold, font * s, shown, {v.center().x, v.center().y - text.line_height(Font::Bold, font * s) * 0.45f}, pal::text, Align::Center);
        return idx;
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
    std::array<bool, 64> face_left{};      // the piece on this square looks left (its last move went left)
    // Which piece of its side stands on each square (0..3), so that the units
    // of a set keep their identity as they move: unit_side[sq] = its side.
    std::array<int8_t, 64> unit{}, unit_side{};
    // Numbers the pieces again (in square order) when the position does not
    // match what we know (a new game, a jump in the history).
    void sync_units(const Yolah& pos) {
        for (int side : {BLACK, WHITE}) {
            uint64_t bb = pos.bitboard(side);
            bool ok = true;
            std::array<bool, 8> seen{};
            for (uint64_t b = bb; b; b &= b - 1) {
                int sq = std::countr_zero(b);
                int u = unit[sq];
                if (unit_side[sq] != side || u < 0 || u > 7 || seen[u]) { ok = false; break; }
                seen[u] = true;
            }
            if (ok) continue;
            int k = 0;
            for (uint64_t b = bb; b; b &= b - 1) {
                int sq = std::countr_zero(b);
                unit[sq] = int8_t(k++);
                unit_side[sq] = int8_t(side);
            }
        }
    }
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
        face_left.fill(false);
        unit.fill(-1);
        unit_side.fill(-1);
        last = Move::none();
        selected = drag_from = -1;
        dragging = false;
    }
    // A move was played (match already updated): animate it.
    void on_move(Move m, int side, double now, FxLayer& fx, const Theme& theme, std::mt19937& rng, float power = 0.4f) {
        last = m;
        selected = -1;
        if (m == Move::none()) return;
        anim = {true, m, side, now, 0.34};
        hole_since[m.from_sq()] = now + 0.08;
        float dx = center(m.to_sq()).x - center(m.from_sq()).x;     // on screen (the board may be flipped)
        face_left[m.to_sq()] = dx < -1 ? true : dx > 1 ? false : face_left[m.from_sq()];
        unit[m.to_sq()] = unit[m.from_sq()];          // (kept on from: an undo finds it there)
        unit_side[m.to_sq()] = unit_side[m.from_sq()];
        fx.add(std::make_unique<Shatter>(center(m.from_sq()), sq(), theme, power, rng));
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
        std::vector<const SpriteDef*> units[2];   // the sprites of the 4 pieces of a side (a set's units); empty: sprite[side]
        bool halo[2] = {false, false};   // the human players
        std::string hole_style = "pit";  // HOLE_STYLES
        bool coords = true;
        std::optional<Move> hint;
        std::string hint_label;
        std::vector<Move> pv;     // drawn as faint numbered arrows (analysis tab)
        bool human_turn = false;
        int winner = -1;          // the game is over and this side won: its pieces play "win"
        const std::map<int, std::string>* tiles = nullptr;          // a puzzle's special squares
        const std::vector<PlacedDecor>* decor = nullptr;            // a puzzle's decorations
    };

    void draw(TextEngine& text, BoardTextures& textures, const Theme& t, const Yolah& pos, const Look& look, double now, float s) {
        using namespace draw;
        const float q = sq();
        sync_units(pos);
        // The sprite of the piece on square sq of `side` (its unit in a set).
        auto piece_sprite = [&](int side, int sq) -> const SpriteDef& {
            const auto& u = look.units[side];
            if (u.empty()) return *look.sprite[side];
            int k = sq >= 0 && unit[sq] >= 0 ? unit[sq] : 0;
            return *u[size_t(k) % u.size()];
        };
        // The theme's planet, under the board (the board stands on its top).
        if (SDL_Texture* pl = scenery_texture(t.planet_image)) {
            float size = area.w * t.planet_size;
            SDL_FRect dst{area.center().x - size / 2, area.y + area.h * t.planet_top, size, size};
            SDL_RenderTexture(R, pl, nullptr, &dst);
        }
        // Frame and shadow.
        Box frame = area.inset(-26 * s);
        if (t.planet_image.empty()) shadow({frame.x, frame.y + 14 * s, frame.w, frame.h}, 22 * s, 46 * s, hex(0x000000, 150));
        if (Art* fa = t.frame_art.empty() ? nullptr : find_art(t.frame_art)) {
            draw_art_frame(*fa, now, frame, 26 * s);
        } else {
            round_box(frame, 22 * s, t.frame_top, t.frame_bottom);
            round_frame(frame.inset(6 * s), 17 * s, std::max(1.0f, 1.5f * s), with_alpha(t.frame_line, 0.55f));
            round_frame(area.inset(-3 * s), 4 * s, std::max(1.0f, 2 * s), with_alpha(hex(0x000000), 0.35f));
        }
        // The squares (flipped boards rotate the texture by 180°).
        SDL_Texture* tex = textures.get(t, int(area.w));
        SDL_FRect dst = area.frect();
        SDL_RenderTextureRotated(R, tex, nullptr, &dst, flipped ? 180.0 : 0.0, nullptr, SDL_FLIP_NONE);
        // Squares drawn in the workshop: the theme's, then the puzzle's special ones.
        Art* tl = t.tile_light.empty() ? nullptr : find_art(t.tile_light);
        Art* td = t.tile_dark.empty() ? nullptr : find_art(t.tile_dark);
        for (int s2 = 0; s2 < 64; s2++) {
            Art* ta = (sq_file(s2) + sq_rank(s2)) % 2 == 1 ? tl : td;   // a1 is dark
            if (look.tiles) {
                auto it = look.tiles->find(s2);
                if (it != look.tiles->end())
                    if (Art* pa = find_art(it->second)) ta = pa;
            }
            if (ta) draw_art(*ta, "idle", now, square(s2));
        }
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
            draw_hole(square(s2), t, k, s, look.hole_style, now, s2);
        }
        // Decorations (a puzzle's), above the squares and the holes, below the pieces.
        if (look.decor)
            for (auto& d : *look.decor)
                if (Art* da = find_art(d.art)) {
                    float x = flipped ? 8 - d.x : d.x, y = flipped ? 8 - d.y : d.y, w = d.scale * q;
                    draw_art(*da, "idle", now + d.x * 0.37 + d.y * 0.61, {area.x + x * q - w / 2, area.y + y * q - w / 2, w, w});
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
        const uint64_t taken = pos.bitboard(BLACK) | pos.bitboard(WHITE) | pos.empty_bitboard();
        auto blocked = [&](int s2) {   // no free square around: the piece cannot move
            for (int df = -1; df <= 1; df++)
                for (int dr = -1; dr <= 1; dr++) {
                    int f = sq_file(s2) + df, r = sq_rank(s2) + dr;
                    if ((df || dr) && f >= 0 && f < 8 && r >= 0 && r < 8 && !(taken >> (r * 8 + f) & 1)) return false;
                }
            return true;
        };
        // From the top of the screen to the bottom: figurines standing above
        // their square overlap the ones behind them.
        std::vector<std::pair<int, int>> pieces;      // (square, side)
        for (int side : {BLACK, WHITE}) {
            uint64_t bb = pos.bitboard(side);
            for (int s2 = 0; s2 < 64; s2++)
                if (bb >> s2 & 1) pieces.push_back({s2, side});
        }
        std::sort(pieces.begin(), pieces.end(), [&](auto& x, auto& y) { return center(x.first).y < center(y.first).y; });
        for (auto [s2, side] : pieces) {
            if (anim.active && now < anim.t0 + anim.dur && s2 == anim.m.to_sq()) continue;
            if (dragging && s2 == drag_from) continue;
            const char* st = look.winner == side ? "win" : blocked(s2) ? "blocked" : "idle";
            draw_piece(piece_sprite(side, s2), side, center(s2), q * 0.4f, 0, now, 1, look.halo[side], st, now + s2 * 0.29, face_left[s2]);
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
            draw_piece(piece_sprite(anim.side, anim.m.to_sq()), anim.side, p + V2{0, -lift * q * 0.12f}, q * 0.4f, lift, now, 1, look.halo[anim.side], "move", now - anim.t0,
                       face_left[anim.m.to_sq()]);
        }
        // The dragged piece follows the mouse.
        if (dragging && drag_from >= 0 && (pos.bitboard(me) >> drag_from & 1))
            draw_piece(piece_sprite(me, drag_from), me, drag_pos, q * 0.4f, 1, now, 1, look.halo[me], "move", now, drag_pos.x < center(drag_from).x - 2);
    }
    V2 drag_pos;

private:
    // The look of a hole: "pit" (the theme's colours), "lava", "water",
    // "void", "ice" or "bush" (HOLE_STYLES); k: its size while it opens.
    void draw_hole(Box b, const Theme& t, float k, float s, const std::string& style, double now, int sq) {
        using namespace draw;
        // A drawing: the puzzle's ("art:<id>"), or the theme's for the default style.
        std::string art_id = style.rfind("art:", 0) == 0 ? style.substr(4) : (style == "pit" ? t.hole_art : "");
        if (Art* ha = art_id.empty() ? nullptr : find_art(art_id)) {
            double age = now - hole_since[sq];
            const ArtAnim* open = ha->anim("open");
            if (open && !open->frames.empty() && age < open->duration()) draw_art(*ha, "open", age, b);
            else draw_art(*ha, "idle", now + sq * 0.23, b);
            return;
        }
        Box h = b.inset(b.w * 0.07f);
        V2 c = h.center();
        h = {c.x - h.w * 0.5f * k, c.y - h.h * 0.5f * k, h.w * k, h.h * k};
        c = h.center();
        float r = h.w * 0.22f;
        const float seed = hash2(sq, 7, 3);
        auto deep = [&] { round_box({h.x, h.y, h.w, h.h * 0.45f}, r, hex(0x000000, 110), hex(0x000000, 0)); };
        if (style == "lava") {
            float p = 0.7f + 0.3f * float(std::sin(now * 2.5 + seed * 6));
            round_box(h.inset(-1.5f * s), r + 1.5f * s, hex(0x2a0c05));
            round_box(h, r, hex(0x5a1205), hex(0x260602));
            circle(c, h.w * 0.46f, hex(0xff9a2a, uint8_t(230 * p)), hex(0xff3d00, 0), 0);
            for (int i = 0; i < 3; i++) {
                float a = seed * 6 + i * 2.1f;
                line(c, c + V2{std::cos(a), std::sin(a)} * (h.w * 0.38f), std::max(1.0f, h.w * 0.035f), hex(0xffe08a, uint8_t(200 * p)));
            }
        } else if (style == "water") {
            round_box(h.inset(-1.5f * s), r + 1.5f * s, hex(0x0a2a40));
            round_box(h, r, hex(0x0f4f7a), hex(0x062033));
            for (int i = 0; i < 2; i++) {
                float f = float(std::fmod(now * 0.5 + seed + i * 0.5, 1.0));
                ring(c, h.w * (0.08f + 0.36f * f), std::max(1.0f, h.w * 0.03f), hex(0x9fdcff, uint8_t(170 * (1 - f))));
            }
            deep();
        } else if (style == "void") {
            round_box(h.inset(-1.5f * s), r + 1.5f * s, hex(0x2b2160));
            round_box(h, r, hex(0x05030f), hex(0x120a35));
            for (int i = 0; i < 6; i++) {
                V2 p{h.x + h.w * (0.15f + 0.7f * hash2(sq, i, 11)), h.y + h.h * (0.15f + 0.7f * hash2(sq, i, 13))};
                float tw = 0.5f + 0.5f * float(std::sin(now * 3 + i * 1.7 + seed * 9));
                circle(p, std::max(0.8f, h.w * 0.025f), hex(0xffffff, uint8_t(90 + 160 * tw)));
            }
        } else if (style == "ice") {
            round_box(h.inset(-1.5f * s), r + 1.5f * s, hex(0x6fa9cf));
            round_box(h, r, hex(0xe6f7ff), hex(0x8cc4e6));
            V2 a = c + V2{-h.w * 0.3f, -h.h * 0.1f}, m = c + V2{h.w * 0.02f, h.h * 0.05f}, z = c + V2{h.w * 0.32f, -h.h * 0.22f};
            float w = std::max(1.0f, h.w * 0.025f);
            line(a, m, w, hex(0x3e7fae));
            line(m, z, w, hex(0x3e7fae));
            line(m, c + V2{h.w * 0.1f, h.h * 0.34f}, w, hex(0x3e7fae));
        } else if (style == "bush") {
            round_box(h.inset(-1.5f * s), r + 1.5f * s, hex(0x0c2a12));
            round_box(h, r, hex(0x24642f), hex(0x0f3317));
            for (int i = 0; i < 7; i++) {
                V2 p{h.x + h.w * (0.2f + 0.6f * hash2(sq, i, 5)), h.y + h.h * (0.2f + 0.6f * hash2(sq, i, 9))};
                circle(p, h.w * (0.12f + 0.08f * hash2(sq, i, 17)), shade(hex(0x3f9a4a), (hash2(sq, i, 21) - 0.5f) * 0.6f));
            }
        } else {
            round_box(h.inset(-1.5f * s), r + 1.5f * s, with_alpha(t.hole_rim, 0.9f), with_alpha(shade(t.hole_rim, 0.25f), 0.9f));
            round_box(h, r, shade(t.hole, -0.3f), shade(t.hole, 0.12f));
            deep();   // inner shadow at the top: the hole is deep
        }
    }
};

// Time per move offered to the AIs.
const std::vector<double> TIMES = {0.5, 1.0, 2.0, 5.0};
const std::vector<std::string> TIME_LABELS = {"0,5 s", "1 s", "2 s", "5 s"};

// ─── @@PUZZLE ───────────────────────────────────────────────────────────────
// A puzzle (assets/puzzles/<id>.json): a position (any placement of the 4
// pieces of each side and of the holes), who starts, which side the player
// takes, the opposing AI, the goal (win by at least `margin` points) and its
// look (board theme, hole style, piece sprites, background image). A level
// (assets/levels/<id>.json) is a sequence of puzzles: each one unlocks the next.
const std::vector<std::pair<std::string, std::string>> HOLE_STYLES = {
    {"pit", "Selon le th\xC3\xA8me"}, {"lava", "Lave"}, {"water", "Eau"}, {"void", "Vide \xC3\xA9toil\xC3\xA9"}, {"ice", "Glace"}, {"bush", "Buisson"}};

struct Puzzle {
    std::string id, name = "Nouveau puzzle", text;
    std::array<char, 64> cells{};   // by square (a1 = 0): '.' free, '#' hole, 'B' black, 'W' white
    int to_move = BLACK, human = BLACK;
    std::string opponent = "dev";
    int margin = 1;                 // win by at least this many points
    int time_index = 1;             // index in TIMES
    std::string theme, hole_style = "pit", sprite_black, sprite_white, background;
    // Drawings of the workshop placed on this puzzle's board.
    std::map<int, std::string> tiles;   // square → a "tile" drawing over it
    using Decor = PlacedDecor;
    std::vector<Decor> decor;
    Puzzle() {
        // The starting position of the game.
        Yolah y;
        for (int sq = 0; sq < 64; sq++)
            cells[sq] = (y.bitboard(BLACK) >> sq & 1) ? 'B' : (y.bitboard(WHITE) >> sq & 1) ? 'W' : '.';
    }
    int count(char c) const { return int(std::count(cells.begin(), cells.end(), c)); }
};

json puzzle_to_json(const Puzzle& p) {
    json rows = json::array();
    for (int r = 7; r >= 0; r--) {
        std::string row;
        for (int f = 0; f < 8; f++) row += p.cells[r * 8 + f];
        rows.push_back(row);
    }
    return {{"name", p.name}, {"text", p.text}, {"board", rows}, {"to move", p.to_move == BLACK ? "black" : "white"},
            {"player", p.human == BLACK ? "black" : "white"}, {"opponent", p.opponent}, {"margin", p.margin},
            {"time index", p.time_index}, {"theme", p.theme}, {"hole style", p.hole_style},
            {"sprite black", p.sprite_black}, {"sprite white", p.sprite_white}, {"background", p.background},
            {"tiles", [&] {
                 json t = json::object();
                 for (auto& [sq, id] : p.tiles) t[std::string{char('a' + sq % 8), char('1' + sq / 8)}] = id;
                 return t;
             }()},
            {"decor", [&] {
                 json d = json::array();
                 for (auto& e : p.decor) d.push_back({{"art", e.art}, {"x", e.x}, {"y", e.y}, {"scale", e.scale}});
                 return d;
             }()}};
}
Puzzle puzzle_from_json(const json& j, const std::string& id) {
    Puzzle p;
    p.id = id;
    p.name = jstr(j, "name", p.name);
    p.text = jstr(j, "text");
    if (j.contains("board") && j["board"].is_array() && j["board"].size() == 8)
        for (int r = 0; r < 8; r++) {
            std::string row = j["board"][r].is_string() ? j["board"][r].get<std::string>() : "";
            for (int f = 0; f < 8; f++) {
                char c = f < int(row.size()) ? row[f] : '.';
                p.cells[(7 - r) * 8 + f] = (c == '#' || c == 'B' || c == 'W') ? c : '.';
            }
        }
    p.to_move = jstr(j, "to move", "black") == "white" ? WHITE : BLACK;
    p.human = jstr(j, "player", "black") == "white" ? WHITE : BLACK;
    p.opponent = jstr(j, "opponent", p.opponent);
    p.margin = std::max(1, int(jnum(j, "margin", 1)));
    p.time_index = std::clamp(int(jnum(j, "time index", 1)), 0, int(TIMES.size()) - 1);
    p.theme = jstr(j, "theme");
    p.hole_style = jstr(j, "hole style", "pit");
    p.sprite_black = jstr(j, "sprite black");
    p.sprite_white = jstr(j, "sprite white");
    p.background = jstr(j, "background");
    if (j.contains("tiles") && j["tiles"].is_object())
        for (auto& [k, v] : j["tiles"].items())
            if (k.size() == 2 && k[0] >= 'a' && k[0] <= 'h' && k[1] >= '1' && k[1] <= '8' && v.is_string())
                p.tiles[(k[1] - '1') * 8 + (k[0] - 'a')] = v.get<std::string>();
    if (j.contains("decor") && j["decor"].is_array())
        for (auto& e : j["decor"])
            if (e.is_object()) p.decor.push_back({jstr(e, "art"), jnum(e, "x", 4), jnum(e, "y", 4), jnum(e, "scale", 1)});
    return p;
}
Yolah puzzle_position(const Puzzle& p) {
    uint64_t b = 0, w = 0, e = 0;
    for (int sq = 0; sq < 64; sq++) {
        uint64_t bit = uint64_t(1) << sq;
        if (p.cells[sq] == 'B') b |= bit;
        else if (p.cells[sq] == 'W') w |= bit;
        else if (p.cells[sq] == '#') e |= bit;
    }
    Yolah y;
    y.set_state(b, w, e, 0, 0, p.to_move == BLACK ? 0 : 1);
    return y;
}
// Why the puzzle cannot be played ("" when it can).
std::string puzzle_problem(const Puzzle& p) {
    if (p.count('B') != 4) return strf("Il faut exactement 4 pions noirs (il y en a %d).", p.count('B'));
    if (p.count('W') != 4) return strf("Il faut exactement 4 pions blancs (il y en a %d).", p.count('W'));
    if (puzzle_position(p).game_over()) return "Personne ne peut bouger : la partie serait d\xC3\xA9j\xC3\xA0 finie.";
    return "";
}
std::vector<std::string> json_files(const fs::path& dir) {   // file names without ".json"
    std::vector<std::string> ids;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".json") ids.push_back(e.path().stem().string());
    std::sort(ids.begin(), ids.end());
    return ids;
}
std::vector<Puzzle> load_puzzles() {
    std::vector<Puzzle> list;
    fs::path dir = g_paths.assets / "puzzles";
    for (auto& id : json_files(dir)) {
        json j = load_json(dir / (id + ".json"));
        if (j.is_object()) list.push_back(puzzle_from_json(j, id));
    }
    return list;
}
// Saves the puzzle; a new one gets an id from its name.
bool save_puzzle(Puzzle& p) {
    fs::path dir = g_paths.assets / "puzzles";
    if (p.id.empty()) {
        std::string base = slug(p.name), id = base;
        for (int k = 2; fs::exists(dir / (id + ".json")); k++) id = base + "_" + std::to_string(k);
        p.id = id;
    }
    return write_file(dir / (p.id + ".json"), puzzle_to_json(p).dump(2));
}

struct Level {
    std::string id, name = "Nouveau niveau", text;
    std::vector<std::string> puzzles;   // ids, in order
};
std::vector<Level> load_levels() {
    std::vector<Level> list;
    fs::path dir = g_paths.assets / "levels";
    for (auto& id : json_files(dir)) {
        if (id == "order") continue;
        json j = load_json(dir / (id + ".json"));
        if (!j.is_object()) continue;
        Level l;
        l.id = id;
        l.name = jstr(j, "name", l.name);
        l.text = jstr(j, "text");
        if (j.contains("puzzles") && j["puzzles"].is_array())
            for (auto& x : j["puzzles"])
                if (x.is_string()) l.puzzles.push_back(x.get<std::string>());
        list.push_back(l);
    }
    // Their order (assets/levels/order.json, written by the editor); levels
    // not listed there come after, by file name.
    json o = load_json(dir / "order.json");
    std::vector<std::string> order;
    if (o.is_object() && o.contains("levels") && o["levels"].is_array())
        for (auto& x : o["levels"]) if (x.is_string()) order.push_back(x.get<std::string>());
    auto rank = [&](const Level& l) {
        auto it = std::find(order.begin(), order.end(), l.id);
        return it == order.end() ? int(order.size()) : int(it - order.begin());
    };
    std::stable_sort(list.begin(), list.end(), [&](const Level& a, const Level& b) { return rank(a) < rank(b); });
    return list;
}
bool save_level_order(const std::vector<Level>& levels) {
    json ids = json::array();
    for (auto& l : levels) ids.push_back(l.id);
    return write_file(g_paths.assets / "levels" / "order.json", json{{"levels", ids}}.dump(2));
}
bool save_level(Level& l) {
    fs::path dir = g_paths.assets / "levels";
    if (l.id.empty()) {
        std::string base = slug(l.name), id = base;
        for (int k = 2; fs::exists(dir / (id + ".json")); k++) id = base + "_" + std::to_string(k);
        l.id = id;
    }
    return write_file(dir / (l.id + ".json"), json{{"name", l.name}, {"text", l.text}, {"puzzles", l.puzzles}}.dump(2));
}
std::vector<std::string> list_backgrounds() {
    std::vector<std::string> files;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(g_paths.assets / "backgrounds", ec)) {
        auto ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") files.push_back(e.path().filename().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ─── @@APP ──────────────────────────────────────────────────────────────────
enum class Scene { Menu, Setup, Game, Themes, Tutorial, Puzzles, Editor, Studio };

// Saved in the preferences folder (settings.json).
struct Settings {
    std::string theme = "bois";
    bool show_eval = true, show_hint = false, show_coords = true, show_pv = false;
    int time_index = 1;                       // index in TIMES
    std::string black = "human", white = "dev";
    std::vector<int> tutorial_done;           // lessons completed
    std::vector<std::string> puzzles_done;    // puzzles solved
    json ai_options = json::object();         // AI id → {setting: value} (AiParam), games only
    std::string piece_set = "auto";           // "auto": the board's set (Theme::piece_set), else a set's id
    json to_json() const {
        return {{"theme", theme}, {"show eval", show_eval}, {"show hint", show_hint}, {"show coordinates", show_coords},
                {"show pv", show_pv}, {"time index", time_index}, {"black", black}, {"white", white},
                {"tutorial done", tutorial_done}, {"puzzles done", puzzles_done}, {"ai options", ai_options},
                {"piece set", piece_set}};
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
        if (j.contains("tutorial done") && j["tutorial done"].is_array()) tutorial_done = j["tutorial done"].get<std::vector<int>>();
        if (j.contains("puzzles done") && j["puzzles done"].is_array()) puzzles_done = j["puzzles done"].get<std::vector<std::string>>();
        if (j.contains("ai options") && j["ai options"].is_object()) ai_options = j["ai options"];
        piece_set = j.value("piece set", piece_set);
    }
};

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
    std::vector<PieceSet> piece_sets;     // assets/sprites/sets.json
    std::string debug_fx;                 // --fx style,power (screenshots): a square breaks at the board's centre
    std::vector<AiSpec> roster;
    Settings settings;
    std::string setup_params;             // the AI whose settings box is open (new-game screen)

    Scene scene = Scene::Menu;
    double scene_t0 = 0;                  // for the fade-in
    SDL_Cursor* cursor_arrow = nullptr;
    SDL_Cursor* cursor_hand = nullptr;
    std::string toast;
    double toast_t = -10;
    V2 toast_at{-1, -1};                  // its centre (x < 0: at the bottom of the window)
    float toast_size = 17;                // its font size (design pixels)

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
        // A puzzle being played (@@SCENE_PUZZLES): no analysis, its own look.
        bool puzzle_mode = false;
        Puzzle puzzle;
        std::string level;                // the level being played ("" = none)
        int level_pos = -1;               // the puzzle's index in it
        bool from_editor = false;         // a test from the editor
        bool intro = false;               // the introduction card is shown
        bool recorded = false;            // the result was saved
    } game;

    // The tutorial scene (@@SCENE_TUTORIAL).
    struct TutorialState {
        int index = 0;                    // the lesson
        Match match;
        BoardView board;
        std::vector<int> targets;         // squares to reach
        int victim = -1;                  // the white piece to shut in
        int moves_done = 0;
        enum class Phase { Play, Success, Failed } phase = Phase::Play;
        double phase_t = 0, reply_at = -1, pass_at = -1;
        std::string feedback;
        bool hint = false;
        Move hint_move = Move::none();
        size_t hint_ply = size_t(-1);     // the position the hint was computed for
    } tuto;

    // The sprite picker (a modal grid over the library), used by the themes
    // scene and the editor: on_pick receives the chosen sprite id.
    float themes_scroll = 0;              // the theme cards (@@SCENE_THEMES)
    struct Picker {
        bool open = false;
        std::string title, current;
        bool halo = false;
        std::function<void(const std::string&)> on_pick;
        float scroll = 0;
    } picker;
    void open_picker(std::string title, std::string current, bool halo, std::function<void(const std::string&)> f) {
        picker = Picker{true, std::move(title), std::move(current), halo, std::move(f), 0};
    }

    // The puzzle editor (@@SCENE_EDITOR).
    struct EditorState {
        Puzzle p;
        std::string saved;                // the puzzle as last saved (JSON): "modified" when it differs
        std::string level;                // the level the puzzle was opened from ("" = none)
        std::string level_selected;       // the level whose name is edited
        std::set<std::string> open_levels;   // levels unfolded in the list
        std::string tester;               // the AI playing the player's side in "AI against AI"
        int tool = 1;                     // 0 free square, 1 hole, 2 black piece, 3 white piece, 4 special square, 5 decoration
        std::string tile_art, decor_art;  // the drawings of the brushes "Case" and "Décor"
        int decor_size = 2;               // index in DECOR_SIZES
        int decor_drag = -1;              // the decoration being moved
        float scroll = 0;
        BoardView board;
        bool painting = false;            // the button went down on the board
        int painted = -1;                 // the last square painted by a drag
        bool confirm_delete = false;
        std::string confirm_level;        // the level whose deletion waits for a second click
        // The "new puzzle" dialog: start from the starting position or from any puzzle.
        bool choose_new = false;
        std::string new_level;            // the level the new puzzle goes into
        float new_scroll = 0;
    } editor;

    // The workshop (@@SCENE_STUDIO): pixel drawings and board themes.
    struct StudioState {
        int tab = 0;                      // 0 drawings, 1 boards
        // Drawings.
        Art edit;                         // the drawing being edited (a copy: g_arts changes on saving)
        bool loaded = false, modified = false;
        std::string anim = "idle";        // the animation being edited
        int frame = 0;
        int tool = 0;                     // STUDIO_TOOLS
        bool mirror = false, grid = true, onion = true, playing = true;
        Rgba color = hex(0xf2b45a);
        float hue = 36, sat = 0.63f, val = 0.95f;   // the colour in HSV (the picker's state)
        std::vector<std::vector<ArtAnim>> undo, redo;
        bool stroke = false;              // the button went down on the canvas
        int last_x = -1, last_y = -1, start_x = 0, start_y = 0;
        std::vector<Pixel> before;        // the frame when a line / rectangle started
        std::string drag;                 // the colour picker part being dragged
        bool new_open = false;            // the "new drawing" dialog
        int new_kind = 0, new_size = 2;
        bool confirm_delete = false;
        float list_scroll = 0;
        std::string back_to_theme;        // opened from the board editor: return there after
        Scene return_to = Scene::Menu;    // where "Retour" goes (the puzzle editor when opened from it)
        // Boards.
        Theme theme;                      // the theme being edited (a copy)
        bool theme_loaded = false, theme_modified = false, confirm_theme_delete = false;
        int color_slot = 0;               // the colour the picker edits
        double tex_changed = -1, tex_built = 0;   // the board texture follows the colours, at most 5 times a second
        BoardView board;
    } studio;

    // The puzzles and levels (@@SCENE_PUZZLES).
    struct PuzzlesState {
        int tab = 0;                      // 0 play, 1 edit the levels
        std::string open_level;           // the level whose puzzles are listed ("" = the levels)
        Level edit;                       // the level being edited
        bool editing = false;
        float scroll = 0, scroll2 = 0;
    } puzzles_ui;
    std::vector<Puzzle> puzzles;
    std::vector<Level> levels;
    void reload_puzzles() {
        puzzles = load_puzzles();
        levels = load_levels();
    }
    const Puzzle* find_puzzle(const std::string& id) const {
        for (auto& p : puzzles)
            if (p.id == id) return &p;
        return nullptr;
    }
    bool solved(const std::string& id) const {
        return std::find(settings.puzzles_done.begin(), settings.puzzles_done.end(), id) != settings.puzzles_done.end();
    }

    // Background images (assets/backgrounds), loaded on first use.
    std::map<std::string, SDL_Texture*> backgrounds;
    SDL_Texture* background(const std::string& file) {
        auto it = backgrounds.find(file);
        if (it != backgrounds.end()) return it->second;
        SDL_Texture* tex = nullptr;
        int w, h, n;
        fs::path p = g_paths.assets / "backgrounds" / file;
        if (unsigned char* px = stbi_load(p.string().c_str(), &w, &h, &n, 4)) {
            tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
            SDL_UpdateTexture(tex, nullptr, px, w * 4);
            SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
            stbi_image_free(px);
        }
        return backgrounds[file] = tex;
    }

    // Files from the system's file dialog and from drag and drop. The dialog's
    // callback may run on another thread: the paths wait here, behind a
    // mutex, until the next frame hands them to the scene (Input::files).
    std::mutex files_mutex;
    std::vector<std::pair<std::string, std::string>> pending_files;
    void push_file(const std::string& purpose, const std::string& path) {
        std::lock_guard lock(files_mutex);
        pending_files.push_back({purpose, path});
    }
    std::vector<std::pair<std::string, std::string>> take_files() {
        std::lock_guard lock(files_mutex);
        return std::exchange(pending_files, {});
    }
    // Opens the system's dialog to choose an image; the answer comes back as
    // a file of Input::files with this purpose (nothing if cancelled).
    std::string dialog_purpose;
    void choose_image(const std::string& purpose) {
        static const SDL_DialogFileFilter filters[] = {{"Images", "png;jpg;jpeg;bmp;gif;tga"}};
        dialog_purpose = purpose;
        SDL_ShowOpenFileDialog(
            [](void* user, const char* const* list, int) {
                App* a = static_cast<App*>(user);
                if (!list) { a->push_file("dialog-error", SDL_GetError()); return; }
                if (list[0]) a->push_file(a->dialog_purpose, list[0]);
            },
            this, window, filters, 1, nullptr, false);
    }
    // Copies an image into assets/<folder>/ (a free name from its own);
    // returns the new file name, "" on failure.
    std::string import_image(const std::string& path, const std::string& folder) {
        fs::path src(path);
        std::string ext = src.extension().string();
        for (auto& c : ext) c = char(std::tolower(uint8_t(c)));
        int w, h, n;
        if (!stbi_info(path.c_str(), &w, &h, &n)) { notify("Ce fichier n'est pas une image lisible : " + src.filename().string()); return ""; }
        fs::path dir = g_paths.assets / folder;
        std::error_code ec;
        fs::create_directories(dir, ec);
        std::string base = slug(src.stem().string()), name = base + ext;
        for (int k = 2; fs::exists(dir / name); k++) name = base + "_" + std::to_string(k) + ext;
        fs::copy_file(src, dir / name, ec);
        if (ec) { notify("Copie impossible : " + ec.message()); return ""; }
        return name;
    }

    // Sprite of each player (assets/sprites/assignments.json): player id
    // ("human" or an AI id) → sprite id; default: the roster's sprite.
    std::map<std::string, std::string> assigned;

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
    bool has_sprite(const std::string& id) const {
        for (auto& sp : sprites)
            if (sp.id == id) return true;
        return false;
    }
    // The sprite of a player: "human" or an AI id.
    const SpriteDef& player_sprite(const std::string& who) const {
        auto it = assigned.find(who);
        if (it != assigned.end() && has_sprite(it->second)) return sprite(it->second);
        if (who == "human") return sprite(has_sprite("fig_human") ? "fig_human" : "human");
        const AiSpec* sp = ai(who);
        return sprite(sp ? sp->sprite : who);
    }
    void assign_sprite(const std::string& who, const std::string& sprite_id) {
        assigned[who] = sprite_id;
        json j = json::object();
        for (auto& [k, v] : assigned) j[k] = v;
        write_file(g_paths.assets / "sprites" / "assignments.json", json{{"players", j}}.dump(2));
    }
    const SpriteDef& slot_sprite(const Slot& sl) const { return player_sprite(sl.human() ? "human" : sl.spec->id); }

    // ── sets of pieces ──
    const PieceSet* piece_set(const std::string& id) const {
        for (auto& ps : piece_sets)
            if (ps.id == id) return &ps;
        return nullptr;
    }
    // The set in use on a board: the one of the settings, or the board's own
    // ("Selon le plateau"), the classic stones when the board has none.
    const PieceSet& active_set(const Theme& t) const {
        std::string id = settings.piece_set == "auto" ? (t.piece_set.empty() ? "classique" : t.piece_set) : settings.piece_set;
        if (const PieceSet* ps = piece_set(id)) return *ps;
        return piece_sets.front();
    }
    // A player's own sprite, never a set (for the set "players").
    const SpriteDef& own_sprite(const std::string& who) const {
        const SpriteDef& sp = player_sprite(who);
        if (sp.id.rfind("set:", 0) != 0) return sp;
        if (who == "human") return sprite(has_sprite("fig_human") ? "fig_human" : "human");
        const AiSpec* ai_ = ai(who);
        return sprite(ai_ ? ai_->sprite : who);
    }
    // The sprites of the four pieces of `side` for sprite sp (a set sprite
    // gives its units), played by `who`.
    std::vector<const SpriteDef*> expand(const SpriteDef& sp, int side, const std::string& who) const {
        if (sp.id.rfind("set:", 0) == 0)
            if (const PieceSet* ps = piece_set(sp.id.substr(4))) return set_units(*ps, side, who);
        return {&sp};
    }
    std::vector<const SpriteDef*> set_units(const PieceSet& ps, int side, const std::string& who) const {
        if (ps.players) return {&own_sprite(who)};
        if (who == "human") return {&sprite(has_sprite("fig_human") ? "fig_human" : "human")};   // the human is always the user, never dressed up
        if (!ps.cast.empty()) {   // one character per player, never a mix: an AI outside the cast takes one of it
            auto it = ps.cast.find(who);
            if (it == ps.cast.end()) it = std::next(ps.cast.begin(), long(std::hash<std::string>{}(who) % ps.cast.size()));
            if (has_sprite(it->second)) return {&sprite(it->second)};
        }
        std::vector<const SpriteDef*> v;
        for (auto& id : side == BLACK ? ps.black : ps.white)
            if (has_sprite(id)) v.push_back(&sprite(id));
        if (v.empty()) v.push_back(&sprite("classic_go"));
        return v;
    }
    // The pieces of `who` playing `side` on board theme t: a set of the
    // player's own (its sprite is a set), else the active set.
    std::vector<const SpriteDef*> board_units(const Theme& t, int side, const std::string& who) const {
        const PieceSet& ps = active_set(t);
        if (ps.players) return expand(player_sprite(who), side, who);
        return set_units(ps, side, who);
    }
    // The figurine of a player in the current theme's set (menus, the choice
    // of the players, the tutorial): the pieces follow the board everywhere.
    const SpriteDef& themed_sprite(const std::string& who, int side = BLACK) const { return *board_units(theme(), side, who).front(); }
    // Fills look.sprite and look.units for two players on theme t; a puzzle's
    // sprite (or set) for a side wins when given.
    void set_look(BoardView::Look& look, const Theme& t, const std::string& who_black, const std::string& who_white,
                  const std::string& force_black = "", const std::string& force_white = "") const {
        for (int side : {BLACK, WHITE}) {
            const std::string& who = side == BLACK ? who_black : who_white;
            const std::string& force = side == BLACK ? force_black : force_white;
            look.units[side] = !force.empty() && has_sprite(force) ? expand(sprite(force), side, who) : board_units(t, side, who);
            look.sprite[side] = look.units[side].front();
        }
    }
    // In a game: the puzzle's sprites and theme when it has some.
    const SpriteDef& game_sprite(int side) const {
        const std::string who = game.slot[side].human() ? "human" : game.slot[side].spec->id;
        if (game.puzzle_mode) {
            const std::string& id = side == BLACK ? game.puzzle.sprite_black : game.puzzle.sprite_white;
            if (!id.empty() && has_sprite(id)) return *expand(sprite(id), side, who).front();
        }
        return *board_units(game_theme(), side, who).front();
    }
    const Theme& theme_or(const std::string& id) const {
        for (auto& t : themes)
            if (!id.empty() && t.id == id) return t;
        return theme();
    }
    const Theme& game_theme() const { return game.puzzle_mode ? theme_or(game.puzzle.theme) : theme(); }
    std::string slot_name(const Slot& sl) const { return sl.human() ? "Vous" : sl.spec->name; }
    void notify(const std::string& msg) {
        toast = msg;
        toast_t = time;
        toast_at = {-1, -1};
        toast_size = 17;
    }
    // A message centred at `at` (pixels), with a bigger font: the board's news.
    void notify_at(const std::string& msg, V2 at, float size = 24) {
        notify(msg);
        toast_at = at;
        toast_size = size;
    }
    void go(Scene sc) {
        if (scene == Scene::Game && sc != Scene::Game) leave_game();
        if (sc == Scene::Puzzles || sc == Scene::Editor) reload_puzzles();
        scene = sc;
        scene_t0 = time;
        fx.clear();
        ui.focus.clear();
        picker.open = false;
    }
    void save_settings() {
        if (screenshot.empty()) write_file(g_paths.prefs / "settings.json", settings.to_json().dump(2));
    }

    // ── game management (@@SCENE_GAME uses these) ──
    void start_game(const Yolah& start = Yolah()) {
        game.puzzle_mode = false;
        launch(start, settings.black, settings.white, TIMES[settings.time_index]);
    }
    // A game from `start` between two players ("human" or AI ids).
    void launch(const Yolah& start, const std::string& black, const std::string& white, double seconds) {
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
        const std::string who[2] = {black, white};
        for (int side : {BLACK, WHITE}) {
            Slot& sl = g.slot[side];
            sl = Slot{};
            sl.who = who[side];
            sl.spec = ai(who[side]);
            if (sl.spec && sl.spec->available) {
                // The player's settings in a game; a puzzle keeps the roster's AI.
                const json none = json::object();
                const json& opts = !g.puzzle_mode && settings.ai_options.contains(sl.who) ? settings.ai_options[sl.who] : none;
                AiSpec eff = with_options(*sl.spec, opts);
                sl.agent = std::make_unique<AiAgent>(game_config(eff, seconds), eff.random_moves);
            }
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
        // How hard the square breaks: the player's strength (Elo 600 → 0, 2250 → 1; the human: 0.6).
        const Slot& sl = g.slot[side];
        float power = sl.human() ? 0.6f : std::clamp((sl.spec->elo - 600) / 1650.0f, 0.0f, 1.0f);
        g.board.on_move(m, side, time, fx, game_theme(), rng, power);
        if (m != Move::none()) {
            float q = g.board.sq();
            V2 c = g.board.center(m.to_sq());
            fx.add(std::make_unique<FloatText>(c + V2{0, -q * 0.3f}, "+1", side == BLACK ? hex(0xb6c2d6) : hex(0xffffff), q * 0.32f));
            fx.add(std::make_unique<Ripple>(c, q * 0.35f, q * 0.62f, game_theme().accent));
        } else {
            notify_at(std::string(side_name(side)) + " ne peuvent plus bouger : ils passent.", g.board.area.center());
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
void start_lesson(App& a, int i);
int first_lesson_to_do(const App& a);
void start_puzzle(App& a, const Puzzle& p, const std::string& level, int pos, bool from_editor, const std::string& player_ai = "");
void draw_image_background(App& a, const std::string& file, uint8_t dim = 70, bool vignette = true, const Theme* th = nullptr);
std::string goal_text(const App& a, const Puzzle& p);

// Drawn behind every scene: the theme's gradient, a vignette, faint shapes.
// th: the board theme whose scenery is drawn (default: the player's).
void draw_background(App& a, bool decorated = true, const Theme* th = nullptr) {
    const Theme& t = th ? *th : a.theme();
    if (SDL_Texture* tex = scenery_texture(t.bg_image)) {   // a scenery: its picture, its stars
        float tw, th;
        SDL_GetTextureSize(tex, &tw, &th);
        float k = std::max(a.w / tw, a.h / th);
        SDL_FRect dst{(a.w - tw * k) / 2, (a.h - th * k) / 2, tw * k, th * k};
        SDL_RenderTexture(a.renderer, tex, nullptr, &dst);
        if (t.stars) draw_star_field({0, 0, float(a.w), float(a.h)}, a.time, a.s);
        if (t.embers) draw_embers({0, 0, float(a.w), float(a.h)}, a.time, a.s);
        if (t.petals) draw_petals({0, 0, float(a.w), float(a.h)}, a.time, a.s);
        if (t.smoke) draw_smoke({0, 0, float(a.w), float(a.h)}, a.time, a.s);
        draw::vignette({0, 0, float(a.w), float(a.h)}, a.s, 60);
        return;
    }
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
        const SpriteDef& sp = a.themed_sprite(i == 0 ? "human" : a.roster[i - 1].id, i % 2 ? WHITE : BLACK);
        draw_piece(sp, i % 2 ? WHITE : BLACK, c, (36 + 6 * std::sin(float(a.time) + i)) * s, 0.4f, a.time, 0.9f * fade, i == 0);
    }
    // Title on a soft light.
    V2 tp{a.w * 0.5f, 75 * s};
    draw::circle({a.w * 0.5f, 150 * s}, 330 * s, with_alpha(a.theme().accent, 0.16f), with_alpha(a.theme().accent, 0), 0);
    a.text.draw_shadowed(Font::Title, 140 * s, "YOLAH", tp, pal::text, Align::Center, hex(0x000000, 150), 6 * s);
    a.text.draw(Font::Regular, 21 * s, "Quatre pions, soixante-quatre cases : chaque pas creuse un trou.", {a.w * 0.5f, 250 * s}, pal::dim, Align::Center);

    struct Item { const char* label; std::function<void()> act; Style st; };
    std::vector<Item> items = {
        {"Jouer", [&] { a.go(Scene::Setup); }, Style::Primary},
        {"Apprendre \xC3\xA0 jouer", [&] { start_lesson(a, first_lesson_to_do(a)); }, Style::Normal},
        {"Puzzles et qu\xC3\xAA" "tes", [&] { a.go(Scene::Puzzles); }, Style::Normal},
        {"\xC3\x89" "diteur de puzzles", [&] { a.go(Scene::Editor); }, Style::Normal},
        {"Th\xC3\xA8mes", [&] { a.go(Scene::Themes); }, Style::Normal},
        // {"Atelier : dessins et plateaux", [&] { a.go(Scene::Studio); }, Style::Normal},   // the workshop, put aside for now
        {"Quitter", [&] { SDL_Event e{}; e.type = SDL_EVENT_QUIT; SDL_PushEvent(&e); }, Style::Ghost},
    };
    float bw = 440 * s, bh = 54 * s, gap = 12 * s, y = 296 * s;
    for (size_t i = 0; i < items.size(); i++) {
        float appear = ease_out_cubic(float((a.time - a.scene_t0) * 3 - i * 0.12f));
        Box b{a.w * 0.5f - bw / 2, y + i * (bh + gap) + (1 - appear) * 30 * s, bw, bh};
        if (a.ui.button(std::string("menu") + std::to_string(i), b, items[i].label, items[i].st, true, 21)) items[i].act();
    }
    a.text.draw(Font::Regular, 14 * s, strf("Yolah %s \xE2\x80\xA2 %zu IA \xE2\x80\xA2 th\xC3\xA8me %s", APP_VERSION, a.roster.size(), a.theme().name.c_str()),
                {a.w * 0.5f, a.h - 34 * s}, pal::faint, Align::Center);
}

// ─── @@SCENE_SETUP ──────────────────────────────────────────────────────────
void draw_ai_params(App& a);
void scene_setup(App& a) {
    const float s = a.s;
    draw_background(a);
    draw_title(a, "Nouvelle partie",
               "Choisissez qui joue les Noirs (ils commencent) et qui joue les Blancs : un humain ou une IA de chaque c\xC3\xB4t\xC3\xA9. Les IA sont class\xC3\xA9" "es de la plus faible \xC3\xA0 la plus forte.");
    const bool params_open = !a.setup_params.empty();
    if (params_open && a.in.key(SDLK_ESCAPE)) { a.in.eat(SDLK_ESCAPE); a.setup_params.clear(); }
    a.ui.blocked = params_open;
    if (back_button(a)) { a.ui.blocked = false; a.go(Scene::Menu); return; }

    struct Entry { std::string id, name, sub, sprite; bool available; std::string reason, description; };
    std::vector<Entry> entries;
    entries.push_back({"human", "Humain", "Vous jouez \xC3\xA0 la souris", "human", true, "", "Cliquez sur un de vos pions puis sur la case d'arriv\xC3\xA9" "e, ou faites-le glisser."});
    for (auto& ai : a.roster) {
        const bool set = a.settings.ai_options.contains(ai.id);
        entries.push_back({ai.id, ai.name, strf("Elo \xE2\x89\x88 %d \xE2\x80\xA2 %s", ai.elo, ai.family.c_str()) + (set ? " \xE2\x80\xA2 r\xC3\xA9gl\xC3\xA9" : ""),
                           ai.sprite, ai.available, ai.reason, ai.description});
    }

    const float top = 122 * s, bottom_bar = 150 * s, desc_h = 52 * s;
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
            draw_piece(a.themed_sprite(e.id, side), side, {r.x + 16 * s + pr, r.center().y}, pr, 0.2f * k, a.time, alpha, e.id == "human");
            float tx = r.x + 30 * s + 2 * pr;
            a.text.draw(Font::Bold, 19 * s, e.name, {tx, r.center().y - 22 * s}, with_alpha(pal::text, alpha));
            a.text.draw(Font::Regular, 14 * s, e.available ? e.sub : "indisponible \xE2\x80\x94 survolez pour savoir pourquoi",
                        {tx, r.center().y + 2 * s}, with_alpha(e.available ? pal::dim : pal::bad, std::max(alpha, 0.7f)));
            bool on_gear = false;
            if (e.id != "human" && e.available) {   // strength gauge, settings button
                float frac = float(i) / (entries.size() - 1);
                Box gb{r.right() - 150 * s, r.center().y - 4 * s, 60 * s, 8 * s};
                draw::round_box(gb, 4 * s, hex(0xffffff, 25));
                draw::round_box({gb.x, gb.y, gb.w * frac, gb.h}, 4 * s, shade(a.theme().accent, -0.2f), a.theme().accent);
                const AiSpec* spec = a.ai(e.id);
                if (spec && !spec->params.empty()) {
                    Box bb{r.right() - 78 * s, r.center().y - 16 * s, 68 * s, 32 * s};
                    on_gear = a.ui.hover(bb);
                    if (a.ui.button("setp" + std::to_string(side) + e.id, bb, "R\xC3\xA9gler", Style::Ghost, true, 13)) a.setup_params = e.id;
                }
            }
            if (h && !on_gear && a.in.released) *choice[side] = e.id;
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
    if (a.ui.button("start", {bar.right() - 224 * s, bar.y + 26 * s, 200 * s, 50 * s}, "Commencer", Style::Primary) || (!params_open && a.in.key(SDLK_RETURN))) {
        a.ui.blocked = false;
        a.setup_params.clear();
        a.save_settings();
        a.start_game();
        return;
    }
    a.ui.blocked = false;
    if (params_open) draw_ai_params(a);
}

// The settings box of an AI (App::setup_params): one row per AiParam, a row of
// choices for a few values, − / + around the value for more. A value equal to
// the default is not stored, so a later change of the roster still applies.
void draw_ai_params(App& a) {
    const float s = a.s;
    const AiSpec* spec = a.ai(a.setup_params);
    if (!spec) { a.setup_params.clear(); return; }
    json& all = a.settings.ai_options;
    json opts = all.contains(spec->id) ? all[spec->id] : json::object();
    draw::rect({0, 0, float(a.w), float(a.h)}, hex(0x000000, 160));
    const float row = 70 * s;
    Box m{a.w * 0.5f - 340 * s, 0, 680 * s, 104 * s + row * spec->params.size() + 96 * s};
    m.y = a.h * 0.5f - m.h / 2;
    a.ui.panel(m, 20, hex(0x1a202b));
    a.text.draw(Font::Black, 24 * s, "R\xC3\xA9glages : " + spec->name, {m.x + 30 * s, m.y + 24 * s}, pal::text);
    a.text.draw(Font::Regular, 14 * s, "Pour les parties ; les puzzles gardent les IA telles qu'elles sont pr\xC3\xA9vues.", {m.x + 30 * s, m.y + 62 * s}, pal::dim);
    float y = m.y + 104 * s;
    for (const AiParam& p : spec->params) {
        int cur = 0;
        const json v = opts.contains(p.key) ? opts[p.key] : p.def;
        for (size_t i = 0; i < p.values.size(); i++)
            if (p.values[i] == v) cur = int(i);
        int sel = cur;
        a.text.draw(Font::Bold, 17 * s, p.label, {m.x + 30 * s, y + 6 * s}, pal::text);
        std::string sub = p.help;
        if (!sub.empty()) sub += " \xE2\x80\xA2 ";
        sub += "d\xC3\xA9" "faut : " + p.value_text(p.def);
        a.text.draw(Font::Regular, 13 * s, sub, {m.x + 30 * s, y + 32 * s}, pal::dim);
        Box c{m.right() - 30 * s - 300 * s, y + 6 * s, 300 * s, 40 * s};
        if (p.values.size() <= 5) {
            std::vector<std::string> labels;
            float widest = 0;
            for (auto& x : p.values) {
                labels.push_back(p.value_text(x));
                widest = std::max(widest, a.text.width(Font::Bold, 15 * s, labels.back()));
            }
            if (widest > c.w / p.values.size() - 12 * s && !p.unit.empty()) {   // too wide: the numbers alone, the unit above
                for (size_t i = 0; i < labels.size(); i++)
                    if (p.values[i].is_number()) labels[i] = strf("%g", p.values[i].get<double>());
                a.text.draw(Font::Regular, 12 * s, "en " + p.unit, {c.right(), c.y - 16 * s}, pal::dim, Align::Right);
            }
            sel = a.ui.segmented("prm" + p.key, c, labels, cur, 15);
        } else {
            if (a.ui.button("prm-" + p.key, {c.x, c.y, 50 * s, c.h}, "\xE2\x80\x93", Style::Normal, cur > 0, 20)) sel = cur - 1;
            if (a.ui.button("prm+" + p.key, {c.right() - 50 * s, c.y, 50 * s, c.h}, "+", Style::Normal, cur + 1 < int(p.values.size()), 20)) sel = cur + 1;
            draw::round_box({c.x + 58 * s, c.y, c.w - 116 * s, c.h}, c.h / 2, hex(0x0f141c, 200));
            a.text.draw(Font::Bold, 17 * s, p.value_text(p.values[cur]), {c.center().x, c.center().y - a.text.line_height(Font::Bold, 17 * s) * 0.42f},
                        p.values[cur] == p.def ? pal::text : pal::accent, Align::Center);
        }
        if (sel != cur) {
            if (p.values[sel] == p.def) opts.erase(p.key);
            else opts[p.key] = p.values[sel];
            if (opts.empty()) all.erase(spec->id);
            else all[spec->id] = opts;
        }
        y += row;
    }
    float by = m.bottom() - 74 * s;
    if (a.ui.button("prm_reset", {m.x + 30 * s, by, 220 * s, 50 * s}, "Valeurs par d\xC3\xA9" "faut", Style::Ghost, all.contains(spec->id), 17)) all.erase(spec->id);
    if (a.ui.button("prm_close", {m.right() - 30 * s - 180 * s, by, 180 * s, 50 * s}, "Fermer", Style::Primary, true, 17) || a.in.key(SDLK_RETURN)) {
        a.in.eat(SDLK_RETURN);
        a.setup_params.clear();
        a.save_settings();
    }
}

// ─── @@SCENE_GAME ───────────────────────────────────────────────────────────
// Each frame: (1) the logic — whose turn, AI searches, passes, end of game,
// analysis; (2) the layout; (3) the drawing: evaluation bar, board, side
// panel (players, tabs, buttons), end-of-game box.
void game_logic(App& a) {
    auto& g = a.game;
    const bool anim = g.board.animating(a.time);
    if (g.intro) return;   // a puzzle's introduction card
    if (g.match.over()) {
        if (!g.over && !anim) {
            g.over = true;
            g.over_t = a.time;
            if (g.puzzle_mode) {   // the goal: win by at least `margin`
                int margin = g.match.score(g.puzzle.human) - g.match.score(1 - g.puzzle.human);
                bool ok = margin >= g.puzzle.margin;
                if (ok && !g.from_editor && !a.solved(g.puzzle.id)) {
                    a.settings.puzzles_done.push_back(g.puzzle.id);
                    a.save_settings();
                }
                if (ok) a.fx.add(std::make_unique<Confetti>(Box{0, 0, float(a.w), float(a.h)}, a.rng));
                return;
            }
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
    draw_piece(a.game_sprite(side), side, {b.x + 18 * s + pr, b.center().y}, pr, 0.3f * k, a.time, 1, sl.human());
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
        draw_piece(a.game_sprite(side), side, {b.x + 24 * s, y + 24 * s}, 22 * s, 0, a.time);
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

// Leaving a puzzle: back to the editor (a test) or to the puzzles.
void leave_puzzle(App& a) {
    if (a.game.from_editor) a.go(Scene::Editor);
    else a.go(Scene::Puzzles);
}

void draw_puzzle_over(App& a, Box area, float k) {
    auto& g = a.game;
    const float s = a.s;
    const Puzzle& p = g.puzzle;
    int margin = g.match.score(p.human) - g.match.score(1 - p.human);
    bool ok = margin >= p.margin;
    Box card{area.center().x - 250 * s, area.center().y - 160 * s + (1 - k) * 40 * s, 500 * s, 320 * s};
    a.ui.panel(card, 22, with_alpha(pal::panel, k));
    const bool ai_test = !g.slot[p.human].human();
    a.text.draw_shadowed(Font::Title, 34 * s, ai_test ? (ok ? "L'IA r\xC3\xA9ussit le puzzle" : "L'IA \xC3\xA9" "choue") : (ok ? "Puzzle r\xC3\xA9ussi !" : "Pas encore\xE2\x80\xA6"), {card.center().x, card.y + 28 * s}, with_alpha(ok ? pal::good : pal::text, k), Align::Center, hex(0x000000, 100), 2 * s);
    a.text.draw(Font::Black, 50 * s, strf("%d \xE2\x80\x93 %d", g.match.score(BLACK), g.match.score(WHITE)), {card.center().x, card.y + 84 * s}, with_alpha(a.game_theme().accent, k), Align::Center);
    a.text.draw(Font::Regular, 16 * s, strf("Avance du joueur : %+d \xE2\x80\xA2 objectif : %+d", margin, p.margin), {card.center().x, card.y + 150 * s}, with_alpha(pal::dim, k), Align::Center);
    // The next puzzle of the level.
    const Level* lvl = nullptr;
    for (auto& l : a.levels) if (l.id == g.level) lvl = &l;
    const Puzzle* next = nullptr;
    std::string next_level = g.level;
    int next_pos = g.level_pos + 1;
    if (ok && lvl && g.level_pos + 1 < int(lvl->puzzles.size())) next = a.find_puzzle(lvl->puzzles[g.level_pos + 1]);
    else if (ok && lvl)   // the last puzzle of the level: the first of the next level
        for (size_t i = 0; i + 1 < a.levels.size(); i++)
            if (a.levels[i].id == lvl->id && !a.levels[i + 1].puzzles.empty()) {
                next = a.find_puzzle(a.levels[i + 1].puzzles.front());
                next_level = a.levels[i + 1].id;
                next_pos = 0;
            }
    float bw = 140 * s, by = card.bottom() - 74 * s;
    if (next) {
        if (a.ui.button("pz_next", {card.x + 24 * s, by, bw, 50 * s}, next_level == g.level ? "Puzzle suivant" : "Niveau suivant", Style::Primary, true, 17)) {
            Puzzle n = *next;
            start_puzzle(a, n, next_level, next_pos, false);
            return;
        }
    } else if (a.ui.button("pz_retry", {card.x + 24 * s, by, bw, 50 * s}, "R\xC3\xA9" "essayer", ok ? Style::Normal : Style::Primary, true, 17)) {
        Puzzle again = p;
        start_puzzle(a, again, g.level, g.level_pos, g.from_editor, ai_test ? g.slot[p.human].who : "");
        a.game.intro = false;   // a retry goes straight to the board
        return;
    }
    if (next && a.ui.button("pz_retry2", {card.center().x - bw / 2, by, bw, 50 * s}, "Rejouer", Style::Normal, true, 17)) { Puzzle again = p; start_puzzle(a, again, g.level, g.level_pos, false); a.game.intro = false; return; }
    if (a.ui.button("pz_leave", {card.right() - 24 * s - bw, by, bw, 50 * s}, g.from_editor ? "\xC3\x89" "diteur" : "Puzzles", Style::Ghost, true, 17)) leave_puzzle(a);
}

void draw_game_over(App& a, Box area) {
    auto& g = a.game;
    const float s = a.s;
    float k = ease_out_cubic(float((a.time - g.over_t) / 0.5));
    a.ui.blocked = false;
    draw::round_box(area, 6 * s, hex(0x000000, uint8_t(110 * k)));
    if (g.puzzle_mode) { draw_puzzle_over(a, area, k); return; }
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
    // A puzzle starts with its introduction: its image, its title, its goal.
    if (g.puzzle_mode && g.intro) {
        draw_image_background(a, g.puzzle.background, 0, false, &a.game_theme());   // the whole window, as it is
        float k = ease_out_cubic(float((a.time - a.scene_t0) / 0.6));
        Box card{a.w * 0.5f - 420 * s, a.h * 0.5f - 145 * s + (1 - k) * 60 * s, 840 * s, 290 * s};   // centred in the window
        a.ui.panel(card, 24, hex(0x10151d, uint8_t(225 * k)));
        a.text.draw_shadowed(Font::Title, 40 * s, g.puzzle.name, {card.center().x, card.y + 26 * s}, with_alpha(pal::text, k), Align::Center, hex(0x000000, 120), 3 * s);
        a.text.paragraph(Font::Regular, 18 * s, g.puzzle.text, {card.x + 50 * s, card.y + 92 * s, card.w - 100 * s, 60 * s}, with_alpha(pal::dim, k), 1.25f, Align::Center);
        a.text.draw(Font::Bold, 18 * s, goal_text(a, g.puzzle), {card.center().x, card.y + 160 * s}, with_alpha(a.game_theme().accent, k), Align::Center);
        if (a.ui.button("pz_start", {card.center().x + 10 * s, card.bottom() - 74 * s, 220 * s, 52 * s}, "Commencer", Style::Primary) || a.in.key(SDLK_RETURN)) {
            g.intro = false;
            a.scene_t0 = a.time;
        }
        if (a.ui.button("pz_back", {card.center().x - 230 * s, card.bottom() - 74 * s, 220 * s, 52 * s}, "Retour", Style::Ghost) || a.in.key(SDLK_ESCAPE)) { leave_puzzle(a); return; }
        return;
    }
    // Keyboard shortcuts.
    if (a.in.key(SDLK_ESCAPE)) {
        if (g.puzzle_mode) leave_puzzle(a);
        else a.go(Scene::Menu);
        return;
    }
    // The analysis is a help: none in a puzzle played from a level (it would
    // give the solution), but there when the editor tests a puzzle.
    const bool help = !g.puzzle_mode || g.from_editor;
    if (a.in.key(SDLK_H) && help) a.settings.show_hint = !a.settings.show_hint;
    if (a.in.key(SDLK_E) && help) a.settings.show_eval = !a.settings.show_eval;
    if (a.in.key(SDLK_F)) g.board.flipped = !g.board.flipped;
    if (a.in.key(SDLK_U) || a.in.key(SDLK_BACKSPACE)) a.undo_move();
    if (a.in.key(SDLK_SPACE) && !g.slot[0].human() && !g.slot[1].human()) g.paused = !g.paused;

    game_logic(a);

    // Analysis: restarted whenever the position changes and something shows it.
    if (!help && g.tab == 1) g.tab = 0;
    const bool show_eval = a.settings.show_eval && help;
    bool need = help && (a.settings.show_eval || a.settings.show_hint || a.settings.show_pv || g.tab == 1);
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
    float block_w = (show_eval ? bar_w + gap : 0) + 2 * frame + B;
    float bx = margin + (left_w - block_w) / 2 + (show_eval ? bar_w + gap : 0) + frame;
    float by = (a.h - B) / 2;
    g.board.area = {std::floor(bx), std::floor(by), B, B};
    Box panel{a.w - margin - panel_w, margin, panel_w, a.h - 2 * margin};

    if (g.puzzle_mode) draw_image_background(a, g.puzzle.background, 70, true, &a.game_theme());
    else draw_background(a, false);
    if (!a.debug_fx.empty()) {   // --fx: one square breaking, for screenshots of the effects
        Theme th = a.game_theme();
        size_t comma = a.debug_fx.find(',');
        th.break_fx = a.debug_fx.substr(0, comma);
        float pw = comma == std::string::npos ? 0.5f : std::stof(a.debug_fx.substr(comma + 1));
        a.fx.add(std::make_unique<Shatter>(g.board.center(27), g.board.sq(), th, pw, a.rng));
        a.debug_fx.clear();
    }
    if (show_eval) draw_eval_bar(a, {g.board.area.x - frame - gap - bar_w, g.board.area.y, bar_w, B}, snap);

    // Board.
    const int side = g.match.to_move();
    bool human_turn = !g.match.over() && g.slot[side].human() && !must_pass(g.match.pos);
    bool modal = g.over && a.time > g.over_t + 0.3;
    auto mv = g.board.input(a.in, g.match.pos, human_turn && !g.board.animating(a.time), modal);
    g.board.drag_pos = a.in.mouse;
    if (mv) a.play_move(*mv);
    BoardView::Look look;
    a.set_look(look, a.game_theme(), g.slot[BLACK].human() ? "human" : g.slot[BLACK].spec->id,
               g.slot[WHITE].human() ? "human" : g.slot[WHITE].spec->id,
               g.puzzle_mode ? g.puzzle.sprite_black : "", g.puzzle_mode ? g.puzzle.sprite_white : "");
    if (g.puzzle_mode) {
        look.hole_style = g.puzzle.hole_style;
        look.tiles = &g.puzzle.tiles;
        look.decor = &g.puzzle.decor;
    }
    if (g.match.over()) {
        int sb = g.match.score(BLACK), sw = g.match.score(WHITE);
        look.winner = sb > sw ? BLACK : sw > sb ? WHITE : -1;
    }
    // (No forced halo for the human any more: the sprites tell who is who.)
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
    g.board.draw(a.text, a.board_textures, a.game_theme(), g.match.pos, look, a.time, s);
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
    float tabs_y = card_top.bottom() + 14 * s;
    if (g.puzzle_mode) {   // the puzzle's goal, always in sight
        a.text.paragraph(Font::Bold, 15 * s, "Objectif : " + goal_text(a, g.puzzle), {panel.x + 24 * s, tabs_y, panel.w - 48 * s, 44 * s}, a.game_theme().accent);
        tabs_y += 48 * s;
    }
    Box tabs{panel.x + 20 * s, tabs_y, panel.w - 40 * s, 38 * s};
    if (help) g.tab = a.ui.tabs("gtabs", tabs, {"Coups", "Analyse", "IA"}, g.tab);
    else g.tab = a.ui.tabs("gtabs2", tabs, {"Coups", "IA"}, g.tab == 2 ? 1 : 0) == 1 ? 2 : 0;   // no analysis tab
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
    } else if (a.ui.button("g_hintb", bbox(0), "Indice", a.settings.show_hint && help ? Style::Primary : Style::Normal, help, 17)) {
        a.settings.show_hint = !a.settings.show_hint;
    }
    if (a.ui.button("g_undo", bbox(1), "Annuler", Style::Normal, !g.match.history.empty(), 17)) a.undo_move();
    if (a.ui.button("g_flip", bbox(2), "Retourner", Style::Normal, true, 17)) g.board.flipped = !g.board.flipped;
    if (a.ui.button("g_menu", bbox(3), g.puzzle_mode ? "Quitter" : "Menu", Style::Ghost, true, 17)) {
        if (g.puzzle_mode) leave_puzzle(a);
        else a.go(Scene::Menu);
        return;
    }

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

void sprite_picker(App& a);

void scene_themes(App& a) {
    const float s = a.s;
    const bool picking = a.picker.open;
    draw_background(a);
    draw_title(a, "Th\xC3\xA8mes", "Le plateau et les pi\xC3\xA8ces de chaque joueur.");
    a.ui.blocked = picking;
    if (picking && a.in.key(SDLK_ESCAPE)) { a.picker.open = false; a.in.eat(SDLK_ESCAPE); }
    if (back_button(a)) { a.save_settings(); a.go(Scene::Menu); return; }
    // Themes: cards with a small live board (with the drawings of the
    // workshop when the theme has some), in a grid that scrolls.
    const float card_w = 236 * s, card_h = 272 * s, gap = 20 * s;
    const int cols = 3;
    Box area{60 * s, 140 * s, cols * card_w + (cols - 1) * gap, a.h - 230 * s};
    if (a.ui.hover(area)) a.themes_scroll = std::max(0.0f, a.themes_scroll - a.in.wheel * 60 * s);
    const int rows = (int(a.themes.size()) + cols - 1) / cols;
    a.themes_scroll = std::clamp(a.themes_scroll, 0.0f, std::max(0.0f, rows * (card_h + gap) - gap - area.h + 12 * s));
    Box clip_box{area.x - 12 * s, area.y - 10 * s, area.w + 24 * s, area.h + 20 * s};
    draw::clip(&clip_box);
    for (size_t i = 0; i < a.themes.size(); i++) {
        const Theme& t = a.themes[i];
        Box c{area.x + (i % cols) * (card_w + gap), area.y + (i / cols) * (card_h + gap) - a.themes_scroll, card_w, card_h};
        if (c.bottom() < area.y - 10 * s || c.y > area.bottom()) continue;
        bool sel = t.id == a.settings.theme, h = a.ui.hover(c) && area.contains(a.in.mouse);
        if (h) a.ui.hovering = true;
        float k = a.ui.anim("th" + t.id, h), ks = a.ui.anim("ths" + t.id, sel);
        Box cc{c.x, c.y - 6 * s * k, c.w, c.h};
        draw::shadow({cc.x, cc.y + 10 * s, cc.w, cc.h}, 18 * s, 30 * s, hex(0x000000, 110));
        draw::round_box(cc, 18 * s, t.bg_top, t.bg_bottom);
        if (ks > 0.01f) draw::round_frame(cc, 18 * s, 3 * s, with_alpha(t.accent, ks));
        Box b{cc.x + 28 * s, cc.y + 22 * s, 180 * s, 180 * s};
        if (Art* fa = t.frame_art.empty() ? nullptr : find_art(t.frame_art)) draw_art_frame(*fa, a.time, b.inset(-10 * s), 10 * s);
        else draw::round_box(b.inset(-10 * s), 10 * s, t.frame_top, t.frame_bottom);
        SDL_FRect dst = b.frect();
        SDL_RenderTexture(a.renderer, a.board_textures.get(t, int(b.w)), nullptr, &dst);
        float q = b.w / 8;
        Art* tl = t.tile_light.empty() ? nullptr : find_art(t.tile_light);
        Art* td = t.tile_dark.empty() ? nullptr : find_art(t.tile_dark);
        if (tl || td)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    if (Art* ta = (x + y) % 2 == 0 ? tl : td) draw_art(*ta, "idle", a.time, {b.x + x * q, b.y + y * q, q, q});
        Art* ha = t.hole_art.empty() ? nullptr : find_art(t.hole_art);
        for (V2 hsq : {V2{3, 3}, V2{4, 2}, V2{2, 5}}) {
            Box hb{b.x + hsq.x * q, b.y + hsq.y * q, q, q};
            if (ha) draw_art(*ha, "idle", a.time + hsq.x, hb);
            else draw::round_box(hb.inset(q * 0.08f), q * 0.2f, shade(t.hole, -0.3f), shade(t.hole, 0.12f));
        }
        {   // two pieces of the set this board would use
            auto ub = a.board_units(t, BLACK, "human"), uw = a.board_units(t, WHITE, "dev");
            draw_piece(*ub.front(), BLACK, {b.x + 2.5f * q, b.y + 2.5f * q}, q * 0.4f, 0, a.time, 1, false, "idle", a.time);
            draw_piece(*uw.front(), WHITE, {b.x + 5.5f * q, b.y + 4.5f * q}, q * 0.4f, 0, a.time, 1, false, "idle", a.time + 0.5);
        }
        a.text.draw(Font::Bold, 18 * s, t.name, {cc.center().x, cc.bottom() - 56 * s}, pal::text, Align::Center);
        a.text.draw(Font::Regular, 13 * s, sel ? "th\xC3\xA8me actuel" : "cliquer pour choisir", {cc.center().x, cc.bottom() - 30 * s}, sel ? t.accent : pal::dim, Align::Center);
        if (h && a.in.released) a.settings.theme = t.id;
    }
    draw::clip(nullptr);
    // The workshop (Atelier) is put aside for now.
    // if (a.ui.button("th_studio", {area.x, a.h - 80 * s, area.w, 46 * s}, "Cr\xC3\xA9" "er ou modifier des plateaux et des sprites dans l'atelier\xE2\x80\xA6", Style::Ghost, true, 16)) {
    //     a.save_settings();
    //     a.go(Scene::Studio);
    //     a.studio.return_to = Scene::Themes;
    //     a.studio.tab = 1;
    //     return;
    // }
    // The players and their sprites.
    Box gal{area.right() + 50 * s, 140 * s, a.w - area.right() - 110 * s, a.h - 180 * s};
    a.ui.panel(gal);
    // The set of pieces: the board's own, or one for every board.
    a.text.draw(Font::Black, 22 * s, "Jeu de pi\xC3\xA8" "ces", {gal.x + 22 * s, gal.y + 18 * s}, pal::text);
    {
        std::vector<std::string> opts = {"Selon le plateau"}, ids = {"auto"};
        for (auto& ps : a.piece_sets) { opts.push_back(ps.name); ids.push_back(ps.id); }
        int cur = 0;
        for (size_t i = 0; i < ids.size(); i++)
            if (ids[i] == a.settings.piece_set) cur = int(i);
        int sel = a.ui.segmented("pset", {gal.x + 22 * s, gal.y + 54 * s, gal.w - 44 * s, 40 * s}, opts, cur, 14);
        if (sel != cur) a.settings.piece_set = ids[size_t(sel)];
        const PieceSet& act = a.active_set(a.theme());
        std::string why = a.settings.piece_set == "auto"
            ? "Chaque plateau prend son jeu : ici \xC2\xAB " + act.name + " \xC2\xBB (pions classiques pour les plateaux sans jeu propre)."
            : "Ce jeu sert sur tous les plateaux.";
        if (!act.description.empty()) why += " " + act.description;
        a.text.paragraph(Font::Regular, 13 * s, why, {gal.x + 22 * s, gal.y + 102 * s, gal.w - 44 * s, 40 * s}, pal::dim);
    }
    a.text.draw(Font::Black, 20 * s, "Sprites des joueurs", {gal.x + 22 * s, gal.y + 150 * s}, pal::text);
    a.text.paragraph(Font::Regular, 13 * s, strf("Avec le jeu \xC2\xAB Robots \xC2\xBB, chaque joueur garde son sprite. Cliquez sur un joueur pour en choisir un parmi les %zu de la biblioth\xC3\xA8que (un jeu entier aussi, dans \xC2\xAB Jeux de pi\xC3\xA8" "ces \xC2\xBB).", a.sprites.size()),
                     {gal.x + 22 * s, gal.y + 180 * s, gal.w - 44 * s, 40 * s}, pal::dim);
    std::vector<std::pair<std::string, std::string>> players = {{"human", "Joueur humain"}};
    for (auto& ai : a.roster) players.push_back({ai.id, ai.name});
    float top = gal.y + 232 * s, rh = std::min(60.0f * s, (gal.h - 320 * s) / players.size());
    for (size_t i = 0; i < players.size(); i++) {
        const auto& [id, name] = players[i];
        const SpriteDef& sp = a.player_sprite(id);
        Box r{gal.x + 12 * s, top + i * rh, gal.w - 24 * s, rh - 4 * s};
        bool h = a.ui.hover(r);
        if (h) a.ui.hovering = true;
        float k = a.ui.anim("pl" + id, h);
        draw::round_box(r, 10 * s, with_alpha(hex(0xffffff), 0.02f + 0.07f * k));
        float pr = rh * 0.36f;
        draw_piece(sp, BLACK, {r.x + 14 * s + pr, r.center().y}, pr, 0.2f * k, a.time, 1, id == "human");
        draw_piece(sp, WHITE, {r.x + 22 * s + 3 * pr, r.center().y}, pr, 0.2f * k, a.time, 1, id == "human");
        float tx = r.x + 40 * s + 4 * pr;
        a.text.draw(Font::Bold, 17 * s, name, {tx, r.center().y - 20 * s}, pal::text);
        a.text.draw(Font::Regular, 13 * s, "sprite : " + sp.name, {tx, r.center().y + 2 * s}, pal::dim);
        a.text.draw(Font::Bold, 15 * s, "Changer \xE2\x86\x92", {r.right() - 12 * s, r.center().y - 9 * s}, mix(pal::faint, a.theme().accent, k), Align::Right);
        if (h && a.in.released) {
            std::string pid = id, pname = name;
            a.open_picker("Sprite de " + name, sp.id, id == "human", [&a, pid, pname](const std::string& sid) {
                a.assign_sprite(pid, sid);
                a.notify(pname + " : " + a.sprite(sid).name);
            });
        }
    }
    if (a.ui.button("export", {gal.x + 22 * s, gal.bottom() - 74 * s, gal.w - 44 * s, 52 * s}, "Exporter tous les sprites en PNG", Style::Primary))
        a.notify(export_sprites(a));
    a.ui.blocked = false;
    if (picking) sprite_picker(a);
}

// The sprite library as a grid, by category (App::picker): a click calls
// picker.on_pick with the sprite's id.
void sprite_picker(App& a) {
    const float s = a.s;
    auto& pk = a.picker;
    draw::rect({0, 0, float(a.w), float(a.h)}, hex(0x000000, 160));
    Box p{a.w * 0.5f - 600 * s, 60 * s, 1200 * s, a.h - 120 * s};
    a.ui.panel(p, 22, hex(0x1a202b));
    a.text.draw(Font::Black, 26 * s, pk.title, {p.x + 30 * s, p.y + 24 * s}, pal::text);
    if (a.ui.button("pick_close", {p.right() - 170 * s, p.y + 20 * s, 140 * s, 46 * s}, "Fermer", Style::Ghost) || a.in.key(SDLK_ESCAPE)) {
        pk.open = false;
        a.in.eat(SDLK_ESCAPE);
        return;
    }
    Box grid{p.x + 24 * s, p.y + 84 * s, p.w - 48 * s, p.h - 190 * s};
    const float cell = 104 * s;
    const int cols = std::max(1, int(grid.w / cell));
    std::vector<std::string> cats;   // the drawings of the workshop first, then in the order of the file
    for (auto& sp : a.sprites) {
        std::string c = sp.category.empty() ? "Divers" : sp.category;
        if (std::find(cats.begin(), cats.end(), c) == cats.end()) cats.push_back(c);
    }
    std::stable_partition(cats.begin(), cats.end(), [](const std::string& c) { return c == "Mes dessins"; });
    if (a.ui.hover(grid)) pk.scroll -= a.in.wheel * cell * 0.8f;
    float y = grid.y - pk.scroll;
    const SpriteDef* hovered = nullptr;
    std::string picked;
    draw::clip(&grid);
    for (auto& cat : cats) {
        a.text.draw(Font::Bold, 17 * s, cat, {grid.x + 6 * s, y + 4 * s}, a.theme().accent);
        y += 34 * s;
        int col = 0;
        for (auto& sp : a.sprites) {
            if ((sp.category.empty() ? "Divers" : sp.category) != cat) continue;
            Box c{grid.x + col * cell, y, cell - 8 * s, cell - 8 * s};
            bool vis = c.bottom() > grid.y && c.y < grid.bottom();
            bool h = vis && a.ui.hover(c) && grid.contains(a.in.mouse);
            if (h) { a.ui.hovering = true; hovered = &sp; }
            if (vis) {
                float k = a.ui.anim("pk" + sp.id, h);
                bool cur = sp.id == pk.current;
                draw::round_box(c, 12 * s, with_alpha(cur ? a.theme().accent : hex(0xffffff), cur ? 0.22f : 0.03f + 0.08f * k));
                if (cur) draw::round_frame(c, 12 * s, 2 * s, a.theme().accent);
                draw_piece(sp, (col % 2) ? WHITE : BLACK, c.center(), c.w * 0.34f, 0.3f * k, a.time, 1, pk.halo);
                if (h && a.in.released) picked = sp.id;
            }
            if (++col == cols) { col = 0; y += cell; }
        }
        if (col) y += cell;
        y += 8 * s;
    }
    draw::clip(nullptr);
    float content = y + pk.scroll - grid.y;
    pk.scroll = std::clamp(pk.scroll, 0.0f, std::max(0.0f, content - grid.h));
    Box info{p.x + 24 * s, p.bottom() - 96 * s, p.w - 48 * s, 76 * s};
    draw::round_box(info, 14 * s, hex(0x0f141c, 200));
    if (hovered) {
        draw_piece(*hovered, BLACK, {info.x + 44 * s, info.center().y}, 28 * s, 0, a.time, 1, pk.halo);
        draw_piece(*hovered, WHITE, {info.x + 108 * s, info.center().y}, 28 * s, 0, a.time, 1, pk.halo);
        a.text.draw(Font::Bold, 20 * s, hovered->name, {info.x + 160 * s, info.y + 14 * s}, pal::text);
        a.text.draw(Font::Regular, 14 * s, strf("embl\xC3\xA8me \xC2\xAB %s \xC2\xBB \xE2\x80\xA2 niveau %d \xE2\x80\xA2 id %s", hovered->emblem.c_str(), hovered->tier, hovered->id.c_str()),
                    {info.x + 160 * s, info.y + 44 * s}, pal::dim);
    } else {
        a.text.draw(Font::Regular, 16 * s, "Survolez un sprite pour le voir sur les deux pierres ; cliquez pour le choisir.", {info.x + 24 * s, info.y + 28 * s}, pal::dim);
    }
    if (!picked.empty()) {
        pk.open = false;
        if (pk.on_pick) pk.on_pick(picked);
    }
}

// ─── @@SCENE_TUTORIAL ───────────────────────────────────────────────────────
// Learn to play. Each lesson is a position and a goal that the code checks
// after the player's move. A board is 8 strings, rank 8 first:
//   B black (the player)  W white  X the white piece to shut in
//   # hole  . free square  * target (a free square to reach)
// The engine needs exactly 4 pieces per side. The positions were checked by a
// small solver (one solution for "reach", the endgame lesson won by 2 with the
// big zone and lost by 2 with the small one).
struct Lesson {
    enum class Goal { Read, Reach, Moves, Immobilize, Win };
    std::string title, text, task, success, failure;
    std::vector<std::string> board;    // empty: the starting position
    Goal goal = Goal::Read;
    int moves = 1;                     // Goal::Moves: the player's moves
    bool replies = false;              // the opponent answers: at random (Moves), perfectly (Win)
    int score = 0;                     // starting score of both sides
    std::string button = "Continuer";
};

const std::vector<Lesson>& lessons() {
    using G = Lesson::Goal;
    static const std::vector<Lesson> L = [] {
        std::vector<Lesson> v;
        Lesson l;
        l = {};
        l.title = "Bienvenue dans Yolah !";
        l.text = "Yolah se joue \xC3\xA0 deux sur un plateau de 8 \xC3\x97 8 cases. Tu joues les Noirs : tes pions sont noirs, ou pos\xC3\xA9s sur un socle noir. L'adversaire joue les Blancs. Chaque camp a 4 pions, et les Noirs commencent toujours.";
        l.task = "Rep\xC3\xA8re tes 4 pions, puis clique sur Continuer.";
        v.push_back(l);

        l = {};
        l.title = "Comme une dame";
        l.text = "Un pion se d\xC3\xA9place en ligne droite ou en diagonale, comme la dame aux \xC3\xA9" "checs, d'autant de cases libres qu'il veut. Clique sur un pion : des points montrent o\xC3\xB9 il peut aller.";
        l.task = "Am\xC3\xA8ne ton pion de d4 sur la case dor\xC3\xA9" "e.";
        l.success = "Parfait ! Toute une diagonale en un seul coup.";
        l.board = {"W......W", "......*.", "........", "........", "...B....", "........", "........", "BBB...WW"};
        l.goal = G::Reach;
        v.push_back(l);

        l = {};
        l.title = "Pas de saut !";
        l.text = "Un pion ne saute jamais : ni par-dessus un autre pion, ni par-dessus un trou (les cases creus\xC3\xA9" "es). Les trous coupent les lignes.";
        l.task = "Un seul de tes pions peut atteindre la case dor\xC3\xA9" "e. Trouve-le !";
        l.success = "Bien vu : les trous barraient la route aux trois autres.";
        l.board = {"W..W...W", "........", "........", "B.#.*...", "....#...", "..#.....", ".......B", "B...B.W."};
        l.goal = G::Reach;
        v.push_back(l);

        l = {};
        l.title = "Chaque pas creuse un trou";
        l.text = "Quand un pion part, la case qu'il quitte s'effondre : elle devient un trou, et plus personne ne pourra y aller. Le plateau r\xC3\xA9tr\xC3\xA9" "cit \xC3\xA0 chaque coup.";
        l.task = "Joue le coup de ton choix et regarde ce qui reste derri\xC3\xA8re toi.";
        l.success = "Tu vois le trou ? Il restera l\xC3\xA0 jusqu'\xC3\xA0 la fin de la partie.";
        l.goal = G::Moves;
        l.moves = 1;
        v.push_back(l);

        l = {};
        l.title = "Un coup, un point";
        l.text = "Chaque coup rapporte 1 point, \xC3\xA0 toi comme \xC3\xA0 l'adversaire. Le score, c'est le nombre de coups jou\xC3\xA9s : le gagnant est celui qui peut bouger le plus longtemps.";
        l.task = "Joue 3 coups. L'adversaire r\xC3\xA9pond \xC3\xA0 chaque fois.";
        l.success = "3 points ! Le score suit le nombre de coups.";
        l.goal = G::Moves;
        l.moves = 3;
        l.replies = true;
        v.push_back(l);

        l = {};
        l.title = "Bloqu\xC3\xA9 ? On passe !";
        l.text = "Un camp qui ne peut plus bouger passe son tour, et l'autre rejoue. La partie ne s'arr\xC3\xAA" "te que quand plus personne ne peut bouger. Ici, les quatre pions blancs sont enferm\xC3\xA9s.";
        l.task = "Joue deux coups : les Blancs passeront entre les deux.";
        l.success = "Les Blancs ont pass\xC3\xA9 : leurs pions ne rapporteront plus rien.";
        l.board = {"W#....#W", "##....##", "........", "...BB...", "...BB...", "........", "##....##", "W#....#W"};
        l.goal = G::Moves;
        l.moves = 2;
        l.replies = true;
        v.push_back(l);

        l = {};
        l.title = "Enfermer un pion";
        l.text = "Un pion entour\xC3\xA9 de trous et de pions ne peut plus bouger : il ne rapporte plus de points. Enfermer les pions adverses, c'est leur voler des coups.";
        l.task = "En un seul coup, enferme le pion blanc du coin h8 (entour\xC3\xA9 de rouge).";
        l.success = "Enferm\xC3\xA9 ! Ce pion ne jouera plus de la partie.";
        l.board = {"WWW...#X", "......#.", "........", "........", "........", ".......B", "........", "BBB....."};
        l.goal = G::Immobilize;
        v.push_back(l);

        l = {};
        l.title = "Choisir son territoire";
        l.text = "En fin de partie, le plateau se d\xC3\xA9" "coupe en zones s\xC3\xA9par\xC3\xA9" "es. Chaque case d'une zone, c'est un coup de plus pour celui qui l'occupe. Ton pion en c5 peut entrer dans l'une de deux zones ; le pion blanc a d\xC3\xA9j\xC3\xA0 la sienne, de 4 cases.";
        l.task = "Gagne cette fin de partie : choisis la bonne zone, puis ne te coupe pas le chemin.";
        l.success = "Gagn\xC3\xA9 ! Tu as gard\xC3\xA9 plus de cases que l'adversaire.";
        l.failure = "Perdu : entre dans la plus grande zone, puis parcours-la case par case.";
        l.board = {"########", "#...####", "#...#W##", "##B##..#", "##.##..#", "##.#####", "B######W", "BB####WW"};
        l.goal = G::Win;
        l.replies = true;
        l.score = 20;
        v.push_back(l);

        l = {};
        l.title = "Pr\xC3\xAAt pour une vraie partie !";
        l.text = "Tu connais toutes les r\xC3\xA8gles. Le secret : garde de la place pour tes pions et prends-en \xC3\xA0 l'adversaire. Commence par l'IA la plus douce, Monte Carlo, puis monte en puissance.";
        l.task = "Lance ta premi\xC3\xA8re partie !";
        l.button = "Jouer contre Monte Carlo";
        v.push_back(l);
        return v;
    }();
    return L;
}

Yolah lesson_position(const Lesson& l, std::vector<int>& targets, int& victim) {
    targets.clear();
    victim = -1;
    if (l.board.size() != 8) return Yolah();
    uint64_t b = 0, w = 0, e = 0;
    for (int r = 0; r < 8; r++)
        for (int f = 0; f < 8; f++) {
            char ch = l.board[r][f];
            int sq = (7 - r) * 8 + f;
            uint64_t bit = uint64_t(1) << sq;
            if (ch == 'B') b |= bit;
            else if (ch == 'W' || ch == 'X') w |= bit;
            else if (ch == '#') e |= bit;
            if (ch == '*') targets.push_back(sq);
            if (ch == 'X') victim = sq;
        }
    Yolah y;
    y.set_state(b, w, e, uint16_t(l.score), uint16_t(l.score), 0);
    return y;
}

// Perfect play in the tiny endgames of the lessons: the final score
// difference for the side to move.
int solve(Yolah& y, int depth = 0) {
    if (y.game_over() || depth > 48) return y.score(y.current_player());
    int best = -10000;
    for (Move m : legal_moves(y)) {
        y.play(m);
        best = std::max(best, -solve(y, depth + 1));
        y.undo(m);
    }
    return best;
}
Move best_move(Yolah y) {
    Move best = Move::none();
    int bv = -10000;
    for (Move m : legal_moves(y)) {
        y.play(m);
        int v = -solve(y);
        y.undo(m);
        if (v > bv) { bv = v; best = m; }
    }
    return best;
}

// Does the move m (already played: `after`) reach the lesson's goal?
bool goal_reached(const Lesson& l, const Yolah& after, Move m, const std::vector<int>& targets, int victim) {
    if (l.goal == Lesson::Goal::Reach)
        return std::find(targets.begin(), targets.end(), int(m.to_sq())) != targets.end();
    if (l.goal == Lesson::Goal::Immobilize) {
        if (victim < 0 || !(after.bitboard(WHITE) >> victim & 1)) return false;
        Yolah::MoveList ml;
        after.moves(uint8_t(WHITE), ml);
        for (Move x : ml)
            if (x != Move::none() && x.from_sq() == victim) return false;
        return true;
    }
    return false;
}

int first_lesson_to_do(const App& a) {
    for (int i = 0; i < int(lessons().size()); i++)
        if (std::find(a.settings.tutorial_done.begin(), a.settings.tutorial_done.end(), i) == a.settings.tutorial_done.end()) return i;
    return 0;
}

void start_lesson(App& a, int i) {
    auto& t = a.tuto;
    t.index = std::clamp(i, 0, int(lessons().size()) - 1);
    t.match.reset(lesson_position(lessons()[t.index], t.targets, t.victim));
    t.board.reset();
    t.moves_done = 0;
    t.phase = App::TutorialState::Phase::Play;
    t.reply_at = t.pass_at = -1;
    t.feedback.clear();
    t.hint = false;
    t.hint_ply = size_t(-1);
    a.go(Scene::Tutorial);
}

void tutorial_play(App& a, Move m) {
    auto& t = a.tuto;
    const int side = t.match.to_move();
    t.match.play(m);
    t.board.on_move(m, side, a.time, a.fx, a.theme(), a.rng);
    if (m != Move::none()) {
        float q = t.board.sq();
        V2 c = t.board.center(m.to_sq());
        a.fx.add(std::make_unique<FloatText>(c + V2{0, -q * 0.3f}, "+1", side == BLACK ? hex(0x7ff0ff) : hex(0xffffff), q * 0.32f));
        a.fx.add(std::make_unique<Ripple>(c, q * 0.35f, q * 0.62f, a.theme().accent));
    } else {
        a.notify_at(std::string(side_name(side)) + " ne peuvent plus bouger : ils passent.", t.board.area.center());
    }
}

void tutorial_result(App& a, bool ok, const std::string& msg) {
    auto& t = a.tuto;
    t.phase = ok ? App::TutorialState::Phase::Success : App::TutorialState::Phase::Failed;
    t.phase_t = a.time;
    t.feedback = msg;
    t.hint = false;
    if (ok) {
        auto& done = a.settings.tutorial_done;
        if (std::find(done.begin(), done.end(), t.index) == done.end()) done.push_back(t.index);
        a.save_settings();
        a.fx.add(std::make_unique<Confetti>(Box{0, 0, float(a.w), float(a.h) * 0.6f}, a.rng));
    }
}

void tutorial_human_move(App& a, Move m) {
    auto& t = a.tuto;
    const Lesson& l = lessons()[t.index];
    tutorial_play(a, m);
    switch (l.goal) {
    case Lesson::Goal::Reach:
        if (goal_reached(l, t.match.pos, m, t.targets, t.victim)) tutorial_result(a, true, l.success);
        else tutorial_result(a, false, "Ce coup arrive en " + sq_name(m.to_sq()) + ", pas sur la case dor\xC3\xA9" "e. On recommence !");
        break;
    case Lesson::Goal::Immobilize:
        if (goal_reached(l, t.match.pos, m, t.targets, t.victim)) tutorial_result(a, true, l.success);
        else tutorial_result(a, false, "Le pion de " + sq_name(t.victim) + " peut encore bouger. On recommence !");
        break;
    case Lesson::Goal::Moves:
        if (++t.moves_done >= l.moves) tutorial_result(a, true, l.success);
        break;
    default: break;   // Win: checked at the end of the game
    }
}

void scene_tutorial(App& a) {
    auto& t = a.tuto;
    using Phase = App::TutorialState::Phase;
    const float s = a.s;
    const auto& L = lessons();
    const Lesson& l = L[t.index];
    if (a.in.key(SDLK_ESCAPE)) { a.go(Scene::Menu); return; }

    // ── logic ──
    const bool anim = t.board.animating(a.time);
    Yolah& pos = t.match.pos;
    if (t.phase == Phase::Play && !anim && l.goal != Lesson::Goal::Read) {
        if (t.match.over()) {
            if (l.goal == Lesson::Goal::Win)
                tutorial_result(a, t.match.score(BLACK) > t.match.score(WHITE), t.match.score(BLACK) > t.match.score(WHITE) ? l.success : l.failure);
        } else if (must_pass(pos)) {
            if (t.pass_at < 0) t.pass_at = a.time + 0.8;
            if (a.time >= t.pass_at) { t.pass_at = -1; tutorial_play(a, Move::none()); }
        } else if (pos.current_player() == WHITE && l.replies) {
            if (t.reply_at < 0) t.reply_at = a.time + 0.55;
            if (a.time >= t.reply_at) {
                t.reply_at = -1;
                Move m;
                if (l.goal == Lesson::Goal::Win) m = best_move(pos);
                else {
                    auto mv = legal_moves(pos);
                    m = mv[a.rng() % mv.size()];
                }
                tutorial_play(a, m);
            }
        }
    }
    if (t.phase == Phase::Failed && l.goal != Lesson::Goal::Win && a.time > t.phase_t + 1.8) start_lesson(a, t.index);

    // ── layout ──
    draw_background(a, false);
    draw_title(a, "Apprendre \xC3\xA0 jouer", "Neuf petites le\xC3\xA7ons, du premier coup \xC3\xA0 la premi\xC3\xA8re victoire.");
    if (back_button(a)) { a.go(Scene::Menu); return; }
    const float margin = 34 * s, list_w = 290 * s, panel_w = 470 * s, frame = 26 * s, top = 130 * s;
    Box list{margin, top, list_w, a.h - top - margin};
    Box panel{a.w - margin - panel_w, top, panel_w, a.h - top - margin};
    float avail_w = panel.x - list.right() - 2 * margin - 2 * frame;
    float B = std::floor(std::min(a.h - top - margin - 2 * frame - 20 * s, avail_w) / 8) * 8;
    t.board.area = {std::floor(list.right() + margin + frame + (avail_w - B) / 2), std::floor(top + frame + (a.h - top - margin - 2 * frame - B) / 2), B, B};

    // Lesson list.
    a.ui.panel(list);
    float rh = std::min(66 * s, (list.h - 20 * s) / L.size());
    for (size_t i = 0; i < L.size(); i++) {
        Box r{list.x + 10 * s, list.y + 10 * s + i * rh, list.w - 20 * s, rh - 4 * s};
        bool cur = int(i) == t.index;
        bool done = std::find(a.settings.tutorial_done.begin(), a.settings.tutorial_done.end(), int(i)) != a.settings.tutorial_done.end();
        bool h = a.ui.hover(r);
        if (h) a.ui.hovering = true;
        float k = a.ui.anim("ls" + std::to_string(i), h), kc = a.ui.anim("lc" + std::to_string(i), cur);
        draw::round_box(r, 10 * s, with_alpha(mix(hex(0xffffff), a.theme().accent, kc), 0.03f + 0.06f * k + 0.14f * kc));
        V2 cc{r.x + 24 * s, r.center().y};
        draw::circle(cc, 16 * s, done ? pal::good : (cur ? a.theme().accent : hex(0x3a4352)));
        if (done) draw_emblem("check", cc, 9 * s, hex(0x0b1a10));
        else a.text.draw(Font::Black, 15 * s, std::to_string(i + 1), {cc.x, cc.y - 9 * s}, cur ? hex(0x1a1206) : pal::text, Align::Center);
        auto lines = a.text.wrap(Font::Bold, 15 * s, L[i].title, r.w - 60 * s);
        float ty = r.center().y - lines.size() * 9 * s;
        for (auto& ln : lines) { a.text.draw(Font::Bold, 15 * s, ln, {r.x + 50 * s, ty}, cur ? pal::text : pal::dim); ty += 18 * s; }
        if (h && a.in.released && !cur) { start_lesson(a, int(i)); return; }
    }

    // Board.
    const bool human_turn = t.phase == Phase::Play && l.goal != Lesson::Goal::Read && !t.match.over() &&
                            pos.current_player() == BLACK && !must_pass(pos) && !anim;
    auto mv = t.board.input(a.in, pos, human_turn, false);
    t.board.drag_pos = a.in.mouse;
    if (mv) tutorial_human_move(a, *mv);
    if (t.hint && t.hint_ply != t.match.history.size()) {
        t.hint_ply = t.match.history.size();
        t.hint_move = Move::none();
        if (l.goal == Lesson::Goal::Win) t.hint_move = best_move(pos);
        else
            for (Move m : legal_moves(pos)) {
                if (m == Move::none()) continue;
                Yolah after = pos;
                after.play(m);
                if (l.goal == Lesson::Goal::Moves || goal_reached(l, after, m, t.targets, t.victim)) { t.hint_move = m; break; }
            }
    }
    BoardView::Look look;
    a.set_look(look, a.theme(), "human", "monte_carlo");
    look.human_turn = human_turn;
    look.coords = true;
    if (t.hint && human_turn && t.hint_move != Move::none()) { look.hint = t.hint_move; look.hint_label = "indice"; }
    t.board.draw(a.text, a.board_textures, a.theme(), pos, look, a.time, s);
    // Targets (golden squares) and the piece to shut in (red ring).
    float pulse = 0.75f + 0.25f * float(std::sin(a.time * 4));
    for (int sq : t.targets) {
        if (!pos.free_squares() || !((pos.free_squares() >> sq) & 1)) continue;
        Box b = t.board.square(sq).inset(t.board.sq() * 0.06f);
        draw::round_box(b, b.w * 0.18f, hex(0xffd36b, uint8_t(70 * pulse)));
        draw::round_frame(b, b.w * 0.18f, std::max(2.0f, b.w * 0.06f), hex(0xffd36b, uint8_t(255 * pulse)));
    }
    if (t.victim >= 0 && (pos.bitboard(WHITE) >> t.victim & 1))
        draw::ring(t.board.center(t.victim), t.board.sq() * 0.5f, t.board.sq() * 0.06f, hex(0xff5a5a, uint8_t(255 * pulse)));
    a.fx.update(float(a.ui.dt));
    a.fx.draw(a.text);

    // The guide's panel.
    a.ui.panel(panel);
    float y = panel.y + 22 * s;
    draw_piece(a.themed_sprite("human"), BLACK, {panel.x + 54 * s, y + 34 * s}, 32 * s, 0.3f, a.time, 1, true);
    a.text.draw(Font::Bold, 14 * s, strf("Le\xC3\xA7on %d sur %zu", t.index + 1, L.size()), {panel.x + 104 * s, y + 6 * s}, a.theme().accent);
    a.text.draw(Font::Black, 24 * s, l.title, {panel.x + 104 * s, y + 26 * s}, pal::text);
    y += 86 * s;
    // Speech bubble.
    auto lines = a.text.wrap(Font::Regular, 17 * s, l.text, panel.w - 80 * s);
    float bh = lines.size() * 25 * s + 30 * s;
    Box bubble{panel.x + 22 * s, y, panel.w - 44 * s, bh};
    draw::poly({V2{panel.x + 46 * s, y + 1}, V2{panel.x + 64 * s, y - 14 * s}, V2{panel.x + 76 * s, y + 1}}, hex(0xf4efe6));
    draw::round_box(bubble, 16 * s, hex(0xf8f4ec), hex(0xebe4d6));
    float ly = bubble.y + 15 * s;
    for (auto& ln : lines) { a.text.draw(Font::Regular, 17 * s, ln, {bubble.x + 18 * s, ly}, hex(0x2a2218)); ly += 25 * s; }
    y = bubble.bottom() + 20 * s;
    // The task.
    a.text.draw(Font::Black, 15 * s, "\xC3\x80 TOI !", {panel.x + 24 * s, y}, a.theme().accent);
    y += 24 * s;
    y += a.text.paragraph(Font::Bold, 18 * s, l.task, {panel.x + 24 * s, y, panel.w - 48 * s, 90 * s}, pal::text) + 10 * s;
    if (l.goal == Lesson::Goal::Moves || l.goal == Lesson::Goal::Win) {
        std::string sc = strf("Score : Noirs %d \xE2\x80\x93 Blancs %d", t.match.score(BLACK), t.match.score(WHITE));
        if (l.goal == Lesson::Goal::Moves) sc += strf("   \xE2\x80\xA2   coups : %d / %d", t.moves_done, l.moves);
        a.text.draw(Font::Regular, 16 * s, sc, {panel.x + 24 * s, y}, pal::dim);
        y += 30 * s;
    }
    // Feedback.
    if (!t.feedback.empty()) {
        bool ok = t.phase == Phase::Success;
        float k = ease_out_back(float((a.time - t.phase_t) / 0.4));
        auto fl = a.text.wrap(Font::Bold, 17 * s, t.feedback, panel.w - 96 * s);
        Box fb{panel.x + 22 * s, y, panel.w - 44 * s, fl.size() * 24 * s + 26 * s};
        Box fbk{fb.center().x - fb.w * 0.5f * k, fb.y, fb.w * k, fb.h};
        draw::round_box(fbk, 14 * s, with_alpha(ok ? pal::good : pal::bad, 0.9f), with_alpha(shade(ok ? pal::good : pal::bad, -0.25f), 0.9f));
        if (k > 0.9f) {
            draw_emblem(ok ? "check" : "cross", {fb.x + 26 * s, fb.y + fb.h / 2}, 12 * s, hex(0xffffff));
            float fy = fb.y + 13 * s;
            for (auto& ln : fl) { a.text.draw(Font::Bold, 17 * s, ln, {fb.x + 52 * s, fy}, hex(0xffffff)); fy += 24 * s; }
        }
    }
    // Buttons.
    Box bar{panel.x + 22 * s, panel.bottom() - 74 * s, panel.w - 44 * s, 52 * s};
    const bool last = t.index + 1 == int(L.size());
    const bool can_next = l.goal == Lesson::Goal::Read || t.phase == Phase::Success;
    float bw3 = (bar.w - 16 * s) / 3;
    if (l.goal != Lesson::Goal::Read) {
        if (a.ui.button("t_hint", {bar.x, bar.y, bw3, bar.h}, "Indice", t.hint ? Style::Primary : Style::Normal, human_turn, 17)) t.hint = !t.hint;
        if (a.ui.button("t_retry", {bar.x + bw3 + 8 * s, bar.y, bw3, bar.h}, "Recommencer", Style::Normal, true, 17)) { start_lesson(a, t.index); return; }
    }
    Box next = l.goal == Lesson::Goal::Read ? bar : Box{bar.x + 2 * (bw3 + 8 * s), bar.y, bw3, bar.h};
    std::string label = last ? l.button : (l.goal == Lesson::Goal::Read ? l.button : "Suivant \xE2\x86\x92");
    if (a.ui.button("t_next", next, label, can_next ? Style::Primary : Style::Normal, can_next, 17) || (can_next && a.in.key(SDLK_RETURN))) {
        if (last) {
            a.settings.black = "human";
            a.settings.white = "monte_carlo";
            a.start_game();
        } else {
            start_lesson(a, t.index + 1);
        }
        return;
    }
}


// ─── @@SCENE_PUZZLES ────────────────────────────────────────────────────────
// Plays a puzzle: a game from its position, the player against its AI, with
// its look; the introduction card first (scene_game shows it while
// game.intro is set), and at the end the goal is checked (draw_game_over).
// player_ai: an AI plays the player's side (the editor's "AI against AI" test).
void start_puzzle(App& a, const Puzzle& p, const std::string& level, int pos, bool from_editor, const std::string& player_ai) {
    std::string who[2];
    who[p.human] = player_ai.empty() ? "human" : player_ai;
    who[1 - p.human] = p.opponent;
    a.game.puzzle_mode = true;   // before launch: the look of the first frames
    a.game.puzzle = p;
    a.launch(puzzle_position(p), who[BLACK], who[WHITE], TIMES[p.time_index]);
    auto& g = a.game;
    g.puzzle_mode = true;
    g.puzzle = p;
    g.level = level;
    g.level_pos = pos;
    g.from_editor = from_editor;
    g.intro = true;
    g.recorded = false;
    g.board.flipped = p.human == WHITE;
}
std::string goal_text(const App& a, const Puzzle& p) {
    const AiSpec* ai = a.ai(p.opponent);
    std::string vs = ai ? ai->name : p.opponent;
    if (p.margin <= 1) return strf("Gagne avec les %s contre %s.", side_name(p.human), vs.c_str());
    return strf("Gagne avec les %s contre %s, avec au moins %d points d'avance.", side_name(p.human), vs.c_str(), p.margin);
}

// The window covered by a background image (cropped to keep its proportions),
// darkened a little so that the board and the panels stay readable.
// vignette: darken the edges (behind a board); off for a full-window picture.
void draw_image_background(App& a, const std::string& file, uint8_t dim, bool vignette, const Theme* theme) {
    SDL_Texture* tex = file.empty() ? nullptr : a.background(file);
    if (!tex) { draw_background(a, false, theme); return; }   // no picture: the scenery of the board's theme
    float tw, th;
    SDL_GetTextureSize(tex, &tw, &th);
    float k = std::max(a.w / tw, a.h / th);
    SDL_FRect dst{(a.w - tw * k) / 2, (a.h - th * k) / 2, tw * k, th * k};
    SDL_RenderTexture(a.renderer, tex, nullptr, &dst);
    draw::rect({0, 0, float(a.w), float(a.h)}, hex(0x000000, dim));
    if (vignette) draw::vignette({0, 0, float(a.w), float(a.h)}, a.s, 50);
}

// A small picture of a puzzle: its squares, holes and pieces.
void draw_mini_board(App& a, Box b, const Puzzle& p) {
    const Theme& t = a.theme_or(p.theme);
    draw::round_box(b.inset(-6 * a.s), 8 * a.s, t.frame_top, t.frame_bottom);
    SDL_FRect dst = b.frect();
    SDL_RenderTexture(a.renderer, a.board_textures.get(t, int(b.w)), nullptr, &dst);
    float q = b.w / 8;
    for (int sq = 0; sq < 64; sq++) {
        Box c{b.x + sq_file(sq) * q, b.y + (7 - sq_rank(sq)) * q, q, q};
        if (p.cells[sq] == '#') draw::round_box(c.inset(q * 0.08f), q * 0.2f, shade(t.hole, -0.3f), shade(t.hole, 0.12f));
        else if (p.cells[sq] == 'B') draw::circle_lit(c.center(), q * 0.38f, c.center() + V2{-q * 0.12f, -q * 0.14f}, hex(0x5b6170), hex(0x0a0c11));
        else if (p.cells[sq] == 'W') draw::circle_lit(c.center(), q * 0.38f, c.center() + V2{-q * 0.12f, -q * 0.14f}, hex(0xffffff), hex(0xb9bec8));
    }
}

void scene_puzzles(App& a) {
    const float s = a.s;
    auto& st = a.puzzles_ui;
    draw_background(a);
    draw_title(a, "Puzzles et qu\xC3\xAA" "tes", "Des positions \xC3\xA0 r\xC3\xA9soudre, regroup\xC3\xA9" "es en niveaux : chaque puzzle r\xC3\xA9ussi ouvre le suivant.");
    a.ui.blocked = !a.ui.focus.empty() && false;
    if (a.ui.focus.empty() && back_button(a)) {
        if (!st.open_level.empty()) st.open_level.clear();
        else { a.go(Scene::Menu); return; }
    }
    st.tab = 0;
    if (a.ui.button("to_editor", {60 * s, 118 * s, 360 * s, 44 * s}, "Cr\xC3\xA9" "er ou modifier des puzzles et des niveaux", Style::Normal, true, 15)) { a.go(Scene::Editor); return; }
    Box area{60 * s, 180 * s, a.w - 120 * s, a.h - 210 * s};

    if (st.tab == 0 && st.open_level.empty()) {
        // The levels, plus every puzzle in one list.
        std::vector<Level> shown = a.levels;
        Level all;
        all.id = "*";
        all.name = "Tous les puzzles";
        all.text = "Chaque puzzle de assets/puzzles, sans ordre impos\xC3\xA9.";
        for (auto& p : a.puzzles) all.puzzles.push_back(p.id);
        shown.push_back(all);
        const float cw = 360 * s, ch = 170 * s, gap = 24 * s;
        int cols = std::max(1, int((area.w + gap) / (cw + gap)));
        for (size_t i = 0; i < shown.size(); i++) {
            const Level& l = shown[i];
            Box c{area.x + (i % cols) * (cw + gap), area.y + (i / cols) * (ch + gap), cw, ch};
            // A level opens when the one before is finished ("Tous les puzzles": always).
            bool unlocked = l.id == "*" || i == 0;
            if (!unlocked) {
                unlocked = true;
                for (auto& id : shown[i - 1].puzzles) unlocked &= a.solved(id);
            }
            bool h = unlocked && a.ui.hover(c);
            if (h) a.ui.hovering = true;
            float k = a.ui.anim("lv" + l.id, h);
            Box cc{c.x, c.y - 4 * s * k, c.w, c.h};
            a.ui.panel(cc, 18);
            int done = 0;
            for (auto& id : l.puzzles) done += a.solved(id);
            a.text.draw(Font::Black, 22 * s, l.name, {cc.x + 22 * s, cc.y + 18 * s}, pal::text);
            a.text.paragraph(Font::Regular, 14 * s, l.text, {cc.x + 22 * s, cc.y + 52 * s, cc.w - 44 * s, 50 * s}, pal::dim);
            a.text.draw(Font::Bold, 15 * s, strf("%zu puzzles \xE2\x80\xA2 %d r\xC3\xA9ussis", l.puzzles.size(), done), {cc.x + 22 * s, cc.bottom() - 52 * s}, a.theme().accent);
            Box bar{cc.x + 22 * s, cc.bottom() - 24 * s, cc.w - 44 * s, 8 * s};
            draw::round_box(bar, 4 * s, hex(0xffffff, 25));
            if (!l.puzzles.empty()) draw::round_box({bar.x, bar.y, bar.w * done / l.puzzles.size(), bar.h}, 4 * s, pal::good);
            if (!unlocked) {
                draw::round_box(cc, 18 * s, hex(0x000000, 130));
                a.text.draw(Font::Black, 17 * s, strf("verrouill\xC3\xA9 : terminez \xC2\xAB %s \xC2\xBB", shown[i - 1].name.c_str()), {cc.center().x, cc.center().y - 10 * s}, pal::dim, Align::Center);
            }
            if (h && a.in.released) { st.open_level = l.id; st.scroll = 0; }
        }
        if (a.levels.empty())
            a.text.draw(Font::Regular, 16 * s, "Aucun niveau pour l'instant : cr\xC3\xA9" "ez-en un dans l'\xC3\xA9" "diteur.", {area.x, area.bottom() - 30 * s}, pal::faint);
    } else if (st.tab == 0) {
        // The puzzles of one level.
        Level l;
        bool any_order = st.open_level == "*";
        if (any_order) { l.name = "Tous les puzzles"; for (auto& p : a.puzzles) l.puzzles.push_back(p.id); }
        else for (auto& x : a.levels) if (x.id == st.open_level) l = x;
        a.text.draw(Font::Black, 24 * s, l.name, {area.x, area.y - 6 * s}, pal::text);
        const float cw = 230 * s, ch = 290 * s, gap = 22 * s;
        int cols = std::max(1, int((area.w + gap) / (cw + gap)));
        Box grid{area.x, area.y + 40 * s, area.w, area.h - 40 * s};
        if (a.ui.hover(grid)) st.scroll = std::max(0.0f, st.scroll - a.in.wheel * 60 * s);
        draw::clip(&grid);
        for (size_t i = 0; i < l.puzzles.size(); i++) {
            const Puzzle* p = a.find_puzzle(l.puzzles[i]);
            Box c{grid.x + (i % cols) * (cw + gap), grid.y + (i / cols) * (ch + gap) - st.scroll, cw, ch};
            bool unlocked = any_order || i == 0 || a.solved(l.puzzles[i - 1]);
            bool ok = p && a.solved(p->id);
            bool h = p && unlocked && a.ui.hover(c) && grid.contains(a.in.mouse);
            if (h) a.ui.hovering = true;
            float k = a.ui.anim("pz" + l.puzzles[i], h);
            a.ui.panel({c.x, c.y - 4 * s * k, c.w, c.h}, 16);
            if (!p) { a.text.draw(Font::Regular, 14 * s, "introuvable : " + l.puzzles[i], {c.x + 16 * s, c.y + 20 * s}, pal::bad); continue; }
            Box mb{c.x + 25 * s, c.y + 22 * s - 4 * s * k, 180 * s, 180 * s};
            draw_mini_board(a, mb, *p);
            if (!unlocked) {
                draw::round_box(mb.inset(-6 * s), 8 * s, hex(0x000000, 150));
                a.text.draw(Font::Black, 18 * s, "verrouill\xC3\xA9", {mb.center().x, mb.center().y - 12 * s}, pal::dim, Align::Center);
            }
            a.text.draw(Font::Black, 15 * s, strf("%zu.", i + 1), {c.x + 16 * s, c.bottom() - 74 * s}, a.theme().accent);
            a.text.draw(Font::Bold, 17 * s, p->name, {c.x + 44 * s, c.bottom() - 76 * s}, pal::text);
            a.text.draw(Font::Regular, 13 * s, ok ? "r\xC3\xA9ussi" : (p->margin > 1 ? strf("gagner de %d+", p->margin) : std::string("gagner")),
                        {c.x + 44 * s, c.bottom() - 50 * s}, ok ? pal::good : pal::dim);
            if (ok) draw_emblem("check", {c.right() - 30 * s, c.bottom() - 56 * s}, 12 * s, pal::good);
            if (h && a.in.released) { draw::clip(nullptr); start_puzzle(a, *p, any_order ? "" : l.id, int(i), false); return; }
        }
        draw::clip(nullptr);
        if (l.puzzles.empty()) a.text.draw(Font::Regular, 16 * s, "Ce niveau est vide.", {grid.x, grid.y}, pal::faint);
    }
}

// ─── @@SCENE_EDITOR ─────────────────────────────────────────────────────────
// One screen for everything: on the left the levels and their puzzles (open,
// add, duplicate, delete, reorder, create a level), in the middle the board
// and its brushes (click or drag), on the right the puzzle's settings and the
// two tests (me against the AI, AI against AI). Every change is saved when the
// puzzle is left or tested.
void editor_save(App& a) {
    auto& ed = a.editor;
    if (!save_puzzle(ed.p)) { a.notify("Impossible d'enregistrer le puzzle."); return; }
    ed.saved = puzzle_to_json(ed.p).dump();
    // A puzzle created in a level belongs to it.
    for (auto& l : a.levels)
        if (l.id == ed.level && std::find(l.puzzles.begin(), l.puzzles.end(), ed.p.id) == l.puzzles.end()) {
            l.puzzles.push_back(ed.p.id);
            save_level(l);
        }
    std::string keep_level = ed.level;
    a.reload_puzzles();
    ed.level = keep_level;
}
bool editor_dirty(const App& a) { return puzzle_to_json(a.editor.p).dump() != a.editor.saved; }
void editor_open(App& a, const Puzzle& p, const std::string& level) {
    auto& ed = a.editor;
    if (editor_dirty(a) && !ed.p.id.empty()) editor_save(a);
    ed.p = p;
    ed.level = level;
    ed.saved = puzzle_to_json(p).dump();
    ed.confirm_delete = false;
}
// A new puzzle in `level`: the starting position, or a copy of `model` (its
// board, its settings and its look) under a new name.
void editor_new(App& a, const std::string& level, const Puzzle* model = nullptr) {
    auto& ed = a.editor;
    if (editor_dirty(a) && !ed.p.id.empty()) editor_save(a);
    ed.p = model ? *model : Puzzle{};
    if (model) { ed.p.id.clear(); ed.p.name = model->name + " (variante)"; }
    ed.level = level;
    editor_save(a);   // it appears at once in the list (and in its level)
    ed.confirm_delete = false;
}

// The "new puzzle" dialog: the starting position, or any existing puzzle as a
// model (by level, then those in no level).
void editor_new_dialog(App& a) {
    const float s = a.s;
    auto& ed = a.editor;
    draw::rect({0, 0, float(a.w), float(a.h)}, hex(0x000000, 160));
    Box p{a.w * 0.5f - 380 * s, 70 * s, 760 * s, a.h - 140 * s};
    a.ui.panel(p, 22, hex(0x1a202b));
    a.text.draw(Font::Black, 24 * s, "Nouveau puzzle", {p.x + 28 * s, p.y + 22 * s}, pal::text);
    a.text.draw(Font::Regular, 15 * s, "Partir de la position de d\xC3\xA9part, ou d'une copie de n'importe quel puzzle (plateau, r\xC3\xA9glages, apparence).",
                {p.x + 28 * s, p.y + 60 * s}, pal::dim);
    if (a.ui.button("nw_close", {p.right() - 150 * s, p.y + 18 * s, 122 * s, 42 * s}, "Annuler", Style::Ghost) || a.in.key(SDLK_ESCAPE)) {
        ed.choose_new = false;
        a.in.eat(SDLK_ESCAPE);
        return;
    }
    Box list{p.x + 20 * s, p.y + 96 * s, p.w - 40 * s, p.h - 116 * s};
    if (a.ui.hover(list)) ed.new_scroll = std::max(0.0f, ed.new_scroll - a.in.wheel * 40 * s);
    float y = list.y - ed.new_scroll;
    const float rh = 54 * s;
    const Puzzle* chosen = nullptr;
    bool blank = false;
    draw::clip(&list);
    auto row = [&](const std::string& id, const std::string& title, const std::string& sub, const Puzzle* pz) {
        Box r{list.x, y, list.w, rh - 6 * s};
        bool h = a.ui.hover(r) && list.contains(a.in.mouse);
        if (h) a.ui.hovering = true;
        draw::round_box(r, 10 * s, with_alpha(hex(0xffffff), h ? 0.1f : 0.03f));
        // A small view of the board.
        Box mb{r.x + 8 * s, r.y + 4 * s, r.h - 8 * s, r.h - 8 * s};
        Puzzle start;
        const Puzzle& src = pz ? *pz : start;
        float q = mb.w / 8;
        draw::rect(mb, a.theme_or(src.theme).light);
        for (int sq = 0; sq < 64; sq++) {
            char c = src.cells[sq];
            V2 cc{mb.x + (sq % 8 + 0.5f) * q, mb.y + (7 - sq / 8 + 0.5f) * q};
            if (c == '#') draw::rect({cc.x - q / 2, cc.y - q / 2, q, q}, hex(0x101318));
            else if (c == 'B') draw::circle(cc, q * 0.4f, hex(0x222630));
            else if (c == 'W') draw::circle(cc, q * 0.4f, hex(0xf4f6fa));
        }
        a.text.draw(Font::Bold, 16 * s, title, {mb.right() + 14 * s, r.y + 7 * s}, pal::text);
        a.text.draw(Font::Regular, 13 * s, sub, {mb.right() + 14 * s, r.y + 27 * s}, pal::dim);
        if (h && a.in.released) { if (pz) chosen = pz; else blank = true; }
        y += rh;
        (void)id;
    };
    row("", "Position de d\xC3\xA9part", "le plateau du d\xC3\xA9" "but de partie", nullptr);
    std::set<std::string> seen;
    for (auto& l : a.levels) {
        a.text.draw(Font::Bold, 14 * s, l.name, {list.x + 4 * s, y + 10 * s}, a.theme().accent);
        y += 36 * s;
        for (auto& id : l.puzzles)
            if (const Puzzle* pz = a.find_puzzle(id)) { seen.insert(id); row(id, pz->name, pz->text.empty() ? id : pz->text, pz); }
    }
    bool header = false;
    for (auto& pz : a.puzzles) {
        if (seen.count(pz.id)) continue;
        if (!header) { a.text.draw(Font::Bold, 14 * s, "Hors niveau", {list.x + 4 * s, y + 10 * s}, pal::faint); y += 36 * s; header = true; }
        row(pz.id, pz.name, pz.text.empty() ? pz.id : pz.text, &pz);
    }
    draw::clip(nullptr);
    float content = y + ed.new_scroll - list.y;
    ed.new_scroll = std::clamp(ed.new_scroll, 0.0f, std::max(0.0f, content - list.h));
    if (chosen || blank) {
        Puzzle model = chosen ? *chosen : Puzzle{};
        ed.choose_new = false;
        editor_new(a, ed.new_level, chosen ? &model : nullptr);
        if (!ed.new_level.empty()) ed.open_levels.insert(ed.new_level);
        a.notify(chosen ? "Variante cr\xC3\xA9\xC3\xA9" "e : " + ed.p.name : std::string("Nouveau puzzle cr\xC3\xA9\xC3\xA9 : peignez le plateau."));
    }
}

void scene_editor(App& a) {
    const float s = a.s;
    auto& ed = a.editor;
    Puzzle& p = ed.p;
    if (p.id.empty() && ed.saved.empty()) {   // first visit: the first puzzle of the first level, unfolded
        for (auto& l : a.levels)
            if (!l.puzzles.empty() && a.find_puzzle(l.puzzles.front())) {
                editor_open(a, *a.find_puzzle(l.puzzles.front()), l.id);
                ed.open_levels.insert(l.id);
                ed.level_selected = l.id;
                break;
            }
        if (ed.saved.empty() && !a.puzzles.empty()) editor_open(a, a.puzzles.front(), "");
    }
    const bool modal = a.picker.open || ed.choose_new;
    draw_image_background(a, p.background, 120, true, &a.theme_or(p.theme));
    a.text.draw_shadowed(Font::Title, 34 * s, "\xC3\x89" "diteur de puzzles", {40 * s, 24 * s}, pal::text, Align::Left, hex(0x000000, 120), 3 * s);
    a.ui.blocked = modal;
    if (!modal && a.ui.focus.empty() && back_button(a)) {
        if (editor_dirty(a)) editor_save(a);
        a.go(Scene::Menu);
        return;
    }
    if (a.ui.focus.empty() && !modal)
        for (int k = 0; k < 6; k++)
            if (a.in.key(SDLK_1 + k)) ed.tool = k;   // keys 1-6: the brushes

    const float margin = 26 * s, tree_w = 330 * s, panel_w = 440 * s, top = 100 * s;
    Box tree{margin, top, tree_w, a.h - top - margin};
    Box panel{a.w - margin - panel_w, top, panel_w, a.h - top - margin};
    Box mid{tree.right() + margin, top, panel.x - tree.right() - 2 * margin, a.h - top - margin};

    // ── left: levels and puzzles ──
    a.ui.panel(tree);
    a.text.draw(Font::Black, 18 * s, "Niveaux et puzzles", {tree.x + 16 * s, tree.y + 14 * s}, pal::text);
    Box list{tree.x + 8 * s, tree.y + 48 * s, tree.w - 16 * s, tree.h - 168 * s};
    if (a.ui.hover(list)) ed.scroll = std::max(0.0f, ed.scroll - a.in.wheel * 40 * s);
    float y = list.y - ed.scroll;
    const float rh = 34 * s;
    std::string action, action_level, action_puzzle;   // done after the loop (the lists may change)
    int move_dir = 0;
    draw::clip(&list);
    auto puzzle_row = [&](const std::string& pid, const std::string& lid, int idx, int count) {
        Box r{list.x + 18 * s, y, list.w - 18 * s, rh - 3 * s};
        const Puzzle* pz = a.find_puzzle(pid);
        bool cur = pid == p.id && lid == ed.level;
        bool h = a.ui.hover(r) && list.contains(a.in.mouse);
        if (h) a.ui.hovering = true;
        draw::round_box(r, 7 * s, with_alpha(cur ? a.theme().accent : hex(0xffffff), cur ? 0.3f : (h ? 0.08f : 0.0f)));
        a.text.draw(cur ? Font::Bold : Font::Regular, 14 * s, pz ? pz->name : pid + " (introuvable)", {r.x + 10 * s, r.y + 8 * s}, pz ? pal::text : pal::bad);
        if (cur && !lid.empty()) {
            // × : out of this level (the puzzle itself stays, in "Hors niveau" if no level has it).
            if (a.ui.button("mx" + pid, {r.right() - 92 * s, r.y + 2 * s, 28 * s, r.h - 4 * s}, "\xC3\x97", Style::Ghost, true, 15)) { action = "unlink"; action_level = lid; action_puzzle = pid; }
            if (idx > 0 && a.ui.button("mu" + pid, {r.right() - 62 * s, r.y + 2 * s, 28 * s, r.h - 4 * s}, "\xE2\x86\x91", Style::Ghost, true, 14)) { action = "move"; action_level = lid; action_puzzle = pid; move_dir = -1; }
            if (idx + 1 < count && a.ui.button("md" + pid, {r.right() - 32 * s, r.y + 2 * s, 28 * s, r.h - 4 * s}, "\xE2\x86\x93", Style::Ghost, true, 14)) { action = "move"; action_level = lid; action_puzzle = pid; move_dir = 1; }
        } else if (h && a.in.released && pz) { action = "open"; action_level = lid; action_puzzle = pid; }
        y += rh;
    };
    std::set<std::string> in_level;
    for (auto& l : a.levels) {
        Box hr{list.x, y, list.w, rh};
        bool open = ed.open_levels.count(l.id) > 0;
        bool sel = ed.level == l.id;
        bool h = a.ui.hover(hr) && list.contains(a.in.mouse);
        if (h) a.ui.hovering = true;
        draw::round_box(hr, 7 * s, with_alpha(hex(0xffffff), sel ? 0.12f : (h ? 0.06f : 0.03f)));
        {   // a small triangle: pointing down when unfolded
            V2 c{hr.x + 15 * s, hr.center().y};
            float t = 5 * s;
            if (open) draw::poly({c + V2{-t, -t * 0.6f}, c + V2{t, -t * 0.6f}, c + V2{0, t * 0.8f}}, pal::dim);
            else draw::poly({c + V2{-t * 0.6f, -t}, c + V2{t * 0.8f, 0}, c + V2{-t * 0.6f, t}}, pal::dim);
        }
        a.text.draw(Font::Bold, 15 * s, strf("%s  (%zu)", l.name.c_str(), l.puzzles.size()), {hr.x + 28 * s, hr.y + 8 * s}, pal::text);
        const int li = int(&l - &a.levels[0]);
        bool on_arrows = false;
        if (ed.level_selected == l.id) {   // the order of the levels: the order in which they are played
            Box up{hr.right() - 62 * s, hr.y + 3 * s, 28 * s, hr.h - 6 * s}, down{hr.right() - 32 * s, hr.y + 3 * s, 28 * s, hr.h - 6 * s};
            on_arrows = up.contains(a.in.mouse) || down.contains(a.in.mouse);
            if (a.ui.button("lu" + l.id, up, "\xE2\x86\x91", Style::Ghost, li > 0, 14)) { action = "lvmove"; action_level = l.id; move_dir = -1; }
            if (a.ui.button("ld" + l.id, down, "\xE2\x86\x93", Style::Ghost, li + 1 < int(a.levels.size()), 14)) { action = "lvmove"; action_level = l.id; move_dir = 1; }
        }
        if (h && a.in.released && !on_arrows) {
            if (open) ed.open_levels.erase(l.id);
            else ed.open_levels.insert(l.id);
            ed.level_selected = l.id;
        }
        y += rh;
        if (open) {
            for (size_t i = 0; i < l.puzzles.size(); i++) {
                in_level.insert(l.puzzles[i]);
                puzzle_row(l.puzzles[i], l.id, int(i), int(l.puzzles.size()));
            }
            Box add{list.x + 18 * s, y, list.w - 18 * s, rh - 4 * s};
            if (a.ui.button("addp" + l.id, add, "+ nouveau puzzle dans ce niveau\xE2\x80\xA6", Style::Ghost, true, 13)) { action = "new"; action_level = l.id; }
            y += rh;
            if (p.id.size() && std::find(l.puzzles.begin(), l.puzzles.end(), p.id) == l.puzzles.end()) {
                if (a.ui.button("addc" + l.id, {list.x + 18 * s, y, list.w - 18 * s, rh - 4 * s}, "+ ajouter \xC2\xAB " + p.name + " \xC2\xBB", Style::Ghost, true, 13)) { action = "add"; action_level = l.id; }
                y += rh;
            }
        } else {
            for (auto& id : l.puzzles) in_level.insert(id);
        }
    }
    // Puzzles in no level.
    std::vector<std::string> loose;
    for (auto& pz : a.puzzles) if (!in_level.count(pz.id)) loose.push_back(pz.id);
    if (!loose.empty()) {
        a.text.draw(Font::Bold, 13 * s, "Hors niveau", {list.x + 6 * s, y + 10 * s}, pal::faint);
        y += rh;
        for (size_t i = 0; i < loose.size(); i++) puzzle_row(loose[i], "", int(i), int(loose.size()));
    }
    draw::clip(nullptr);
    float content = y + ed.scroll - list.y;
    ed.scroll = std::clamp(ed.scroll, 0.0f, std::max(0.0f, content - list.h));
    // The selected level: its name, editable.
    Level* lsel = nullptr;
    for (auto& l : a.levels) if (l.id == ed.level_selected) lsel = &l;
    float by = tree.bottom() - 112 * s;
    if (lsel) {
        const float dw = 104 * s;
        if (a.ui.text_field("lv_name", {tree.x + 12 * s, by, tree.w - 32 * s - dw, 40 * s}, lsel->name, "Nom du niveau")) save_level(*lsel);
        bool confirm = ed.confirm_level == lsel->id;
        if (a.ui.button("lv_del", {tree.right() - 12 * s - dw, by, dw, 40 * s}, confirm ? "Confirmer ?" : "Supprimer", Style::Danger, true, 13)) {
            if (confirm) { action = "dellevel"; action_level = lsel->id; }
            else ed.confirm_level = lsel->id;
        }
        if (confirm) a.text.draw(Font::Regular, 12 * s, "Le niveau dispara\xC3\xAEt, ses puzzles restent.", {tree.x + 14 * s, by - 18 * s}, pal::bad);
    } else {
        a.text.paragraph(Font::Regular, 13 * s, "Cliquez sur un niveau pour l'ouvrir, le renommer et le d\xC3\xA9placer (\xE2\x86\x91 \xE2\x86\x93 : l'ordre de jeu).", {tree.x + 16 * s, by + 4 * s, tree.w - 32 * s, 40 * s}, pal::faint);
    }
    if (a.ui.button("lv_new", {tree.x + 12 * s, tree.bottom() - 60 * s, tree.w - 24 * s, 48 * s}, "+ Nouveau niveau", Style::Normal, true, 16)) action = "newlevel";

    // ── middle: brushes and board ──
    const char* names[6] = {"Libre", "Trou", "Noir", "Blanc", "Case", "D\xC3\xA9" "cor"};
    float tbw = (mid.w - 5 * 6 * s) / 6;
    for (int i = 0; i < 6; i++) {
        Box r{mid.x + i * (tbw + 6 * s), mid.y, tbw, 44 * s};
        if (a.ui.button("tool" + std::to_string(i), r, "", ed.tool == i ? Style::Primary : Style::Normal)) ed.tool = i;
        V2 ic{r.x + 20 * s, r.center().y};
        if (i == 0) draw::round_box({ic.x - 10 * s, ic.y - 10 * s, 20 * s, 20 * s}, 4 * s, a.theme_or(p.theme).light);
        else if (i == 1) draw::round_box({ic.x - 10 * s, ic.y - 10 * s, 20 * s, 20 * s}, 4 * s, hex(0x0d0f12));
        else if (i <= 3) draw::circle_lit(ic, 10 * s, ic + V2{-3 * s, -3 * s}, i == 2 ? hex(0x5b6170) : hex(0xffffff), i == 2 ? hex(0x0a0c11) : hex(0xb9bec8));
        else {
            Art* ia = find_art(i == 4 ? ed.tile_art : ed.decor_art);
            Box ib{ic.x - 11 * s, ic.y - 11 * s, 22 * s, 22 * s};
            if (ia) draw_art(*ia, "idle", a.time, ib);
            else draw::round_frame(ib, 4 * s, 1.5f * s, pal::dim);
        }
        a.text.draw(Font::Bold, 15 * s, names[i], {r.x + 38 * s, r.center().y - 9 * s}, ed.tool == i ? hex(0x1a1206) : pal::text);
    }
    // The second row: what the brush does.
    const std::vector<float> DECOR_SIZES = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f};
    Box row2{mid.x, mid.y + 50 * s, mid.w, 38 * s};
    auto art_choice = [&](const char* id, const char* kind, std::string& cur, Box b, const char* label) {
        auto list = arts_of(kind);
        if (list.empty()) {
            a.text.draw(Font::Regular, 14 * s, strf("Aucun dessin de type \xC2\xAB %s \xC2\xBB : cr\xC3\xA9" "ez-en dans l'atelier.", art_kind(kind).label), {b.x, b.y + 10 * s}, pal::faint);
            return;
        }
        std::vector<std::string> nm;
        int idx = 0;
        for (size_t i = 0; i < list.size(); i++) { nm.push_back(list[i]->name); if (list[i]->id == cur) idx = int(i); }
        cur = list[a.ui.cycler(id, b, label, nm, idx, 14)]->id;
    };
    if (ed.tool <= 3) {
        float bw2 = (row2.w - 12 * s) / 3;
        if (a.ui.button("ed_fill", {row2.x, row2.y, bw2, row2.h}, "Tout noircir", Style::Ghost, true, 14)) for (auto& c : p.cells) if (c == '.') c = '#';
        if (a.ui.button("ed_clear", {row2.x + bw2 + 6 * s, row2.y, bw2, row2.h}, "Tout vider", Style::Ghost, true, 14)) for (auto& c : p.cells) if (c == '#') c = '.';
        if (a.ui.button("ed_start", {row2.x + 2 * (bw2 + 6 * s), row2.y, bw2, row2.h}, "D\xC3\xA9part", Style::Ghost, true, 14)) p.cells = Puzzle().cells;
    } else if (ed.tool == 4) {
        art_choice("ed_tile", "tile", ed.tile_art, {row2.x, row2.y, row2.w * 0.5f, row2.h}, "Case");
        a.text.draw(Font::Regular, 13 * s, "clic : peindre \xE2\x80\xA2 clic droit : effacer", {row2.x + row2.w * 0.52f, row2.y + 11 * s}, pal::faint);
        if (a.ui.button("ed_tiles_clear", {row2.right() - 120 * s, row2.y, 120 * s, row2.h}, "Tout effacer", Style::Ghost, !p.tiles.empty(), 13)) p.tiles.clear();
    } else {
        art_choice("ed_decor", "decor", ed.decor_art, {row2.x, row2.y, row2.w * 0.4f, row2.h}, "D\xC3\xA9" "cor");
        std::vector<std::string> sz;
        for (float k : DECOR_SIZES) sz.push_back(strf("%g case%s", k, k > 1 ? "s" : ""));
        ed.decor_size = a.ui.cycler("ed_dsize", {row2.x + row2.w * 0.42f, row2.y, row2.w * 0.3f, row2.h}, "Taille", sz, ed.decor_size, 14);
        if (a.ui.button("ed_decor_clear", {row2.right() - 120 * s, row2.y, 120 * s, row2.h}, "Tout retirer", Style::Ghost, !p.decor.empty(), 13)) p.decor.clear();
    }
    const float frame = 26 * s;
    float bavail_h = mid.h - 96 * s - 40 * s - 2 * frame;
    float B = std::floor(std::min(bavail_h, mid.w - 2 * frame) / 8) * 8;
    ed.board.area = {std::floor(mid.x + (mid.w - B) / 2), std::floor(mid.y + 96 * s + frame), B, B};
    // Painting: a press on the board starts it; while the button is held, the
    // brush paints every square the mouse passes over (pieces: on the press only).
    int hover_sq = modal ? -1 : ed.board.at(a.in.mouse);
    if (a.in.pressed || a.in.rpressed) ed.painting = hover_sq >= 0;
    if (!a.in.down && !a.in.rdown) { ed.painting = false; ed.painted = -1; }
    if (ed.tool <= 3 && ed.painting && hover_sq >= 0 && hover_sq != ed.painted && (a.in.pressed || ed.tool <= 1)) {
        ed.painted = hover_sq;
        const char want = ed.tool == 0 ? '.' : ed.tool == 1 ? '#' : ed.tool == 2 ? 'B' : 'W';
        char& c = p.cells[hover_sq];
        if ((want == 'B' || want == 'W') && c != want && p.count(want) >= 4)
            a.notify(strf("D\xC3\xA9j\xC3\xA0 4 pions %s : retirez-en un d'abord (pinceau Libre).", want == 'B' ? "noirs" : "blancs"));
        else if (a.in.pressed && (want == 'B' || want == 'W') && c == want) c = '.';   // a click on a piece removes it
        else c = want;
    }
    if (ed.tool == 4 && ed.painting && hover_sq >= 0) {   // special squares: painted (left) or erased (right)
        if (a.in.rdown) p.tiles.erase(hover_sq);
        else if (!ed.tile_art.empty()) p.tiles[hover_sq] = ed.tile_art;
    }
    if (ed.tool == 5 && !modal) {   // decorations: anywhere on the board or around it
        const float q = ed.board.sq();
        Box zone = ed.board.area.inset(-frame - 10 * s);
        V2 m{(a.in.mouse.x - ed.board.area.x) / q, (a.in.mouse.y - ed.board.area.y) / q};
        int hit = -1;
        for (int i = int(p.decor.size()) - 1; i >= 0 && hit < 0; i--)
            if (std::abs(m.x - p.decor[i].x) < p.decor[i].scale * 0.5f && std::abs(m.y - p.decor[i].y) < p.decor[i].scale * 0.5f) hit = i;
        if (zone.contains(a.in.mouse)) {
            a.ui.hovering = true;
            if (a.in.rpressed && hit >= 0) p.decor.erase(p.decor.begin() + hit);
            else if (a.in.pressed) {
                if (hit >= 0) ed.decor_drag = hit;
                else if (!ed.decor_art.empty()) {
                    p.decor.push_back({ed.decor_art, m.x, m.y, DECOR_SIZES[ed.decor_size]});
                    ed.decor_drag = int(p.decor.size()) - 1;
                }
            }
        }
        if (ed.decor_drag >= 0 && ed.decor_drag < int(p.decor.size()) && a.in.down) {
            p.decor[ed.decor_drag].x = std::clamp(m.x, -1.0f, 9.0f);
            p.decor[ed.decor_drag].y = std::clamp(m.y, -1.0f, 9.0f);
        }
        if (!a.in.down) ed.decor_drag = -1;
    }
    if (hover_sq >= 0) a.ui.hovering = true;
    Yolah shown;
    {
        uint64_t b = 0, w = 0, e = 0;
        for (int sq = 0; sq < 64; sq++) {
            uint64_t bit = uint64_t(1) << sq;
            if (p.cells[sq] == 'B') b |= bit;
            else if (p.cells[sq] == 'W') w |= bit;
            else if (p.cells[sq] == '#') e |= bit;
        }
        shown.set_state(b, w, e, 0, 0, p.to_move == BLACK ? 0 : 1);
    }
    ed.board.flipped = false;
    const AiSpec* opp = a.ai(p.opponent);
    // The sprite of each side: the puzzle's, or the first unit of the set of its board.
    const SpriteDef& sb = !p.sprite_black.empty() ? a.sprite(p.sprite_black)
                                                  : *a.board_units(a.theme_or(p.theme), BLACK, p.human == BLACK ? "human" : (opp ? opp->id : "dev")).front();
    const SpriteDef& sw = !p.sprite_white.empty() ? a.sprite(p.sprite_white)
                                                  : *a.board_units(a.theme_or(p.theme), WHITE, p.human == WHITE ? "human" : (opp ? opp->id : "dev")).front();
    BoardView::Look look;
    a.set_look(look, a.theme_or(p.theme), p.human == BLACK ? "human" : (opp ? opp->id : "dev"),
               p.human == WHITE ? "human" : (opp ? opp->id : "dev"), p.sprite_black, p.sprite_white);
    look.hole_style = p.hole_style;
    look.tiles = &p.tiles;
    look.decor = &p.decor;
    ed.board.draw(a.text, a.board_textures, a.theme_or(p.theme), shown, look, a.time, s);
    if (hover_sq >= 0 && ed.tool <= 4) draw::round_frame(ed.board.square(hover_sq).inset(2 * s), 6 * s, 3 * s, hex(0xffd36b, 220));
    if (ed.tool == 5)   // the decorations' boxes
        for (size_t i = 0; i < p.decor.size(); i++) {
            const float q = ed.board.sq(), w2 = p.decor[i].scale * q;
            Box db{ed.board.area.x + p.decor[i].x * q - w2 / 2, ed.board.area.y + p.decor[i].y * q - w2 / 2, w2, w2};
            draw::round_frame(db, 4 * s, int(i) == ed.decor_drag ? 2.5f * s : 1.2f * s, hex(0x5ee7ff, int(i) == ed.decor_drag ? 230 : 120));
        }
    std::string problem = puzzle_problem(p);
    std::string status = strf("Noirs %d/4 \xE2\x80\xA2 Blancs %d/4 \xE2\x80\xA2 %d cases libres  \xE2\x80\x94  ", p.count('B'), p.count('W'), p.count('.')) +
                         (problem.empty() ? std::string("pr\xC3\xAAt \xC3\xA0 jouer") : problem);
    a.text.draw(Font::Bold, 15 * s, status, {mid.center().x, ed.board.area.bottom() + frame + 4 * s}, problem.empty() ? pal::good : pal::bad, Align::Center);

    // ── right: settings and tests ──
    a.ui.panel(panel);
    float x = panel.x + 20 * s, w = panel.w - 40 * s;
    y = panel.y + 14 * s;
    a.text.draw(Font::Bold, 13 * s, editor_dirty(a) ? "Titre  \xE2\x80\xA2 modifi\xC3\xA9 (enregistr\xC3\xA9 en changeant de puzzle ou en testant)" : "Titre", {x, y}, pal::dim);
    a.ui.text_field("ed_name", {x, y + 18 * s, w, 36 * s}, p.name, "Titre du puzzle", 16);
    y += 60 * s;
    a.text.draw(Font::Bold, 13 * s, "Introduction (montr\xC3\xA9" "e avant le puzzle)", {x, y}, pal::dim);
    a.ui.text_field("ed_text", {x, y + 18 * s, w, 36 * s}, p.text, "Une phrase pour pr\xC3\xA9senter le d\xC3\xA9" "fi", 15);
    y += 62 * s;
    std::vector<std::string> ai_names, ai_ids;
    int ai_idx = 0;
    for (auto& ai : a.roster)
        if (ai.available) { if (ai.id == p.opponent) ai_idx = int(ai_ids.size()); ai_ids.push_back(ai.id); ai_names.push_back(ai.name); }
    if (!ai_ids.empty()) p.opponent = ai_ids[a.ui.cycler("ed_ai", {x, y, w, 36 * s}, "IA adverse", ai_names, ai_idx)];
    y += 42 * s;
    a.text.draw(Font::Regular, 16 * s, "Le joueur a", {x, y + 8 * s}, pal::dim);
    p.human = a.ui.segmented("ed_human", {x + w * 0.38f, y, w * 0.62f, 34 * s}, {"les Noirs", "les Blancs"}, p.human, 15);
    y += 40 * s;
    a.text.draw(Font::Regular, 16 * s, "Commence", {x, y + 8 * s}, pal::dim);
    p.to_move = a.ui.segmented("ed_tomove", {x + w * 0.38f, y, w * 0.62f, 34 * s}, {"Noirs", "Blancs"}, p.to_move, 15);
    y += 40 * s;
    std::vector<std::string> goals;
    for (int m = 1; m <= 12; m++) goals.push_back(m == 1 ? "Gagner" : strf("Gagner de %d+", m));
    p.margin = 1 + a.ui.cycler("ed_goal", {x, y, w, 34 * s}, "Objectif", goals, p.margin - 1);
    y += 40 * s;
    a.text.draw(Font::Regular, 16 * s, "Temps de l'IA", {x, y + 8 * s}, pal::dim);
    p.time_index = a.ui.segmented("ed_time", {x + w * 0.38f, y, w * 0.62f, 34 * s}, TIME_LABELS, p.time_index, 14);
    y += 46 * s;
    std::vector<std::string> th_names = {"Th\xC3\xA8me du joueur"};
    int th_idx = 0;
    for (size_t i = 0; i < a.themes.size(); i++) { th_names.push_back(a.themes[i].name); if (a.themes[i].id == p.theme) th_idx = int(i) + 1; }
    int ti = a.ui.cycler("ed_theme", {x, y, w, 34 * s}, "Th\xC3\xA8me", th_names, th_idx);
    if (ti != th_idx) {   // a new theme: its board, holes, scenery and pieces (the puzzle's own choices are reset)
        p.theme = ti == 0 ? "" : a.themes[ti - 1].id;
        p.hole_style = "pit";
        p.background.clear();
        p.sprite_black.clear();
        p.sprite_white.clear();
    }
    y += 40 * s;
    std::vector<std::string> hs_names, hs_ids;
    int hs_idx = 0;
    for (auto& [id, name] : HOLE_STYLES) { hs_ids.push_back(id); hs_names.push_back(name); }
    for (Art* ha : arts_of("hole")) { hs_ids.push_back("art:" + ha->id); hs_names.push_back(ha->name + " (dessin)"); }
    for (size_t i = 0; i < hs_ids.size(); i++) if (hs_ids[i] == p.hole_style) hs_idx = int(i);
    p.hole_style = hs_ids[a.ui.cycler("ed_holes", {x, y, w, 34 * s}, "Trous", hs_names, hs_idx)];
    y += 40 * s;
    std::vector<std::string> bgs = list_backgrounds(), bg_names = {"Selon le th\xC3\xA8me"};
    int bg_idx = 0;
    for (size_t i = 0; i < bgs.size(); i++) { bg_names.push_back(bgs[i]); if (bgs[i] == p.background) bg_idx = int(i) + 1; }
    const float lw = 96 * s;
    int bi = a.ui.cycler("ed_bg", {x, y, w - lw - 8 * s, 34 * s}, "Image de fond", bg_names, bg_idx);
    p.background = bi == 0 ? "" : bgs[bi - 1];
    if (a.ui.button("ed_bg_load", {x + w - lw, y, lw, 34 * s}, "Charger\xE2\x80\xA6", Style::Normal, true, 13)) a.choose_image("puzzle-background");
    y += 42 * s;
    for (int side : {BLACK, WHITE}) {
        Box r{x + side * (w / 2 + 4 * s), y, w / 2 - 4 * s, 46 * s};
        const std::string& id = side == BLACK ? p.sprite_black : p.sprite_white;
        const SpriteDef& cur = side == BLACK ? sb : sw;
        if (a.ui.button(side == BLACK ? "ed_sb" : "ed_sw", r, "", Style::Normal)) {
            Puzzle* pp = &p;
            bool black = side == BLACK;
            a.open_picker(strf("Sprite des pions %s du puzzle", black ? "noirs" : "blancs"), cur.id, p.human == side,
                          [pp, black](const std::string& sid) { (black ? pp->sprite_black : pp->sprite_white) = sid; });
        }
        draw_piece(cur, side, {r.x + 24 * s, r.center().y}, 16 * s, 0, a.time, 1, p.human == side);
        a.text.draw(Font::Bold, 13 * s, side == BLACK ? "Pions noirs" : "Pions blancs", {r.x + 46 * s, r.y + 7 * s}, pal::text);
        a.text.draw(Font::Regular, 12 * s, id.empty() ? "par d\xC3\xA9" "faut" : cur.name, {r.x + 46 * s, r.y + 25 * s}, pal::dim);
    }
    y += 54 * s;
    // The workshop (Atelier) is put aside for now.
    // if (a.ui.button("ed_studio", {x, y, w, 36 * s}, "Atelier : dessiner pions, d\xC3\xA9" "cors, cases, plateaux\xE2\x80\xA6", Style::Ghost, true, 14)) {
    //     if (editor_dirty(a)) editor_save(a);
    //     a.go(Scene::Studio);
    //     a.studio.return_to = Scene::Editor;
    //     return;
    // }
    // Tests.
    Box tb{x, panel.bottom() - 178 * s, w, 104 * s};
    draw::round_box(tb, 12 * s, hex(0x0f141c, 200));
    a.text.draw(Font::Bold, 14 * s, "Tester ce puzzle", {tb.x + 12 * s, tb.y + 8 * s}, pal::dim);
    std::vector<std::string> me_names, me_ids;
    int me_idx = 0;
    for (auto& ai : a.roster)
        if (ai.available) { if (ai.id == ed.tester) me_idx = int(me_ids.size()); me_ids.push_back(ai.id); me_names.push_back(ai.name); }
    if (ed.tester.empty() && !me_ids.empty()) me_idx = int(me_ids.size()) - 1;
    float half = (tb.w - 30 * s) / 2;
    if (a.ui.button("ed_play", {tb.x + 10 * s, tb.y + 32 * s, half, 62 * s}, "Moi contre l'IA", Style::Primary, problem.empty(), 16)) {
        editor_save(a);
        start_puzzle(a, p, "", -1, true);
        return;
    }
    Box ab{tb.x + 20 * s + half, tb.y + 32 * s, half, 62 * s};
    if (!me_ids.empty()) {
        if (a.ui.button("ed_ai_vs_ai", {ab.x, ab.y, ab.w, 34 * s}, "IA contre IA", Style::Normal, problem.empty(), 15)) {
            editor_save(a);
            start_puzzle(a, p, "", -1, true, me_ids[me_idx]);
            return;
        }
        // The AI playing the player's side.
        int k = a.ui.cycler("ed_tester", {ab.x, ab.y + 38 * s, ab.w, 24 * s}, "", me_names, me_idx, 12);
        ed.tester = me_ids[k];
    }
    // File actions.
    Box fb{x, panel.bottom() - 62 * s, w, 46 * s};
    float fw = (w - 2 * 8 * s) / 3;
    if (a.ui.button("ed_save", {fb.x, fb.y, fw, fb.h}, "Enregistrer", Style::Normal, true, 15)) { editor_save(a); a.notify("Enregistr\xC3\xA9 : assets/puzzles/" + p.id + ".json"); }
    if (a.ui.button("ed_dup", {fb.x + fw + 8 * s, fb.y, fw, fb.h}, "Dupliquer", Style::Normal, true, 15)) action = "dup";
    // An image given to the editor (dialog or dropped on the window): the puzzle's background.
    for (auto& [purpose, path] : a.in.files) {
        if (purpose == "dialog-error") { a.notify("Pas de s\xC3\xA9lecteur de fichiers ici : glissez l'image sur la fen\xC3\xAAtre."); continue; }
        if (purpose != "puzzle-background" && purpose != "drop") continue;
        std::string name = a.import_image(path, "backgrounds");
        if (!name.empty()) { p.background = name; a.notify("Image de fond : " + name + " (copi\xC3\xA9" "e dans assets/backgrounds)"); }
    }
    if (a.ui.button("ed_del", {fb.x + 2 * (fw + 8 * s), fb.y, fw, fb.h}, ed.confirm_delete ? "Confirmer ?" : "Supprimer", Style::Danger, !p.id.empty(), 15)) {
        if (ed.confirm_delete) action = "delete";
        else ed.confirm_delete = true;
    }
    a.ui.blocked = false;
    if (a.picker.open) sprite_picker(a);
    if (ed.choose_new) { editor_new_dialog(a); return; }

    // ── actions on the lists ──
    if (action == "open") {
        if (const Puzzle* pz = a.find_puzzle(action_puzzle)) { Puzzle copy = *pz; editor_open(a, copy, action_level); }
    } else if (action == "new") {
        ed.choose_new = true;
        ed.new_level = action_level;
        ed.new_scroll = 0;
    } else if (action == "unlink") {
        for (auto& l : a.levels)
            if (l.id == action_level) {
                l.puzzles.erase(std::remove(l.puzzles.begin(), l.puzzles.end(), action_puzzle), l.puzzles.end());
                save_level(l);
            }
        if (ed.level == action_level) ed.level.clear();
        a.notify("Retir\xC3\xA9 du niveau (le puzzle existe toujours).");
    } else if (action == "dellevel") {
        std::error_code ec;
        fs::remove(g_paths.assets / "levels" / (action_level + ".json"), ec);
        a.levels.erase(std::remove_if(a.levels.begin(), a.levels.end(), [&](const Level& l) { return l.id == action_level; }), a.levels.end());
        save_level_order(a.levels);
        if (ed.level == action_level) ed.level.clear();
        ed.level_selected.clear();
        ed.confirm_level.clear();
        a.reload_puzzles();
        a.notify("Niveau supprim\xC3\xA9 ; ses puzzles sont dans \xC2\xAB Hors niveau \xC2\xBB s'ils n'ont plus de niveau.");
    } else if (action == "add") {
        for (auto& l : a.levels)
            if (l.id == action_level) { l.puzzles.push_back(p.id); save_level(l); }
        ed.level = action_level;
    } else if (action == "move") {
        for (auto& l : a.levels)
            if (l.id == action_level) {
                auto it = std::find(l.puzzles.begin(), l.puzzles.end(), action_puzzle);
                int i = int(it - l.puzzles.begin()), j = i + move_dir;
                if (it != l.puzzles.end() && j >= 0 && j < int(l.puzzles.size())) { std::swap(l.puzzles[i], l.puzzles[j]); save_level(l); }
            }
    } else if (action == "lvmove") {
        for (size_t i = 0; i < a.levels.size(); i++)
            if (a.levels[i].id == action_level) {
                int j = int(i) + move_dir;
                if (j >= 0 && j < int(a.levels.size())) { std::swap(a.levels[i], a.levels[j]); save_level_order(a.levels); }
                break;
            }
    } else if (action == "newlevel") {
        Level l;
        save_level(l);
        a.levels.push_back(l);
        save_level_order(a.levels);   // a new level comes last
        a.reload_puzzles();
        ed.open_levels.insert(l.id);
        ed.level_selected = l.id;
        a.ui.focus = "lv_name";
        a.notify("Niveau cr\xC3\xA9\xC3\xA9 : donnez-lui un nom, puis ajoutez des puzzles.");
    } else if (action == "dup") {
        if (editor_dirty(a) && !p.id.empty()) editor_save(a);
        Puzzle copy = p;
        copy.id.clear();
        copy.name += " (copie)";
        ed.p = copy;
        editor_save(a);
        a.notify("Copie cr\xC3\xA9\xC3\xA9" "e : " + ed.p.name);
    } else if (action == "delete") {
        std::error_code ec;
        fs::remove(g_paths.assets / "puzzles" / (p.id + ".json"), ec);
        for (auto& l : a.levels) {
            auto it = std::remove(l.puzzles.begin(), l.puzzles.end(), p.id);
            if (it != l.puzzles.end()) { l.puzzles.erase(it, l.puzzles.end()); save_level(l); }
        }
        a.notify("Puzzle supprim\xC3\xA9 : " + p.name);
        a.reload_puzzles();
        ed.saved.clear();
        ed.p = Puzzle{};
        ed.p.id.clear();
        if (!a.puzzles.empty()) editor_open(a, a.puzzles.front(), "");
        ed.confirm_delete = false;
    }
}

// ─── @@SCENE_STUDIO ─────────────────────────────────────────────────────────
// The workshop. Tab "Dessins": a pixel-art editor (pencil, eraser, fill,
// colour picker, line, rectangle, mirror, undo) for the drawings of @@ART,
// with their animations (frames, speed, loop, onion skin, preview), and the
// import of any image (file dialog or drag and drop). Tab "Plateaux": the
// board themes (assets/themes): colours, pattern, and the drawings used for
// the squares, the holes and the frame, on a live board.

// Colours: HSV (h in degrees, s and v in [0, 1]) ↔ RGB.
Rgba hsv_to_rgb(float h, float s, float v, uint8_t a = 255) {
    h = std::fmod(std::fmod(h, 360.0f) + 360.0f, 360.0f);
    float c = v * s, x = c * (1 - std::abs(std::fmod(h / 60.0f, 2.0f) - 1)), m = v - c;
    float r = 0, g = 0, b = 0;
    if (h < 60) r = c, g = x;
    else if (h < 120) r = x, g = c;
    else if (h < 180) g = c, b = x;
    else if (h < 240) g = x, b = c;
    else if (h < 300) r = x, b = c;
    else r = c, b = x;
    return {uint8_t((r + m) * 255 + 0.5f), uint8_t((g + m) * 255 + 0.5f), uint8_t((b + m) * 255 + 0.5f), a};
}
void rgb_to_hsv(Rgba c, float& h, float& s, float& v) {
    float r = c.r / 255.0f, g = c.g / 255.0f, b = c.b / 255.0f;
    float mx = std::max({r, g, b}), mn = std::min({r, g, b}), d = mx - mn;
    v = mx;
    s = mx > 0 ? d / mx : 0;
    if (d <= 0) return;   // grey: keep the hue
    if (mx == r) h = 60 * std::fmod((g - b) / d, 6.0f);
    else if (mx == g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h < 0) h += 360;
}

// A quad with one colour per corner (SDL_RenderGeometry interpolates them).
void quad(Box b, Rgba tl, Rgba tr, Rgba br, Rgba bl) {
    SDL_Vertex v[4] = {{{b.x, b.y}, fcolor(tl), {0, 0}}, {{b.right(), b.y}, fcolor(tr), {0, 0}},
                       {{b.right(), b.bottom()}, fcolor(br), {0, 0}}, {{b.x, b.bottom()}, fcolor(bl), {0, 0}}};
    int idx[6] = {0, 1, 2, 0, 2, 3};
    SDL_RenderGeometry(draw::R, nullptr, v, 4, idx, 6);
}
// A checkerboard: what transparent pixels look like in image editors.
void checker(Box b, float cell) {
    draw::rect(b, hex(0x9aa1ab));
    int nx = int(std::ceil(b.w / cell)), ny = int(std::ceil(b.h / cell));
    for (int y = 0; y < ny; y++)
        for (int x = (y % 2); x < nx; x += 2)
            draw::rect({b.x + x * cell, b.y + y * cell, std::min(cell, b.right() - (b.x + x * cell)), std::min(cell, b.bottom() - (b.y + y * cell))}, hex(0xc9ced6));
}

// The colour picker: a saturation × value square, a hue bar and an opacity
// bar. `drag` remembers which part the mouse is dragging (it may leave it).
bool color_picker(App& a, Box b, Rgba& col, float& hue, float& sat, float& val, std::string& drag, const std::string& id) {
    const float s = a.s;
    const float bar = 18 * s, gap = 10 * s;
    Box sv{b.x, b.y, b.w, b.h - 2 * (bar + gap)};
    Box hb{b.x, sv.bottom() + gap, b.w, bar}, ab{b.x, hb.bottom() + gap, b.w, bar};
    Rgba pure = hsv_to_rgb(hue, 1, 1);
    quad(sv, hex(0xffffff), pure, pure, hex(0xffffff));
    quad(sv, hex(0x000000, 0), hex(0x000000, 0), hex(0x000000), hex(0x000000));
    for (int i = 0; i < 6; i++)
        quad({hb.x + hb.w * i / 6, hb.y, hb.w / 6 + 0.5f, hb.h}, hsv_to_rgb(i * 60.0f, 1, 1), hsv_to_rgb((i + 1) * 60.0f, 1, 1),
             hsv_to_rgb((i + 1) * 60.0f, 1, 1), hsv_to_rgb(i * 60.0f, 1, 1));
    checker(ab, ab.h / 2);
    Rgba opaque = col;
    opaque.a = 255;
    quad(ab, with_alpha(opaque, 0), opaque, opaque, with_alpha(opaque, 0));
    // Markers.
    V2 m{sv.x + sat * sv.w, sv.y + (1 - val) * sv.h};
    draw::ring(m, 7 * s, 2 * s, hex(0xffffff));
    draw::ring(m, 9 * s, 1 * s, hex(0x000000, 160));
    for (auto [bx, t] : {std::pair<Box, float>{hb, hue / 360}, {ab, col.a / 255.0f}}) {
        float x = bx.x + t * bx.w;
        draw::round_frame({x - 4 * s, bx.y - 3 * s, 8 * s, bx.h + 6 * s}, 3 * s, 2 * s, hex(0xffffff));
    }
    // Input.
    if (a.in.pressed && !a.ui.blocked) {
        if (sv.contains(a.in.mouse)) drag = id + "sv";
        else if (hb.contains(a.in.mouse)) drag = id + "h";
        else if (ab.contains(a.in.mouse)) drag = id + "a";
    }
    if (!a.in.down) {
        if (drag.rfind(id, 0) == 0) drag.clear();
        return false;
    }
    float fx = clamp01((a.in.mouse.x - b.x) / b.w);
    if (drag == id + "sv") { sat = fx; val = 1 - clamp01((a.in.mouse.y - sv.y) / sv.h); }
    else if (drag == id + "h") hue = fx * 359.9f;
    else if (drag == id + "a") col.a = uint8_t(fx * 255 + 0.5f);
    else return false;
    uint8_t al = col.a;
    col = hsv_to_rgb(hue, sat, val, al);
    return true;
}

// 32 colours to start with (dark to light, by hue), then transparent.
const std::vector<uint32_t> STUDIO_PALETTE = {
    0x0b0d12, 0x262b36, 0x4a5262, 0x7d8696, 0xb3bac6, 0xe6e9ee, 0xffffff, 0x5a2e1c,
    0x8a4b2a, 0xc47a3f, 0xf2b45a, 0xffe08a, 0x7a1f2b, 0xc2343f, 0xff6b6b, 0xff9ec0,
    0x6b2a7d, 0xa654c7, 0xd59bf0, 0x1f2d6b, 0x3456c2, 0x5ea2ff, 0x5ee7ff, 0x0f5a5a,
    0x1f9a8a, 0x4ade80, 0xb6f27a, 0x2b4a1c, 0x4f7f2a, 0x8fb83f, 0xf0e2c0, 0x9c7a54};

const std::vector<std::pair<const char*, const char*>> STUDIO_TOOLS = {
    {"Crayon", "P"}, {"Gomme", "E"}, {"Remplir", "F"}, {"Pipette", "I"}, {"Ligne", "L"}, {"Rectangle", "R"}};

// ── the drawing being edited ──
ArtAnim& studio_anim(App& a) {
    auto& st = a.studio;
    st.edit.complete();
    ArtAnim* an = st.edit.anim(st.anim);
    if (!an) { st.anim = art_kind(st.edit.kind).anims[0].first; an = st.edit.anim(st.anim); }
    return *an;
}
// The frame being edited (an animation without frames gets one).
std::vector<Pixel>& studio_frame(App& a) {
    auto& st = a.studio;
    ArtAnim& an = studio_anim(a);
    if (an.frames.empty()) an.frames.push_back(std::vector<Pixel>(size_t(st.edit.size) * st.edit.size, 0));
    st.frame = std::clamp(st.frame, 0, int(an.frames.size()) - 1);
    return an.frames[st.frame];
}
void studio_push_undo(App& a) {
    auto& st = a.studio;
    st.undo.push_back(st.edit.anims);
    if (st.undo.size() > 60) st.undo.erase(st.undo.begin());
    st.redo.clear();
    st.modified = true;
    st.edit.dirty = true;
}
void studio_changed(App& a) { a.studio.edit.dirty = true; a.studio.modified = true; }

void studio_save(App& a) {
    auto& st = a.studio;
    if (!st.loaded) return;
    if (!save_art(st.edit)) { a.notify("Impossible d'enregistrer le dessin."); return; }
    auto copy = std::make_unique<Art>();
    load_art(st.edit.id, *copy);
    if (Art* old = find_art(st.edit.id); old && old->tex) SDL_DestroyTexture(old->tex);
    g_arts[st.edit.id] = std::move(copy);
    register_art_sprites(a.sprites);
    st.modified = false;
}
void studio_open(App& a, const std::string& id) {
    auto& st = a.studio;
    if (st.modified) studio_save(a);
    Art* src = find_art(id);
    if (!src) return;
    if (st.edit.tex) SDL_DestroyTexture(st.edit.tex);
    st.edit = Art{};
    load_art(id, st.edit);
    st.loaded = true;
    st.modified = false;
    st.anim = art_kind(st.edit.kind).anims[0].first;
    st.frame = 0;
    st.undo.clear();
    st.redo.clear();
    st.confirm_delete = false;
}
void studio_new(App& a, const std::string& kind, int size) {
    auto& st = a.studio;
    if (st.modified) studio_save(a);
    if (st.edit.tex) SDL_DestroyTexture(st.edit.tex);
    st.edit = Art{};
    st.edit.kind = kind;
    st.edit.size = size;
    st.edit.name = std::string(art_kind(kind).label) + " " + std::to_string(arts_of(kind).size() + 1);
    st.edit.on_stone = kind == "piece";
    st.edit.complete();
    st.loaded = true;
    st.anim = art_kind(kind).anims[0].first;
    st.frame = 0;
    st.undo.clear();
    st.redo.clear();
    studio_save(a);   // it appears at once in the list
}

// Pixel operations on one frame (size × size).
void plot(std::vector<Pixel>& f, int n, int x, int y, Pixel c, bool mirror) {
    if (x >= 0 && y >= 0 && x < n && y < n) f[size_t(y) * n + x] = c;
    if (mirror && n - 1 - x != x && y >= 0 && y < n && n - 1 - x >= 0 && n - 1 - x < n) f[size_t(y) * n + (n - 1 - x)] = c;
}
void plot_line(std::vector<Pixel>& f, int n, int x0, int y0, int x1, int y1, Pixel c, bool mirror) {
    int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
    for (;;) {   // Bresenham
        plot(f, n, x0, y0, c, mirror);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}
void plot_rect(std::vector<Pixel>& f, int n, int x0, int y0, int x1, int y1, Pixel c, bool mirror) {
    plot_line(f, n, x0, y0, x1, y0, c, mirror);
    plot_line(f, n, x1, y0, x1, y1, c, mirror);
    plot_line(f, n, x1, y1, x0, y1, c, mirror);
    plot_line(f, n, x0, y1, x0, y0, c, mirror);
}
void flood_fill(std::vector<Pixel>& f, int n, int x, int y, Pixel c) {
    if (x < 0 || y < 0 || x >= n || y >= n) return;
    Pixel target = f[size_t(y) * n + x];
    if (target == c) return;
    std::vector<std::pair<int, int>> todo = {{x, y}};
    while (!todo.empty()) {
        auto [px, py] = todo.back();
        todo.pop_back();
        if (px < 0 || py < 0 || px >= n || py >= n || f[size_t(py) * n + px] != target) continue;
        f[size_t(py) * n + px] = c;
        todo.insert(todo.end(), {{px + 1, py}, {px - 1, py}, {px, py + 1}, {px, py - 1}});
    }
}
// An image (w × h RGBA) fitted into an n × n frame, centred, its proportions
// kept: each pixel of the frame averages the pixels it covers (weighted by
// their opacity), or takes the nearest one when the image is smaller.
std::vector<Pixel> fit_image(const unsigned char* px, int w, int h, int x0, int ww, int n) {
    std::vector<Pixel> f(size_t(n) * n, 0);
    float k = float(n) / std::max(ww, h);
    int dw = std::max(1, int(ww * k + 0.5f)), dh = std::max(1, int(h * k + 0.5f));
    int ox = (n - dw) / 2, oy = (n - dh) / 2;
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            int sx0 = int(x / k), sx1 = std::max(sx0 + 1, int((x + 1) / k)), sy0 = int(y / k), sy1 = std::max(sy0 + 1, int((y + 1) / k));
            double r = 0, g = 0, b = 0, al = 0;
            int cnt = 0;
            for (int sy = sy0; sy < std::min(sy1, h); sy++)
                for (int sx = sx0; sx < std::min(sx1, ww); sx++) {
                    const unsigned char* p = px + (size_t(sy) * w + x0 + sx) * 4;
                    double wa = p[3] / 255.0;
                    r += p[0] * wa; g += p[1] * wa; b += p[2] * wa; al += wa;
                    cnt++;
                }
            if (!cnt || al <= 0) continue;
            Rgba c{uint8_t(r / al + 0.5), uint8_t(g / al + 0.5), uint8_t(b / al + 0.5), uint8_t(al / cnt * 255 + 0.5)};
            f[size_t(oy + y) * n + ox + x] = pack(c);
        }
    return f;
}
// Imports an image into the current animation: a strip of square frames
// (width = k × height) replaces its frames, any other image replaces the
// current frame.
void studio_import(App& a, const std::string& path) {
    auto& st = a.studio;
    int w, h, n;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!px) { a.notify("Image illisible : " + fs::path(path).filename().string()); return; }
    studio_push_undo(a);
    ArtAnim& an = studio_anim(a);
    if (w > h && w % h == 0 && w / h <= 64) {
        an.frames.clear();
        for (int k = 0; k < w / h; k++) an.frames.push_back(fit_image(px, w, h, k * h, h, st.edit.size));
        st.frame = 0;
        a.notify(strf("%d images import\xC3\xA9" "es dans l'animation", w / h));
    } else {
        studio_frame(a) = fit_image(px, w, h, 0, w, st.edit.size);
        a.notify("Image import\xC3\xA9" "e dans l'image courante");
    }
    stbi_image_free(px);
    studio_changed(a);
}
// Draws frame k of animation index ai of an art in a box.
void draw_art_frame_k(Art& art, int ai, int k, Box b, float alpha = 1) {
    if (ai < 0 || ai >= int(art.anims.size()) || k < 0 || k >= int(art.anims[ai].frames.size())) return;
    art.update_texture();
    SDL_FRect src{float(k * art.size), float(ai * art.size), float(art.size), float(art.size)}, dst = b.frect();
    SDL_SetTextureAlphaModFloat(art.tex, alpha);
    SDL_RenderTexture(draw::R, art.tex, &src, &dst);
}
// A small preview of a drawing, as it is used: a piece on a square, a
// square, a hole, a frame around squares, a decoration.
void art_preview(App& a, Art& art, Box b, const std::string& anim, double t) {
    const Theme& th = a.theme();
    draw::rect(b, th.light);
    if (art.kind == "piece") {
        SpriteDef sp;
        sp.art = art.id;
        float r = std::min(b.h * 0.4f, b.w * 0.18f);
        // The art may not be saved yet: draw it from the copy.
        V2 c1{b.x + b.w * 0.3f, b.center().y}, c2{b.x + b.w * 0.7f, b.center().y};
        for (auto [c, side] : {std::pair<V2, int>{c1, BLACK}, {c2, WHITE}}) {
            draw::circle(c + V2{r * 0.06f, r * 0.12f}, r, hex(0x000000, 90), hex(0x000000, 0), r * 0.3f);
            if (art.on_stone) {
                V2 light = c + V2{-r * 0.35f, -r * 0.4f};
                if (side == BLACK) draw::circle_lit(c, r, light, hex(0x5b6170), hex(0x0a0c11));
                else draw::circle_lit(c, r, light, hex(0xffffff), hex(0xb9bec8));
                draw_art(art, anim, t, {c.x - r * 0.78f, c.y - r * 0.78f, r * 1.56f, r * 1.56f});
            } else {
                draw_art(art, anim, t, {c.x - r * 1.1f, c.y - r * 1.1f, r * 2.2f, r * 2.2f});
            }
        }
    } else if (art.kind == "tile" || art.kind == "hole") {
        float q = b.h / 3;
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 6 && b.x + x * q < b.right(); x++) {
                Box c{b.x + x * q, b.y + y * q, q, q};
                draw::rect(c, (x + y) % 2 ? th.dark : th.light);
                if (art.kind == "tile" || (x + y) % 3 == 1) draw_art(art, anim, t + x * 0.3, c);
            }
    } else if (art.kind == "frame") {
        draw::rect(b, hex(0x0f141c));
        Box inner = b.inset(b.h * 0.18f);
        draw_art_frame(art, t, inner.inset(-b.h * 0.12f), b.h * 0.12f);
        float q = inner.h / 4;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x * q < inner.w - 0.5f; x++)
                draw::rect({inner.x + x * q, inner.y + y * q, std::min(q, inner.right() - inner.x - x * q), q}, (x + y) % 2 ? th.dark : th.light);
    } else {
        float q = b.h / 3;
        for (int y = 0; y < 3; y++)
            for (int x = 0; x * q < b.w; x++) draw::rect({b.x + x * q, b.y + y * q, std::min(q, b.right() - b.x - x * q), q}, (x + y) % 2 ? th.dark : th.light);
        draw_art(art, anim, t, {b.center().x - b.h * 0.45f, b.y + b.h * 0.05f, b.h * 0.9f, b.h * 0.9f});
    }
}

// ── tab "Dessins" ──
void studio_drawings(App& a) {
    const float s = a.s;
    auto& st = a.studio;
    if (!st.loaded) {
        auto all = arts_of("piece", false);
        if (all.empty())
            for (auto& k : ART_KINDS) { auto v = arts_of(k.id, false); if (!v.empty()) { all = v; break; } }
        if (!all.empty()) studio_open(a, all.front()->id);
        else studio_new(a, "piece", 32);
    }
    Art& art = st.edit;
    const int n = art.size;
    const bool modal = st.new_open;
    a.ui.blocked = modal;
    const bool typing = !a.ui.focus.empty();
    const SDL_Keymod mods = SDL_GetModState();
    const bool ctrl = mods & SDL_KMOD_CTRL;

    const float margin = 26 * s, top = 150 * s;
    Box lib{margin, top, 270 * s, a.h - top - margin};
    Box tools{lib.right() + 16 * s, top, 118 * s, a.h - top - margin};
    Box right{a.w - margin - 380 * s, top, 380 * s, a.h - top - margin};
    Box mid{tools.right() + 16 * s, top, right.x - tools.right() - 32 * s, a.h - top - margin};

    // ── the library ──
    a.ui.panel(lib);
    Box list{lib.x + 8 * s, lib.y + 12 * s, lib.w - 16 * s, lib.h - 140 * s};
    if (a.ui.hover(list)) st.list_scroll = std::max(0.0f, st.list_scroll - a.in.wheel * 40 * s);
    float y = list.y - st.list_scroll;
    std::string open_id;
    draw::clip(&list);
    for (auto& k : ART_KINDS) {
        auto v = arts_of(k.id, false);
        a.text.draw(Font::Bold, 13 * s, strf("%s (%zu)", k.label, v.size()), {list.x + 6 * s, y + 6 * s}, a.theme().accent);
        y += 28 * s;
        for (Art* x : v) {
            Box r{list.x, y, list.w, 40 * s};
            bool cur = x->id == art.id;
            bool h = a.ui.hover(r) && list.contains(a.in.mouse);
            if (h) a.ui.hovering = true;
            draw::round_box(r, 8 * s, with_alpha(cur ? a.theme().accent : hex(0xffffff), cur ? 0.28f : (h ? 0.08f : 0.0f)));
            Box th{r.x + 6 * s, r.y + 4 * s, 32 * s, 32 * s};
            checker(th, 8 * s);
            Art& shown = cur ? art : *x;
            draw_art(shown, "idle", a.time, th);
            a.text.draw(cur ? Font::Bold : Font::Regular, 14 * s, cur ? art.name : x->name, {th.right() + 10 * s, r.y + 11 * s}, pal::text);
            if (h && a.in.released && !cur) open_id = x->id;
            y += 42 * s;
        }
        y += 6 * s;
    }
    draw::clip(nullptr);
    st.list_scroll = std::clamp(st.list_scroll, 0.0f, std::max(0.0f, y + st.list_scroll - list.y - list.h));
    float bw = (lib.w - 30 * s) / 2;
    if (a.ui.button("st_new", {lib.x + 10 * s, lib.bottom() - 120 * s, lib.w - 20 * s, 42 * s}, "+ Nouveau dessin", Style::Primary, true, 15)) {
        st.new_open = true;
        for (size_t i = 0; i < ART_KINDS.size(); i++) if (art.kind == ART_KINDS[i].id) st.new_kind = int(i);
    }
    if (a.ui.button("st_dup", {lib.x + 10 * s, lib.bottom() - 70 * s, bw, 40 * s}, "Dupliquer", Style::Normal, true, 14)) {
        studio_save(a);
        Art copy = art;
        copy.tex = nullptr;
        copy.id.clear();
        copy.name += " (copie)";
        if (st.edit.tex) SDL_DestroyTexture(st.edit.tex);
        st.edit = copy;
        st.edit.dirty = true;
        studio_save(a);
        a.notify("Copie : " + st.edit.name);
    }
    if (a.ui.button("st_del", {lib.x + 20 * s + bw, lib.bottom() - 70 * s, bw, 40 * s}, st.confirm_delete ? "Confirmer ?" : "Supprimer", Style::Danger, true, 14)) {
        if (!st.confirm_delete) st.confirm_delete = true;
        else {
            std::error_code ec;
            fs::remove(art_dir() / (art.id + ".json"), ec);
            fs::remove(art_dir() / (art.id + ".png"), ec);
            if (Art* old = find_art(art.id); old && old->tex) SDL_DestroyTexture(old->tex);
            g_arts.erase(art.id);
            register_art_sprites(a.sprites);
            a.notify("Dessin supprim\xC3\xA9 : " + art.name);
            st.loaded = false;
            st.modified = false;
            st.confirm_delete = false;
            return;
        }
    }

    // ── tools ──
    a.ui.panel(tools);
    float ty = tools.y + 10 * s;
    for (size_t i = 0; i < STUDIO_TOOLS.size(); i++) {
        Box r{tools.x + 8 * s, ty, tools.w - 16 * s, 38 * s};
        if (a.ui.button("tool_" + std::to_string(i), r, strf("%s (%s)", STUDIO_TOOLS[i].first, STUDIO_TOOLS[i].second), int(i) == st.tool ? Style::Primary : Style::Normal, true, 13)) st.tool = int(i);
        ty += 42 * s;
    }
    ty += 8 * s;
    auto small_toggle = [&](const char* id, const char* label, bool& v) {
        Box r{tools.x + 8 * s, ty, tools.w - 16 * s, 30 * s};
        if (a.ui.button(id, r, label, v ? Style::Primary : Style::Ghost, true, 12)) v = !v;
        ty += 34 * s;
    };
    small_toggle("st_mirror", "Miroir", st.mirror);
    small_toggle("st_grid", "Grille", st.grid);
    small_toggle("st_onion", "Pelure", st.onion);
    ty += 8 * s;
    auto action_button = [&](const char* id, const char* label) {
        Box r{tools.x + 8 * s, ty, tools.w - 16 * s, 30 * s};
        ty += 34 * s;
        return a.ui.button(id, r, label, Style::Normal, true, 12);
    };
    if (action_button("st_undo", "Annuler") || (!typing && ctrl && a.in.key(SDLK_Z) && !(mods & SDL_KMOD_SHIFT))) {
        if (!st.undo.empty()) { st.redo.push_back(art.anims); art.anims = st.undo.back(); st.undo.pop_back(); studio_changed(a); }
    }
    if (action_button("st_redo", "R\xC3\xA9tablir") || (!typing && ctrl && (a.in.key(SDLK_Y) || (a.in.key(SDLK_Z) && (mods & SDL_KMOD_SHIFT))))) {
        if (!st.redo.empty()) { st.undo.push_back(art.anims); art.anims = st.redo.back(); st.redo.pop_back(); studio_changed(a); }
    }
    std::vector<Pixel>& fr = studio_frame(a);
    if (action_button("st_fliph", "Retourner H")) {
        studio_push_undo(a);
        for (int yy = 0; yy < n; yy++) std::reverse(fr.begin() + size_t(yy) * n, fr.begin() + size_t(yy + 1) * n);
    }
    if (action_button("st_flipv", "Retourner V")) {
        studio_push_undo(a);
        for (int yy = 0; yy < n / 2; yy++) std::swap_ranges(fr.begin() + size_t(yy) * n, fr.begin() + size_t(yy + 1) * n, fr.begin() + size_t(n - 1 - yy) * n);
    }
    if (action_button("st_rot", "Tourner")) {
        studio_push_undo(a);
        std::vector<Pixel> r(fr.size());
        for (int yy = 0; yy < n; yy++)
            for (int xx = 0; xx < n; xx++) r[size_t(xx) * n + (n - 1 - yy)] = fr[size_t(yy) * n + xx];
        fr = r;
    }
    if (action_button("st_clear", "Effacer")) { studio_push_undo(a); std::fill(fr.begin(), fr.end(), 0); }
    if (action_button("st_import", "Importer\xE2\x80\xA6")) a.choose_image("studio-import");
    for (auto& [purpose, path] : a.in.files) {
        if (purpose == "dialog-error") a.notify("Pas de s\xC3\xA9lecteur de fichiers ici : glissez l'image sur la fen\xC3\xAAtre.");
        else if (purpose == "studio-import" || purpose == "drop") studio_import(a, path);
    }
    if (!typing && !ctrl)
        for (size_t i = 0; i < STUDIO_TOOLS.size(); i++)
            if (a.in.key(SDL_Keycode(std::tolower(STUDIO_TOOLS[i].second[0])))) st.tool = int(i);

    // ── canvas ── (the frame again: an import may have replaced the frames)
    std::vector<Pixel>& frc = studio_frame(a);
    ArtAnim& an = studio_anim(a);
    const int ai = int(&an - art.anims.data());
    const float strip_h = 150 * s;
    float C = std::floor(std::min(mid.w, mid.h - strip_h - 16 * s) / n) * n;
    float cell = C / n;
    Box canvas{std::floor(mid.x + (mid.w - C) / 2), mid.y, C, C};
    draw::shadow({canvas.x, canvas.y + 8 * s, canvas.w, canvas.h}, 6 * s, 24 * s, hex(0x000000, 120));
    checker(canvas, std::max(cell, 8 * s));
    if (st.onion && st.frame > 0) draw_art_frame_k(art, ai, st.frame - 1, canvas, 0.3f);
    draw_art_frame_k(art, ai, st.frame, canvas);
    if (st.grid && cell >= 6 * s) {
        for (int i = 1; i < n; i++) {
            Rgba gc = hex(0x000000, i % 8 == 0 ? 90 : 40);
            draw::rect({canvas.x + i * cell, canvas.y, 1, canvas.h}, gc);
            draw::rect({canvas.x, canvas.y + i * cell, canvas.w, 1}, gc);
        }
    }
    if (st.mirror) draw::rect({canvas.center().x - 1, canvas.y, 2, canvas.h}, hex(0x5ee7ff, 160));
    if (art.kind == "frame") {   // the 3 × 3 cut of a frame
        for (int i = 1; i <= 2; i++) {
            draw::rect({canvas.x + canvas.w * i / 3, canvas.y, 1, canvas.h}, hex(0xff9ec0, 160));
            draw::rect({canvas.x, canvas.y + canvas.h * i / 3, canvas.w, 1}, hex(0xff9ec0, 160));
        }
    }
    int px = int(std::floor((a.in.mouse.x - canvas.x) / cell)), py = int(std::floor((a.in.mouse.y - canvas.y) / cell));
    bool on = !modal && canvas.contains(a.in.mouse);
    if (on) {
        a.ui.hovering = true;
        draw::rect({canvas.x + px * cell, canvas.y + py * cell, cell, cell}, with_alpha(st.tool == 1 ? hex(0xffffff) : st.color, 0.5f));
        draw::round_frame({canvas.x + px * cell, canvas.y + py * cell, cell, cell}, 1, std::max(1.0f, s), hex(0xffffff, 200));
    }
    const bool erase = st.tool == 1 || a.in.rdown;
    const Pixel paint = erase ? 0 : pack(st.color);
    if (on && (a.in.pressed || a.in.rpressed)) {
        st.stroke = true;
        if (st.tool == 3) {   // colour picker
            Pixel c = frc[size_t(py) * n + px];
            if (c >> 24) { st.color = unpack(c); rgb_to_hsv(st.color, st.hue, st.sat, st.val); }
            st.stroke = false;
        } else {
            studio_push_undo(a);
            st.start_x = st.last_x = px;
            st.start_y = st.last_y = py;
            st.before = frc;
            if (st.tool == 2 && !a.in.rpressed) { flood_fill(frc, n, px, py, paint); st.stroke = false; }
            else if (st.tool <= 1 || a.in.rpressed) plot(frc, n, px, py, paint, st.mirror);
        }
    }
    if (st.stroke && (a.in.down || a.in.rdown)) {
        int cx = std::clamp(px, 0, n - 1), cy = std::clamp(py, 0, n - 1);
        if (st.tool <= 1 || st.tool == 2 || st.tool == 3 || a.in.rdown) {
            if (cx != st.last_x || cy != st.last_y) plot_line(frc, n, st.last_x, st.last_y, cx, cy, paint, st.mirror);
        } else {
            frc = st.before;
            if (st.tool == 4) plot_line(frc, n, st.start_x, st.start_y, cx, cy, paint, st.mirror);
            else plot_rect(frc, n, st.start_x, st.start_y, cx, cy, paint, st.mirror);
        }
        st.last_x = cx;
        st.last_y = cy;
        studio_changed(a);
    }
    if (!a.in.down && !a.in.rdown) st.stroke = false;

    // ── animation strip ──
    Box strip{mid.x, canvas.bottom() + 16 * s, mid.w, mid.bottom() - canvas.bottom() - 16 * s};
    a.ui.panel(strip, 12);
    const ArtKind& kind = art_kind(art.kind);
    std::vector<std::string> anim_labels;
    int anim_sel = 0;
    for (size_t i = 0; i < kind.anims.size(); i++) {
        anim_labels.push_back(kind.anims[i].second);
        if (st.anim == kind.anims[i].first) anim_sel = int(i);
    }
    float aw = std::min(strip.w - 350 * s, 120 * s * anim_labels.size());
    int ns = a.ui.segmented("st_anim", {strip.x + 10 * s, strip.y + 10 * s, aw, 32 * s}, anim_labels, anim_sel, 13);
    if (ns != anim_sel) { st.anim = kind.anims[ns].first; st.frame = 0; }
    // Speed and loop.
    std::vector<std::string> fps_names;
    const std::vector<float> FPS = {1, 2, 4, 6, 8, 10, 12, 15, 20, 24, 30};
    int fi = 0;
    for (size_t i = 0; i < FPS.size(); i++) { fps_names.push_back(strf("%g img/s", FPS[i])); if (std::abs(FPS[i] - an.fps) < 0.01f) fi = int(i); }
    int nf = a.ui.cycler("st_fps", {strip.right() - 330 * s, strip.y + 10 * s, 222 * s, 32 * s}, "Vitesse", fps_names, fi, 13);
    if (nf != fi) { an.fps = FPS[nf]; studio_changed(a); }
    if (a.ui.button("st_loop", {strip.right() - 100 * s, strip.y + 10 * s, 90 * s, 32 * s}, an.loop ? "En boucle" : "Une fois", an.loop ? Style::Primary : Style::Normal, true, 12)) { an.loop = !an.loop; studio_changed(a); }
    // Frames.
    float fx = strip.x + 10 * s, fy = strip.y + 52 * s, fs = strip.h - 62 * s;
    for (int k = 0; k < int(an.frames.size()); k++) {
        Box r{fx + k * (fs + 6 * s), fy, fs, fs};
        if (r.right() > strip.right() - 250 * s) break;
        bool cur = k == st.frame, h = a.ui.hover(r);
        if (h) a.ui.hovering = true;
        checker(r, 6 * s);
        draw_art_frame_k(art, ai, k, r);
        draw::round_frame(r, 4 * s, cur ? 3 * s : 1, cur ? a.theme().accent : with_alpha(hex(0xffffff), h ? 0.5f : 0.15f));
        a.text.draw(Font::Bold, 11 * s, std::to_string(k + 1), {r.x + 4 * s, r.y + 2 * s}, hex(0x10141c));
        if (h && a.in.released) st.frame = k;
    }
    Box fb{strip.right() - 240 * s, fy, 230 * s, fs};
    float bw3 = (fb.w - 12 * s) / 3, bh2 = (fb.h - 6 * s) / 2;
    if (a.ui.button("st_fadd", {fb.x, fb.y, bw3, bh2}, "+ Vide", Style::Normal, true, 12)) {
        studio_push_undo(a);
        an.frames.insert(an.frames.begin() + st.frame + 1, std::vector<Pixel>(size_t(n) * n, 0));
        st.frame++;
    }
    if (a.ui.button("st_fdup", {fb.x + bw3 + 6 * s, fb.y, bw3, bh2}, "+ Copie", Style::Normal, true, 12)) {
        studio_push_undo(a);
        an.frames.insert(an.frames.begin() + st.frame + 1, an.frames[st.frame]);
        st.frame++;
    }
    if (a.ui.button("st_fdel", {fb.x + 2 * (bw3 + 6 * s), fb.y, bw3, bh2}, "Retirer", Style::Danger, an.frames.size() > 1 || ai > 0, 12)) {
        studio_push_undo(a);
        an.frames.erase(an.frames.begin() + st.frame);
        st.frame = std::max(0, st.frame - 1);
    }
    if (a.ui.button("st_fl", {fb.x, fb.y + bh2 + 6 * s, bw3, bh2}, "\xE2\x86\x90", Style::Ghost, st.frame > 0, 14)) {
        studio_push_undo(a);
        std::swap(an.frames[st.frame], an.frames[st.frame - 1]);
        st.frame--;
    }
    if (a.ui.button("st_fr", {fb.x + bw3 + 6 * s, fb.y + bh2 + 6 * s, bw3, bh2}, "\xE2\x86\x92", Style::Ghost, st.frame + 1 < int(an.frames.size()), 14)) {
        studio_push_undo(a);
        std::swap(an.frames[st.frame], an.frames[st.frame + 1]);
        st.frame++;
    }
    if (a.ui.button("st_play", {fb.x + 2 * (bw3 + 6 * s), fb.y + bh2 + 6 * s, bw3, bh2}, st.playing ? "Pause" : "Lire", Style::Ghost, true, 12)) st.playing = !st.playing;
    if (!typing && !modal) {
        if (a.in.key(SDLK_LEFT) && !an.frames.empty()) st.frame = (st.frame + int(an.frames.size()) - 1) % int(an.frames.size());
        if (a.in.key(SDLK_RIGHT) && !an.frames.empty()) st.frame = (st.frame + 1) % int(an.frames.size());
        if (a.in.key(SDLK_SPACE)) st.playing = !st.playing;
    }

    // ── right panel ──
    a.ui.panel(right);
    float x = right.x + 18 * s, w = right.w - 36 * s;
    y = right.y + 12 * s;
    a.text.draw(Font::Bold, 12 * s, st.modified ? "Nom  \xE2\x80\xA2 modifi\xC3\xA9 (enregistr\xC3\xA9 en changeant de dessin)" : "Nom", {x, y}, pal::dim);
    if (a.ui.text_field("st_name", {x, y + 16 * s, w, 34 * s}, art.name, "Nom du dessin", 15)) st.modified = true;
    y += 58 * s;
    std::vector<std::string> kinds;
    int ki = 0;
    for (size_t i = 0; i < ART_KINDS.size(); i++) { kinds.push_back(ART_KINDS[i].label); if (art.kind == ART_KINDS[i].id) ki = int(i); }
    int nk = a.ui.cycler("st_kind", {x, y, w, 30 * s}, "Type", kinds, ki, 13);
    if (nk != ki) {   // new animations may be added: the references above are stale, next frame
        studio_push_undo(a);
        art.kind = ART_KINDS[nk].id;
        art.complete();
        st.anim = art_kind(art.kind).anims[0].first;
        st.frame = 0;
        a.ui.blocked = false;
        return;
    }
    const std::vector<int> SIZES = {16, 24, 32, 48, 64};
    std::vector<std::string> size_names;
    int si = 0;
    for (size_t i = 0; i < SIZES.size(); i++) { size_names.push_back(strf("%d px", SIZES[i])); if (SIZES[i] == n) si = int(i); }
    int nsz = a.ui.cycler("st_size", {x, y + 36 * s, w, 30 * s}, "Taille", size_names, si, 13);
    if (nsz != si) {   // every frame resampled (nearest pixel)
        studio_push_undo(a);
        int m = SIZES[nsz];
        for (auto& an2 : art.anims)
            for (auto& f : an2.frames) {
                std::vector<Pixel> g(size_t(m) * m);
                for (int yy = 0; yy < m; yy++)
                    for (int xx = 0; xx < m; xx++) g[size_t(yy) * m + xx] = f[size_t(yy * n / m) * n + xx * n / m];
                f = g;
            }
        art.size = m;
        studio_changed(a);
        a.ui.blocked = false;
        return;   // the canvas uses the old size this frame
    }
    y += 74 * s;
    if (art.kind == "piece") {
        if (a.ui.toggle("st_stone", {x, y, w, 28 * s}, "Sur la pierre (noire ou blanche)", art.on_stone, 14)) studio_changed(a);
        y += 34 * s;
    } else {
        const char* help = art.kind == "frame" ? "Le cadre est coup\xC3\xA9 en 3 \xC3\x97 3 : coins, bords, centre."
                         : art.kind == "hole" ? "\xC2\xAB Apparition \xC2\xBB : jou\xC3\xA9" "e une fois quand le trou se creuse."
                         : art.kind == "tile" ? "Une case du plateau (claire, fonc\xC3\xA9" "e ou sp\xC3\xA9" "ciale)."
                         : "Pos\xC3\xA9 sur le plateau ou autour, dans l'\xC3\xA9" "diteur de puzzles.";
        a.text.draw(Font::Regular, 12 * s, help, {x, y + 6 * s}, pal::faint);
        y += 34 * s;
    }
    // Colour.
    Box cp{x, y, w - 70 * s, 168 * s};
    color_picker(a, cp, st.color, st.hue, st.sat, st.val, st.drag, "st");
    Box cur{cp.right() + 12 * s, y, 58 * s, 58 * s};
    checker(cur, 8 * s);
    draw::rect(cur, st.color);
    a.text.draw(Font::Regular, 11 * s, color_string(st.color), {cur.center().x, cur.bottom() + 4 * s}, pal::dim, Align::Center);
    y += 176 * s;
    const int PC = 11;
    float sw = (w - (PC - 1) * 4 * s) / PC;
    for (size_t i = 0; i <= STUDIO_PALETTE.size(); i++) {
        Box c{x + (i % PC) * (sw + 4 * s), y + (i / PC) * (sw + 4 * s), sw, sw};
        bool transparent = i == STUDIO_PALETTE.size();
        if (transparent) { checker(c, sw / 4); }
        else draw::round_box(c, 4 * s, hex(STUDIO_PALETTE[i]));
        bool h = a.ui.hover(c);
        if (h) { a.ui.hovering = true; draw::round_frame(c, 4 * s, 2 * s, hex(0xffffff)); }
        if (h && a.in.released) {
            st.color = transparent ? Rgba{0, 0, 0, 0} : hex(STUDIO_PALETTE[i]);
            rgb_to_hsv(st.color, st.hue, st.sat, st.val);
            if (transparent) st.tool = 1;
        }
    }
    y += float((STUDIO_PALETTE.size() + PC) / PC) * (sw + 4 * s) + 6 * s;
    // Preview.
    Box pv{x, y, w, std::max(60 * s, right.bottom() - y - 70 * s)};
    double t = st.playing ? a.time : st.frame / std::max(0.1f, an.fps) + 0.001;
    if (!an.loop && st.playing) t = std::fmod(a.time, an.duration() + 0.8);
    art_preview(a, art, pv, st.anim, t);
    draw::round_frame(pv, 6 * s, 1, pal::panel_border);
    if (a.ui.button("st_save", {x, right.bottom() - 58 * s, w, 44 * s}, "Enregistrer", Style::Primary, true, 16)) {
        studio_save(a);
        a.notify("Enregistr\xC3\xA9 : assets/art/" + art.id + ".png");
    }
    if (!typing && ctrl && a.in.key(SDLK_S)) { studio_save(a); a.notify("Enregistr\xC3\xA9"); }
    a.ui.blocked = false;

    if (!open_id.empty()) studio_open(a, open_id);

    // ── the "new drawing" dialog ──
    if (st.new_open) {
        draw::rect({0, 0, float(a.w), float(a.h)}, hex(0x000000, 160));
        Box d{a.w * 0.5f - 330 * s, a.h * 0.5f - 160 * s, 660 * s, 320 * s};
        a.ui.panel(d, 20, hex(0x1a202b));
        a.text.draw(Font::Black, 24 * s, "Nouveau dessin", {d.x + 26 * s, d.y + 22 * s}, pal::text);
        std::vector<std::string> kl;
        for (auto& k : ART_KINDS) kl.push_back(k.label);
        a.text.draw(Font::Regular, 15 * s, "Type", {d.x + 26 * s, d.y + 82 * s}, pal::dim);
        st.new_kind = a.ui.segmented("nw_kind", {d.x + 120 * s, d.y + 74 * s, d.w - 146 * s, 36 * s}, kl, st.new_kind, 14);
        a.text.draw(Font::Regular, 15 * s, "Taille", {d.x + 26 * s, d.y + 138 * s}, pal::dim);
        st.new_size = a.ui.segmented("nw_size", {d.x + 120 * s, d.y + 130 * s, d.w - 146 * s, 36 * s}, size_names, st.new_size, 14);
        a.text.draw(Font::Regular, 13 * s, ART_KINDS[st.new_kind].id == std::string("frame") ? "Un cadre : une taille multiple de 3 (24 ou 48) se coupe proprement en 3 \xC3\x97 3."
                                                                                            : "16 ou 32 pixels : un vrai style pixel art ; 64 : plus de d\xC3\xA9tails.",
                    {d.x + 26 * s, d.y + 186 * s}, pal::faint);
        if (a.ui.button("nw_ok", {d.right() - 200 * s, d.bottom() - 70 * s, 174 * s, 48 * s}, "Cr\xC3\xA9" "er", Style::Primary) || a.in.key(SDLK_RETURN)) {
            st.new_open = false;
            studio_new(a, ART_KINDS[st.new_kind].id, SIZES[st.new_size]);
        }
        if (a.ui.button("nw_cancel", {d.right() - 390 * s, d.bottom() - 70 * s, 174 * s, 48 * s}, "Annuler", Style::Ghost) || a.in.key(SDLK_ESCAPE)) {
            st.new_open = false;
            a.in.eat(SDLK_ESCAPE);
        }
    }
}

// ── tab "Plateaux" ──
void studio_theme_save(App& a) {
    auto& st = a.studio;
    if (!st.theme_loaded) return;
    Theme& t = st.theme;
    if (t.file.empty()) {
        std::string base = slug(t.name), id = base;
        for (int k = 2; fs::exists(g_paths.assets / "themes" / (id + ".json")); k++) id = base + "_" + std::to_string(k);
        t.id = id;
        t.file = id + ".json";
    }
    if (!write_file(g_paths.assets / "themes" / t.file, theme_to_json(t).dump(2))) { a.notify("Impossible d'enregistrer le th\xC3\xA8me."); return; }
    a.themes = load_themes(g_paths.assets / "themes");
    a.board_textures.clear();
    st.theme_modified = false;
}
void studio_theme_open(App& a, const Theme& t) {
    auto& st = a.studio;
    if (st.theme_modified) studio_theme_save(a);
    st.theme = t;
    st.theme_loaded = true;
    st.theme_modified = false;
    st.confirm_theme_delete = false;
    st.tex_changed = a.time;
}

void studio_boards(App& a) {
    const float s = a.s;
    auto& st = a.studio;
    if (!st.theme_loaded) studio_theme_open(a, a.theme());
    Theme& t = st.theme;
    const float margin = 26 * s, top = 150 * s;
    Box lib{margin, top, 270 * s, a.h - top - margin};
    Box right{a.w - margin - 470 * s, top, 470 * s, a.h - top - margin};
    Box mid{lib.right() + 26 * s, top, right.x - lib.right() - 52 * s, a.h - top - margin};

    // ── the themes ──
    a.ui.panel(lib);
    float y = lib.y + 12 * s;
    const Theme* open = nullptr;
    for (auto& th : a.themes) {
        Box r{lib.x + 8 * s, y, lib.w - 16 * s, 44 * s};
        bool cur = th.file == t.file;
        bool h = a.ui.hover(r);
        if (h) a.ui.hovering = true;
        draw::round_box(r, 8 * s, with_alpha(cur ? a.theme().accent : hex(0xffffff), cur ? 0.28f : (h ? 0.08f : 0.0f)));
        Box sw{r.x + 8 * s, r.y + 8 * s, 28 * s, 28 * s};
        const Theme& shown = cur ? t : th;
        draw::rect({sw.x, sw.y, sw.w / 2, sw.h / 2}, shown.light);
        draw::rect({sw.x + sw.w / 2, sw.y, sw.w / 2, sw.h / 2}, shown.dark);
        draw::rect({sw.x, sw.y + sw.h / 2, sw.w / 2, sw.h / 2}, shown.dark);
        draw::rect({sw.x + sw.w / 2, sw.y + sw.h / 2, sw.w / 2, sw.h / 2}, shown.light);
        a.text.draw(cur ? Font::Bold : Font::Regular, 14 * s, shown.name, {sw.right() + 10 * s, r.y + 13 * s}, pal::text);
        if (h && a.in.released && !cur) open = &th;
        y += 48 * s;
    }
    if (a.ui.button("th_new", {lib.x + 10 * s, lib.bottom() - 120 * s, lib.w - 20 * s, 42 * s}, "+ Nouveau th\xC3\xA8me", Style::Primary, true, 15)) {
        studio_theme_save(a);
        Theme copy = t;
        copy.file.clear();
        copy.name = t.name + " (copie)";
        st.theme = copy;
        st.theme_modified = true;
        studio_theme_save(a);
        a.notify("Nouveau th\xC3\xA8me : " + st.theme.name);
    }
    bool can_delete = a.themes.size() > 1;
    if (a.ui.button("th_del", {lib.x + 10 * s, lib.bottom() - 70 * s, lib.w - 20 * s, 40 * s}, st.confirm_theme_delete ? "Confirmer la suppression ?" : "Supprimer ce th\xC3\xA8me", Style::Danger, can_delete, 14)) {
        if (!st.confirm_theme_delete) st.confirm_theme_delete = true;
        else {
            std::error_code ec;
            fs::remove(g_paths.assets / "themes" / t.file, ec);
            a.themes = load_themes(g_paths.assets / "themes");
            a.board_textures.clear();
            st.theme_loaded = false;
            st.theme_modified = false;
            a.notify("Th\xC3\xA8me supprim\xC3\xA9 : " + t.name);
            return;
        }
    }

    // ── a live board ──
    // The board texture is computed pixel by pixel: while a colour changes,
    // it is rebuilt at most 5 times a second (under a key of its own).
    Theme shown = t;
    shown.id = "__studio_" + std::to_string(int(st.tex_built * 1000));
    if (st.tex_changed > st.tex_built && a.time - st.tex_built > 0.2) {
        a.board_textures.clear();
        st.tex_built = a.time;
        shown.id = "__studio_" + std::to_string(int(st.tex_built * 1000));
    }
    const float frame = 26 * s;
    float B = std::floor(std::min(mid.w, mid.h) - 2 * frame - 30 * s);
    B = std::floor(B / 8) * 8;
    st.board.area = {std::floor(mid.x + (mid.w - B) / 2), std::floor(mid.y + frame + 4 * s), B, B};
    Yolah y0;
    {   // A position with holes and pieces of both sides, one of them shut in.
        uint64_t b = 0, w = 0, e = 0;
        for (int sq : {9, 18, 36, 43}) b |= uint64_t(1) << sq;
        for (int sq : {14, 29, 52, 63}) w |= uint64_t(1) << sq;
        for (int sq : {0, 1, 8, 10, 17, 19, 21, 27, 28, 35, 44, 45, 53, 54, 55, 62, 22, 30, 38, 46, 47})
            if (!((b | w) >> sq & 1)) e |= uint64_t(1) << sq;
        y0.set_state(b, w, e, 4, 4, 8);
    }
    BoardView::Look look;
    a.set_look(look, shown, "human", "dev");
    st.board.draw(a.text, a.board_textures, shown, y0, look, a.time, s);
    if (a.ui.hover(st.board.area) && a.in.released) {   // a click: a hole opens there (to see its animation)
        int sq = st.board.at(a.in.mouse);
        if (sq >= 0) st.board.hole_since[sq] = a.time;
    }
    a.text.draw(Font::Regular, 13 * s, "Cliquez sur un trou pour revoir son apparition.", {mid.center().x, st.board.area.bottom() + frame + 2 * s}, pal::faint, Align::Center);

    // ── settings ──
    a.ui.panel(right);
    float x = right.x + 18 * s, w = right.w - 36 * s;
    y = right.y + 12 * s;
    a.text.draw(Font::Bold, 12 * s, st.theme_modified ? "Nom  \xE2\x80\xA2 modifi\xC3\xA9" : "Nom", {x, y}, pal::dim);
    if (a.ui.text_field("th_name", {x, y + 16 * s, w, 34 * s}, t.name, "Nom du th\xC3\xA8me", 15)) st.theme_modified = true;
    y += 58 * s;
    const std::vector<std::pair<std::string, std::string>> PATTERNS = {{"wood", "Bois"}, {"marble", "Marbre"}, {"stone", "Pierre"}, {"neon", "N\xC3\xA9on"}, {"flat", "Uni"}};
    std::vector<std::string> pl;
    int pi = 0;
    for (size_t i = 0; i < PATTERNS.size(); i++) { pl.push_back(PATTERNS[i].second); if (PATTERNS[i].first == t.pattern) pi = int(i); }
    int np = a.ui.segmented("th_pat", {x, y, w, 30 * s}, pl, pi, 13);
    if (np != pi) { t.pattern = PATTERNS[np].first; st.theme_modified = true; st.tex_changed = a.time; }
    y += 36 * s;
    std::vector<std::string> strengths = {"Motif discret", "Motif l\xC3\xA9ger", "Motif moyen", "Motif marqu\xC3\xA9", "Motif fort"};
    const float STRENGTH[] = {0.15f, 0.3f, 0.5f, 0.75f, 1.0f};
    int sti = 0;
    for (int i = 0; i < 5; i++) if (std::abs(STRENGTH[i] - t.strength) < std::abs(STRENGTH[sti] - t.strength)) sti = i;
    int nst = a.ui.cycler("th_str", {x, y, w, 30 * s}, "Motif", strengths, sti, 13);
    if (nst != sti) { t.strength = STRENGTH[nst]; st.theme_modified = true; st.tex_changed = a.time; }
    y += 38 * s;
    // The drawings of the workshop.
    struct Slot { const char* label; const char* kind; std::string* id; };
    Slot slots[] = {{"Case claire", "tile", &t.tile_light}, {"Case fonc\xC3\xA9" "e", "tile", &t.tile_dark}, {"Trous", "hole", &t.hole_art}, {"Cadre", "frame", &t.frame_art}};
    for (auto& sl : slots) {
        std::vector<std::string> names = {"Aucun (couleurs)"}, ids = {""};
        int idx = 0;
        for (Art* x2 : arts_of(sl.kind)) { names.push_back(x2->name); ids.push_back(x2->id); if (x2->id == *sl.id) idx = int(ids.size()) - 1; }
        int ni = a.ui.cycler(std::string("th_art_") + sl.label, {x, y, w - 84 * s, 30 * s}, sl.label, names, idx, 13);
        if (ni != idx) { *sl.id = ids[ni]; st.theme_modified = true; }
        if (a.ui.button(std::string("th_edit_") + sl.label, {x + w - 76 * s, y, 76 * s, 30 * s}, sl.id->empty() ? "Dessiner" : "\xC3\x89" "diter", Style::Normal, true, 12)) {
            studio_theme_save(a);
            if (sl.id->empty()) {   // a new drawing of this kind, then used by the theme
                studio_new(a, sl.kind, sl.kind == std::string("frame") ? 24 : 16);
                *sl.id = st.edit.id;
                st.theme_modified = true;
                studio_theme_save(a);
            } else {
                studio_open(a, *sl.id);
            }
            st.back_to_theme = t.file;
            st.tab = 0;
            return;
        }
        y += 36 * s;
    }
    y += 4 * s;
    // The colours: a grid of swatches; the selected one is edited below.
    struct Col { const char* label; Rgba* c; bool texture; };
    Col cols[] = {{"Case claire", &t.light, true}, {"Case fonc\xC3\xA9" "e", &t.dark, true}, {"Cadre (haut)", &t.frame_top, false},
                  {"Cadre (bas)", &t.frame_bottom, false}, {"Filet du cadre", &t.frame_line, false}, {"Coordonn\xC3\xA9" "es", &t.coords, false},
                  {"Trou", &t.hole, false}, {"Bord du trou", &t.hole_rim, false}, {"Dernier coup", &t.last_move, false},
                  {"S\xC3\xA9lection", &t.select, false}, {"Coups possibles", &t.legal, false}, {"Conseil", &t.hint, false},
                  {"Accent", &t.accent, false}, {"Fond (haut)", &t.bg_top, false}, {"Fond (bas)", &t.bg_bottom, false}};
    const int NC = int(std::size(cols));
    float cw = (w - 8 * s) / 3, ch = 30 * s;
    for (int i = 0; i < NC; i++) {
        Box r{x + (i % 3) * (cw + 4 * s), y + (i / 3) * (ch + 4 * s), cw, ch};
        bool cur = i == st.color_slot, h = a.ui.hover(r);
        if (h) a.ui.hovering = true;
        draw::round_box(r, 6 * s, with_alpha(hex(0xffffff), cur ? 0.16f : (h ? 0.08f : 0.03f)));
        if (cur) draw::round_frame(r, 6 * s, 2 * s, a.theme().accent);
        Box sw{r.x + 5 * s, r.y + 5 * s, 20 * s, 20 * s};
        checker(sw, 5 * s);
        draw::rect(sw, *cols[i].c);
        a.text.draw(Font::Regular, 11 * s, cols[i].label, {sw.right() + 6 * s, r.y + 8 * s}, pal::text);
        if (h && a.in.released) { st.color_slot = i; rgb_to_hsv(*cols[i].c, st.hue, st.sat, st.val); }
    }
    y += ((NC + 2) / 3) * (ch + 4 * s) + 6 * s;
    Col& cc = cols[std::clamp(st.color_slot, 0, NC - 1)];
    Box cp{x, y, w, std::max(80 * s, right.bottom() - y - 66 * s)};
    if (color_picker(a, cp, *cc.c, st.hue, st.sat, st.val, st.drag, "th")) {
        st.theme_modified = true;
        if (cc.texture) st.tex_changed = a.time;
    }
    if (a.ui.button("th_save", {x, right.bottom() - 56 * s, w, 44 * s}, "Enregistrer le th\xC3\xA8me", Style::Primary, true, 16)) {
        studio_theme_save(a);
        a.notify("Th\xC3\xA8me enregistr\xC3\xA9 : assets/themes/" + t.file);
    }
    if (open) studio_theme_open(a, *open);
}

void scene_studio(App& a) {
    const float s = a.s;
    auto& st = a.studio;
    draw_background(a, false);
    a.text.draw_shadowed(Font::Title, 34 * s, "Atelier", {40 * s, 24 * s}, pal::text, Align::Left, hex(0x000000, 120), 3 * s);
    a.text.draw(Font::Regular, 15 * s, st.tab == 0 ? "Dessinez les pions, les d\xC3\xA9" "cors, les cases, les trous et les cadres, image par image ; glissez une image sur la fen\xC3\xAAtre pour l'importer."
                                                : "Les th\xC3\xA8mes du plateau : couleurs, motif, et les dessins de l'atelier pour les cases, les trous et le cadre.",
                {42 * s, 74 * s}, pal::dim);
    if (!st.new_open && a.ui.focus.empty() && back_button(a)) {
        if (st.modified) studio_save(a);
        if (st.theme_modified) studio_theme_save(a);
        Scene back = st.return_to;
        st.return_to = Scene::Menu;
        a.go(back);
        return;
    }
    int nt = a.ui.tabs("st_tabs", {40 * s, 104 * s, 420 * s, 36 * s}, {"Dessins", "Plateaux"}, st.tab);
    if (nt != st.tab) {
        if (st.modified) studio_save(a);
        if (st.theme_modified) studio_theme_save(a);
        if (nt == 1 && !st.back_to_theme.empty()) {   // back to the theme that opened the drawing
            for (auto& th : a.themes)
                if (th.file == st.back_to_theme) studio_theme_open(a, th);
            st.back_to_theme.clear();
        }
        st.tab = nt;
        a.ui.focus.clear();
    }
    if (st.tab == 0) studio_drawings(a);
    else studio_boards(a);
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
    a->piece_sets = load_piece_sets(g_paths.assets / "sprites" / "sets.json");
    add_set_sprites(a->sprites, a->piece_sets);
    load_all_arts();
    register_art_sprites(a->sprites);
    a->roster = load_roster(g_paths.assets / "ai_roster.json");
    {
        json j = load_json(g_paths.assets / "sprites" / "assignments.json");
        if (j.is_object() && j.contains("players") && j["players"].is_object())
            for (auto& [k, v] : j["players"].items())
                if (v.is_string()) a->assigned[k] = v.get<std::string>();
    }
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
    a->debug_fx = arg_value(argc, argv, "--fx");
    if (!arg_value(argc, argv, "--pieces").empty()) a->settings.piece_set = arg_value(argc, argv, "--pieces");   // a set for this run
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
    } else if (scene == "setup") {
        a->go(Scene::Setup);
        a->setup_params = arg_value(argc, argv, "--params");   // its settings box open
    }
    else if (scene == "themes") {
        a->go(Scene::Themes);
        std::string who = arg_value(argc, argv, "--pick");   // open the sprite picker for this player
        if (!who.empty()) a->open_picker("Sprite de " + who, a->player_sprite(who).id, who == "human", nullptr);
    }
    else if (scene == "tutorial" || scene == "rules") start_lesson(*a, std::stoi(arg_value(argc, argv, "--lesson", "0")));
    else if (scene == "puzzles") {
        a->go(Scene::Puzzles);
        a->puzzles_ui.open_level = arg_value(argc, argv, "--level");
        a->puzzles_ui.tab = std::stoi(arg_value(argc, argv, "--ptab", "0"));
        if (a->puzzles_ui.tab == 1 && !a->levels.empty()) { a->puzzles_ui.edit = a->levels.front(); a->puzzles_ui.editing = true; }
        std::string pid = arg_value(argc, argv, "--puzzle");
        if (const Puzzle* pz = a->find_puzzle(pid)) {
            Puzzle copy = *pz;
            start_puzzle(*a, copy, "", -1, false);
            if (arg_flag(argc, argv, "--no-intro")) a->game.intro = false;
        }
    }
    else if (scene == "editor") a->go(Scene::Editor);
    else if (scene == "studio") {
        a->go(Scene::Studio);
        a->studio.tab = std::stoi(arg_value(argc, argv, "--tab", "0"));
        std::string art = arg_value(argc, argv, "--art");   // open this drawing
        if (find_art(art)) studio_open(*a, art);
    }
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
        if (e->button.button == SDL_BUTTON_RIGHT) { a->in.rdown = true; a->in.rpressed = true; a->in.mouse = {e->button.x, e->button.y}; }
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (e->button.button == SDL_BUTTON_LEFT) { a->in.down = false; a->in.released = true; a->in.mouse = {e->button.x, e->button.y}; }
        if (e->button.button == SDL_BUTTON_RIGHT) a->in.rdown = false;
        break;
    // A file dropped on the window (SDL sends one event per file).
    case SDL_EVENT_DROP_FILE:
        if (e->drop.data) a->push_file("drop", e->drop.data);
        break;
    case SDL_EVENT_MOUSE_WHEEL: a->in.wheel += e->wheel.y; break;
    case SDL_EVENT_KEY_DOWN:
        if (!e->key.repeat) a->in.keys.push_back(e->key.key);
        if (e->key.key == SDLK_BACKSPACE) a->in.backspaces++;
        break;
    case SDL_EVENT_TEXT_INPUT: a->in.text += e->text.text; break;
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
    a->in.files = a->take_files();

    // A scene that returned between draw::clip(&box) and draw::clip(nullptr)
    // (a click that changes scene) must not crop the next frame.
    SDL_SetRenderClipRect(a->renderer, nullptr);
    SDL_SetRenderDrawColor(a->renderer, 0, 0, 0, 255);
    SDL_RenderClear(a->renderer);
    switch (a->scene) {
    case Scene::Menu: scene_menu(*a); break;
    case Scene::Setup: scene_setup(*a); break;
    case Scene::Game: scene_game(*a); break;
    case Scene::Themes: scene_themes(*a); break;
    case Scene::Tutorial: scene_tutorial(*a); break;
    case Scene::Puzzles: scene_puzzles(*a); break;
    case Scene::Editor: scene_editor(*a); break;
    case Scene::Studio: scene_studio(*a); break;
    }
    if (a->scene != Scene::Game) {
        a->fx.update(float(a->dt));
        a->fx.draw(a->text);
    }
    // A message on the board (a side that must pass) goes as soon as the game is
    // over: the scores take its place.
    if (a->toast_at.x >= 0 && ((a->scene == Scene::Game && a->game.match.over())
                               || (a->scene == Scene::Tutorial && a->tuto.match.over())))
        a->toast.clear();
    // Toast message at the bottom.
    float tk = float(a->time - a->toast_t);
    if (tk < 4.0f && !a->toast.empty()) {
        float al = clamp01(std::min(tk * 5, (4.0f - tk) * 2));
        float s = a->s, fs = a->toast_size * s;
        float w = a->text.width(Font::Bold, fs, a->toast) + 2.8f * fs, h = 2.7f * fs;
        V2 c = a->toast_at.x >= 0 ? a->toast_at : V2{a->w * 0.5f, a->h - 67 * s};
        Box b{c.x - w / 2, c.y - h / 2 + (1 - al) * 20 * s, w, h};
        draw::shadow({b.x, b.y + 6 * s, b.w, b.h}, h / 2, 20 * s, hex(0x000000, uint8_t(120 * al)));
        draw::round_box(b, h / 2, with_alpha(hex(0x10151d), 0.95f * al));
        draw::round_frame(b, h / 2, 1, with_alpha(pal::accent, 0.6f * al));
        a->text.draw(Font::Bold, fs, a->toast, {b.center().x, b.center().y - a->text.line_height(Font::Bold, fs) * 0.45f}, with_alpha(pal::text, al), Align::Center);
    }
    // Fade-in after a scene change.
    float fade = 1 - clamp01(float(a->time - a->scene_t0) / 0.25f);
    if (fade > 0) draw::rect({0, 0, float(a->w), float(a->h)}, hex(0x000000, uint8_t(255 * fade)));
    SDL_SetCursor(a->ui.hovering ? a->cursor_hand : a->cursor_arrow);
    // Typed text only arrives while text input is on: on while a field has the focus.
    if (!a->ui.focus.empty() != SDL_TextInputActive(a->window)) {
        if (!a->ui.focus.empty()) SDL_StartTextInput(a->window);
        else SDL_StopTextInput(a->window);
    }

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
    a->in.pressed = a->in.released = a->in.rpressed = false;
    a->in.wheel = 0;
    a->in.files.clear();
    a->in.keys.clear();
    a->in.text.clear();
    a->in.backspaces = 0;
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
