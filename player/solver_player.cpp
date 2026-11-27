#include "solver_player.h"
#include <bit>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {
EndgameSolver::Options solver_options(const json& j) {
    EndgameSolver::Options o;
    o.pass_rule = false;
    o.tt_bits = j.value("tt bits", 22);
    o.threads = j.value("nb threads", 1);
    o.regions = j.value("regions", true);   // private regions as exact reserves (Points goal only)   // shared work (EndgameSolver::split); the app caps it to its cores
    // The app's target puzzles: "goal" collect / first, "targets" a 64-bit mask (a decimal string).
    const std::string goal = j.value("goal", std::string("points"));
    o.goal = goal == "collect" ? EndgameGoal::Collect : goal == "first" ? EndgameGoal::First : EndgameGoal::Points;
    if (j.contains("targets")) o.targets = j["targets"].is_string() ? std::stoull(j["targets"].get<std::string>()) : j["targets"].get<uint64_t>();
    return o;
}
}

SolverPlayer::SolverPlayer(const json& j)
    : cfg(j), microseconds(j.value("microseconds", uint64_t(5000000))), max_free(j.value("max free", 36)), options(solver_options(j)), solver(options) {
    if (j.contains("fallback") && j["fallback"].is_object()) {
        json f = j["fallback"];
        f["microseconds"] = microseconds;   // the same time per move as the solver's
        fallback = Player::create(f);
    }
}

Move SolverPlayer::play(Yolah yolah) {
    Yolah::MoveList moves;
    yolah.moves(moves);
    if (moves.size() == 0) return Move::none();
    if (moves.size() == 1) { last = "only move"; return moves[0]; }
    if (std::popcount(yolah.free_squares()) <= max_free) {
        // The proof, stopped by a watchdog after `microseconds`.
        std::atomic_bool stop{false};
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
        std::thread watchdog([&] {
            std::unique_lock lk(m);
            if (!cv.wait_for(lk, std::chrono::microseconds(microseconds), [&] { return done; })) stop = true;
        });
        const EndgameSolver::Result r = solver.solve(yolah, false, &stop);
        {
            std::lock_guard lk(m);
            done = true;
        }
        cv.notify_all();
        watchdog.join();
        if (r.complete && r.move != Move::none()) {
            last = "exact: final lead " + std::to_string(r.value);
            return r.move;
        }
    }
    if (options.goal != EndgameGoal::Points) {
        // No proof, a target goal: a move onto a target if there is one (First: it
        // wins at once); else the fallback, which plays for points.
        for (Move m : moves)
            if (m != Move::none() && (options.targets >> m.to_sq() & 1)) { last = "target taken (no proof)"; return m; }
    }
    if (fallback) { last = "fallback"; return fallback->play(yolah); }
    last = "no proof, first move";
    return moves[0];
}
