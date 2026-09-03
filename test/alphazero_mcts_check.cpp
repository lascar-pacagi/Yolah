#include "alphazero_mcts_check.h"
#include "alphazero_mcts_player.h"
#include "misc.h"
#include <cmath>
#include <iostream>
#include <memory>

using std::cout;

namespace test {

namespace {
    Yolah random_position(PRNG& prng, size_t max_plies) {
        Yolah y;
        Yolah::MoveList moves;
        const size_t n = reduce(prng.rand<uint32_t>(), static_cast<uint32_t>(max_plies));
        for (size_t i = 0; i < n && !y.game_over(); i++) {
            y.moves(moves);
            y.play(moves[reduce(prng.rand<uint32_t>(), moves.size())]);
        }
        return y;
    }

    bool check_result(const az::SearchResult& r, uint64_t nb_simulations, bool reused) {
        bool ok = true;
        auto fail = [&](const std::string& what) { cout << "  FAIL: " << what << "\n"; ok = false; };
        if (r.best_move == Move::none() && r.children.size() > 1) fail("no move chosen");
        uint64_t sum = 0;
        float pi = 0.0f;
        for (const az::ChildStat& c : r.children) {
            sum += c.visits;
            if (c.q < -1.0001f || c.q > 1.0001f) fail("Q out of range");
            if (c.prior < 0.0f || c.prior > 1.0001f) fail("prior out of range");
        }
        for (float p : r.policy) pi += p;
        if (!r.policy.empty() && std::fabs(pi - 1.0f) > 1e-3f) fail("policy does not sum to 1");
        if (!reused && sum != r.root_visits) fail("children visits != root visits");
        if (!reused && r.nb_simulations != nb_simulations) fail("simulation budget not honoured exactly");
        if (r.nb_simulations < nb_simulations) fail("fewer simulations than requested");
        if (!r.children.empty() && r.children[0].move != r.best_move &&
            r.children[0].visits != r.children[1].visits)
            fail("best move is not the most visited child");
        return ok;
    }
}

bool alphazero_search_check(const json& player_config, uint64_t nb_simulations) {
    json cfg = player_config;
    cfg["microseconds"] = 0;
    cfg["nb simulations"] = nb_simulations;
    cfg["verbose"] = false;
    cfg["reuse tree"] = false;
    cfg["temperature"] = 0.0;
    AlphaZeroMCTSPlayer player(cfg);
    cout << "[search check] " << player.info() << "\n";
    bool ok = true;
    PRNG prng(7);
    for (int k = 0; k < 4; k++) {
        Yolah y = k == 0 ? Yolah() : random_position(prng, 40);
        if (y.game_over()) continue;
        player.play(y);
        const az::SearchResult& r = player.last_result();
        cout << "position " << k << " (ply " << y.nb_plies() << "):\n" << r.to_string(5);
        ok &= check_result(r, nb_simulations, false);
    }
    cout << "  " << (ok ? "OK" : "FAILED") << std::endl;
    return ok;
}

bool alphazero_reuse_check(const json& player_config, uint64_t nb_simulations) {
    json cfg = player_config;
    cfg["microseconds"] = 0;
    cfg["nb simulations"] = nb_simulations;
    cfg["verbose"] = false;
    cfg["reuse tree"] = true;
    AlphaZeroMCTSPlayer player(cfg);
    cout << "[reuse check] " << player.info() << "\n";
    PRNG prng(11);
    Yolah y;
    Yolah::MoveList moves;
    bool ok = true;
    size_t reused_plies = 0;
    for (int ply = 0; ply < 6 && !y.game_over(); ply++) {
        Move m = player.play(y);
        const az::SearchResult& r = player.last_result();
        ok &= check_result(r, nb_simulations, ply > 0);
        const uint64_t new_nodes = r.tree_nodes;
        cout << "  ply " << y.nb_plies() << ": " << m << "  root visits " << r.root_visits
             << " (this search: " << r.nb_simulations << "), nodes " << new_nodes << "\n";
        if (ply > 0 && r.root_visits > r.nb_simulations) ++reused_plies;
        y.play(m);
        if (y.game_over()) break;
        // random opponent reply
        y.moves(moves);
        y.play(moves[reduce(prng.rand<uint32_t>(), moves.size())]);
    }
    if (reused_plies == 0) { cout << "  FAIL: no subtree was reused\n"; ok = false; }
    cout << "  reused subtrees on " << reused_plies << " plies  " << (ok ? "OK" : "FAILED") << std::endl;
    return ok;
}

double alphazero_vs(const json& player_config, const json& opponent_config, size_t nb_games, bool verbose) {
    double score = 0.0;
    for (size_t g = 0; g < nb_games; g++) {
        const bool az_black = (g % 2) == 0;
        std::unique_ptr<Player> az = Player::create(player_config);
        std::unique_ptr<Player> opp = Player::create(opponent_config);
        Player* black = az_black ? az.get() : opp.get();
        Player* white = az_black ? opp.get() : az.get();
        Yolah y;
        while (!y.game_over()) {
            Player* p = y.current_player() == Yolah::BLACK ? black : white;
            Move m = p->play(y);
            if (verbose) cout << (y.current_player() == Yolah::BLACK ? "black " : "white ") << m << "\n";
            y.play(m);
        }
        az->game_over(y);
        opp->game_over(y);
        const auto [bs, ws] = y.score();
        const int az_diff = az_black ? int(bs) - int(ws) : int(ws) - int(bs);
        score += az_diff > 0 ? 1.0 : (az_diff == 0 ? 0.5 : 0.0);
        cout << "game " << g + 1 << ": alphazero as " << (az_black ? "black" : "white")
             << ", score " << bs << "/" << ws << " → " << (az_diff > 0 ? "win" : az_diff == 0 ? "draw" : "loss")
             << std::endl;
    }
    cout << "[alphazero vs " << opponent_config.value("name", std::string("?")) << "] "
         << score / nb_games * 100 << " %" << std::endl;
    return score / nb_games;
}

} // namespace test
