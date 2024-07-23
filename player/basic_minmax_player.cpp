#include "basic_minmax_player.h"
#include <cstring>
#include "cem.h"
#include <random>
#include <chrono>
#include <array>
#include "monte_carlo_player.h"
#include "MCTS_player.h"
#include "MCTS_mem_player.h"

BasicMinMaxPlayer::BasicMinMaxPlayer(uint8_t depth, heuristic_eval h) : depth(depth), heuristic(h) {
}

void BasicMinMaxPlayer::sort_moves(Yolah& yolah, Yolah::MoveList& moves) {
    std::vector<std::pair<int32_t, Move>> tmp;
    size_t nb_moves = moves.size();
    auto current_player = yolah.current_player();
    for (size_t i = 0; i < nb_moves; i++) {
        yolah.play(moves[i]);
        tmp.emplace_back(heuristic(current_player, yolah), moves[i]);
        yolah.undo(moves[i]);
    }
    std::sort(begin(tmp), end(tmp), [](const auto& p1, const auto& p2) {
        return p1.first > p2.first;
    });
    for (size_t i = 0; i < nb_moves; i++) {
        moves[i] = tmp[i].second;
    }
}

int32_t BasicMinMaxPlayer::negamax(Yolah& yolah, int32_t alpha, int32_t beta, uint8_t depth) {
    nb_nodes++;
    if (yolah.game_over()) {
        int32_t score = yolah.score(yolah.current_player());
        return score + (score >= 0 ? heuristic::MAX_VALUE : heuristic::MIN_VALUE);
    }
    if (depth == 0) {
        return heuristic(yolah.current_player(), yolah);
    }
    Yolah::MoveList moves;
    yolah.moves(moves);
    sort_moves(yolah, moves);
    for (const Move& m : moves) {
        yolah.play(m);
        int32_t v = -negamax(yolah, -beta, -alpha, depth - 1);
        yolah.undo(m);
        if (v >= beta) {
            return v;
        }
        if (v > alpha) {
            alpha = v;
        }
    }
    return alpha;
}

int32_t BasicMinMaxPlayer::search(Yolah& yolah, Move& res) {
    res = Move::none();
    if (yolah.game_over()) {
        return yolah.score(yolah.current_player()) * heuristic::MAX_VALUE;
    }
    Yolah::MoveList moves;
    yolah.moves(moves);
    int32_t alpha = -std::numeric_limits<int32_t>::max();
    int32_t beta = std::numeric_limits<int32_t>::max();
    sort_moves(yolah, moves);
    for (const Move& m : moves) {
        yolah.play(m);
        int32_t v = -negamax(yolah, alpha, beta, depth - 1);
        yolah.undo(m);        
        if (v > alpha) {
            alpha = v;
            res = m;
        }
    }
    return alpha;
}

Move BasicMinMaxPlayer::play(Yolah yolah) {
    Move m;
    nb_nodes = 0;
    auto value = search(yolah, m);
    std::cout << "value  : " << value << '\n';
    std::cout << "# nodes: " << nb_nodes << std::endl;
    return m;
}

std::string BasicMinMaxPlayer::info() {
    return "basic minmax player";
}

json BasicMinMaxPlayer::config() {
    json j;
    j["name"] = "BasicMinMaxPlayer";
    j["depth"] = depth;
    return j;
}
