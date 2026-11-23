// final_mcts_check.cpp — doc/final_mcts.{h,cpp} against the real player
// (AlphaZeroMCTSPlayer, player/alphazero_mcts*): one search thread, a fixed
// number of simulations, the network in fp32, on the positions of
// test/bench_positions.txt (plus a second search after two plies, for the
// tree reuse). For each search: the move, and for each root move the visits
// and Q, must be identical.
//
// Not part of the normal build. Build (from build-torch/, after `make
// yolah_tournament` so that the libraries are up to date) with the compile and
// link lines of alphazero_check, see doc/final_mcts.h « VÉRIFICATION »:
//     bash ../test/final_mcts_check.sh
// Run from config/ (the weights path is relative):
//     cd config && ../build-torch/final_mcts_check [nb simulations] [nb positions]
//
// Last check (2026-11-23, RTX 3080, fp32, 800 simulations): 209 / 209 positions
// identical, 411 searches (every move, visit count, Q and play value), 10 min.
#include "../doc/final_mcts.h"
#include "alphazero_mcts_player.h"
#include "magic.h"
#include "zobrist.h"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    magic::init();
    zobrist::init();
    const uint64_t sims = argc > 1 ? std::stoull(argv[1]) : 800;
    const size_t max_pos = argc > 2 ? std::stoul(argv[2]) : 1000;
    const std::string weights = "../nnue/cnn_resnet_256x30_value_policy.ts";

    json cfg = json::parse(std::ifstream("hybrid/resnet.cfg"));
    cfg["nb threads"] = 1;
    cfg["microseconds"] = 0;
    cfg["nb simulations"] = sims;
    cfg["fp16"] = false;
    cfg["weights"] = weights;
    AlphaZeroMCTSPlayer real(cfg);

    final_mcts::Params p;
    p.nb_threads = 1;
    p.microseconds = 0;
    p.nb_simulations = sims;
    final_mcts::Player mine(weights, p, /*fp16=*/false);

    std::ifstream in("../test/bench_positions.txt");
    std::string line;
    size_t nb = 0, same = 0, searches = 0;
    while (std::getline(in, line) && nb < max_pos) {
        if (line.empty()) continue;
        Yolah y = Yolah::from_json(line);
        if (y.game_over()) continue;
        ++nb;
        bool ok = true;
        // two searches per position: the second one after two plies (the
        // played move and the opponent's reply), from the reused graph
        for (int k = 0; k < 2 && !y.game_over(); k++) {
            const Move a = real.play(y);
            const Move b = mine.play(y);
            ++searches;
            const auto& ra = real.last_result();
            const auto& rb = mine.last_result();
            bool eq = a == b && ra.children.size() == rb.children.size() && ra.nb_simulations == rb.nb_simulations;
            for (size_t i = 0; eq && i < ra.children.size(); i++)
                eq = ra.children[i].move == rb.children[i].move && ra.children[i].visits == rb.children[i].visits &&
                     ra.children[i].q == rb.children[i].q && ra.children[i].play_value == rb.children[i].play_value;
            if (!eq) {
                ok = false;
                std::cout << "position " << nb << " search " << k << ": DIFFERENT (" << a << " vs " << b << ")\n";
                for (size_t i = 0; i < std::min<size_t>(5, std::min(ra.children.size(), rb.children.size())); i++)
                    std::cout << "   " << ra.children[i].move << " " << ra.children[i].visits << " " << ra.children[i].q
                              << "   |   " << rb.children[i].move << " " << rb.children[i].visits << " " << rb.children[i].q << "\n";
                break;
            }
            y.play(a);
            if (y.game_over()) break;
            Yolah::MoveList ml;
            y.moves(ml);
            y.play(ml[0]);                  // the opponent's reply: its first legal move
        }
        same += ok;
        real.game_over(y);
        mine.game_over();
    }
    std::cout << same << " / " << nb << " positions identical (" << searches << " searches, " << sims << " simulations each)\n";
    return same == nb ? 0 : 1;
}
