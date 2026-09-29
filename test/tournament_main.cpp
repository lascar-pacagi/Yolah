// tournament_main.cpp — the grading tournament as its own executable (used by
// nnue/tournament.sh on the cluster). Same as `Yolah --tournament`, without
// boost::program_options: in the cluster image, libtorch (needed by the GPU
// player) imposes the pre-C++11 std::string ABI, which the system Boost
// libraries do not use.
//
//   yolah_tournament --players list.txt [--results games.csv] [--time 1000000]
//       [--threads 4] [--parallel 6] [--gpu-parallel 1] [--rounds 0]
//       [--opening-plies 4] [--hours 0] [--seed 1]
#include "tournament.h"
#include "magic.h"
#include "zobrist.h"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    magic::init();
    zobrist::init();
    test::TournamentOptions opt;
    std::string players;
    for (int i = 1; i < argc; i++) {
        const std::string o = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << o << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (o == "--players") players = next();
        else if (o == "--results") opt.results = next();
        else if (o == "--time") opt.microseconds = std::stoull(next());
        else if (o == "--threads") opt.threads = std::stoul(next());
        else if (o == "--parallel") opt.parallel = std::stoul(next());
        else if (o == "--gpu-parallel") opt.gpu_parallel = std::stoul(next());
        else if (o == "--rounds") opt.rounds = std::stoul(next());
        else if (o == "--opening-plies") opt.opening_plies = std::stoul(next());
        else if (o == "--hours") opt.hours = std::stod(next());
        else if (o == "--seed") opt.seed = std::stoull(next());
        else { std::cerr << "unknown option " << o << " (see tournament_main.cpp)\n"; return 2; }
    }
    if (players.empty()) { std::cerr << "--players list.txt expected\n"; return 2; }
    try {
        opt.configs = test::read_players_list(players);
        test::tournament(opt);
    } catch (const std::exception& e) {
        std::cerr << "tournament: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
