#ifndef TOURNAMENT_H
#define TOURNAMENT_H
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace test {
    // The original round robin: every pair plays nb_games games with each
    // colour, all pairs at once, and prints the percentages.
    void tournament(const std::vector<std::string>& players_configs, size_t nb_random_moves, size_t nb_games);

    // A long, resumable round robin for grading many players (see tournament.cpp).
    struct TournamentOptions {
        std::vector<std::string> configs;   // player config files
        std::string results = "tournament_games.csv";   // one line per game; appended, resumed
        uint64_t microseconds = 1000000;    // time per move for every player (0 = keep each config's)
        size_t   threads = 4;               // search threads per player (0 = keep each config's)
        size_t   parallel = 6;              // CPU games played at the same time
        size_t   gpu_parallel = 1;          // games involving a GPU player played at the same time
        size_t   rounds = 0;                // 0 = until stopped (signal or hours)
        size_t   opening_plies = 4;         // random plies before each pair of games
        double   hours = 0;                 // stop after this long (0 = never)
        uint64_t seed = 1;                  // openings: the same for a given (seed, round, pair)
    };
    // Returns when all the rounds are played, after `hours`, or on SIGINT /
    // SIGTERM (the games in progress are dropped; the finished ones are in
    // the results file).
    void tournament(const TournamentOptions& options);

    // The players list file: one configuration path per line; blank lines and
    // lines starting with # are ignored; relative paths are relative to the file.
    std::vector<std::string> read_players_list(const std::string& path);
}

#endif
