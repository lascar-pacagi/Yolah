#include "variability_timer_multithreads.h"
#include <thread>
#include <chrono>
#include <vector>
#include "game.h"
#include "misc.h"

namespace test {
    void variability_timer_multithreads(size_t nb_threads, uint64_t microseconds) {
        std::vector<size_t> nb_iterations(nb_threads);
        {
            std::vector<std::jthread> threads;
            for (size_t i = 0; i < nb_threads; i++) {
                threads.emplace_back([i, &nb_iterations, &microseconds]{
                    using namespace std::chrono;
                    PRNG prng(42);
                    const steady_clock::time_point start = steady_clock::now();
                    duration<uint64_t, std::micro> mu;        
                    size_t n = 0;
                    for (;;) {
                        Yolah yolah;
                        Yolah::MoveList moves;
                        while (!yolah.game_over()) {
                            yolah.moves(moves);
                            if (moves.size() == 0) continue;
                            Move m = moves[prng.rand<size_t>() % moves.size()];
                            yolah.play(m);
                        }
                        n++;
                        mu = std::chrono::duration_cast<std::chrono::microseconds>(steady_clock::now() - start);
                        if (mu.count() > microseconds) break;
                    }
                    nb_iterations[i] = n;
                });
            }
        }
        for (auto n : nb_iterations) {
            std::cout << n << '\n';
        }
    }
}

