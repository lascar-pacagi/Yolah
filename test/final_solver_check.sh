#!/bin/bash
# final_solver_check.sh — builds and runs test/final_solver_check.cpp: doc/final_solver.cpp
# against the editor's solver (player/endgame_solver.cpp, taken from the app's engine
# library) and a plain negamax. Run `ninja` in app/build first (the library must be
# up to date). Argument: the number of random positions (default 2000).
set -euo pipefail
cd "$(dirname "$0")/.."
out=app/build/final_solver_check
g++ -std=c++20 -O2 -march=native -Igame -Iplayer -Imisc -Idoc \
    test/final_solver_check.cpp doc/final_solver.cpp app/build/libyolah_engine.a -fopenmp -o $out
./$out "${1:-2000}"
