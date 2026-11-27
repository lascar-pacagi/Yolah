#!/bin/bash
# final_ybwc_check.sh — builds and runs test/final_ybwc_check.cpp: doc/final_ybwc.cpp
# against the editor's parallel solver (EndgameSolver, player/endgame_solver.cpp)
# and a plain negamax. Argument: the number of random positions (default 1000).
set -euo pipefail
cd "$(dirname "$0")/.."
out=app/build/final_ybwc_check
g++ -std=c++20 -O2 -march=native -Igame -Iplayer -Imisc -Idoc \
    test/final_ybwc_check.cpp doc/final_ybwc.cpp player/endgame_solver.cpp player/solo_region.cpp \
    game/game.cpp game/magic.cpp game/move.cpp game/zobrist.cpp misc/misc.cpp -pthread -o $out
./$out "${1:-1000}"
