#!/bin/bash
# final_retro_check.sh — builds and runs test/final_retro_check.cpp: doc/final_retro.cpp
# against player/solo_region.cpp and the exact endgame solver. Argument: games (default 1000).
set -euo pipefail
cd "$(dirname "$0")/.."
out=app/build/final_retro_check
g++ -std=c++20 -O2 -march=native -Igame -Iplayer -Imisc -Idoc \
    test/final_retro_check.cpp doc/final_retro.cpp player/solo_region.cpp player/endgame_solver.cpp \
    game/game.cpp game/magic.cpp game/move.cpp game/zobrist.cpp misc/misc.cpp -pthread -o $out
./$out "${1:-1000}"
