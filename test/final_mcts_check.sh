#!/bin/bash
# final_mcts_check.sh — builds build-torch/final_mcts_check (test/final_mcts_check.cpp)
# with the compile and link lines CMake made for alphazero_check (same libraries,
# same flags), its own main and doc/final_mcts.cpp in place of alphazero_check's main.
# Run from build-torch/ after `make alphazero_check` (the objects must be up to date).
set -euo pipefail
cd "$(dirname "$0")/../build-torch"
d=CMakeFiles/alphazero_check.dir
o=CMakeFiles/final_mcts_check.dir
mkdir -p $o
flags=$(sed -n 's/^CXX_\(DEFINES\|INCLUDES\|FLAGS\) = //p' $d/flags.make | tr '\n' ' ')
for f in ../test/final_mcts_check.cpp ../doc/final_mcts.cpp; do
    c++ $flags -I../doc -c "$f" -o "$o/$(basename "$f").o" &
done
wait
link=$(sed -e "s#$d/test/alphazero_check_main.cpp.o#$o/final_mcts_check.cpp.o $o/final_mcts.cpp.o#" -e 's/-o alphazero_check/-o final_mcts_check/' $d/link.txt)
eval "$link"
echo "built $(pwd)/final_mcts_check"
