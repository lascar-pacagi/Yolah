#!/bin/bash
# solo_retro_gpu.sh — builds and runs test/solo_retro_gpu.cu (the table of private
# regions by retrograde analysis, on the GPU). Arguments: N (free squares, default 8),
# K (pieces, default 2).
set -euo pipefail
cd "$(dirname "$0")/.."
out=app/build/solo_retro_gpu
obj=app/build/solo_retro_gpu_obj
mkdir -p $obj
for f in test/solo_retro_gpu_cpu.cpp player/solo_region.cpp game/magic.cpp misc/misc.cpp; do
    g++ -c -O2 -std=c++20 -march=native -Igame -Iplayer -Imisc "$f" -o "$obj/$(basename "${f%.cpp}").o"
done
nvcc -O3 -std=c++20 -arch=native test/solo_retro_gpu.cu $obj/*.o -o $out
./$out "${1:-8}" "${2:-2}"
