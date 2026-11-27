#!/bin/bash
# final_retro_gpu_check.sh — builds doc/final_retro_gpu.cu (the commented copy) and
# test/solo_retro_gpu.cu (the real one), runs both at N free squares (default 8) and
# compares their outputs: the same layers, the same counts, no error in either.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-8}"
obj=app/build/solo_retro_gpu_obj
mkdir -p $obj
for f in test/solo_retro_gpu_cpu.cpp player/solo_region.cpp game/magic.cpp misc/misc.cpp; do
    g++ -c -O2 -std=c++20 -march=native -Igame -Iplayer -Imisc "$f" -o "$obj/$(basename "${f%.cpp}").o"
done
nvcc -O3 -std=c++20 -arch=native doc/final_retro_gpu.cu $obj/*.o -o app/build/final_retro_gpu
nvcc -O3 -std=c++20 -arch=native test/solo_retro_gpu.cu $obj/*.o -o app/build/solo_retro_gpu
strip_times() { sed -E 's/[0-9]+\.[0-9]+ s//g'; }
doc=$(app/build/final_retro_gpu "$N" 2 | strip_times)
real=$(app/build/solo_retro_gpu "$N" 2 | strip_times)
echo "$doc"
if [[ "$doc" == "$real" ]]; then echo "final_retro_gpu: same output as test/solo_retro_gpu.cu"; else echo "final_retro_gpu: OUTPUTS DIFFER"; diff <(echo "$doc") <(echo "$real"); exit 1; fi
