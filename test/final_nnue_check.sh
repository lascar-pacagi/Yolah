#!/bin/bash
# final_nnue_check.sh — builds and runs test/final_nnue_check.cpp: doc/final_nnue.cpp
# against the dev player's network (NNUE_Persp, nnue/nnue_persp.cpp, taken from the
# app's engine library, compiled with the same flags here). Run `ninja` in app/build
# first (the library must be up to date). Arguments: the weights file (default: the
# dev player's) and the number of random games (default 300).
set -euo pipefail
cd "$(dirname "$0")/.."
out=app/build/final_nnue_check
g++ -std=c++20 -O2 -march=native -mavx2 -ffast-math -Igame -Iplayer -Imisc -Innue -Idoc \
    test/final_nnue_check.cpp doc/final_nnue.cpp app/build/libyolah_engine.a -fopenmp -o $out
./$out "${1:-models/nnue_persp_k8_them_g046_qat_ib4f_ref2u_az20_distill.quantized.txt}" "${2:-300}"
