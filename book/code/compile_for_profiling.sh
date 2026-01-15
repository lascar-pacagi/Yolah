#!/bin/bash
# Compile chapter02 with optimal settings for profiling
# This ensures clean call stacks and accurate performance data

SOURCE=${1:-chapter02.cpp}
OUTPUT=${2:-chapter02}

echo "Compiling $SOURCE for profiling..."

g++-13 -std=c++23 \
  -O3 \
  -march=native \
  -mavx2 \
  -ffast-math \
  -funroll-loops \
  -fopenmp \
  -fno-omit-frame-pointer \
  -g \
  -gdwarf-4 \
  "$SOURCE" \
  -o "$OUTPUT" \
  -lbenchmark \
  -lpthread

if [ $? -eq 0 ]; then
  echo "✓ Compiled successfully: $OUTPUT"
  echo ""
  echo "Binary info:"
  ls -lh "$OUTPUT"
  echo ""
  echo "Debug sections:"
  objdump -h "$OUTPUT" | grep debug | head -n 5
  echo ""
  echo "Ready for profiling with:"
  echo "  perf record -F 999 --call-graph dwarf -g ./$OUTPUT"
else
  echo "✗ Compilation failed"
  exit 1
fi
