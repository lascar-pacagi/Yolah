#!/bin/bash
# Quick script to compile and run the magic number benchmark

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Compiling magic_bench...${NC}"
g++ -std=c++23 -O3 -march=native -Wall -Wextra magic_bench.cpp -o magic_bench -lbenchmark -lpthread

echo -e "${GREEN}Compilation successful!${NC}"
echo -e "${BLUE}Running benchmark...${NC}"
echo ""

./magic_bench "$@"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
