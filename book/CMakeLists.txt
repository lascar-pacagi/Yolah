cmake_minimum_required(VERSION 3.20)
project(MagicBench CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Google Benchmark
find_package(benchmark REQUIRED)

# Build the benchmark executable
add_executable(magic_bench magic_bench.cpp)

# Compiler flags for optimization
target_compile_options(magic_bench PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# Link Google Benchmark
target_link_libraries(magic_bench PRIVATE benchmark::benchmark)
