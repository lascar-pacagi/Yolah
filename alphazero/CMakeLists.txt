cmake_minimum_required(VERSION 3.20.0)
project(AlphaZero LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Support for modern GPUs

# Find required packages
find_package(CUDAToolkit REQUIRED)

# AlphaZero library
add_library(alphazero STATIC
    alphazero_network.cu
    alphazero_mcts.cpp
    self_play.cpp
)

target_include_directories(alphazero PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(alphazero PUBLIC
    game
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
    CUDA::curand
)

# CUDA compilation options
target_compile_options(alphazero PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -Xcompiler=-fopenmp
        -Xcompiler=-fPIC
    >
)

# C++ compilation options
target_compile_options(alphazero PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -march=native
        -fopenmp
        -fPIC
    >
)

# Self-play executable
add_executable(alphazero_selfplay selfplay_main.cpp)
target_link_libraries(alphazero_selfplay PRIVATE alphazero)
