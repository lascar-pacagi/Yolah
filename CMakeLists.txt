cmake_minimum_required(VERSION 3.20.0)

# Enable CUDA support (optional - only if CUDA is available)
option(ENABLE_CUDA "Enable CUDA support for AlphaZero" OFF)
# libtorch backend for the AlphaZero MCTS player's ResNet (GPU inference).
# Point CMAKE_PREFIX_PATH at a libtorch install, e.g. the one inside a pip
# torch package:  -DENABLE_TORCH=ON -DCMAKE_PREFIX_PATH=<site-packages>/torch
option(ENABLE_TORCH "Enable the libtorch (GPU) network backend" OFF)

if(ENABLE_CUDA)
    project(Yolah CXX CUDA)
    enable_language(CUDA)
else()
    project(Yolah CXX)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable Link-Time Optimization (LTO/IPO)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)

if(ipo_supported)
    message(STATUS "LTO/IPO is supported and enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
else()
    message(WARNING "LTO/IPO is not supported: ${ipo_output}")
endif()

find_package(OpenMP)
find_package(Boost 1.71 COMPONENTS program_options REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)
if(ENABLE_TORCH)
    # TorchConfig.cmake of a CUDA-enabled libtorch probes the installed GPUs by
    # compiling a CUDA test project, which needs a matching CUDA toolkit and
    # trips over our C++23 dialect. Giving it the architecture list skips the
    # probe (8.6 = RTX 30xx; 7.5 = Turing; 8.9 = RTX 40xx; 9.0 = H100).
    if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architectures for the libtorch probe")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "libtorch backend enabled: ${TORCH_LIBRARIES}")
endif()

add_executable(Yolah main.cpp)
target_compile_options(Yolah PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp -flto=auto)
target_link_options(Yolah PRIVATE -flto=auto)

if(OpenMP_CXX_FOUND)
    target_link_libraries(Yolah PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(Yolah PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(Yolah PRIVATE ${Boost_LIBRARIES})
target_link_libraries(Yolah PRIVATE TBB::tbb)
target_link_libraries(Yolah PRIVATE Threads::Threads)
add_subdirectory(game)
target_link_libraries(Yolah PRIVATE game)
add_subdirectory(misc)
target_link_libraries(Yolah PRIVATE misc)
add_subdirectory(test)
target_link_libraries(Yolah PRIVATE test)
add_subdirectory(player)
target_link_libraries(Yolah PRIVATE player)
add_subdirectory(ui)
target_link_libraries(Yolah PRIVATE ui)
add_subdirectory(optimization)
target_link_libraries(Yolah PRIVATE optimization)
add_subdirectory(data)
target_link_libraries(Yolah PRIVATE data)
add_subdirectory(nnue)
target_link_libraries(Yolah PRIVATE nnue)

# Self-check / benchmark of the AlphaZero MCTS player and its network backends:
#   ./alphazero_check [--weights f.bin] [--test-vectors f.test.bin] [--backend cpu|torch] ...
add_executable(alphazero_check test/alphazero_check_main.cpp)
target_compile_options(alphazero_check PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)
# With LTO the code is generated at link time: repeat the target flags there.
target_link_options(alphazero_check PRIVATE -march=native -mavx2 -ffast-math -fopenmp)
target_include_directories(alphazero_check PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(alphazero_check PRIVATE game misc test player ui optimization data nnue
    ${Boost_LIBRARIES} TBB::tbb Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero_check PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ENABLE_TORCH)
    target_link_libraries(alphazero_check PRIVATE ${TORCH_LIBRARIES})
endif()

# The C++ half of the AlphaZero learning loop (self-play games, evaluation
# matches); driven by nnue/alphazero_learn.py:
#   ./alphazero_learn selfplay --config ../config/alphazero_mcts_selfplay_player.cfg --work <dir>
#   ./alphazero_learn match --config ../config/alphazero_mcts_eval_player.cfg --a new.ts --b old.ts
add_executable(alphazero_learn test/alphazero_learn_main.cpp)
target_compile_options(alphazero_learn PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)
target_link_options(alphazero_learn PRIVATE -march=native -mavx2 -ffast-math -fopenmp)
target_include_directories(alphazero_learn PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(alphazero_learn PRIVATE game misc test player ui optimization data nnue
    ${Boost_LIBRARIES} TBB::tbb Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(alphazero_learn PRIVATE OpenMP::OpenMP_CXX)
endif()
if(ENABLE_TORCH)
    target_link_libraries(alphazero_learn PRIVATE ${TORCH_LIBRARIES})
endif()

# AlphaZero (only if CUDA is enabled)
if(ENABLE_CUDA)
    add_subdirectory(alphazero)
    target_link_libraries(Yolah PRIVATE alphazero)
endif()
