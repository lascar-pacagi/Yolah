# ────────────────────────────────────────────────────────────────────────────
# Yolah, the graphical application (SDL3).
#
#   cmake -S app -B app/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build app/build -j
#   ./app/build/yolah_app            (run from anywhere: assets are found next
#                                     to the executable or in app/assets)
#
# One source file (yolah_app.cpp), plus the engine and the AI players of the
# repository (game/, player/, nnue/, misc/...), compiled into a static library.
#
# SDL3: an installed SDL3 is used if CMake finds it (default hint ~/opt/SDL3,
# or -DCMAKE_PREFIX_PATH=...); otherwise it is downloaded and compiled
# (-DYOLAH_FETCH_SDL3=ON forces this).
# Images and fonts: stb_image, stb_image_write, stb_truetype (third_party/,
# single headers, public domain). No other library.
#
# Options:
#   -DYOLAH_APP_TORCH=ON   libtorch backend for the ResNet players (MCTS ResNet,
#                          AlphaZero) on the GPU; without it they run on the CPU
#                          (the .bin weights of nnue/resnet_export.py), slower.
#                          Needs -DCMAKE_PREFIX_PATH=<libtorch>.
# ────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.20)
project(YolahApp CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(YOLAH_APP_TORCH "libtorch (GPU) backend for the ResNet players" OFF)
option(YOLAH_FETCH_SDL3 "Download and build SDL3 instead of using an installed one" OFF)
set(YOLAH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "The Yolah repository (engine and AI sources)")

# ── SDL3 ────────────────────────────────────────────────────────────────────
if(NOT YOLAH_FETCH_SDL3)
    find_package(SDL3 CONFIG QUIET HINTS "$ENV{HOME}/opt/SDL3")
endif()
if(NOT SDL3_FOUND)
    message(STATUS "SDL3 not found: downloading and building it")
    include(FetchContent)
    FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-3.2.30
        GIT_SHALLOW TRUE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL3)
endif()

# ── the engine and the AI players ───────────────────────────────────────────
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB QUIET)

set(R "${YOLAH_ROOT}")
set(ENGINE_SOURCES
    ${R}/game/game.cpp ${R}/game/magic.cpp ${R}/game/move.cpp ${R}/game/zobrist.cpp
    ${R}/misc/misc.cpp
    ${R}/player/random_player.cpp ${R}/player/monte_carlo_player.cpp ${R}/player/client_player.cpp
    ${R}/player/MCTS_mem_nn_player.cpp ${R}/player/human_player.cpp
    ${R}/player/heuristic.cpp ${R}/player/basic_minmax_player.cpp ${R}/player/transposition_table.cpp
    ${R}/player/heuristic_weights_learner.cpp ${R}/player/player.cpp
    ${R}/player/minmax_player.cpp ${R}/player/minmax_player2.cpp ${R}/player/minmax_nnue_player.cpp
    ${R}/player/minmax_nnue_quantized_player.cpp ${R}/player/yolah_features.cpp
    ${R}/player/minmax_nnue_baseline_player.cpp ${R}/player/minmax_nnue_dev_player.cpp
    ${R}/player/endgame_solver.cpp ${R}/player/search_table.cpp
    ${R}/player/features_net_player.cpp
    ${R}/player/alphazero_mcts.cpp ${R}/player/alphazero_mcts_player.cpp
    ${R}/nnue/nnue.cpp ${R}/nnue/nnue_quantized.cpp ${R}/nnue/nnue_persp.cpp ${R}/nnue/ffnn.cpp ${R}/nnue/ffnn_float.cpp
    ${R}/nnue/nn_evaluator.cpp ${R}/nnue/resnet.cpp ${R}/nnue/batched_evaluator.cpp
    ${R}/optimization/cem.cpp ${R}/optimization/nelder_mead.cpp
    ${R}/data/generate_games.cpp
)
if(YOLAH_APP_TORCH)
    list(APPEND ENGINE_SOURCES ${R}/nnue/resnet_torch.cpp)
endif()
add_library(yolah_engine STATIC ${ENGINE_SOURCES})
target_include_directories(yolah_engine PUBLIC
    ${R}/game ${R}/misc ${R}/player ${R}/nnue ${R}/optimization ${R}/data ${R}/eigen ${R}/vcl)
target_compile_options(yolah_engine PRIVATE -march=native -mavx2 -ffast-math -funroll-loops -w)
target_link_libraries(yolah_engine PUBLIC OpenMP::OpenMP_CXX Threads::Threads)
if(TBB_FOUND)
    target_link_libraries(yolah_engine PUBLIC TBB::tbb)
endif()
if(YOLAH_APP_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(yolah_engine PUBLIC ENABLE_TORCH)
    target_include_directories(yolah_engine PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(yolah_engine PUBLIC ${TORCH_LIBRARIES})
endif()

# ── the application ─────────────────────────────────────────────────────────
add_executable(yolah_app yolah_app.cpp)
target_include_directories(yolah_app PRIVATE third_party)
target_compile_options(yolah_app PRIVATE -march=native -Wall -Wno-unused-function)
target_compile_definitions(yolah_app PRIVATE
    YOLAH_APP_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    YOLAH_APP_ROOT="${YOLAH_ROOT}")
if(YOLAH_APP_TORCH)
    target_compile_definitions(yolah_app PRIVATE YOLAH_APP_TORCH)
endif()
target_link_libraries(yolah_app PRIVATE yolah_engine SDL3::SDL3)
