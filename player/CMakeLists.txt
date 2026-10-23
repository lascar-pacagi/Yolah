set(PLAYER_SOURCES
    random_player.cpp monte_carlo_player.cpp client_player.cpp
    MCTS_mem_nn_player.cpp human_player.cpp
    heuristic.cpp basic_minmax_player.cpp transposition_table.cpp
    heuristic_weights_learner.cpp player.cpp
    minmax_player.cpp minmax_player2.cpp minmax_nnue_player.cpp minmax_nnue_quantized_player.cpp yolah_features.cpp
    minmax_nnue_baseline_player.cpp minmax_nnue_dev_player.cpp endgame_solver.cpp search_table.cpp
    features_net_player.cpp
    alphazero_mcts.cpp alphazero_mcts_player.cpp alphazero_selfplay.cpp
)

# Add AlphaZero player if CUDA is enabled
if(ENABLE_CUDA)
    list(APPEND PLAYER_SOURCES alphazero_player.cpp)
endif()

add_library(player OBJECT ${PLAYER_SOURCES})
target_compile_options(player PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)
target_include_directories(player PUBLIC .)

include_directories(../misc ../game ../optimization ../nnue ../eigen ../data)

if(ENABLE_CUDA)
    include_directories(../alphazero)
    target_compile_definitions(player PRIVATE ENABLE_CUDA)
endif()

# The search debugger (test/search_trace_main.cpp, ui/search_debugger.html):
# the same player sources compiled with the trace hooks of
# minmax_nnue_dev_player.cpp switched on. Only built with
# `make search_trace` (the other programs keep the library above, without hooks).
add_library(player_trace OBJECT EXCLUDE_FROM_ALL ${PLAYER_SOURCES})
target_compile_options(player_trace PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)
target_compile_definitions(player_trace PRIVATE YOLAH_SEARCH_TRACE)
target_include_directories(player_trace PUBLIC .)
if(ENABLE_CUDA)
    target_compile_definitions(player_trace PRIVATE ENABLE_CUDA)
endif()
