set(PLAYER_SOURCES
    random_player.cpp monte_carlo_player.cpp client_player.cpp
    MCTS_mem_nn_player.cpp human_player.cpp
    heuristic.cpp basic_minmax_player.cpp transposition_table.cpp
    heuristic_weights_learner.cpp player.cpp
    minmax_player.cpp minmax_player2.cpp minmax_nnue_player.cpp minmax_nnue_quantized_player.cpp
)

# Add AlphaZero player if CUDA is enabled
if(ENABLE_CUDA)
    list(APPEND PLAYER_SOURCES alphazero_player.cpp)
endif()

add_library(player OBJECT ${PLAYER_SOURCES})
target_include_directories(player PUBLIC .)

include_directories(../misc ../game ../optimization ../nnue ../eigen)

if(ENABLE_CUDA)
    include_directories(../alphazero)
    target_compile_definitions(player PRIVATE ENABLE_CUDA)
endif()
