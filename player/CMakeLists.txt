add_library(player OBJECT 
    random_player.cpp monte_carlo_player.cpp client_player.cpp 
    MCTS_player.cpp MCTS_mem_player.cpp human_player.cpp
    heuristic.cpp basic_minmax_player.cpp transposition_table.cpp
    heuristic_weights_learner.cpp player.cpp
    minmax_player.cpp minmax_nnue_player.cpp minmax_nnue_quantized_player.cpp
)
target_include_directories(player PUBLIC .)

include_directories(../misc ../game ../optimization ../nnue ../eigen)
