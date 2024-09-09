add_library(player OBJECT 
    random_player.cpp monte_carlo_player.cpp client_player.cpp 
    MCTS_player.cpp MCTS_mem_player.cpp human_player.cpp
    heuristic.cpp basic_minmax_player.cpp transposition_table.cpp minmax_player_v1.cpp
    heuristic_weights_learner.cpp player.cpp tt.cpp minmax_player_v2.cpp minmax_player_v3.cpp
    minmax_player_v4.cpp minmax_player_v5.cpp minmax_player_v6.cpp minmax_player_v7.cpp
    minmax_player_v8.cpp minmax_player_v9.cpp minmax_player_v10.cpp minmax_player_v11.cpp
    minmax_player.cpp
)
target_include_directories(player PUBLIC .)

include_directories(../misc ../game ../optimization)
