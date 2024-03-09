add_library(player OBJECT 
    random_player.cpp monte_carlo_player.cpp client_player.cpp
)
target_include_directories(player PUBLIC .)

include_directories(../misc ../game)
