add_library(test OBJECT 
    random_game.cpp human_player_test.cpp play.cpp cem_test.cpp
)
target_include_directories(test PUBLIC .)

include_directories(../misc ../game ../player ../ui ../ga)
