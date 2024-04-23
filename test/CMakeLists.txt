add_library(test OBJECT 
    random_game.cpp human_player.cpp play.cpp
)
target_include_directories(test PUBLIC .)

include_directories(../misc ../game ../player ../ui)
