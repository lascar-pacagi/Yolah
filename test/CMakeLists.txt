add_library(test OBJECT 
    random_game.cpp
)
target_include_directories(test PUBLIC .)

include_directories(../misc ../game)
