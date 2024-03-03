add_library(game OBJECT 
    game.cpp magic.cpp move.cpp
)
target_include_directories(game PUBLIC .)

include_directories(../misc)
