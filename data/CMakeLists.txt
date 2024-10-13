add_library(data OBJECT 
    generate_games.cpp analyze_games.cpp
)
target_include_directories(data PUBLIC .)

include_directories(../game ../player ../misc)
