add_library(optimization OBJECT cem.cpp nelder_mead.cpp)
target_include_directories(optimization PUBLIC .)

include_directories(../misc ../game ../player)
