add_library(ga OBJECT cem.cpp)
target_include_directories(ga PUBLIC .)

include_directories(../misc ../game ../player)
