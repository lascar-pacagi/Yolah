add_library(server OBJECT 
    server.cpp
)
target_include_directories(server PUBLIC .)

include_directories(../misc ../game)
