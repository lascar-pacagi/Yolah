add_library(ui OBJECT 
    ascii_observer.cpp ws_observer.cpp do_nothing_observer.cpp
)
target_include_directories(ui PUBLIC .)

include_directories(../misc ../game)