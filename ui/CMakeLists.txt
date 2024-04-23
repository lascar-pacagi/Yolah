add_library(ui OBJECT 
    ascii_observer.cpp html_observer.cpp
)
target_include_directories(ui PUBLIC .)

include_directories(../misc ../game)