cmake_minimum_required(VERSION 3.20.0)
project(Yolah CXX)
set (CMAKE_CXX_STANDARD 20)

add_executable(Yolah main.cpp)

add_subdirectory(game)
target_link_libraries(Yolah PRIVATE game)
add_subdirectory(misc)
target_link_libraries(Yolah PRIVATE misc)
add_subdirectory(test)
target_link_libraries(Yolah PRIVATE test)
