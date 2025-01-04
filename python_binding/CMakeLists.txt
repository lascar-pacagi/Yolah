cmake_minimum_required(VERSION 3.20.0)
project(yolah_bind CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../misc ../game ../player)

find_package(pybind11 REQUIRED)
pybind11_add_module(yolah_bind yolah_bind.cpp ../game/game.cpp ../game/zobrist.cpp ../game/magic.cpp ../player/heuristic.cpp)
