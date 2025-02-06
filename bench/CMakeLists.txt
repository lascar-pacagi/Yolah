cmake_minimum_required(VERSION 3.20.0)
project(Benchmarks CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.71 COMPONENTS program_options REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

include_directories(../misc ../game ../nnue ../vcl ../eigen)
add_executable(Benchmarks main.cpp ../game/game.cpp ../game/zobrist.cpp ../game/magic.cpp)
target_compile_options(Benchmarks PRIVATE -Wall -O3 -march=native -ffast-math -funroll-loops)

target_include_directories(Benchmarks PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(Benchmarks PRIVATE ${Boost_LIBRARIES})
target_link_libraries(Benchmarks PRIVATE TBB::tbb)
target_link_libraries(Benchmarks PRIVATE Threads::Threads)
target_link_libraries(Benchmarks PRIVATE benchmark::benchmark)
