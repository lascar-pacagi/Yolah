add_library(nnue OBJECT nnue.cpp nnue_quantized.cpp logic_net.cpp logic_net_learning.cpp compare_models.cpp)

target_compile_options(nnue PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc ../data ../player)
