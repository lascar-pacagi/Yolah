add_library(nnue OBJECT nnue.cpp nnue_quantized.cpp)

target_compile_options(nnue PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc)
