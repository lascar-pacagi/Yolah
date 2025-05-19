add_library(nnue OBJECT nnue.cpp nnue_1024x64x32x3.cpp nnue_1024x64x32x3_2.cpp nnue_quantized.cpp nnue_q1024x64x32x3_8bits.cpp nnue_q768x160x64x3.cpp)

target_compile_options(nnue PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc)
