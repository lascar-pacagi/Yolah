add_library(nnue OBJECT nnue.cpp nnue_quantized.cpp)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc)
