add_library(nnue OBJECT nnue.cpp)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc)
