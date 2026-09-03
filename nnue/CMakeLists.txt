set(NNUE_SOURCES nnue.cpp nnue_quantized.cpp logic_net.cpp logic_net_learning.cpp compare_models.cpp ffnn.cpp ffnn_float.cpp
    nn_evaluator.cpp resnet.cpp batched_evaluator.cpp)
# libtorch backend of the value/policy ResNet (see resnet_torch.h)
if(ENABLE_TORCH)
    list(APPEND NNUE_SOURCES resnet_torch.cpp)
endif()
add_library(nnue OBJECT ${NNUE_SOURCES})
if(ENABLE_TORCH)
    target_compile_definitions(nnue PUBLIC ENABLE_TORCH)
    target_include_directories(nnue PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(nnue PUBLIC ${TORCH_LIBRARIES})
endif()

target_compile_options(nnue PRIVATE -Wall -march=native -mavx2 -ffast-math -funroll-loops -fopenmp)

target_include_directories(nnue PUBLIC .)

include_directories(../game ../vcl ../eigen ../misc ../data ../player)
