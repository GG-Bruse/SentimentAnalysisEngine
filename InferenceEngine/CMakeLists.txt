cmake_minimum_required(VERSION 3.10)
project(InferenceEngine CXX)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# 定义库版本相关变量
set(LIB_MAJOR_VERSION "1")
set(LIB_MINOR_VERSION "0")
set(LIB_PATCH_VERSION "0")
set(LIB_VERSION_STRING "${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${LIB_PATCH_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# local
file(GLOB_RECURSE SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIBS ${LIBS} -L${CMAKE_CURRENT_SOURCE_DIR}/lib/)

# cuda
include_directories(/usr/local/cuda/include/)
# set(LIBS ${LIBS} -L/usr/local/cuda/lib64/)

# TensorRT
include_directories(/data3/a100_project/TensorRT-8.5.2.2/include/)

set(LIBS ${LIBS} -lpthread -liniparser -lnvinfer_plugin -lnvinfer -lutf8proc -licuuc -licuio -licudata -licui18n)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

add_library(emotion_classification_engine SHARED ${SOURCE})
set_target_properties(emotion_classification_engine PROPERTIES
    COMPILE_FLAGS "-Wno-deprecated-declarations"
)
target_link_libraries(emotion_classification_engine ${LIBS})

