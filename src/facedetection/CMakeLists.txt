cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set (TARGET_DETECTION nvds_facedetection)
project(${TARGET_DETECTION} LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")
file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/nvds_retinaface.cpp
                ${CMAKE_CURRENT_LIST_DIR}/decode_plugin_kernel.cu)

add_library(${TARGET_DETECTION} SHARED ${SOURCES})
target_include_directories(${TARGET_DETECTION} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/plugin)

# cuda
target_compile_features(${TARGET_DETECTION} PUBLIC cxx_std_17)
# set_target_properties(${TARGET_DETECTION} PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(${TARGET_DETECTION} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${TARGET_DETECTION} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# cuda runtime
target_include_directories(${TARGET_DETECTION} PUBLIC /usr/local/cuda/include)
target_link_libraries(${TARGET_DETECTION} -L/usr/local/cuda/lib64 -lcuda -lcudart)

# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
target_include_directories(${TARGET_DETECTION} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${TARGET_DETECTION} PkgConfig::GST_PKG)
target_compile_options(${TARGET_DETECTION} PUBLIC ${GST_PKG_CFLAGS_OTHER})

# deepstream
target_include_directories(${TARGET_DETECTION} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${TARGET_DETECTION} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${TARGET_DETECTION} -lnvdsgst_meta -lnvds_meta -lnvinfer)