cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set (TARGET_YOLOV5 nvds_yolov5)
project(${TARGET_YOLOV5} LANGUAGES CXX CUDA)

file(GLOB YOLOV5_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/*.cu
                        ${CMAKE_CURRENT_LIST_DIR}/layers/*.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/layers/*.cu)
add_library(${TARGET_YOLOV5} SHARED ${YOLOV5_SOURCES})

# cuda
target_compile_features(${TARGET_YOLOV5} PUBLIC cxx_std_11)
set_target_properties(${TARGET_YOLOV5} PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(${TARGET_YOLOV5} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${TARGET_YOLOV5} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# cuda runtime
target_include_directories(${TARGET_YOLOV5} PUBLIC /usr/local/cuda/include)
target_link_libraries(${TARGET_YOLOV5} -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas)

# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
target_include_directories(${TARGET_YOLOV5} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${TARGET_YOLOV5} PkgConfig::GST_PKG)
target_compile_options(${TARGET_YOLOV5} PUBLIC ${GST_PKG_CFLAGS_OTHER})

# deepstream
target_include_directories(${TARGET_YOLOV5} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${TARGET_YOLOV5} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${TARGET_YOLOV5} -lnvdsgst_meta -lnvds_meta -lnvinfer -lstdc++fs -lnvinfer_plugin -lnvparsers )