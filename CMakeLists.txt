cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.5/bin/nvcc)

project(experiment LANGUAGES CXX CUDA)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/*.cu
                    ${PROJECT_SOURCE_DIR}/samples/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/samples
                    ${PROJECT_SOURCE_DIR}/utils)
add_executable(${PROJECT_NAME} ${SOURCES})

# cuda
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -fno-strict-aliasing --ptxas-options=-v --generate-line-info")
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
target_include_directories(${PROJECT_NAME} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PkgConfig::GST_PKG)
target_compile_options(${PROJECT_NAME} PUBLIC ${GST_PKG_CFLAGS_OTHER})

# deepstream
target_include_directories(${PROJECT_NAME} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${PROJECT_NAME} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${PROJECT_NAME} -lnvdsgst_meta -lnvds_meta -lnvinfer)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs)

# Subdirectory

add_subdirectory(src)