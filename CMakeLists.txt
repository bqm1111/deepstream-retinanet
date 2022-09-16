cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.5/bin/nvcc)

project(experiment LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -pthread")
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/*.cu
                    ${PROJECT_SOURCE_DIR}/common/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/pipeline/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/deepsort/*.cpp
                    ${PROJECT_SOURCE_DIR}/common/QDTConfig/*.cpp
                    ${PROJECT_SOURCE_DIR}/common/QDTLog/*.cpp
                    )
                    
include_directories(
${PROJECT_SOURCE_DIR}/utils
${PROJECT_SOURCE_DIR}/common
${PROJECT_SOURCE_DIR}/common/QDTConfig
${PROJECT_SOURCE_DIR}/common/QDTLog
${PROJECT_SOURCE_DIR}/src/pipeline
${PROJECT_SOURCE_DIR}/src/deepsort
${PROJECT_SOURCE_DIR}/src/mot
)
add_executable(${PROJECT_NAME} ${SOURCES})

# cuda
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 json-glib/json-glib.h: No such file or directoryCACHE STRING "CUDA architectures")
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
target_link_libraries(${PROJECT_NAME} -lnvdsgst_meta -lnvds_meta -lnvinfer -lnvinfer_plugin -lnvds_inferutils -lnvds_utils -lm -lnvds_infercustomparser)
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs -lcurl)

# Json-glib
pkg_check_modules(JSON-GLIB REQUIRED json-glib-1.0)
include_directories(${JSON-GLIB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${JSON-GLIB_LIBRARIES})

# Eigen 
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})

# Subdirectory
add_subdirectory(src)