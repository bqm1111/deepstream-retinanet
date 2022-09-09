cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set (TARGET_FACEFEATURE nvds_parsenone)
project(${TARGET_FACEFEATURE} LANGUAGES CXX)

add_library(${TARGET_FACEFEATURE} SHARED
    ${CMAKE_CURRENT_LIST_DIR}/nvds_facefeature.cpp
)

# deepstream
target_include_directories(${TARGET_FACEFEATURE} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${TARGET_FACEFEATURE} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${TARGET_FACEFEATURE} -lnvdsgst_meta -lnvds_meta -lnvinfer)