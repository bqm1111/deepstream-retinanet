set(ALIGNMENT_ELEMENT gstnvfacealign)
project(${ALIGNMENT_ELEMENT} LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")
add_library(${ALIGNMENT_ELEMENT} SHARED
    ${CMAKE_CURRENT_LIST_DIR}/gstnvfacealign.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gstnvdsfacealign_allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/alignment_logic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/alignment_logic_kernel.cu
    ${CMAKE_CURRENT_LIST_DIR}/nvdsfacealign_property_parser.cpp
)
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC ${SRC_COMMON_DIR})

add_compile_definitions(DS_VERSION="6.0.0")
set_target_properties(${ALIGNMENT_ELEMENT} PROPERTIES PUBLIC_HEADER 
    ${CMAKE_CURRENT_LIST_DIR}/include/nvdsfacealign_interface.h
)

# cuda
target_compile_features(${ALIGNMENT_ELEMENT} PUBLIC cxx_std_17)
# set_target_properties(${ALIGNMENT_ELEMENT} PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(${ALIGNMENT_ELEMENT} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${ALIGNMENT_ELEMENT} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# cuda runtime
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC /usr/local/cuda/include)
target_link_libraries(${ALIGNMENT_ELEMENT} -L/usr/local/cuda/lib64 -lcuda -lcudart)

# deepstream
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${ALIGNMENT_ELEMENT} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${ALIGNMENT_ELEMENT} -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform)

# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${ALIGNMENT_ELEMENT} PkgConfig::GST_PKG)
target_compile_options(${ALIGNMENT_ELEMENT} PUBLIC ${GST_PKG_CFLAGS_OTHER})

# opencv
add_compile_definitions(WITH_OPENCV)
find_package( OpenCV 4.3.0 REQUIRED )
target_include_directories(${ALIGNMENT_ELEMENT} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ALIGNMENT_ELEMENT} ${OpenCV_LIBS})

