cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set (KAFKA_PROTOCOL nvds_kafka_proto)
project(${KAFKA_PROTOCOL} LANGUAGES CXX CUDA)

file(GLOB KAFKA_PROTOCOL_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(${KAFKA_PROTOCOL} SHARED ${KAFKA_PROTOCOL_SOURCES}
)
# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
target_include_directories(${KAFKA_PROTOCOL} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${KAFKA_PROTOCOL} PkgConfig::GST_PKG)
target_compile_options(${KAFKA_PROTOCOL} PUBLIC ${GST_PKG_CFLAGS_OTHER})

target_include_directories(${KAFKA_PROTOCOL} PUBLIC /usr/local/include/librdkafka)
# deepstream
target_include_directories(${KAFKA_PROTOCOL} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${KAFKA_PROTOCOL} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${KAFKA_PROTOCOL} -lrdkafka -ljansson -lnvds_logger -lpthread -lcrypto)