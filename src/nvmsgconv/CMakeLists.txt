cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set (CUSTOM_MSGCONV nvmsgconv)
project(${CUSTOM_MSGCONV} LANGUAGES CXX CUDA)

file(GLOB CUSTOM_MSGCONV_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                                ${CMAKE_CURRENT_LIST_DIR}/custom_schema/*.cpp)
add_library(${CUSTOM_MSGCONV} SHARED ${CUSTOM_MSGCONV_SOURCES}
)

target_include_directories(${CUSTOM_MSGCONV} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/custom_schema)

pkg_check_modules(JsonGlib REQUIRED IMPORTED_TARGET json-glib-1.0)
pkg_check_modules(Uuid REQUIRED IMPORTED_TARGET uuid)

# gstreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST_PKG REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
target_include_directories(${CUSTOM_MSGCONV} PUBLIC ${GST_PKG_INCLUDE_DIRS})
target_link_libraries(${CUSTOM_MSGCONV} PkgConfig::GST_PKG)
target_compile_options(${CUSTOM_MSGCONV} PUBLIC ${GST_PKG_CFLAGS_OTHER})

target_include_directories(${CUSTOM_MSGCONV} PUBLIC /usr/local/include/librdkafka ${Uuid_INCLUDE_DIRS} 
                                            ${JsonGlib_INCLUDE_DIRS})
# deepstream
target_include_directories(${CUSTOM_MSGCONV} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${CUSTOM_MSGCONV} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${CUSTOM_MSGCONV} -lrdkafka -ljansson -lnvds_logger -lpthread -lcrypto
                                        ${JsonGlib_LIBRARIES} ${Uuid_LIBRARIES})