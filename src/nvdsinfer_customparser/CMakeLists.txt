set (NVDS_CUSTOMPARSER nvds_infercustomparser)
add_library(${NVDS_CUSTOMPARSER} SHARED 
    ${CMAKE_CURRENT_LIST_DIR}/nvdsinfer_custombboxparser.cpp
)

# deepstream
target_include_directories(${NVDS_CUSTOMPARSER} PUBLIC /opt/nvidia/deepstream/deepstream/sources/includes)
target_link_directories(${NVDS_CUSTOMPARSER} PUBLIC /opt/nvidia/deepstream/deepstream/lib)
target_link_libraries(${NVDS_CUSTOMPARSER} -lnvdsgst_meta -lnvds_meta)