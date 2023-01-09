set (TARGET_DEEPSORT deepsort)
add_library(deepsort SHARED 
    ${CMAKE_CURRENT_LIST_DIR}/adapter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/boostmatrixadapter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hungarianoper.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kalmanfilter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linear_assignment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/munkres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nn_matching.cpp
    ${CMAKE_CURRENT_LIST_DIR}/track.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/errmsg.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED )

target_include_directories(${TARGET_DEEPSORT} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(${TARGET_DEEPSORT} PUBLIC ${EIGEN3_INCLUDE_DIR})