cmake_minimum_required(VERSION 3.10)
project(box_tracking_demo CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===========================================================================
# Find OpenCV -- the only external dependency.
# ===========================================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc video highgui videoio calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")

# ===========================================================================
# Tracking library (standalone, no mediapipe dependency)
# ===========================================================================
add_library(box_tracker_lib STATIC
  src/box_tracker.cc
)

target_include_directories(box_tracker_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(box_tracker_lib PUBLIC
  ${OpenCV_LIBS}
)

# ===========================================================================
# Demo executable
# ===========================================================================
add_executable(box_tracking_demo
  src/main.cc
)

target_include_directories(box_tracking_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(box_tracking_demo PRIVATE
  box_tracker_lib
  ${OpenCV_LIBS}
)
