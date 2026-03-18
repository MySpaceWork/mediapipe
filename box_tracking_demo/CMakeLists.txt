cmake_minimum_required(VERSION 3.10)
project(box_tracking_demo CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(STATIC_BUILD "Build a fully static executable" OFF)

# ===========================================================================
# Find OpenCV -- the only external dependency.
# ===========================================================================
if(STATIC_BUILD)
  # For static build, prefer static libraries.
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  set(OpenCV_STATIC ON)
  set(BUILD_SHARED_LIBS OFF)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video highgui videoio calib3d)
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

if(STATIC_BUILD)
  # Link C++ runtime statically and add system libs needed for static OpenCV.
  target_link_options(box_tracking_demo PRIVATE -static-libgcc -static-libstdc++)
  # Common system libraries that static OpenCV needs.
  target_link_libraries(box_tracking_demo PRIVATE
    pthread dl m z
  )
endif()
