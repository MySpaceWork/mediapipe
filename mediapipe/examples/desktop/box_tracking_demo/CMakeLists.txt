cmake_minimum_required(VERSION 3.16)
project(box_tracking_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===========================================================================
# MediaPipe source root (four levels up from this CMakeLists.txt)
# ===========================================================================
get_filename_component(MEDIAPIPE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
message(STATUS "MediaPipe root: ${MEDIAPIPE_ROOT}")

# ===========================================================================
# Find required packages
# ===========================================================================
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d features2d highgui imgcodecs videoio)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

# Try to find absl via CMake config or pkg-config.
# If using system-installed abseil, find_package should work.
# Otherwise, we fall back to manually linking.
find_package(absl QUIET)
if (NOT absl_FOUND)
  message(STATUS "absl not found via find_package. Looking for system libs.")
  # Try pkg-config fallback
  find_package(PkgConfig)
  if (PkgConfig_FOUND)
    pkg_check_modules(ABSL REQUIRED
      absl_base
      absl_strings
      absl_str_format
      absl_synchronization
      absl_time
      absl_flat_hash_map
      absl_flat_hash_set
      absl_node_hash_map
      absl_node_hash_set
      absl_memory
      absl_algorithm
      absl_hash
      absl_container_common
    )
  endif()
endif()

# ===========================================================================
# Generate protobuf C++ sources
# ===========================================================================
# All .proto files needed by the tracking module.
set(TRACKING_PROTO_DIR "${MEDIAPIPE_ROOT}/mediapipe/util/tracking")
set(PROTO_FILES
  ${TRACKING_PROTO_DIR}/tone_models.proto
  ${TRACKING_PROTO_DIR}/tone_estimation.proto
  ${TRACKING_PROTO_DIR}/region_flow.proto
  ${TRACKING_PROTO_DIR}/motion_models.proto
  ${TRACKING_PROTO_DIR}/camera_motion.proto
  ${TRACKING_PROTO_DIR}/motion_estimation.proto
  ${TRACKING_PROTO_DIR}/motion_saliency.proto
  ${TRACKING_PROTO_DIR}/push_pull_filtering.proto
  ${TRACKING_PROTO_DIR}/region_flow_computation.proto
  ${TRACKING_PROTO_DIR}/motion_analysis.proto
  ${TRACKING_PROTO_DIR}/flow_packager.proto
  ${TRACKING_PROTO_DIR}/tracking.proto
  ${TRACKING_PROTO_DIR}/box_tracker.proto
  ${TRACKING_PROTO_DIR}/box_detector.proto
  ${TRACKING_PROTO_DIR}/tracked_detection_manager_config.proto
)

# Also need the framework proto(s) that some deps reference.
set(FRAMEWORK_PROTO_DIR "${MEDIAPIPE_ROOT}/mediapipe/framework")
list(APPEND PROTO_FILES
  ${FRAMEWORK_PROTO_DIR}/formats/rect.proto
)

# Output directory for generated proto C++ files.
set(PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

set(PROTO_SRCS "")
set(PROTO_HDRS "")

foreach(proto_file ${PROTO_FILES})
  # Get relative path from MEDIAPIPE_ROOT.
  file(RELATIVE_PATH proto_rel "${MEDIAPIPE_ROOT}" "${proto_file}")
  get_filename_component(proto_dir "${proto_rel}" DIRECTORY)
  get_filename_component(proto_name "${proto_file}" NAME_WE)

  set(gen_src "${PROTO_OUTPUT_DIR}/${proto_dir}/${proto_name}.pb.cc")
  set(gen_hdr "${PROTO_OUTPUT_DIR}/${proto_dir}/${proto_name}.pb.h")

  file(MAKE_DIRECTORY "${PROTO_OUTPUT_DIR}/${proto_dir}")

  add_custom_command(
    OUTPUT "${gen_src}" "${gen_hdr}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      --proto_path=${MEDIAPIPE_ROOT}
      --cpp_out=${PROTO_OUTPUT_DIR}
      ${proto_file}
    DEPENDS ${proto_file}
    COMMENT "Generating protobuf C++ for ${proto_rel}"
  )

  list(APPEND PROTO_SRCS "${gen_src}")
  list(APPEND PROTO_HDRS "${gen_hdr}")
endforeach()

# Proto library target.
add_library(tracking_protos STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(tracking_protos PUBLIC
  ${PROTO_OUTPUT_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(tracking_protos PUBLIC ${Protobuf_LIBRARIES})

# ===========================================================================
# MediaPipe framework deps library (minimal subset needed by tracking)
# ===========================================================================
set(FRAMEWORK_DEPS_DIR "${MEDIAPIPE_ROOT}/mediapipe/framework/deps")

add_library(mediapipe_deps STATIC
  ${FRAMEWORK_DEPS_DIR}/threadpool_std_thread_impl.cc
  ${FRAMEWORK_DEPS_DIR}/registration.cc
  ${FRAMEWORK_DEPS_DIR}/status.cc
  ${FRAMEWORK_DEPS_DIR}/status_builder.cc
  ${FRAMEWORK_DEPS_DIR}/statusor.cc
  ${FRAMEWORK_DEPS_DIR}/ret_check.cc
  ${FRAMEWORK_DEPS_DIR}/file_helpers.cc
  ${FRAMEWORK_DEPS_DIR}/file_path.cc
)
target_include_directories(mediapipe_deps PUBLIC
  ${MEDIAPIPE_ROOT}
  ${PROTO_OUTPUT_DIR}
)
target_link_libraries(mediapipe_deps PUBLIC
  glog::glog
  ${Protobuf_LIBRARIES}
)

if (absl_FOUND)
  target_link_libraries(mediapipe_deps PUBLIC
    absl::base
    absl::strings
    absl::str_format
    absl::synchronization
    absl::time
    absl::flat_hash_map
    absl::flat_hash_set
    absl::node_hash_map
    absl::node_hash_set
    absl::memory
    absl::status
    absl::statusor
    absl::algorithm_container
    absl::hash
  )
else()
  target_include_directories(mediapipe_deps PUBLIC ${ABSL_INCLUDE_DIRS})
  target_link_libraries(mediapipe_deps PUBLIC ${ABSL_LIBRARIES})
endif()

# ===========================================================================
# MediaPipe tracking library
# ===========================================================================
set(TRACKING_SRC_DIR "${MEDIAPIPE_ROOT}/mediapipe/util/tracking")

add_library(mediapipe_tracking STATIC
  ${TRACKING_SRC_DIR}/tracking.cc
  ${TRACKING_SRC_DIR}/box_tracker.cc
  ${TRACKING_SRC_DIR}/box_detector.cc
  ${TRACKING_SRC_DIR}/region_flow_computation.cc
  ${TRACKING_SRC_DIR}/region_flow.cc
  ${TRACKING_SRC_DIR}/region_flow_visualization.cc
  ${TRACKING_SRC_DIR}/motion_estimation.cc
  ${TRACKING_SRC_DIR}/motion_analysis.cc
  ${TRACKING_SRC_DIR}/motion_models.cc
  ${TRACKING_SRC_DIR}/motion_models_cv.cc
  ${TRACKING_SRC_DIR}/motion_saliency.cc
  ${TRACKING_SRC_DIR}/camera_motion.cc
  ${TRACKING_SRC_DIR}/flow_packager.cc
  ${TRACKING_SRC_DIR}/tone_estimation.cc
  ${TRACKING_SRC_DIR}/tone_models.cc
  ${TRACKING_SRC_DIR}/image_util.cc
  ${TRACKING_SRC_DIR}/measure_time.cc
  ${TRACKING_SRC_DIR}/parallel_invoker.cc
  ${TRACKING_SRC_DIR}/streaming_buffer.cc
  ${TRACKING_SRC_DIR}/tracking_visualization_utilities.cc
  ${TRACKING_SRC_DIR}/tracked_detection.cc
  ${TRACKING_SRC_DIR}/tracked_detection_manager.cc
)

target_include_directories(mediapipe_tracking PUBLIC
  ${MEDIAPIPE_ROOT}
  ${PROTO_OUTPUT_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(mediapipe_tracking PUBLIC
  tracking_protos
  mediapipe_deps
  ${OpenCV_LIBS}
  Eigen3::Eigen
  glog::glog
)

if (absl_FOUND)
  target_link_libraries(mediapipe_tracking PUBLIC
    absl::base
    absl::strings
    absl::str_format
    absl::synchronization
    absl::time
    absl::flat_hash_map
    absl::flat_hash_set
    absl::node_hash_map
    absl::node_hash_set
    absl::memory
    absl::algorithm_container
    absl::hash
  )
else()
  target_link_libraries(mediapipe_tracking PUBLIC ${ABSL_LIBRARIES})
endif()

# ===========================================================================
# Demo executable
# ===========================================================================
add_executable(box_tracking_demo
  ${CMAKE_CURRENT_SOURCE_DIR}/box_tracking_demo.cc
)

target_include_directories(box_tracking_demo PRIVATE
  ${MEDIAPIPE_ROOT}
  ${PROTO_OUTPUT_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(box_tracking_demo PRIVATE
  mediapipe_tracking
  ${OpenCV_LIBS}
  glog::glog
)
