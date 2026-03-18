// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Box Tracking Demo using MediaPipe's tracking module.
//
// Usage:
//   ./box_tracking_demo <video_file>
//
// Controls:
//   - Draw a rectangle with the mouse on the first frame to select a target.
//   - Press ENTER/SPACE to start tracking.
//   - Press 'r' to reset and draw a new box.
//   - Press 'q' or ESC to quit.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/flow_packager.h"
#include "mediapipe/util/tracking/motion_analysis.h"
#include "mediapipe/util/tracking/region_flow_computation.h"
#include "mediapipe/util/tracking/tracking.h"

namespace {

// Global state for mouse callback.
struct MouseState {
  cv::Point start;
  cv::Point end;
  bool drawing = false;
  bool done = false;
};

MouseState g_mouse;

void OnMouse(int event, int x, int y, int /*flags*/, void* /*userdata*/) {
  switch (event) {
    case cv::EVENT_LBUTTONDOWN:
      g_mouse.start = cv::Point(x, y);
      g_mouse.end = cv::Point(x, y);
      g_mouse.drawing = true;
      g_mouse.done = false;
      break;
    case cv::EVENT_MOUSEMOVE:
      if (g_mouse.drawing) {
        g_mouse.end = cv::Point(x, y);
      }
      break;
    case cv::EVENT_LBUTTONUP:
      g_mouse.end = cv::Point(x, y);
      g_mouse.drawing = false;
      g_mouse.done = true;
      break;
  }
}

// Converts a pixel-space cv::Rect to a normalized TimedBox in [0, 1].
mediapipe::TimedBox RectToTimedBox(const cv::Rect& rect, int frame_width,
                                   int frame_height, int64_t time_msec) {
  mediapipe::TimedBox box;
  box.left = static_cast<float>(rect.x) / frame_width;
  box.top = static_cast<float>(rect.y) / frame_height;
  box.right = static_cast<float>(rect.x + rect.width) / frame_width;
  box.bottom = static_cast<float>(rect.y + rect.height) / frame_height;
  box.time_msec = time_msec;
  return box;
}

// Converts a normalized TimedBox to a pixel-space cv::Rect.
cv::Rect TimedBoxToRect(const mediapipe::TimedBox& box, int frame_width,
                        int frame_height) {
  int x = static_cast<int>(box.left * frame_width);
  int y = static_cast<int>(box.top * frame_height);
  int w = static_cast<int>((box.right - box.left) * frame_width);
  int h = static_cast<int>((box.bottom - box.top) * frame_height);
  return cv::Rect(x, y, w, h);
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
    return 1;
  }

  const std::string video_path = argv[1];
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
    return 1;
  }

  const int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) {
    std::cerr << "Error: Cannot read FPS from video." << std::endl;
    return 1;
  }
  const double frame_duration_ms = 1000.0 / fps;

  std::cout << "Video: " << frame_width << "x" << frame_height
            << " @ " << fps << " fps" << std::endl;

  // ---------------------------------------------------------------
  // Phase 1: Read first frame and let user draw a bounding box.
  // ---------------------------------------------------------------
  cv::Mat first_frame;
  cap >> first_frame;
  if (first_frame.empty()) {
    std::cerr << "Error: Cannot read first frame." << std::endl;
    return 1;
  }

  const std::string window_name = "Box Tracking Demo";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(window_name, OnMouse, nullptr);

  std::cout << "Draw a rectangle on the object you want to track, "
            << "then press ENTER or SPACE." << std::endl;

  cv::Rect selected_rect;
  while (true) {
    cv::Mat display = first_frame.clone();
    if (g_mouse.drawing || g_mouse.done) {
      cv::rectangle(display, g_mouse.start, g_mouse.end,
                    cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(window_name, display);
    int key = cv::waitKey(30) & 0xFF;
    if (key == 27 || key == 'q') {  // ESC or q
      std::cout << "Quit." << std::endl;
      return 0;
    }
    if ((key == 13 || key == 32) && g_mouse.done) {  // ENTER or SPACE
      selected_rect = cv::Rect(g_mouse.start, g_mouse.end);
      if (selected_rect.width > 0 && selected_rect.height > 0) {
        break;
      }
      std::cout << "Invalid rectangle. Please draw again." << std::endl;
      g_mouse.done = false;
    }
  }

  std::cout << "Selected box: " << selected_rect << std::endl;

  // ---------------------------------------------------------------
  // Phase 2: Pre-process the entire video to build TrackingData.
  //
  // Pipeline: RegionFlowComputation -> MotionEstimation -> FlowPackager
  //           -> TrackingDataChunk
  // ---------------------------------------------------------------
  std::cout << "Processing video frames for optical flow..." << std::endl;

  // Reset video to beginning.
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  // Configure RegionFlowComputation.
  mediapipe::RegionFlowComputationOptions flow_options;
  flow_options.set_image_format(
      mediapipe::RegionFlowComputationOptions::FORMAT_RGB);
  auto* tracking_opts = flow_options.mutable_tracking_options();
  tracking_opts->set_klt_tracker_implementation(
      mediapipe::TrackingOptions::KLT_OPENCV);
  // Use long tracks for better feature continuity.
  tracking_opts->set_long_tracks(true);
  tracking_opts->set_internal_tracking_direction(
      mediapipe::TrackingOptions::FORWARD);

  mediapipe::RegionFlowComputation flow_computation(flow_options, frame_width,
                                                    frame_height);

  // Configure MotionEstimation.
  mediapipe::MotionEstimationOptions motion_options;
  mediapipe::MotionEstimation motion_estimation(motion_options, frame_width,
                                                frame_height);

  // Configure FlowPackager.
  mediapipe::FlowPackagerOptions packager_options;
  mediapipe::FlowPackager flow_packager(packager_options);

  // Build a single TrackingDataChunk for the whole video.
  mediapipe::TrackingDataChunk tracking_chunk;
  std::vector<cv::Mat> frames;

  int frame_idx = 0;
  cv::Mat frame;
  while (cap.read(frame)) {
    frames.push_back(frame.clone());
    int64_t timestamp_usec =
        static_cast<int64_t>(frame_idx * frame_duration_ms * 1000.0);

    // Convert to RGB if needed (OpenCV reads BGR).
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    flow_computation.AddImage(rgb_frame, timestamp_usec);

    // Retrieve region flow feature list (first frame returns empty flow).
    std::unique_ptr<mediapipe::RegionFlowFeatureList> feature_list(
        flow_computation.RetrieveRegionFlowFeatureList(
            false,  // no feature descriptor
            false,  // no match descriptor
            nullptr, nullptr));

    if (feature_list == nullptr) {
      // First frame -- add empty tracking data.
      auto* item = tracking_chunk.add_item();
      item->set_timestamp_usec(timestamp_usec);
      ++frame_idx;
      continue;
    }

    // Estimate camera motion.
    std::vector<mediapipe::RegionFlowFeatureList*> feature_ptrs = {
        feature_list.get()};
    std::vector<mediapipe::CameraMotion> camera_motions;
    motion_estimation.EstimateMotionsParallel(false, &feature_ptrs,
                                             &camera_motions);

    // Pack flow into TrackingData.
    mediapipe::TrackingData tracking_data;
    const mediapipe::CameraMotion* motion_ptr =
        camera_motions.empty() ? nullptr : &camera_motions[0];
    flow_packager.PackFlow(*feature_list, motion_ptr, &tracking_data);

    // Add to chunk.
    auto* item = tracking_chunk.add_item();
    item->set_timestamp_usec(timestamp_usec);
    *item->mutable_tracking_data() = tracking_data;

    ++frame_idx;
    if (frame_idx % 50 == 0) {
      std::cout << "  Processed " << frame_idx << " frames..." << std::endl;
    }
  }

  tracking_chunk.set_first_chunk(true);
  tracking_chunk.set_last_chunk(true);

  const int total_frames = frames.size();
  std::cout << "Done. Total frames: " << total_frames << std::endl;

  if (total_frames < 2) {
    std::cerr << "Error: Video too short." << std::endl;
    return 1;
  }

  // ---------------------------------------------------------------
  // Phase 3: Run BoxTracker using the pre-computed TrackingData.
  // ---------------------------------------------------------------
  std::cout << "Running box tracker..." << std::endl;

  mediapipe::BoxTrackerOptions tracker_options;
  tracker_options.set_num_tracking_workers(1);
  auto* track_step = tracker_options.mutable_track_step_options();
  track_step->set_tracking_degrees(
      mediapipe::TrackStepOptions::TRACKING_DEGREE_OBJECT_ROTATION_SCALE);

  std::vector<const mediapipe::TrackingDataChunk*> chunks = {&tracking_chunk};
  mediapipe::BoxTracker box_tracker(chunks, /*copy_data=*/false,
                                    tracker_options);

  // Create initial box from the user's selection.
  const int64_t start_time_msec = 0;
  mediapipe::TimedBox initial_box =
      RectToTimedBox(selected_rect, frame_width, frame_height, start_time_msec);

  const int kBoxId = 1;
  box_tracker.NewBoxTrack(initial_box, kBoxId);

  // Wait for tracking to complete.
  box_tracker.WaitForAllOngoingTracks();

  std::cout << "Tracking complete. Playing back results..." << std::endl;

  // ---------------------------------------------------------------
  // Phase 4: Visualize tracking results.
  // ---------------------------------------------------------------
  for (int f = 0; f < total_frames; ++f) {
    int64_t time_msec =
        static_cast<int64_t>(f * frame_duration_ms);

    mediapipe::TimedBox result_box;
    bool ok = box_tracker.GetTimedPosition(kBoxId, time_msec, &result_box);

    cv::Mat display = frames[f].clone();
    if (ok && result_box.confidence > 0.1f) {
      cv::Rect box_rect = TimedBoxToRect(result_box, frame_width, frame_height);

      // Clamp to frame bounds.
      box_rect &= cv::Rect(0, 0, frame_width, frame_height);

      // Color by confidence: green (high) -> red (low).
      float conf = std::min(1.0f, std::max(0.0f, result_box.confidence));
      cv::Scalar color(0, static_cast<int>(255 * conf),
                       static_cast<int>(255 * (1.0f - conf)));
      cv::rectangle(display, box_rect, color, 2);

      // Draw confidence text.
      char conf_text[32];
      snprintf(conf_text, sizeof(conf_text), "conf: %.2f", conf);
      cv::putText(display, conf_text, box_rect.tl() + cv::Point(0, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    } else {
      cv::putText(display, "Tracking lost", cv::Point(10, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    // Frame counter.
    char frame_text[64];
    snprintf(frame_text, sizeof(frame_text), "Frame: %d / %d", f + 1,
             total_frames);
    cv::putText(display, frame_text, cv::Point(10, frame_height - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::imshow(window_name, display);

    int key = cv::waitKey(static_cast<int>(frame_duration_ms)) & 0xFF;
    if (key == 27 || key == 'q') {
      break;
    }
    if (key == 'r') {
      // Reset -- user wants to pick a new box. For simplicity, just quit.
      std::cout << "Reset requested. Rerun to pick a new box." << std::endl;
      break;
    }
  }

  cv::destroyAllWindows();
  std::cout << "Done." << std::endl;
  return 0;
}
