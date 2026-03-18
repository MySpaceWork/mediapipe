// Box Tracking Demo -- Standalone application.
//
// Usage:
//   ./box_tracking_demo <video_file>
//
// Controls:
//   - Draw a rectangle with the mouse on the first frame.
//   - Press ENTER or SPACE to start tracking.
//   - Press 'r' to reset and draw a new box.
//   - Press 'q' or ESC to quit.
//   - Press 'p' to pause/resume.

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "box_tracker.h"

namespace {

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

}  // namespace

int main(int argc, char** argv) {
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
  const int total_frames =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int delay_ms = (fps > 0) ? static_cast<int>(1000.0 / fps) : 33;

  std::cout << "Video: " << frame_width << "x" << frame_height << " @ " << fps
            << " fps, " << total_frames << " frames" << std::endl;

  // Read first frame.
  cv::Mat first_frame;
  cap >> first_frame;
  if (first_frame.empty()) {
    std::cerr << "Error: Cannot read first frame." << std::endl;
    return 1;
  }

  const std::string window_name = "Box Tracking Demo (MediaPipe Algorithm)";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(window_name, OnMouse, nullptr);

  // ---------------------------------------------------------------
  // Phase 1: Let user draw a bounding box on the first frame.
  // ---------------------------------------------------------------
  std::cout << "Draw a rectangle around the target, then press ENTER/SPACE."
            << std::endl;

  cv::Rect selected_rect;
  while (true) {
    cv::Mat display = first_frame.clone();

    // Draw instruction text.
    cv::putText(display,
                "Draw a box around the target, press ENTER to confirm",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);

    if (g_mouse.drawing || g_mouse.done) {
      cv::rectangle(display, g_mouse.start, g_mouse.end,
                    cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(window_name, display);

    int key = cv::waitKey(30) & 0xFF;
    if (key == 27 || key == 'q') {
      std::cout << "Quit." << std::endl;
      return 0;
    }
    if ((key == 13 || key == 32) && g_mouse.done) {
      selected_rect = cv::Rect(g_mouse.start, g_mouse.end);
      // Ensure positive width/height regardless of draw direction.
      if (selected_rect.width < 0) {
        selected_rect.x += selected_rect.width;
        selected_rect.width = -selected_rect.width;
      }
      if (selected_rect.height < 0) {
        selected_rect.y += selected_rect.height;
        selected_rect.height = -selected_rect.height;
      }
      if (selected_rect.width > 5 && selected_rect.height > 5) {
        break;
      }
      std::cout << "Rectangle too small. Please draw again." << std::endl;
      g_mouse.done = false;
    }
  }

  std::cout << "Selected: " << selected_rect << std::endl;

  // ---------------------------------------------------------------
  // Phase 2: Track frame by frame.
  // ---------------------------------------------------------------
  tracking::TrackerConfig config;
  tracking::BoxTracker tracker(config);

  // Process the first frame to initialize the flow computation.
  tracker.ProcessFrame(first_frame);
  tracker.SetBox(selected_rect, frame_width, frame_height);

  std::cout << "Tracking started. Press 'q' to quit, 'r' to reset, "
            << "'p' to pause." << std::endl;

  bool paused = false;
  int frame_idx = 1;  // We already read frame 0.

  while (true) {
    if (!paused) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty()) {
        std::cout << "End of video." << std::endl;
        // Wait for user to quit.
        cv::waitKey(0);
        break;
      }

      // Track.
      tracker.ProcessFrame(frame);

      // Get result.
      cv::Rect tracked_rect;
      float confidence = 0;
      bool ok = tracker.GetTrackedBox(tracked_rect, confidence);

      // Visualize.
      cv::Mat display = frame.clone();

      if (ok) {
        // Color by confidence: green (high) -> red (low).
        float c = std::min(1.0f, std::max(0.0f, confidence));
        cv::Scalar color(0, static_cast<int>(255 * c),
                         static_cast<int>(255 * (1.0f - c)));
        cv::rectangle(display, tracked_rect, color, 2);

        char text[64];
        snprintf(text, sizeof(text), "conf: %.2f", confidence);
        cv::putText(display, text,
                    tracked_rect.tl() + cv::Point(0, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
      } else {
        cv::putText(display, "Tracking lost", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
      }

      // Frame info.
      char info[128];
      snprintf(info, sizeof(info), "Frame %d / %d  |  q:quit  r:reset  p:pause",
               frame_idx, total_frames);
      cv::putText(display, info, cv::Point(10, frame_height - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

      cv::imshow(window_name, display);
      ++frame_idx;
    }

    int key = cv::waitKey(paused ? 30 : delay_ms) & 0xFF;
    if (key == 27 || key == 'q') {
      break;
    }
    if (key == 'r') {
      // Reset: go back to beginning, let user pick new box.
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      cap >> first_frame;
      tracker.Reset();
      g_mouse = MouseState();
      frame_idx = 0;
      paused = false;

      std::cout << "Reset. Draw a new box." << std::endl;

      // Re-enter box selection loop.
      while (true) {
        cv::Mat display = first_frame.clone();
        cv::putText(display,
                    "Draw a box around the target, press ENTER to confirm",
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
        if (g_mouse.drawing || g_mouse.done) {
          cv::rectangle(display, g_mouse.start, g_mouse.end,
                        cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow(window_name, display);

        int k = cv::waitKey(30) & 0xFF;
        if (k == 27 || k == 'q') {
          cv::destroyAllWindows();
          return 0;
        }
        if ((k == 13 || k == 32) && g_mouse.done) {
          selected_rect = cv::Rect(g_mouse.start, g_mouse.end);
          if (selected_rect.width < 0) {
            selected_rect.x += selected_rect.width;
            selected_rect.width = -selected_rect.width;
          }
          if (selected_rect.height < 0) {
            selected_rect.y += selected_rect.height;
            selected_rect.height = -selected_rect.height;
          }
          if (selected_rect.width > 5 && selected_rect.height > 5) break;
          g_mouse.done = false;
        }
      }

      tracker.ProcessFrame(first_frame);
      tracker.SetBox(selected_rect, frame_width, frame_height);
      frame_idx = 1;
    }
    if (key == 'p') {
      paused = !paused;
      std::cout << (paused ? "Paused" : "Resumed") << std::endl;
    }
  }

  cv::destroyAllWindows();
  std::cout << "Done." << std::endl;
  return 0;
}
