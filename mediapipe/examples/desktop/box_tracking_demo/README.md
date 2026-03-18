# Box Tracking Demo

A standalone C++ demo using MediaPipe's object tracking module. Loads a video,
lets you draw a bounding box on the first frame with the mouse, then tracks
that box through the entire video using MediaPipe's KLT optical flow + IRLS
motion estimation pipeline.

## Algorithm

The demo uses the full MediaPipe tracking pipeline:

1. **Feature Extraction**: Harris corners extracted on a grid.
2. **Optical Flow**: Pyramidal Lucas-Kanade (KLT) via OpenCV.
3. **Camera Motion Estimation**: Cascading IRLS models
   (Translation -> Similarity -> Homography).
4. **Object Tracking**: `MotionBox` with RANSAC initialization + IRLS object
   motion estimation, bidirectional tracking via `BoxTracker`.

## Prerequisites

Install the following on Ubuntu (tested on 20.04 / 22.04):

```bash
# Build tools
sudo apt-get install -y cmake g++ pkg-config

# OpenCV (3.x or 4.x)
sudo apt-get install -y \
  libopencv-dev \
  libopencv-core-dev \
  libopencv-imgproc-dev \
  libopencv-video-dev \
  libopencv-calib3d-dev \
  libopencv-features2d-dev \
  libopencv-highgui-dev \
  libopencv-imgcodecs-dev \
  libopencv-videoio-dev

# Eigen3
sudo apt-get install -y libeigen3-dev

# Protobuf (development libraries + protoc compiler)
sudo apt-get install -y libprotobuf-dev protobuf-compiler

# glog
sudo apt-get install -y libgoogle-glog-dev

# Abseil (if available as a system package)
# On Ubuntu 22.04+:
sudo apt-get install -y libabsl-dev
# On Ubuntu 20.04, you may need to build abseil from source:
#   git clone https://github.com/abseil/abseil-cpp.git
#   cd abseil-cpp && mkdir build && cd build
#   cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DABSL_BUILD_TESTING=OFF \
#         -DABSL_USE_GOOGLETEST_HEAD=OFF -DCMAKE_CXX_STANDARD=17 ..
#   make -j$(nproc) && sudo make install
```

## Build

From this directory (`mediapipe/examples/desktop/box_tracking_demo/`):

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

If CMake cannot find abseil via `find_package`, it falls back to `pkg-config`.
If that also fails, ensure abseil is installed and `CMAKE_PREFIX_PATH` is set:

```bash
cmake -DCMAKE_PREFIX_PATH=/usr/local ..
```

## Run

```bash
./box_tracking_demo /path/to/video.mp4
```

### Controls

| Key | Action |
|---|---|
| Mouse drag | Draw bounding box on the first frame |
| ENTER / SPACE | Confirm selection and start tracking |
| r | Reset (exit playback) |
| q / ESC | Quit |

## How It Works

1. **Phase 1 - Box Selection**: The first frame is displayed. Draw a rectangle
   around the target object with your mouse, then press ENTER.

2. **Phase 2 - Flow Computation**: The entire video is processed frame-by-frame.
   For each frame, `RegionFlowComputation` extracts features and runs KLT
   optical flow. `MotionEstimation` estimates the camera motion model.
   `FlowPackager` packs the results into `TrackingData` stored in a
   `TrackingDataChunk`.

3. **Phase 3 - Box Tracking**: `BoxTracker` takes the pre-computed
   `TrackingDataChunk` and the user-selected initial box, then runs
   bidirectional tracking (forward and backward from the initial frame).

4. **Phase 4 - Visualization**: The video plays back with the tracked box
   overlaid. Box color indicates confidence (green = high, red = low).

## Notes

- The demo processes the entire video upfront before playback. For long videos
  this may take a while.
- Tracking quality depends on video content -- textured objects with moderate
  motion work best.
- The tracking uses `TRACKING_DEGREE_OBJECT_ROTATION_SCALE` which handles
  translation, rotation, and scale changes. For perspective tracking (e.g.,
  planar surfaces), change to `TRACKING_DEGREE_OBJECT_PERSPECTIVE`.
