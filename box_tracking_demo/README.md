# Box Tracking Demo

A standalone C++ object tracking demo implementing the same algorithm used by
MediaPipe's tracking module. **No MediaPipe dependency** -- only OpenCV is
required.

## Algorithm

This project implements a simplified version of the MediaPipe box tracking
pipeline:

1. **Feature Extraction**: Harris/Shi-Tomasi corners extracted on a uniform grid
   (`cv::goodFeaturesToTrack`).
2. **Pyramidal Lucas-Kanade (KLT) Optical Flow**: Features tracked between
   consecutive frames (`cv::calcOpticalFlowPyrLK`).
3. **Forward-backward Verification**: Features tracked forward then backward;
   those with high round-trip error are discarded.
4. **RANSAC Outlier Rejection**: Random sample consensus removes flow outliers
   that don't match the dominant motion.
5. **Camera Motion Estimation**: Homography estimated via RANSAC
   (`cv::findHomography`) to model global camera motion.
6. **Motion Decomposition**: Each feature's motion is split into background
   (camera) and object (foreground) components.
7. **IRLS Object Motion Estimation**: Iteratively Reweighted Least Squares
   estimates the object's translation, down-weighting outliers at each
   iteration.
8. **Inlier Scoring + Spring Correction**: Confidence computed from inlier
   ratio; box center pulled toward inlier center of mass to prevent drift.

## Dependencies

- **OpenCV** (>= 3.0) -- the only dependency.

## Build

### Ubuntu / Debian

```bash
# Install OpenCV
sudo apt-get install -y libopencv-dev

# Build
cd box_tracking_demo
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# Install OpenCV via Homebrew
brew install opencv

# Build
cd box_tracking_demo
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows (vcpkg)

```powershell
vcpkg install opencv4
cd box_tracking_demo
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

## Run

```bash
./box_tracking_demo /path/to/video.mp4
```

### Controls

| Key | Action |
|-----|--------|
| Mouse drag | Draw bounding box on the first frame |
| ENTER / SPACE | Confirm box selection and start tracking |
| p | Pause / resume playback |
| r | Reset: go back to frame 1 and draw a new box |
| q / ESC | Quit |

## Project Structure

```
box_tracking_demo/
  CMakeLists.txt          # Build configuration (CMake)
  README.md               # This file
  include/
    box_tracker.h         # Tracker API and data structures
  src/
    box_tracker.cc        # Tracking algorithm implementation
    main.cc               # Demo application with OpenCV GUI
```

## How It Works

### Frame Processing Pipeline

For each video frame:

```
Input Frame (BGR)
      |
      v
  Convert to Grayscale
      |
      v
  Extract Grid Features (goodFeaturesToTrack)
      |
      v
  KLT Optical Flow (calcOpticalFlowPyrLK, forward + backward)
      |
      v
  RANSAC Flow Outlier Rejection
      |
      v
  Camera Homography Estimation (findHomography + RANSAC)
      |
      v
  Motion Decomposition (camera vs. object)
      |
      v
  Box Motion Estimation:
    1. Select features inside/near box
    2. Weight by spatial proximity + motion consistency
    3. RANSAC translation initialization
    4. IRLS translation refinement
    5. Inlier scoring + spring correction
      |
      v
  Updated Box Position + Confidence
```

### Relation to MediaPipe

This implementation mirrors the core algorithms from MediaPipe's
`mediapipe/util/tracking/` module:

| MediaPipe Component | This Project |
|---|---|
| `RegionFlowComputation` | `FlowComputation` class |
| `MotionEstimation` | `CameraMotionEstimator` class |
| `MotionBox::TrackStep` | `MotionBoxTracker::TrackStep` |
| `MotionBox::TranslationIrlsInitialization` | `MotionBoxTracker::RansacTranslationInit` |
| `MotionBox::EstimateTranslation` | `MotionBoxTracker::EstimateTranslation` |
| `BoxTracker` | `BoxTracker` (high-level) |

Key simplifications vs. the full MediaPipe implementation:
- Translation-only object motion (MediaPipe also supports similarity/homography)
- Single-pass forward tracking (MediaPipe does bidirectional)
- No protobuf serialization (all in-memory)
- No chunk-based caching (processes frames online)
- No box re-detection / re-acquisition

## License

Apache License 2.0 (same as MediaPipe).
