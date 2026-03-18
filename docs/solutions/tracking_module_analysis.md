# MediaPipe Tracking Module - Algorithm & Implementation Analysis

## Overview

The MediaPipe tracking module (`mediapipe/util/tracking/`) implements a complete object tracking pipeline based on **sparse optical flow** with robust motion estimation. Rather than relying on deep-learning-based trackers, it uses a classical computer vision approach built on feature tracking, camera motion estimation, and per-object motion modeling with iteratively reweighted least squares (IRLS).

The pipeline can be decomposed into four major stages:

1. **Feature Extraction & Optical Flow** (`RegionFlowComputation`)
2. **Camera Motion Estimation** (`MotionEstimation`)
3. **Object-level Box Tracking** (`MotionBox` / `BoxTracker`)
4. **Box Re-detection / Re-acquisition** (`BoxDetectorInterface`)

---

## 1. Feature Extraction & Optical Flow

**File:** [`region_flow_computation.h`](../../mediapipe/util/tracking/region_flow_computation.h) / [`region_flow_computation.cc`](../../mediapipe/util/tracking/region_flow_computation.cc)

### Algorithm: Lucas-Kanade (KLT) Optical Flow

The foundational tracking primitive is the **Kanade-Lucas-Tomasi (KLT)** sparse optical flow tracker, invoked via OpenCV's `cv::calcOpticalFlowPyrLK`. The implementation:

- Extracts **Harris-like corner features** from each frame on a grid basis.
- Tracks these features between consecutive frames using **pyramidal Lucas-Kanade** optical flow (multi-scale).
- Performs **forward-backward verification** -- features are tracked from frame 1 to frame 2, then back from frame 2 to frame 1. Features with high round-trip error are discarded.
- Supports **gain/tone correction** between frames (via `ToneEstimation`) to handle illumination changes.

### Region-based RANSAC Outlier Rejection

After optical flow, features are grouped into spatial grid regions. Within each region, a **RANSAC-based outlier rejection** step is performed:

- For each region, multiple RANSAC rounds pick a random feature's flow vector as a hypothesis.
- All features whose flow agrees within an error threshold are counted as inliers.
- The best inlier set is kept. Multiple inlier sets can be extracted (controlled by `top_inlier_sets`), useful when a region contains multiple independently-moving objects.

This produces a `RegionFlowFrame` or `RegionFlowFeatureList` -- a set of spatially-consistent tracked features with their positions, flow vectors, and quality scores.

### Key Data Structures

- **`TrackedFeature`**: position, flow, tracking error, corner response, octave, track ID, IRLS weight.
- **`RegionFlowFrame`**: collection of region flows, each containing features within a spatial bin.
- **`RegionFlowFeatureList`**: flat list of features with optional descriptors.

---

## 2. Camera Motion Estimation

**File:** [`motion_estimation.h`](../../mediapipe/util/tracking/motion_estimation.h) / [`motion_estimation.cc`](../../mediapipe/util/tracking/motion_estimation.cc)

### Algorithm: IRLS (Iteratively Reweighted Least Squares) with Cascading Models

Camera motion is estimated by fitting parametric motion models to the tracked features, using a **cascade of increasing degrees of freedom**:

| Model | DoF | Description |
|---|---|---|
| `TranslationModel` | 2 | Pure translation (dx, dy) |
| `LinearSimilarityModel` | 4 | Translation + rotation + uniform scale |
| `AffineModel` | 6 | Full affine transform |
| `Homography` | 8 | Full perspective transform |
| `MixtureHomography` | 8*N | Spatially-varying perspective (N blocks) |

The estimation proceeds bottom-up:

1. Start with the simplest model (translation).
2. Run IRLS to robustly fit the model, down-weighting outlier features.
3. Check stability -- if the model is reliable, proceed to the next higher-DoF model.
4. Each higher model is initialized using the result of the lower model.

The IRLS procedure uses **anisotropic error weighting** via an error system aligned to the dominant motion direction, giving different sensitivity along the motion direction versus perpendicular to it.

### Stability Analysis

Each model fit is assessed for stability. If a higher-DoF model is unstable (e.g., not enough inlier support, numerical issues), the system falls back to the last stable lower-DoF model. The final result is recorded as a `CameraMotion` proto with the highest stable model.

---

## 3. Object-level Box Tracking

**File:** [`tracking.h`](../../mediapipe/util/tracking/tracking.h) / [`tracking.cc`](../../mediapipe/util/tracking/tracking.cc)

### Core Class: `MotionBox`

The `MotionBox` class tracks a rectangular (or quadrilateral) region across frames. It is the heart of the object tracking logic.

### Algorithm: Feature-based Motion Estimation with IRLS

Given a `MotionVectorFrame` (which separates each feature's motion into **background** and **object** components), the tracking proceeds as follows for each frame step:

#### Step 1: Feature Selection (`GetVectorsAndWeights`)

- Find all features within or near the current box region (with optional expansion).
- Assign each feature a **prior weight** based on:
  - **Spatial proximity** -- Gaussian weighting from box center.
  - **Track continuity** -- features that were inliers in the previous frame get higher weight.
  - **Motion consistency** -- features whose motion matches the box's velocity from the previous frame are preferred.
  - **Background discrimination** -- features with significant object-relative motion (not just camera motion) are boosted.

#### Step 2: RANSAC Initialization (`TranslationIrlsInitialization`)

- Multiple RANSAC rounds are run to find the dominant object motion hypothesis.
- Each round picks a random feature's object motion, measures agreement across all features using an anisotropic error metric.
- The best inlier set seeds the IRLS weights, suppressing outliers early.

#### Step 3: Object Motion Estimation (`EstimateObjectMotion`)

Depending on the configured tracking degrees of freedom:

- **Translation** (2 DoF): Weighted mean of object motion vectors, refined with IRLS.
- **Similarity** (4 DoF): Least-squares fit of a linear similarity (translation + rotation + scale) via `LinearSimilarityL2Solve`.
- **Homography** (8 DoF): Least-squares fit of a homography via `HomographyL2Solve`, for perspective tracking (e.g., planar surfaces).
- **PnP Homography** (6 DoF): `cv::solvePnP` for 3D perspective tracking when aspect ratio is known.

The system cascades: it always estimates translation first, then attempts higher-DoF models only if enough continued inliers exist. If a higher model is unstable, it falls back to translation.

#### Step 4: Inlier/Outlier Scoring (`ScoreAndRecordInliers`)

After motion estimation, every feature is scored as inlier or outlier based on how well it matches the estimated motion. Statistics computed include:

- **Continued inliers**: Features that were inliers in both the current and previous frame (track stability).
- **Swapped inliers**: Features that switched from outlier to inlier (potential tracking drift indicator).
- **Motion inliers**: Features agreeing with the box's velocity (handles fast-moving objects with short-lived tracks).
- **Inlier density**: Local density of inliers (detects whether inliers are well-distributed across the box).

#### Step 5: Box Position Update

The box position, size, and rotation are updated based on the estimated motion model. Additional corrections are applied:

- **Spring force**: Pulls the box center toward the inlier center of mass, preventing drift.
- **Tracking confidence**: A scalar in [0, 1] computed from inlier statistics, used downstream to decide if tracking is reliable.

### Key Data Structures

- **`MotionBoxState`**: Complete state of a tracked box -- position, size, rotation, velocity, inlier/outlier IDs, tracking status, confidence, internal state.
- **`MotionVectorFrame`**: Per-frame collection of `MotionVector`s, each decomposed into background (camera) and object (foreground) components.
- **`MotionVector`**: A tracked feature's position and decomposed motion.

---

## 4. Box Tracker (Multi-box, Bidirectional)

**File:** [`box_tracker.h`](../../mediapipe/util/tracking/box_tracker.h) / [`box_tracker.cc`](../../mediapipe/util/tracking/box_tracker.cc)

### Algorithm: Checkpoint-based Bidirectional Tracking

`BoxTracker` manages multiple tracked boxes over time using pre-computed `TrackingDataChunk`s:

1. Given an initial `TimedBox` (position + timestamp), the tracker:
   - Creates a `MotionBox` and resets it at the initial frame.
   - Launches **forward tracking** (from the start frame to later frames).
   - Launches **backward tracking** (from the start frame to earlier frames).
2. Tracking is **asynchronous** -- uses a thread pool to run forward/backward tracks concurrently.
3. Results are stored per **checkpoint** in a `Path` (map of checkpoint -> `PathSegment`).
4. `GetTimedPosition` retrieves the box at any timestamp via **temporal interpolation** (linear blend between known states).

### Chunk-based Architecture

Tracking data is organized into time-ordered chunks (`TrackingDataChunk`), each containing per-frame `TrackingData`. Chunks can be:
- Loaded from disk (cache directory) for offline processing.
- Passed directly in memory for streaming/online processing.

This design enables efficient tracking over long videos without loading all data into memory.

---

## 5. Box Re-detection (Re-acquisition)

**File:** [`box_detector.h`](../../mediapipe/util/tracking/box_detector.h) / [`box_detector.cc`](../../mediapipe/util/tracking/box_detector.cc)

### Algorithm: Feature Descriptor Matching + RANSAC Homography + PnP

When a tracked box is lost (e.g., due to occlusion), the `BoxDetectorInterface` attempts to re-detect it:

1. **Index building**: When a box is first tracked, features within the box and their descriptors (from OpenCV's feature detector, e.g., ORB) are stored in an index.
2. **Matching**: On subsequent frames, frame features are matched against the stored index using descriptor matching.
3. **Geometric verification**: `cv::findHomography` with **RANSAC** (100 max iterations) verifies the match geometrically. The homography's determinant and perspective factor are checked for validity.
4. **PnP pose estimation** (optional): When box aspect ratio and frame aspect ratio are known, `cv::solvePnP` is used to estimate the 3D pose of the planar target, followed by `cv::projectPoints` to recover the 2D quad corners. This enables more accurate perspective tracking.

---

## 6. Supporting Components

### Motion Analysis (`MotionAnalysis`)

**File:** [`motion_analysis.h`](../../mediapipe/util/tracking/motion_analysis.h)

Orchestrates the full pipeline: feature extraction -> motion estimation -> saliency computation. Buffers frames internally with adaptive overlap for temporal consistency.

### Camera Motion Utilities (`camera_motion.h`)

**File:** [`camera_motion.h`](../../mediapipe/util/tracking/camera_motion.h)

Helper functions to extract specific motion models from `CameraMotion` protos, compose/invert camera motions, and subtract camera motion from features to isolate foreground motion.

### Motion Saliency (`MotionSaliency`)

**File:** [`motion_saliency.h`](../../mediapipe/util/tracking/motion_saliency.h)

Computes saliency points by finding **modes** (clusters) in the feature distribution based on IRLS weights. Used for video stabilization and retargeting.

### Flow Packager (`FlowPackager`)

**File:** [`flow_packager.h`](../../mediapipe/util/tracking/flow_packager.h)

Serializes/deserializes `RegionFlowFeatureList` + `CameraMotion` into compact `TrackingData` format for efficient storage and streaming.

### Tracked Detection Manager (`TrackedDetectionManager`)

**File:** [`tracked_detection_manager.h`](../../mediapipe/util/tracking/tracked_detection_manager.h)

Manages a set of active detections, handling:
- Duplicate detection removal (based on bounding box overlap/area ratio).
- Obsolete detection cleanup (not updated within a timeout).
- Out-of-view detection removal.

### Motion Models (`motion_models.h`)

**File:** [`motion_models.h`](../../mediapipe/util/tracking/motion_models.h)

Template-based abstraction (`ModelAdapter<Model>`) for all motion models. Provides uniform APIs for: creating from parameters, transforming points, inverting, composing, and normalizing models. Covers Translation, LinearSimilarity, Affine, Homography, and MixtureHomography.

### Filtering Utilities

- [`low_pass_filter.h`](../../mediapipe/util/filtering/low_pass_filter.h): Exponential smoothing low-pass filter.
- [`relative_velocity_filter.h`](../../mediapipe/util/filtering/relative_velocity_filter.h): Adaptive smoothing based on velocity (1-euro filter variant).

---

## Algorithm Summary Table

| Stage | Algorithm | Implementation |
|---|---|---|
| Feature extraction | Harris corners on grid | `RegionFlowComputation` |
| Optical flow | Pyramidal Lucas-Kanade (KLT) | OpenCV `calcOpticalFlowPyrLK` |
| Flow outlier rejection | Per-region RANSAC | `DetermineRegionFlowInliers` |
| Camera motion estimation | Cascading IRLS (Translation -> Similarity -> Affine -> Homography) | `MotionEstimation` |
| Object motion estimation | RANSAC init + IRLS (Translation / Similarity / Homography / PnP) | `MotionBox::EstimateObjectMotion` |
| Box tracking | Feature-weighted bidirectional tracking with spring correction | `BoxTracker::TrackingImpl` + `MotionBox::TrackStep` |
| Box re-detection | Feature descriptor matching + RANSAC homography + PnP | `BoxDetectorInterface::DetectAndAddBox` |
| Detection management | Overlap-based dedup + timeout cleanup | `TrackedDetectionManager` |

---

## Architecture Diagram

```
Frame N-1, Frame N
       |
       v
+---------------------------+
| RegionFlowComputation     |
| - Harris feature extract  |
| - KLT optical flow        |
| - Region RANSAC filtering |
+---------------------------+
       |
       v
  RegionFlowFeatureList
       |
       v
+---------------------------+
| MotionEstimation          |
| - IRLS cascading models   |
| - Stability analysis      |
+---------------------------+
       |
       v
  CameraMotion (background model)
       |
       v
+---------------------------+
| FlowPackager              |
| - Pack into TrackingData  |
+---------------------------+
       |
       v
  TrackingDataChunk (stored per time window)
       |
       v
+---------------------------+       +---------------------------+
| BoxTracker                |       | BoxDetectorInterface      |
| - Bidirectional tracking  |<----->| - Feature index           |
| - Checkpoint management   |       | - RANSAC homography       |
| - Thread pool             |       | - PnP re-detection        |
+---------------------------+       +---------------------------+
       |
       v
  MotionBox (per object)
  - GetVectorsAndWeights
  - TranslationIrlsInitialization (RANSAC)
  - EstimateObjectMotion (IRLS)
  - ScoreAndRecordInliers
       |
       v
  TimedBox (position, rotation, confidence)
```

---

## Design Highlights

1. **Classical CV approach**: The entire pipeline uses hand-crafted algorithms (KLT, RANSAC, IRLS) rather than neural networks, making it lightweight and suitable for real-time mobile/embedded deployment.

2. **Hierarchical motion decomposition**: Features' total motion is decomposed into camera (background) and object (foreground) components, enabling robust tracking even with significant camera motion.

3. **Graceful degradation**: The cascading model estimation (Translation -> Similarity -> Homography) with stability checks ensures the tracker always uses the most complex model it can reliably estimate, falling back to simpler models when needed.

4. **Temporal consistency**: Track IDs, inlier history, and velocity-based priors ensure temporal coherence. The spring force mechanism prevents gradual drift.

5. **Bidirectional tracking**: Tracking runs both forward and backward from the initial position, maximizing the tracked time range.

6. **Streaming architecture**: Chunk-based data organization supports both offline batch processing and online streaming use cases.
