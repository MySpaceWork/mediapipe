// Box Tracker -- Standalone implementation inspired by MediaPipe's tracking
// module.

#include "box_tracker.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tracking {

// ============================================================================
// FlowComputation
// ============================================================================

FlowComputation::FlowComputation(const TrackerConfig& config)
    : config_(config) {}

void FlowComputation::Reset() {
  prev_gray_ = cv::Mat();
  prev_points_.clear();
  prev_track_ids_.clear();
  has_prev_ = false;
  next_track_id_ = 0;
}

void FlowComputation::ExtractGridFeatures(const cv::Mat& gray,
                                           std::vector<cv::Point2f>& pts) {
  pts.clear();

  const int cell_w = gray.cols / config_.grid_cols;
  const int cell_h = gray.rows / config_.grid_rows;
  const int features_per_cell =
      std::max(1, config_.max_features / (config_.grid_cols * config_.grid_rows));

  for (int gy = 0; gy < config_.grid_rows; ++gy) {
    for (int gx = 0; gx < config_.grid_cols; ++gx) {
      cv::Rect roi(gx * cell_w, gy * cell_h, cell_w, cell_h);
      roi &= cv::Rect(0, 0, gray.cols, gray.rows);
      if (roi.width <= 0 || roi.height <= 0) continue;

      cv::Mat cell = gray(roi);
      std::vector<cv::Point2f> corners;
      cv::goodFeaturesToTrack(cell, corners, features_per_cell, 0.01, 5);

      for (auto& pt : corners) {
        pt.x += roi.x;
        pt.y += roi.y;
        pts.push_back(pt);
      }
    }
  }
}

std::vector<TrackedFeature> FlowComputation::ProcessFrame(
    const cv::Mat& gray_frame) {
  std::vector<TrackedFeature> result;

  if (!has_prev_) {
    gray_frame.copyTo(prev_gray_);
    ExtractGridFeatures(prev_gray_, prev_points_);
    prev_track_ids_.resize(prev_points_.size());
    for (size_t i = 0; i < prev_points_.size(); ++i) {
      prev_track_ids_[i] = next_track_id_++;
    }
    has_prev_ = true;
    return result;  // No flow for the first frame.
  }

  if (prev_points_.empty()) {
    gray_frame.copyTo(prev_gray_);
    ExtractGridFeatures(prev_gray_, prev_points_);
    prev_track_ids_.resize(prev_points_.size());
    for (size_t i = 0; i < prev_points_.size(); ++i) {
      prev_track_ids_[i] = next_track_id_++;
    }
    return result;
  }

  // Forward KLT tracking.
  std::vector<cv::Point2f> next_points;
  std::vector<uchar> status;
  std::vector<float> errors;
  cv::calcOpticalFlowPyrLK(
      prev_gray_, gray_frame, prev_points_, next_points, status, errors,
      config_.klt_window, config_.pyramid_levels,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                       0.01));

  // Forward-backward verification.
  std::vector<cv::Point2f> back_points;
  std::vector<uchar> back_status;
  std::vector<float> back_errors;
  cv::calcOpticalFlowPyrLK(
      gray_frame, prev_gray_, next_points, back_points, back_status,
      back_errors, config_.klt_window, config_.pyramid_levels,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                       0.01));

  // Collect valid features.
  for (size_t i = 0; i < prev_points_.size(); ++i) {
    if (!status[i] || !back_status[i]) continue;

    // Forward-backward check: round-trip error must be small.
    float fb_dist = cv::norm(prev_points_[i] - back_points[i]);
    if (fb_dist > config_.fb_verify_threshold) continue;

    TrackedFeature feat;
    feat.position = next_points[i];
    feat.flow = next_points[i] - prev_points_[i];
    feat.track_id = prev_track_ids_[i];
    feat.irls_weight = 1.0f;
    feat.is_inlier = true;
    result.push_back(feat);
  }

  // RANSAC outlier rejection on flow vectors (per-region simplified).
  if (result.size() > 4) {
    std::default_random_engine rng(42);
    std::uniform_int_distribution<int> dist(0, result.size() - 1);
    float best_count = 0;
    cv::Point2f best_flow(0, 0);

    for (int r = 0; r < config_.ransac_rounds; ++r) {
      int idx = dist(rng);
      cv::Point2f hyp_flow = result[idx].flow;
      float count = 0;
      for (const auto& f : result) {
        float d = cv::norm(f.flow - hyp_flow);
        if (d < config_.ransac_inlier_threshold) {
          count += 1.0f;
        }
      }
      if (count > best_count) {
        best_count = count;
        best_flow = hyp_flow;
      }
    }

    // Mark outliers.
    for (auto& f : result) {
      float d = cv::norm(f.flow - best_flow);
      if (d >= config_.ransac_inlier_threshold * 2.0f) {
        f.is_inlier = false;
        f.irls_weight = 0.01f;
      }
    }
  }

  // Update state for next frame.
  // Keep tracked points and add new features in empty grid cells.
  std::vector<cv::Point2f> kept_points;
  std::vector<int> kept_ids;
  for (const auto& f : result) {
    if (f.is_inlier) {
      kept_points.push_back(f.position);
      kept_ids.push_back(f.track_id);
    }
  }

  // Extract new features to maintain feature count.
  std::vector<cv::Point2f> new_features;
  ExtractGridFeatures(gray_frame, new_features);

  // Merge: keep existing tracked features, add new ones that are far enough.
  const float min_dist_sq = 25.0f;  // 5 pixels
  for (const auto& nf : new_features) {
    bool too_close = false;
    for (const auto& kp : kept_points) {
      float dx = nf.x - kp.x;
      float dy = nf.y - kp.y;
      if (dx * dx + dy * dy < min_dist_sq) {
        too_close = true;
        break;
      }
    }
    if (!too_close && kept_points.size() < static_cast<size_t>(config_.max_features)) {
      kept_points.push_back(nf);
      kept_ids.push_back(next_track_id_++);
    }
  }

  gray_frame.copyTo(prev_gray_);
  prev_points_ = kept_points;
  prev_track_ids_ = kept_ids;

  return result;
}

// ============================================================================
// CameraMotionEstimator
// ============================================================================

CameraMotionEstimator::CameraMotionEstimator(const TrackerConfig& config)
    : config_(config) {}

cv::Matx33f CameraMotionEstimator::Estimate(
    std::vector<TrackedFeature>& features, int frame_width, int frame_height) {
  if (features.size() < 4) {
    return cv::Matx33f::eye();
  }

  // Collect point pairs for homography estimation.
  std::vector<cv::Point2f> src_pts, dst_pts;
  for (const auto& f : features) {
    if (!f.is_inlier) continue;
    cv::Point2f prev = f.position - f.flow;
    src_pts.push_back(prev);
    dst_pts.push_back(f.position);
  }

  if (src_pts.size() < 4) {
    return cv::Matx33f::eye();
  }

  // Use OpenCV's findHomography with RANSAC.
  cv::Mat mask;
  cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 3.0, mask);

  if (H.empty()) {
    return cv::Matx33f::eye();
  }

  // Update inlier status based on homography RANSAC.
  int inlier_idx = 0;
  for (auto& f : features) {
    if (!f.is_inlier) continue;
    if (inlier_idx < mask.rows && mask.at<uchar>(inlier_idx) == 0) {
      f.irls_weight *= 0.1f;
    }
    ++inlier_idx;
  }

  cv::Matx33f result;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      result(i, j) = static_cast<float>(H.at<double>(i, j));

  return result;
}

// ============================================================================
// MotionBoxTracker
// ============================================================================

MotionBoxTracker::MotionBoxTracker(const TrackerConfig& config)
    : config_(config) {}

void MotionBoxTracker::Init(const BoxState& initial_state) {
  state_ = initial_state;
  state_.tracked = true;
  state_.confidence = 1.0f;
  initialized_ = true;
}

void MotionBoxTracker::GetVectorsAndWeights(
    const std::vector<MotionVector>& vectors, const BoxState& box,
    std::vector<const MotionVector*>& selected, std::vector<float>& weights) {
  selected.clear();
  weights.clear();

  // Expand the search area slightly beyond the box.
  const float expand = 0.05f;
  const float left = box.x - expand;
  const float right = box.x + box.width + expand;
  const float top = box.y - expand;
  const float bottom = box.y + box.height + expand;

  const cv::Point2f box_center = box.center();
  const float inv_sigma_x =
      1.0f / std::max(0.01f, config_.spatial_sigma * box.width);
  const float inv_sigma_y =
      1.0f / std::max(0.01f, config_.spatial_sigma * box.height);

  for (const auto& mv : vectors) {
    if (mv.pos.x < left || mv.pos.x > right || mv.pos.y < top ||
        mv.pos.y > bottom) {
      continue;
    }

    selected.push_back(&mv);

    // Spatial Gaussian weight.
    float dx = (mv.pos.x - box_center.x) * inv_sigma_x;
    float dy = (mv.pos.y - box_center.y) * inv_sigma_y;
    float spatial_w = std::exp(-0.5f * (dx * dx + dy * dy));

    // Motion consistency weight (how close is this vector's motion to the
    // box's previous velocity).
    float motion_w = 1.0f;
    if (std::abs(state_.dx) > 1e-6f || std::abs(state_.dy) > 1e-6f) {
      float mdx = mv.object.x - state_.dx;
      float mdy = mv.object.y - state_.dy;
      float motion_dist =
          std::sqrt(mdx * mdx + mdy * mdy) /
          std::max(0.001f,
                   config_.motion_sigma *
                       std::sqrt(state_.dx * state_.dx + state_.dy * state_.dy));
      motion_w = std::exp(-0.5f * motion_dist * motion_dist);
    }

    weights.push_back(spatial_w * motion_w);
  }

  // Normalize weights.
  float sum = 0;
  for (float w : weights) sum += w;
  if (sum > 0) {
    for (float& w : weights) w /= sum;
  }
}

void MotionBoxTracker::RansacTranslationInit(
    const std::vector<const MotionVector*>& vectors,
    std::vector<float>& weights) {
  const int n = vectors.size();
  if (n == 0) return;

  std::vector<uint8_t> best_inliers(n, 1);
  float best_score = 0;

  std::uniform_int_distribution<int> dist(0, n - 1);

  for (int r = 0; r < config_.ransac_rounds; ++r) {
    int idx = dist(rng_);
    cv::Point2f hyp = vectors[idx]->object;
    float score = 0;

    for (int i = 0; i < n; ++i) {
      float d = cv::norm(vectors[i]->object - hyp);
      if (d < 0.02f) {  // ~2% of normalized frame
        score += weights[i];
      }
    }

    if (score > best_score) {
      best_score = score;
      for (int i = 0; i < n; ++i) {
        float d = cv::norm(vectors[i]->object - hyp);
        best_inliers[i] = (d < 0.02f) ? 1 : 0;
      }
    }
  }

  // Down-weight outliers.
  for (int i = 0; i < n; ++i) {
    if (!best_inliers[i]) {
      weights[i] *= 1e-4f;
    }
  }
}

cv::Point2f MotionBoxTracker::EstimateTranslation(
    const std::vector<const MotionVector*>& vectors,
    const std::vector<float>& prior_weights, std::vector<float>& weights) {
  const int n = vectors.size();
  weights = prior_weights;

  cv::Point2f translation(0, 0);

  for (int iter = 0; iter < config_.irls_iterations; ++iter) {
    // Weighted mean of object motion.
    float sum_w = 0;
    cv::Point2f sum_motion(0, 0);
    for (int i = 0; i < n; ++i) {
      sum_motion += weights[i] * vectors[i]->object;
      sum_w += weights[i];
    }

    if (sum_w > 1e-8f) {
      translation = sum_motion * (1.0f / sum_w);
    }

    // Update IRLS weights: w = prior / max(eps, |residual|).
    for (int i = 0; i < n; ++i) {
      float residual = cv::norm(vectors[i]->object - translation);
      weights[i] = prior_weights[i] / std::max(1e-6f, residual);
    }

    // Normalize.
    float wsum = 0;
    for (float w : weights) wsum += w;
    if (wsum > 0) {
      for (float& w : weights) w /= wsum;
    }
  }

  return translation;
}

float MotionBoxTracker::ScoreInliers(
    const std::vector<const MotionVector*>& vectors,
    const std::vector<float>& weights, const cv::Point2f& translation,
    BoxState& next_state) {
  int inliers = 0;
  float inlier_sum = 0;
  cv::Point2f inlier_center(0, 0);

  for (size_t i = 0; i < vectors.size(); ++i) {
    float residual = cv::norm(vectors[i]->object - translation);
    bool is_inlier = (residual < 0.03f);  // ~3% of frame

    if (is_inlier) {
      ++inliers;
      inlier_sum += weights[i];
      inlier_center += vectors[i]->pos;
    }
  }

  next_state.num_inliers = inliers;
  float confidence = 0;

  if (inliers > 0) {
    inlier_center *= (1.0f / inliers);

    // Confidence based on inlier ratio and count.
    float inlier_ratio =
        static_cast<float>(inliers) / std::max(1, (int)vectors.size());
    confidence = std::min(1.0f, inlier_ratio / config_.min_inlier_ratio);

    // Apply spring force toward inlier center.
    cv::Point2f box_center = next_state.center();
    cv::Point2f diff = inlier_center - box_center;
    float diff_mag = cv::norm(diff);
    if (diff_mag > 0.01f) {
      next_state.x += diff.x * config_.spring_force;
      next_state.y += diff.y * config_.spring_force;
    }
  }

  return confidence;
}

BoxState MotionBoxTracker::TrackStep(const FrameTrackingData& data,
                                      float aspect_ratio) {
  if (!initialized_) {
    BoxState invalid;
    invalid.tracked = false;
    return invalid;
  }

  BoxState next_state = state_;

  // Step 1: Get relevant motion vectors and their prior weights.
  std::vector<const MotionVector*> selected;
  std::vector<float> prior_weights;
  GetVectorsAndWeights(data.motion_vectors, state_, selected, prior_weights);

  if (selected.size() < 3) {
    next_state.tracked = false;
    next_state.confidence *= config_.confidence_decay;
    state_ = next_state;
    return next_state;
  }

  // Step 2: RANSAC initialization.
  RansacTranslationInit(selected, prior_weights);

  // Step 3: IRLS translation estimation.
  std::vector<float> weights;
  cv::Point2f translation =
      EstimateTranslation(selected, prior_weights, weights);

  // Step 4: Apply motion to box.
  next_state.x += translation.x;
  next_state.y += translation.y;
  next_state.dx = translation.x;
  next_state.dy = translation.y;

  // Step 5: Score inliers and compute confidence.
  float confidence = ScoreInliers(selected, weights, translation, next_state);
  next_state.confidence = confidence;
  next_state.tracked = (confidence > 0.1f);

  // Clamp to [0, 1] domain.
  next_state.x = std::max(0.0f, std::min(1.0f - next_state.width, next_state.x));
  next_state.y =
      std::max(0.0f, std::min(1.0f - next_state.height, next_state.y));

  state_ = next_state;
  return next_state;
}

// ============================================================================
// BoxTracker (high-level)
// ============================================================================

BoxTracker::BoxTracker(const TrackerConfig& config)
    : config_(config),
      flow_(config),
      camera_estimator_(config),
      box_tracker_(config) {}

void BoxTracker::Reset() {
  flow_.Reset();
  box_tracker_ = MotionBoxTracker(config_);
  tracking_active_ = false;
  frame_width_ = 0;
  frame_height_ = 0;
}

void BoxTracker::SetBox(const cv::Rect& rect, int frame_width,
                        int frame_height) {
  frame_width_ = frame_width;
  frame_height_ = frame_height;

  BoxState state;
  state.x = static_cast<float>(rect.x) / frame_width;
  state.y = static_cast<float>(rect.y) / frame_height;
  state.width = static_cast<float>(rect.width) / frame_width;
  state.height = static_cast<float>(rect.height) / frame_height;
  state.confidence = 1.0f;
  state.tracked = true;

  box_tracker_.Init(state);
  tracking_active_ = true;
}

void BoxTracker::ProcessFrame(const cv::Mat& frame) {
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  if (frame_width_ == 0) {
    frame_width_ = frame.cols;
    frame_height_ = frame.rows;
  }

  // Step 1: Compute optical flow features.
  std::vector<TrackedFeature> features = flow_.ProcessFrame(gray);

  if (features.empty() || !tracking_active_) {
    return;
  }

  // Step 2: Estimate camera motion.
  cv::Matx33f camera_H =
      camera_estimator_.Estimate(features, frame_width_, frame_height_);

  // Step 3: Decompose features into camera + object motion.
  FrameTrackingData tracking_data;
  tracking_data.camera_homography = camera_H;

  const float inv_w = 1.0f / frame_width_;
  const float inv_h = 1.0f / frame_height_;

  for (const auto& f : features) {
    MotionVector mv;
    mv.pos = cv::Point2f(f.position.x * inv_w, f.position.y * inv_h);
    mv.track_id = f.track_id;

    // Compute camera motion at this point's location.
    cv::Point2f prev_pos = f.position - f.flow;
    cv::Vec3f p(prev_pos.x, prev_pos.y, 1.0f);
    cv::Vec3f transformed = camera_H * p;
    if (std::abs(transformed[2]) > 1e-6f) {
      transformed /= transformed[2];
    }
    cv::Point2f camera_dest(transformed[0], transformed[1]);
    cv::Point2f camera_flow = camera_dest - prev_pos;

    mv.background =
        cv::Point2f(camera_flow.x * inv_w, camera_flow.y * inv_h);
    mv.object = cv::Point2f(f.flow.x * inv_w - mv.background.x,
                             f.flow.y * inv_h - mv.background.y);

    tracking_data.motion_vectors.push_back(mv);
  }

  // Step 4: Track the box.
  float aspect = static_cast<float>(frame_width_) / frame_height_;
  BoxState result = box_tracker_.TrackStep(tracking_data, aspect);

  if (!result.tracked) {
    // Tracking lost.
    tracking_active_ = (result.confidence > 0.05f);
  }
}

bool BoxTracker::GetTrackedBox(cv::Rect& out_rect,
                                float& out_confidence) const {
  if (!tracking_active_ || !box_tracker_.is_initialized()) {
    return false;
  }

  const BoxState& s = box_tracker_.state();
  if (!s.tracked && s.confidence < 0.05f) {
    return false;
  }

  int x = static_cast<int>(s.x * frame_width_);
  int y = static_cast<int>(s.y * frame_height_);
  int w = static_cast<int>(s.width * frame_width_);
  int h = static_cast<int>(s.height * frame_height_);

  out_rect = cv::Rect(x, y, w, h) & cv::Rect(0, 0, frame_width_, frame_height_);
  out_confidence = s.confidence;
  return true;
}

}  // namespace tracking
