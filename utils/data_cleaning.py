"""
Member 2: Data Cleaning & Normalization Module
Removes noise, smooths points, normalizes size/position, handles missing landmarks
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class LandmarkCleaner:
    """Cleans and stabilizes landmark data"""

    def __init__(self, window_size: int = 5, threshold: float = 0.05):
        """
        Args:
            window_size: Size of smoothing window
            threshold: Threshold for outlier detection (relative to expected range)
        """
        self.window_size = window_size
        self.threshold = threshold

    def remove_outliers(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Remove outliers using median absolute deviation (MAD)
        
        Args:
            landmarks: Shape (num_frames, num_points, 3) - [x, y, confidence]
            
        Returns:
            Cleaned landmarks with outliers replaced by interpolation
        """
        cleaned = landmarks.copy()
        num_points = landmarks.shape[1]

        for point_idx in range(num_points):
            point_trajectory = landmarks[:, point_idx, :2]  # x, y only
            
            # Calculate median absolute deviation
            median = np.median(point_trajectory, axis=0)
            mad = np.median(np.abs(point_trajectory - median), axis=0)
            
            # Identify outliers (> 3 MAD from median)
            outlier_mask = np.any(
                np.abs(point_trajectory - median) > 3 * (mad + 1e-6),
                axis=1
            )
            
            # Interpolate outliers
            for frame_idx in np.where(outlier_mask)[0]:
                # Find nearest valid frames
                valid_frames = np.where(~outlier_mask)[0]
                if len(valid_frames) > 0:
                    nearest = valid_frames[np.argmin(np.abs(valid_frames - frame_idx))]
                    cleaned[frame_idx, point_idx, :2] = cleaned[nearest, point_idx, :2]

        return cleaned

    def smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            
        Returns:
            Smoothed landmarks
        """
        from scipy.signal import savgol_filter

        smoothed = landmarks.copy()
        num_points = landmarks.shape[1]

        for point_idx in range(num_points):
            for coord_idx in range(2):  # x, y only
                trajectory = landmarks[:, point_idx, coord_idx]
                
                # Only smooth if we have enough frames
                if len(trajectory) >= self.window_size + 2:
                    order = min(3, self.window_size - 1)
                    try:
                        smoothed[:, point_idx, coord_idx] = savgol_filter(
                            trajectory, 
                            window_length=self.window_size,
                            polyorder=order
                        )
                    except Exception:
                        # Fallback to moving average if Savitzky-Golay fails
                        smoothed[:, point_idx, coord_idx] = self._moving_average(
                            trajectory, 
                            window_size=3
                        )

        return smoothed

    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Simple moving average"""
        result = data.copy()
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            result[i] = np.mean(data[start:end])
        
        return result

    def handle_missing_landmarks(
        self, 
        landmarks: np.ndarray,
        confidence: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing or low-confidence landmarks
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            confidence: Shape (num_frames, num_points)
            confidence_threshold: Minimum confidence to keep a point
            
        Returns:
            Cleaned landmarks and confidence scores
        """
        cleaned_landmarks = landmarks.copy()
        cleaned_confidence = confidence.copy()
        num_frames = landmarks.shape[0]
        num_points = landmarks.shape[1]

        for frame_idx in range(num_frames):
            for point_idx in range(num_points):
                if confidence[frame_idx, point_idx] < confidence_threshold:
                    # Try to interpolate from neighbors
                    valid_frames = np.where(
                        confidence[:, point_idx] >= confidence_threshold
                    )[0]
                    
                    if len(valid_frames) > 0:
                        # Linear interpolation from nearest frame
                        nearest_idx = valid_frames[
                            np.argmin(np.abs(valid_frames - frame_idx))
                        ]
                        cleaned_landmarks[frame_idx, point_idx] = \
                            cleaned_landmarks[nearest_idx, point_idx]
                        # Mark as interpolated (lower confidence)
                        cleaned_confidence[frame_idx, point_idx] = 0.3
                    else:
                        # No valid frames - set to zero
                        cleaned_landmarks[frame_idx, point_idx] = [0, 0, 0]
                        cleaned_confidence[frame_idx, point_idx] = 0.0

        return cleaned_landmarks, cleaned_confidence


class LandmarkNormalizer:
    """Normalizes landmark positions and sizes"""

    def __init__(self):
        self.hand_center_mean = None
        self.hand_size_mean = None

    def normalize(
        self,
        landmarks: np.ndarray,
        method: str = "position_scale"
    ) -> np.ndarray:
        """
        Normalize landmarks
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            method: 'position_scale' or 'hand_based'
            
        Returns:
            Normalized landmarks
        """
        if method == "position_scale":
            return self._normalize_position_scale(landmarks)
        elif method == "hand_based":
            return self._normalize_hand_based(landmarks)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _normalize_position_scale(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize by centering and scaling
        """
        normalized = landmarks.copy()
        num_frames = landmarks.shape[0]

        for frame_idx in range(num_frames):
            frame_points = normalized[frame_idx, :, :2]  # x, y only
            
            # Get valid points
            valid_mask = ~np.all(frame_points == 0, axis=1)
            valid_points = frame_points[valid_mask]
            
            if len(valid_points) > 0:
                # Center
                center = np.mean(valid_points, axis=0)
                frame_points -= center
                
                # Scale by bounding box
                bbox_size = np.max(np.abs(valid_points))
                if bbox_size > 0:
                    frame_points /= bbox_size
                
                normalized[frame_idx, :, :2] = frame_points

        return normalized

    def _normalize_hand_based(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize based on hand size (distance between wrist and middle finger)
        """
        normalized = landmarks.copy()
        
        for frame_idx in range(len(landmarks)):
            # Wrist is usually landmark 0, middle finger tip is landmark 12
            if len(normalized[frame_idx]) > 12:
                wrist = normalized[frame_idx, 0, :2]
                finger_tip = normalized[frame_idx, 12, :2]
                
                hand_size = np.linalg.norm(finger_tip - wrist)
                
                if hand_size > 0:
                    # Center on wrist
                    normalized[frame_idx, :, :2] -= wrist
                    # Scale by hand size
                    normalized[frame_idx, :, :2] /= hand_size

        return normalized

    def fit(self, landmarks_list: List[np.ndarray]) -> None:
        """
        Fit normalization parameters on a set of sequences
        
        Args:
            landmarks_list: List of landmark arrays
        """
        all_landmarks = np.concatenate(landmarks_list, axis=0)
        self.hand_center_mean = np.nanmean(all_landmarks[:, :, :2], axis=(0, 1))
        self.hand_size_mean = np.nanmean([
            np.linalg.norm(lm[:, :2])
            for lm in landmarks_list
        ])

    def denormalize(self, normalized_landmarks: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        if self.hand_center_mean is None or self.hand_size_mean is None:
            raise ValueError("Normalizer not fitted yet")
        
        denormalized = normalized_landmarks.copy()
        denormalized[:, :, :2] *= self.hand_size_mean
        denormalized[:, :, :2] += self.hand_center_mean
        return denormalized


class LandmarkValidator:
    """Validates cleaned landmark data"""

    @staticmethod
    def validate_sequence(
        landmarks: np.ndarray,
        confidence: np.ndarray,
        min_valid_ratio: float = 0.7
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate a landmark sequence
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            confidence: Shape (num_frames, num_points)
            min_valid_ratio: Minimum ratio of valid points required
            
        Returns:
            (is_valid, metrics_dict)
        """
        num_frames = landmarks.shape[0]
        num_points = landmarks.shape[1]
        
        # Check confidence
        avg_confidence = np.mean(confidence)
        valid_ratio = np.sum(confidence > 0.5) / (num_frames * num_points)
        
        # Check for extreme jumps (velocity)
        velocities = []
        for point_idx in range(num_points):
            point_trajectory = landmarks[:, point_idx, :2]
            valid_indices = np.where(np.sum(confidence[:, point_idx:point_idx+1], axis=1) > 0)[0]
            
            if len(valid_indices) > 1:
                diffs = np.diff(point_trajectory[valid_indices], axis=0)
                velocities.append(np.linalg.norm(diffs, axis=1))
        
        if velocities:
            max_velocity = np.max(np.concatenate(velocities))
        else:
            max_velocity = 0.0
        
        # Determine validity
        is_valid = (
            valid_ratio >= min_valid_ratio and
            max_velocity < 1.0  # Reasonable movement threshold
        )
        
        metrics = {
            "avg_confidence": float(avg_confidence),
            "valid_ratio": float(valid_ratio),
            "max_velocity": float(max_velocity),
            "num_frames": num_frames,
            "num_points": num_points
        }
        
        return is_valid, metrics


def load_landmarks_from_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load landmarks from JSON file
    
    Returns:
        landmarks: (num_frames, num_points, 3)
        confidence: (num_frames, num_points)
        frame_ids: frame identifiers
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        frames = data
    else:
        frames = [data]
    
    landmarks_list = []
    confidence_list = []
    frame_ids = []
    
    for frame in frames:
        all_landmarks = []
        all_confidence = []
        
        # Process hands
        for hand in frame.get('hands', []):
            lms = np.array(hand.get('landmarks', []))
            if len(lms) > 0:
                all_landmarks.append(lms)
                # Extract confidence (3rd column) or assume high confidence
                conf = lms[:, 2] if lms.shape[1] > 2 else np.ones(len(lms))
                all_confidence.append(conf)
        
        # Process faces
        for face in frame.get('faces', []):
            lms = np.array(face.get('landmarks', []))
            if len(lms) > 0:
                all_landmarks.append(lms)
                conf = lms[:, 2] if lms.shape[1] > 2 else np.ones(len(lms))
                all_confidence.append(conf)
        
        if all_landmarks:
            # Concatenate all landmarks
            frame_lms = np.concatenate(all_landmarks, axis=0)
            frame_conf = np.concatenate(all_confidence, axis=0)
            
            landmarks_list.append(frame_lms)
            confidence_list.append(frame_conf)
            frame_ids.append(frame.get('frame_id', len(frame_ids)))
    
    # Stack into (num_frames, num_points, 3) format
    # Pad to same number of points if needed
    max_points = max(lm.shape[0] for lm in landmarks_list)
    
    padded_landmarks = []
    padded_confidence = []
    
    for lm, conf in zip(landmarks_list, confidence_list):
        if lm.shape[0] < max_points:
            pad_size = max_points - lm.shape[0]
            lm = np.vstack([lm, np.zeros((pad_size, 3))])
            conf = np.hstack([conf, np.zeros(pad_size)])
        padded_landmarks.append(lm)
        padded_confidence.append(conf)
    
    landmarks = np.array(padded_landmarks)
    confidence = np.array(padded_confidence)
    
    return landmarks, confidence, frame_ids


def save_cleaned_landmarks(
    landmarks: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """Save cleaned landmarks to JSON"""
    frames = []
    
    for frame_idx, (lm, conf) in enumerate(zip(landmarks, confidence)):
        frame_data = {
            "frame_id": frame_idx,
            "landmarks": lm.tolist(),
            "confidence": conf.tolist()
        }
        if metadata:
            frame_data.update(metadata)
        frames.append(frame_data)
    
    with open(output_path, 'w') as f:
        json.dump(frames, f, indent=2)
