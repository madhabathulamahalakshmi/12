"""
Data Augmentation Module for Member 2
Implements flip, rotate, and scale transformations
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class AugmentationStrategy(Enum):
    """Available augmentation strategies"""
    HORIZONTAL_FLIP = "horizontal_flip"
    VERTICAL_FLIP = "vertical_flip"
    ROTATE = "rotate"
    SCALE = "scale"
    NOISE = "noise"
    TIME_WARP = "time_warp"


class LandmarkAugmenter:
    """Augments landmark sequences for training"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def augment_sequence(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        strategy: AugmentationStrategy,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to a landmark sequence
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            confidence: Shape (num_frames, num_points)
            strategy: Which augmentation to apply
            **kwargs: Strategy-specific parameters
            
        Returns:
            (augmented_landmarks, augmented_confidence)
        """
        if strategy == AugmentationStrategy.HORIZONTAL_FLIP:
            return self.horizontal_flip(landmarks, confidence)
        elif strategy == AugmentationStrategy.VERTICAL_FLIP:
            return self.vertical_flip(landmarks, confidence)
        elif strategy == AugmentationStrategy.ROTATE:
            angle = kwargs.get('angle', 15)
            return self.rotate(landmarks, confidence, angle)
        elif strategy == AugmentationStrategy.SCALE:
            scale_range = kwargs.get('scale_range', (0.8, 1.2))
            return self.scale(landmarks, confidence, scale_range)
        elif strategy == AugmentationStrategy.NOISE:
            noise_level = kwargs.get('noise_level', 0.01)
            return self.add_noise(landmarks, confidence, noise_level)
        elif strategy == AugmentationStrategy.TIME_WARP:
            factor = kwargs.get('factor', 0.8)
            return self.time_warp(landmarks, confidence, factor)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def horizontal_flip(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flip landmarks horizontally (mirror across y-axis)"""
        flipped = landmarks.copy()
        # Flip x coordinates
        flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
        
        # Swap left/right hands if applicable
        # This is a simple implementation - adjust based on your landmark order
        return flipped, confidence.copy()

    def vertical_flip(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flip landmarks vertically (mirror across x-axis)"""
        flipped = landmarks.copy()
        # Flip y coordinates
        flipped[:, :, 1] = 1.0 - flipped[:, :, 1]
        return flipped, confidence.copy()

    def rotate(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate landmarks around center
        
        Args:
            angle: Rotation angle in degrees
        """
        rotated = landmarks.copy()
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Center the landmarks
        center = np.nanmean(landmarks[:, :, :2], axis=(0, 1), keepdims=True)
        
        for frame_idx in range(len(landmarks)):
            for point_idx in range(landmarks.shape[1]):
                x, y = landmarks[frame_idx, point_idx, :2]
                
                # Translate to origin
                x -= center[0, 0, 0]
                y -= center[0, 0, 1]
                
                # Rotate
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                # Translate back
                rotated[frame_idx, point_idx, 0] = x_rot + center[0, 0, 0]
                rotated[frame_idx, point_idx, 1] = y_rot + center[0, 0, 1]
        
        return rotated, confidence.copy()

    def scale(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale landmarks randomly within range
        
        Args:
            scale_range: (min_scale, max_scale)
        """
        scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
        
        scaled = landmarks.copy()
        # Center and scale
        center = np.nanmean(landmarks[:, :, :2], axis=(0, 1), keepdims=True)
        
        for frame_idx in range(len(landmarks)):
            for point_idx in range(landmarks.shape[1]):
                # Center
                scaled[frame_idx, point_idx, 0] -= center[0, 0, 0]
                scaled[frame_idx, point_idx, 1] -= center[0, 0, 1]
                
                # Scale
                scaled[frame_idx, point_idx, 0] *= scale_factor
                scaled[frame_idx, point_idx, 1] *= scale_factor
                
                # Translate back
                scaled[frame_idx, point_idx, 0] += center[0, 0, 0]
                scaled[frame_idx, point_idx, 1] += center[0, 0, 1]
        
        return scaled, confidence.copy()

    def add_noise(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        noise_level: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to landmarks"""
        noisy = landmarks.copy()
        noise = self.rng.normal(0, noise_level, size=landmarks.shape)
        
        # Only add noise to x, y coordinates (not confidence)
        noisy[:, :, :2] += noise[:, :, :2]
        
        # Clip to valid range [0, 1]
        noisy[:, :, :2] = np.clip(noisy[:, :, :2], 0, 1)
        
        return noisy, confidence.copy()

    def time_warp(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        factor: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp time dimension (stretch/compress temporal axis)
        
        Args:
            factor: Compression factor (< 1.0 for compression, > 1.0 for stretching)
        """
        num_frames = len(landmarks)
        new_length = max(1, int(num_frames * factor))
        
        # Create new time indices
        old_indices = np.linspace(0, num_frames - 1, new_length)
        
        # Interpolate landmarks
        warped_lms = []
        warped_conf = []
        
        for new_idx, old_idx in enumerate(old_indices):
            lower_idx = int(np.floor(old_idx))
            upper_idx = int(np.ceil(old_idx))
            lower_idx = min(lower_idx, num_frames - 1)
            upper_idx = min(upper_idx, num_frames - 1)
            
            if lower_idx == upper_idx:
                warped_lms.append(landmarks[lower_idx])
                warped_conf.append(confidence[lower_idx])
            else:
                alpha = old_idx - lower_idx
                interp_lm = (
                    (1 - alpha) * landmarks[lower_idx] +
                    alpha * landmarks[upper_idx]
                )
                interp_conf = (
                    (1 - alpha) * confidence[lower_idx] +
                    alpha * confidence[upper_idx]
                )
                warped_lms.append(interp_lm)
                warped_conf.append(interp_conf)
        
        warped_landmarks = np.array(warped_lms)
        warped_confidence = np.array(warped_conf)
        
        return warped_landmarks, warped_confidence

    def batch_augment(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        num_augmentations: int = 3,
        strategies: Optional[List[AugmentationStrategy]] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate multiple augmented versions of a sequence
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            confidence: Shape (num_frames, num_points)
            num_augmentations: Number of augmented versions to create
            strategies: List of strategies to use (random if None)
            
        Returns:
            (list of augmented landmarks, list of augmented confidence)
        """
        if strategies is None:
            strategies = [
                AugmentationStrategy.HORIZONTAL_FLIP,
                AugmentationStrategy.ROTATE,
                AugmentationStrategy.SCALE,
                AugmentationStrategy.NOISE,
            ]
        
        augmented_lms = []
        augmented_conf = []
        
        for _ in range(num_augmentations):
            strategy = self.rng.choice(strategies)
            
            # Apply augmentation with random parameters
            if strategy == AugmentationStrategy.ROTATE:
                angle = self.rng.uniform(-30, 30)
                aug_lm, aug_cf = self.augment_sequence(
                    landmarks, confidence, strategy, angle=angle
                )
            elif strategy == AugmentationStrategy.SCALE:
                aug_lm, aug_cf = self.augment_sequence(
                    landmarks, confidence, strategy,
                    scale_range=(0.7, 1.3)
                )
            elif strategy == AugmentationStrategy.NOISE:
                noise_level = self.rng.uniform(0.005, 0.02)
                aug_lm, aug_cf = self.augment_sequence(
                    landmarks, confidence, strategy,
                    noise_level=noise_level
                )
            else:
                aug_lm, aug_cf = self.augment_sequence(
                    landmarks, confidence, strategy
                )
            
            augmented_lms.append(aug_lm)
            augmented_conf.append(aug_cf)
        
        return augmented_lms, augmented_conf


class AugmentationPipeline:
    """Complete augmentation pipeline with configurable strategies"""

    def __init__(self, seed: Optional[int] = None):
        self.augmenter = LandmarkAugmenter(seed=seed)
        self.strategies = []

    def add_strategy(
        self,
        strategy: AugmentationStrategy,
        probability: float = 0.5,
        **kwargs
    ) -> 'AugmentationPipeline':
        """Add a strategy to the pipeline"""
        self.strategies.append({
            'strategy': strategy,
            'probability': probability,
            'kwargs': kwargs
        })
        return self

    def apply(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all strategies in pipeline"""
        result_lm = landmarks.copy()
        result_conf = confidence.copy()
        
        for strategy_config in self.strategies:
            if self.augmenter.rng.random() < strategy_config['probability']:
                result_lm, result_conf = self.augmenter.augment_sequence(
                    result_lm,
                    result_conf,
                    strategy_config['strategy'],
                    **strategy_config['kwargs']
                )
        
        return result_lm, result_conf

    def get_standard_pipeline(self) -> 'AugmentationPipeline':
        """Get recommended pipeline for sign language"""
        return (
            AugmentationPipeline(seed=self.augmenter.rng.randint(0, 10000))
            .add_strategy(AugmentationStrategy.HORIZONTAL_FLIP, probability=0.3)
            .add_strategy(AugmentationStrategy.ROTATE, probability=0.5, angle=15)
            .add_strategy(AugmentationStrategy.SCALE, probability=0.5, scale_range=(0.8, 1.2))
            .add_strategy(AugmentationStrategy.NOISE, probability=0.3, noise_level=0.01)
        )
