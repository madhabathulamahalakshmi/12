"""
Sequence Processor Module for Member 2
Converts raw landmarks into model-ready sequences
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .data_cleaning import (
    LandmarkCleaner,
    LandmarkNormalizer,
    LandmarkValidator,
    load_landmarks_from_json,
    save_cleaned_landmarks
)
from .data_augmentation import LandmarkAugmenter, AugmentationStrategy


class SequenceProcessor:
    """Processes raw landmark sequences into model-ready format"""

    def __init__(
        self,
        window_size: int = 5,
        confidence_threshold: float = 0.5,
        normalization_method: str = "position_scale"
    ):
        self.cleaner = LandmarkCleaner(window_size=window_size)
        self.normalizer = LandmarkNormalizer()
        self.validator = LandmarkValidator()
        self.augmenter = LandmarkAugmenter()
        
        self.confidence_threshold = confidence_threshold
        self.normalization_method = normalization_method
        self.processing_log = []

    def process_sequence(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        apply_augmentation: bool = False,
        num_augmentations: int = 3
    ) -> Dict[str, any]:
        """
        Process a single landmark sequence
        
        Args:
            landmarks: Shape (num_frames, num_points, 3)
            confidence: Shape (num_frames, num_points)
            apply_augmentation: Whether to generate augmented versions
            num_augmentations: Number of augmented versions
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'original_shape': landmarks.shape,
            'steps': {},
            'augmented_sequences': [],
            'metrics': {}
        }

        # Step 1: Remove outliers
        cleaned_lm = self.cleaner.remove_outliers(landmarks)
        results['steps']['outlier_removal'] = cleaned_lm.shape
        self._log("Outlier removal completed")

        # Step 2: Smooth landmarks
        smoothed_lm = self.cleaner.smooth_landmarks(cleaned_lm)
        results['steps']['smoothing'] = smoothed_lm.shape
        self._log("Smoothing completed")

        # Step 3: Handle missing landmarks
        final_lm, final_conf = self.cleaner.handle_missing_landmarks(
            smoothed_lm,
            confidence,
            confidence_threshold=self.confidence_threshold
        )
        results['steps']['missing_handling'] = final_lm.shape
        self._log("Missing landmark handling completed")

        # Step 4: Normalize
        normalized_lm = self.normalizer.normalize(
            final_lm,
            method=self.normalization_method
        )
        results['steps']['normalization'] = normalized_lm.shape
        self._log(f"Normalization ({self.normalization_method}) completed")

        # Step 5: Validate
        is_valid, metrics = self.validator.validate_sequence(
            normalized_lm,
            final_conf
        )
        results['valid'] = is_valid
        results['metrics'] = metrics
        self._log(f"Validation: {'PASS' if is_valid else 'FAIL'}")

        # Store final cleaned sequence
        results['cleaned_landmarks'] = normalized_lm
        results['cleaned_confidence'] = final_conf

        # Step 6: Data augmentation (optional)
        if apply_augmentation:
            aug_lms, aug_confs = self.augmenter.batch_augment(
                normalized_lm,
                final_conf,
                num_augmentations=num_augmentations
            )
            results['augmented_sequences'] = [
                {'landmarks': lm, 'confidence': conf}
                for lm, conf in zip(aug_lms, aug_confs)
            ]
            self._log(f"Data augmentation: {num_augmentations} versions created")

        return results

    def process_json_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        apply_augmentation: bool = False
    ) -> Dict[str, any]:
        """
        Process landmarks from JSON file
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to save cleaned landmarks (optional)
            apply_augmentation: Whether to augment
            
        Returns:
            Processing results
        """
        self._log(f"Processing file: {input_path}")

        # Load
        landmarks, confidence, frame_ids = load_landmarks_from_json(input_path)
        self._log(f"Loaded: {landmarks.shape}")

        # Process
        results = self.process_sequence(
            landmarks,
            confidence,
            apply_augmentation=apply_augmentation,
            num_augmentations=3
        )

        # Save
        if output_path:
            save_cleaned_landmarks(
                results['cleaned_landmarks'],
                results['cleaned_confidence'],
                output_path,
                metadata={
                    'original_file': input_path,
                    'frame_ids': frame_ids
                }
            )
            self._log(f"Saved to: {output_path}")

        return results

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        apply_augmentation: bool = False
    ) -> List[Dict[str, any]]:
        """
        Process multiple JSON files from a directory
        
        Args:
            input_dir: Directory containing raw landmark files
            output_dir: Directory to save cleaned files
            apply_augmentation: Whether to augment
            
        Returns:
            List of processing results for each file
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        json_files = list(input_path.glob('*.json'))
        self._log(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            out_file = output_path / f"cleaned_{json_file.name}"
            
            try:
                result = self.process_json_file(
                    str(json_file),
                    output_path=str(out_file),
                    apply_augmentation=apply_augmentation
                )
                result['input_file'] = str(json_file)
                result['output_file'] = str(out_file)
                result['success'] = True
                results.append(result)
            except Exception as e:
                self._log(f"ERROR processing {json_file}: {str(e)}")
                results.append({
                    'input_file': str(json_file),
                    'success': False,
                    'error': str(e)
                })

        return results

    def create_fixed_length_sequence(
        self,
        landmarks: np.ndarray,
        confidence: np.ndarray,
        target_length: int = 30,
        padding_strategy: str = "repeat"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert variable-length sequence to fixed length
        
        Args:
            target_length: Target number of frames
            padding_strategy: 'repeat', 'pad', or 'interpolate'
            
        Returns:
            (fixed_landmarks, fixed_confidence)
        """
        current_length = len(landmarks)

        if current_length == target_length:
            return landmarks, confidence

        elif current_length < target_length:
            if padding_strategy == "repeat":
                # Repeat last frame
                repeat_count = target_length - current_length
                landmarks = np.vstack([
                    landmarks,
                    np.tile(landmarks[-1:], (repeat_count, 1, 1))
                ])
                confidence = np.vstack([
                    confidence,
                    np.tile(confidence[-1:], (repeat_count, 1))
                ])
            elif padding_strategy == "pad":
                # Pad with zeros
                pad_width = target_length - current_length
                landmarks = np.pad(
                    landmarks,
                    ((0, pad_width), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
                confidence = np.pad(
                    confidence,
                    ((0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            elif padding_strategy == "interpolate":
                # Interpolate
                indices = np.linspace(0, current_length - 1, target_length)
                landmarks = np.array([
                    np.interp(indices, np.arange(current_length), landmarks[:, i, j])
                    if not np.all(landmarks[:, i, j] == 0) else np.zeros(target_length)
                    for i in range(landmarks.shape[1])
                    for j in range(3)
                ]).reshape(target_length, landmarks.shape[1], 3)
                confidence = np.array([
                    np.interp(indices, np.arange(current_length), confidence[:, i])
                    for i in range(confidence.shape[1])
                ]).T

        else:  # current_length > target_length
            # Subsample
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            landmarks = landmarks[indices]
            confidence = confidence[indices]

        return landmarks[:target_length], confidence[:target_length]

    def get_processing_report(self) -> str:
        """Get human-readable processing report"""
        report = "=" * 60 + "\n"
        report += "PROCESSING LOG\n"
        report += "=" * 60 + "\n"
        for log_entry in self.processing_log:
            report += log_entry + "\n"
        report += "=" * 60 + "\n"
        return report

    def _log(self, message: str) -> None:
        """Add log entry"""
        self.processing_log.append(message)


class SequenceDataset:
    """Dataset class for model-ready sequences"""

    def __init__(self, sequences: List[np.ndarray], labels: Optional[List[int]] = None):
        """
        Args:
            sequences: List of landmark sequences (all same length)
            labels: Optional list of class labels
        """
        self.sequences = np.array(sequences)
        self.labels = labels
        self.num_samples = len(sequences)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int]]:
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]

    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Export as numpy arrays"""
        return self.sequences, np.array(self.labels) if self.labels else None

    def save(self, filepath: str) -> None:
        """Save dataset to file"""
        data = {
            'sequences': self.sequences.tolist(),
            'labels': self.labels
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath: str) -> 'SequenceDataset':
        """Load dataset from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        sequences = np.array(data['sequences'])
        labels = data['labels']
        return SequenceDataset(sequences, labels)

    def split(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple['SequenceDataset', 'SequenceDataset']:
        """Split into train/test"""
        np.random.seed(seed)
        indices = np.random.permutation(self.num_samples)
        train_size = int(self.num_samples * train_ratio)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_seqs = self.sequences[train_indices]
        train_labels = [self.labels[i] for i in train_indices] if self.labels else None

        test_seqs = self.sequences[test_indices]
        test_labels = [self.labels[i] for i in test_indices] if self.labels else None

        return (
            SequenceDataset(train_seqs, train_labels),
            SequenceDataset(test_seqs, test_labels)
        )
