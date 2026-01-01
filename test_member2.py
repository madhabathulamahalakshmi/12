"""
Member 2 Data Cleaning & Normalization - Testing & Demonstration
This script tests all cleaning and normalization functions
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

from utils.data_cleaning import (
    LandmarkCleaner,
    LandmarkNormalizer,
    LandmarkValidator,
    load_landmarks_from_json,
    save_cleaned_landmarks
)
from utils.data_augmentation import (
    LandmarkAugmenter,
    AugmentationStrategy,
    AugmentationPipeline
)
from utils.sequence_processor import (
    SequenceProcessor,
    SequenceDataset
)


def create_synthetic_data(num_frames: int = 30, num_points: int = 21) -> tuple:
    """Create synthetic landmark data for testing"""
    landmarks = np.random.rand(num_frames, num_points, 3) * 0.5 + 0.25
    confidence = np.random.rand(num_frames, num_points)
    
    # Add some noise/missing data
    landmarks[::3, :, :2] += np.random.normal(0, 0.05, (len(range(0, num_frames, 3)), num_points, 2))
    confidence[::5, :] *= 0.3  # Some frames with low confidence
    
    return landmarks, confidence


def print_separator(title: str = ""):
    """Print a nice separator"""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)
    else:
        print("=" * 70)


def test_landmark_cleaner():
    """Test landmark cleaning functionality"""
    print_separator("TEST 1: Landmark Cleaning")
    
    landmarks, confidence = create_synthetic_data(30, 21)
    print(f"Original shape: {landmarks.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Average confidence: {np.mean(confidence):.3f}")
    
    cleaner = LandmarkCleaner(window_size=5)
    
    # Step 1: Outlier removal
    print("\n[Step 1] Removing outliers...")
    cleaned = cleaner.remove_outliers(landmarks)
    print(f"✓ Outlier removal completed")
    
    # Step 2: Smoothing
    print("\n[Step 2] Smoothing landmarks...")
    smoothed = cleaner.smooth_landmarks(cleaned)
    print(f"✓ Smoothing completed")
    
    # Step 3: Handle missing
    print("\n[Step 3] Handling missing landmarks...")
    final_lm, final_conf = cleaner.handle_missing_landmarks(smoothed, confidence, confidence_threshold=0.5)
    print(f"✓ Missing landmarks handled")
    print(f"  Points recovered: {np.sum(final_conf > 0)}/{final_conf.size}")
    
    return final_lm, final_conf


def test_landmark_normalizer(landmarks: np.ndarray, confidence: np.ndarray):
    """Test landmark normalization"""
    print_separator("TEST 2: Landmark Normalization")
    
    normalizer = LandmarkNormalizer()
    
    # Test position-scale normalization
    print("Testing 'position_scale' normalization method...")
    normalized_ps = normalizer.normalize(landmarks, method="position_scale")
    print(f"✓ Normalized using position-scale")
    print(f"  Range X: [{np.min(normalized_ps[:, :, 0]):.3f}, {np.max(normalized_ps[:, :, 0]):.3f}]")
    print(f"  Range Y: [{np.min(normalized_ps[:, :, 1]):.3f}, {np.max(normalized_ps[:, :, 1]):.3f}]")
    
    # Test hand-based normalization
    print("\nTesting 'hand_based' normalization method...")
    normalized_hb = normalizer.normalize(landmarks, method="hand_based")
    print(f"✓ Normalized using hand-based method")
    print(f"  Range X: [{np.min(normalized_hb[:, :, 0]):.3f}, {np.max(normalized_hb[:, :, 0]):.3f}]")
    print(f"  Range Y: [{np.min(normalized_hb[:, :, 1]):.3f}, {np.max(normalized_hb[:, :, 1]):.3f}]")
    
    return normalized_ps


def test_landmark_validator(landmarks: np.ndarray, confidence: np.ndarray):
    """Test landmark validation"""
    print_separator("TEST 3: Landmark Validation")
    
    validator = LandmarkValidator()
    
    is_valid, metrics = validator.validate_sequence(landmarks, confidence)
    
    print(f"Validation Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return is_valid, metrics


def test_data_augmentation(landmarks: np.ndarray, confidence: np.ndarray):
    """Test data augmentation"""
    print_separator("TEST 4: Data Augmentation")
    
    augmenter = LandmarkAugmenter(seed=42)
    
    strategies = [
        (AugmentationStrategy.HORIZONTAL_FLIP, {}),
        (AugmentationStrategy.VERTICAL_FLIP, {}),
        (AugmentationStrategy.ROTATE, {'angle': 15}),
        (AugmentationStrategy.SCALE, {'scale_range': (0.8, 1.2)}),
        (AugmentationStrategy.NOISE, {'noise_level': 0.01}),
    ]
    
    print("Testing individual augmentation strategies:\n")
    
    for strategy, kwargs in strategies:
        try:
            aug_lm, aug_conf = augmenter.augment_sequence(
                landmarks, confidence, strategy, **kwargs
            )
            print(f"✓ {strategy.value:20s} - Output shape: {aug_lm.shape}")
        except Exception as e:
            print(f"✗ {strategy.value:20s} - Error: {str(e)}")
    
    # Batch augmentation
    print("\n\nBatch Augmentation (3 versions):")
    aug_list, conf_list = augmenter.batch_augment(landmarks, confidence, num_augmentations=3)
    print(f"✓ Generated {len(aug_list)} augmented sequences")
    for i, (lm, cf) in enumerate(zip(aug_list, conf_list)):
        print(f"  Version {i+1}: {lm.shape}")


def test_sequence_processor():
    """Test the complete sequence processing pipeline"""
    print_separator("TEST 5: Complete Sequence Processing Pipeline")
    
    landmarks, confidence = create_synthetic_data(30, 21)
    
    processor = SequenceProcessor(
        window_size=5,
        confidence_threshold=0.5,
        normalization_method="position_scale"
    )
    
    print("Processing sequence with augmentation...")
    results = processor.process_sequence(
        landmarks,
        confidence,
        apply_augmentation=True,
        num_augmentations=2
    )
    
    print("\n✓ Processing completed")
    print(f"\nProcessing Steps:")
    for step, shape in results['steps'].items():
        print(f"  {step:30s} → {shape}")
    
    print(f"\nValidation: {'✓ PASS' if results['valid'] else '✗ FAIL'}")
    
    print(f"\nMetrics:")
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nAugmented sequences created: {len(results['augmented_sequences'])}")
    
    print("\n--- Processing Log ---")
    print(processor.get_processing_report())
    
    return results


def test_fixed_length_sequences(landmarks: np.ndarray, confidence: np.ndarray):
    """Test fixed-length sequence conversion"""
    print_separator("TEST 6: Fixed-Length Sequence Conversion")
    
    processor = SequenceProcessor()
    
    target_length = 30
    print(f"Original length: {len(landmarks)}")
    print(f"Target length: {target_length}")
    
    strategies = ['repeat', 'pad', 'interpolate']
    
    for strategy in strategies:
        try:
            fixed_lm, fixed_conf = processor.create_fixed_length_sequence(
                landmarks, confidence, target_length=target_length, padding_strategy=strategy
            )
            print(f"✓ {strategy:15s} → {fixed_lm.shape}")
        except Exception as e:
            print(f"✗ {strategy:15s} → Error: {str(e)}")


def test_sample_file_processing():
    """Test processing actual sample file if it exists"""
    print_separator("TEST 7: Processing Sample Landmark File")
    
    sample_path = Path("data/samples/landmarks_sample.json")
    
    if not sample_path.exists():
        print(f"⚠ Sample file not found: {sample_path}")
        print("Skipping file processing test")
        return
    
    print(f"Processing: {sample_path}")
    
    processor = SequenceProcessor()
    
    try:
        output_path = Path("data/cleaned_landmarks/cleaned_sample.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = processor.process_json_file(
            str(sample_path),
            output_path=str(output_path),
            apply_augmentation=False
        )
        
        print(f"\n✓ File processing completed")
        print(f"  Original shape: {results['original_shape']}")
        print(f"  Validation: {'PASS' if results['valid'] else 'FAIL'}")
        print(f"  Output file: {output_path}")
        
    except Exception as e:
        print(f"✗ Error processing file: {str(e)}")


def test_dataset_class():
    """Test the SequenceDataset class"""
    print_separator("TEST 8: SequenceDataset Class")
    
    # Create synthetic dataset
    sequences = [create_synthetic_data(30, 21)[0] for _ in range(10)]
    labels = [0, 1] * 5
    
    dataset = SequenceDataset(sequences, labels)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sample shape: {dataset[0][0].shape}")
    print(f"Labels: {dataset[0][1]}, {dataset[1][1]}")
    
    # Test split
    train_ds, test_ds = dataset.split(train_ratio=0.8)
    print(f"\n✓ Split into train/test:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    
    # Test save/load
    save_path = Path("data/test_dataset.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset.save(str(save_path))
    print(f"\n✓ Dataset saved to: {save_path}")
    
    loaded_ds = SequenceDataset.load(str(save_path))
    print(f"✓ Dataset loaded: {len(loaded_ds)} samples")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MEMBER 2: DATA CLEANING & NORMALIZATION - TEST SUITE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Test 1: Cleaning
        final_lm, final_conf = test_landmark_cleaner()
        
        # Test 2: Normalization
        normalized_lm = test_landmark_normalizer(final_lm, final_conf)
        
        # Test 3: Validation
        is_valid, metrics = test_landmark_validator(normalized_lm, final_conf)
        
        # Test 4: Augmentation
        test_data_augmentation(normalized_lm, final_conf)
        
        # Test 5: Full pipeline
        test_sequence_processor()
        
        # Test 6: Fixed length
        test_fixed_length_sequences(final_lm, final_conf)
        
        # Test 7: Sample file
        test_sample_file_processing()
        
        # Test 8: Dataset
        test_dataset_class()
        
        print_separator("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print_separator(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
