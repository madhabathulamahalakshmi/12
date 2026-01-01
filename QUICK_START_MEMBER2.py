"""
Quick Start Guide for Member 2
Simple examples to get started immediately
"""

# ============================================================
# EXAMPLE 1: Process Raw Landmarks with Full Pipeline
# ============================================================

from utils.sequence_processor import SequenceProcessor

processor = SequenceProcessor(
    window_size=5,
    confidence_threshold=0.5,
    normalization_method="position_scale"
)

# Process a single file
results = processor.process_json_file(
    input_path="data/raw_landmarks/session_1767171443.json",
    output_path="data/cleaned_landmarks/cleaned_session_1767171443.json",
    apply_augmentation=True  # Generate augmented versions
)

print("Processing Status:")
print(f"  Valid: {results['valid']}")
print(f"  Shape: {results['original_shape']}")
print(f"  Avg Confidence: {results['metrics']['avg_confidence']:.3f}")

# ============================================================
# EXAMPLE 2: Process All Sessions in Batch
# ============================================================

from pathlib import Path

results_list = processor.process_batch(
    input_dir="data/raw_landmarks/",
    output_dir="data/cleaned_landmarks/",
    apply_augmentation=True
)

successful = sum(1 for r in results_list if r['success'])
print(f"\nBatch Processing: {successful}/{len(results_list)} files ✓")

# ============================================================
# EXAMPLE 3: Create Fixed-Length Sequences for Model
# ============================================================

import numpy as np
from utils.data_cleaning import load_landmarks_from_json
from utils.sequence_processor import SequenceDataset

processor = SequenceProcessor()
sequences = []
labels = []

for json_file in Path("data/cleaned_landmarks/").glob("*.json"):
    # Load cleaned landmarks
    landmarks, confidence, _ = load_landmarks_from_json(str(json_file))
    
    # Convert to fixed length (30 frames)
    fixed_lm, fixed_conf = processor.create_fixed_length_sequence(
        landmarks, confidence,
        target_length=30,
        padding_strategy="repeat"
    )
    
    sequences.append(fixed_lm)
    labels.append(0)  # Replace with actual label

# Create and save dataset
dataset = SequenceDataset(sequences, labels)
dataset.save("data/training_dataset.json")
print(f"\n✓ Created dataset with {len(dataset)} sequences")

# ============================================================
# EXAMPLE 4: Augment Data for Better Training
# ============================================================

from utils.data_augmentation import LandmarkAugmenter, AugmentationStrategy

augmenter = LandmarkAugmenter(seed=42)

# Load sample data
landmarks, confidence, _ = load_landmarks_from_json(
    "data/samples/landmarks_sample.json"
)

# Generate 5 augmented versions
aug_list, conf_list = augmenter.batch_augment(
    landmarks, confidence,
    num_augmentations=5,
    strategies=[
        AugmentationStrategy.ROTATE,
        AugmentationStrategy.SCALE,
        AugmentationStrategy.NOISE,
        AugmentationStrategy.HORIZONTAL_FLIP
    ]
)

print(f"\n✓ Generated {len(aug_list)} augmented sequences")

# ============================================================
# EXAMPLE 5: Custom Cleaning Pipeline
# ============================================================

from utils.data_cleaning import (
    LandmarkCleaner, LandmarkNormalizer, LandmarkValidator
)

cleaner = LandmarkCleaner(window_size=5)
normalizer = LandmarkNormalizer()
validator = LandmarkValidator()

# Step 1: Clean
cleaned = cleaner.remove_outliers(landmarks)
smoothed = cleaner.smooth_landmarks(cleaned)
final_lm, final_conf = cleaner.handle_missing_landmarks(
    smoothed, confidence, confidence_threshold=0.5
)

# Step 2: Normalize
normalized = normalizer.normalize(final_lm, method="position_scale")

# Step 3: Validate
is_valid, metrics = validator.validate_sequence(normalized, final_conf)

print(f"\n✓ Cleaning Pipeline Complete")
print(f"  Valid: {is_valid}")
print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")

# ============================================================
# EXAMPLE 6: Process & Save Cleaned Data
# ============================================================

from utils.data_cleaning import save_cleaned_landmarks

# Process
processor = SequenceProcessor()
results = processor.process_sequence(
    landmarks, confidence,
    apply_augmentation=False
)

# Save
save_cleaned_landmarks(
    results['cleaned_landmarks'],
    results['cleaned_confidence'],
    output_path="data/my_cleaned_landmarks.json",
    metadata={
        'source': 'session_123',
        'processing_date': '2024-01-01'
    }
)

print(f"\n✓ Saved cleaned landmarks to: data/my_cleaned_landmarks.json")

# ============================================================
# EXAMPLE 7: Get Processing Report
# ============================================================

processor = SequenceProcessor()
results = processor.process_json_file(
    "data/samples/landmarks_sample.json",
    apply_augmentation=False
)

print("\n" + processor.get_processing_report())

# ============================================================
# EXAMPLE 8: Use Different Normalization Methods
# ============================================================

normalizer = LandmarkNormalizer()

# Method 1: Position-Scale (Center and scale by bounding box)
norm_ps = normalizer.normalize(landmarks, method="position_scale")
print(f"\nPosition-Scale Normalization:")
print(f"  X range: [{np.min(norm_ps[:, :, 0]):.3f}, {np.max(norm_ps[:, :, 0]):.3f}]")
print(f"  Y range: [{np.min(norm_ps[:, :, 1]):.3f}, {np.max(norm_ps[:, :, 1]):.3f}]")

# Method 2: Hand-Based (Normalize by wrist-to-finger distance)
norm_hb = normalizer.normalize(landmarks, method="hand_based")
print(f"\nHand-Based Normalization:")
print(f"  X range: [{np.min(norm_hb[:, :, 0]):.3f}, {np.max(norm_hb[:, :, 0]):.3f}]")
print(f"  Y range: [{np.min(norm_hb[:, :, 1]):.3f}, {np.max(norm_hb[:, :, 1]):.3f}]")

# ============================================================
# EXAMPLE 9: Train/Test Split
# ============================================================

from utils.sequence_processor import SequenceDataset

# Create dataset
sequences = np.random.rand(20, 30, 21, 3)  # 20 sequences
labels = [0, 1] * 10
dataset = SequenceDataset(sequences, labels)

# Split
train_ds, test_ds = dataset.split(train_ratio=0.8, seed=42)

print(f"\nDataset Split:")
print(f"  Train: {len(train_ds)} samples")
print(f"  Test:  {len(test_ds)} samples")

# ============================================================
# EXAMPLE 10: Quality Check on Multiple Files
# ============================================================

from utils.data_cleaning import LandmarkValidator

validator = LandmarkValidator()

print("\nQuality Check on All Cleaned Files:")
print("-" * 50)

for json_file in Path("data/cleaned_landmarks/").glob("*.json"):
    landmarks, confidence, _ = load_landmarks_from_json(str(json_file))
    is_valid, metrics = validator.validate_sequence(landmarks, confidence)
    
    status = "✓ PASS" if is_valid else "✗ FAIL"
    print(f"{json_file.name}: {status}")
    print(f"  Confidence: {metrics['avg_confidence']:.3f}")
    print(f"  Valid ratio: {metrics['valid_ratio']:.3f}")

print("-" * 50)
