
# How to Run & Check Member 2's Work

**Document Version:** 1.0  
**Date:** January 1, 2026  
**Purpose:** Step-by-step guide to verify Member 2 functionality  
**Audience:** Member 2, Team leads, QA testers

---

## **Quick Check (30 seconds)**

```powershell
.\.venv\Scripts\Activate.ps1
python test_member2.py
```

**Expected Output:** `✓ All tests passed!`

If you see this → Member 2 is working ✓

---

## **Method 1: Run the Test Suite (Recommended) ✅**

This is the best way to verify all Member 2 functionality at once.

### **Steps:**

```powershell
# 1. Navigate to project folder
cd "C:\Users\P C\Desktop\new\camera"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Run all tests
python test_member2.py
```

### **What It Tests:**

| Test # | Name | Checks |
|--------|------|--------|
| 1 | Landmark Cleaning | Outlier removal, smoothing, missing data handling |
| 2 | Normalization | Position-scale and hand-based normalization |
| 3 | Validation | Confidence scoring, velocity checking |
| 4 | Augmentation | All 6 augmentation strategies |
| 5 | Complete Pipeline | End-to-end cleaning + normalization |
| 6 | Fixed-Length Conversion | Padding, repeat, interpolate strategies |
| 7 | File Processing | JSON I/O and file handling |
| 8 | Dataset Class | Creation, splitting, save/load |

### **Expected Output:**

```
Running Member 2 Test Suite
===========================

✓ Test 1: Landmark Cleaning - PASS
✓ Test 2: Normalization - PASS
✓ Test 3: Validation - PASS
✓ Test 4: Augmentation - PASS
✓ Test 5: Complete Pipeline - PASS
✓ Test 6: Fixed-Length Conversion - PASS
✓ Test 7: File Processing - PASS
✓ Test 8: Dataset Class - PASS

✓ All 8 tests passed!
```

### **Success Criteria:**

✅ All 8 tests show PASS  
✅ No error messages  
✅ Output files created  

---

## **Method 2: Run Quick Start Examples**

Use copy-paste code examples to test specific functionality.

### **Example 1: Process a Single File**

```python
from utils.sequence_processor import SequenceProcessor

processor = SequenceProcessor(
    window_size=5,
    confidence_threshold=0.5,
    normalization_method="position_scale"
)

results = processor.process_json_file(
    input_path="data/raw_landmarks/session_1767171443.json",
    output_path="data/cleaned_landmarks/cleaned_session.json",
    apply_augmentation=True
)

print(f"Valid: {results['valid']}")
print(f"Confidence: {results['metrics']['avg_confidence']:.3f}")
```

**What it does:**
- Loads raw landmarks from JSON
- Cleans the data (removes outliers, smooths)
- Normalizes coordinates
- Saves cleaned output
- Shows quality metrics

**Expected output:**
```
Valid: True
Confidence: 0.892
```

---

### **Example 2: Batch Process All Sessions**

```python
results_list = processor.process_batch(
    input_dir="data/raw_landmarks/",
    output_dir="data/cleaned_landmarks/",
    apply_augmentation=True
)

successful = sum(1 for r in results_list if r['success'])
print(f"Processed: {successful}/{len(results_list)} files ✓")
```

**What it does:**
- Processes all JSON files in a folder
- Cleans each one
- Saves outputs
- Reports success rate

**Expected output:**
```
Processed: 3/3 files ✓
```

---

### **Example 3: Create Dataset for Training**

```python
from pathlib import Path
from utils.data_cleaning import load_landmarks_from_json
from utils.sequence_processor import SequenceDataset

processor = SequenceProcessor()
sequences = []
labels = []

for json_file in Path("data/cleaned_landmarks/").glob("*.json"):
    landmarks, confidence, _ = load_landmarks_from_json(str(json_file))
    
    fixed_lm, fixed_conf = processor.create_fixed_length_sequence(
        landmarks, confidence,
        target_length=30,
        padding_strategy="repeat"
    )
    
    sequences.append(fixed_lm)
    labels.append(0)

dataset = SequenceDataset(sequences, labels)
dataset.save("data/training_dataset.json")
print(f"✓ Created dataset with {len(dataset)} sequences")
```

**What it does:**
- Loads all cleaned files
- Converts to fixed length (30 frames)
- Creates a dataset
- Saves for machine learning

**Expected output:**
```
✓ Created dataset with 3 sequences
```

---

### **Example 4: Augment Data**

```python
from utils.data_augmentation import LandmarkAugmenter, AugmentationStrategy

augmenter = LandmarkAugmenter(seed=42)

landmarks, confidence, _ = load_landmarks_from_json(
    "data/samples/landmarks_sample.json"
)

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

print(f"✓ Generated {len(aug_list)} augmented versions")
```

**What it does:**
- Takes original data
- Creates 5 variations (rotated, scaled, noisy, flipped)
- Returns augmented versions

**Expected output:**
```
✓ Generated 5 augmented versions
```

---

### **Example 5: Validate Data Quality**

```python
from utils.data_cleaning import LandmarkValidator

validator = LandmarkValidator(confidence_threshold=0.5)
is_valid, score = validator.validate_sequence(landmarks)

print(f"Valid: {is_valid}")
print(f"Score: {score:.3f}")
print(f"All confidence > 0.5: {validator.check_confidence(landmarks)[0]}")
```

**What it does:**
- Checks if data quality is acceptable
- Validates confidence scores
- Checks for anomalies

**Expected output:**
```
Valid: True
Score: 0.892
All confidence > 0.5: True
```

---

## **Method 3: Check Output Files**

Verify that Member 2 created the expected output files.

### **View Created Files:**

```powershell
# See all cleaned files
Get-ChildItem -Path data/cleaned_landmarks/ -Name "*.json"

# See augmented files
Get-ChildItem -Path data/cleaned_landmarks/ -Name "*augmented*"

# List everything
Get-ChildItem -Path data/cleaned_landmarks/
```

### **Expected Structure:**

```
data/
├── cleaned_landmarks/
│   ├── cleaned_session_1767171443.json          ← Cleaned data
│   ├── cleaned_session_1767171443_augmented_1.json
│   ├── cleaned_session_1767171443_augmented_2.json
│   ├── cleaned_session_1767172563.json
│   └── cleaned_session_1767282269.json
└── training_dataset.json                        ← Ready for ML
```

### **File Contents:**

Each cleaned JSON file contains:
```json
{
    "sequence": [[[x1, y1, z1], [x2, y2, z2], ...]],  // 30 frames × 21 points
    "confidence": [[c1, c2, ...]],                      // Confidence scores
    "metadata": {
        "original_length": 47,
        "normalized": true,
        "smoothed": true
    }
}
```

---

## **Method 4: Verify Each Component Individually**

Test specific Member 2 functions one at a time.

### **Test Cleaning**

```python
from utils.data_cleaning import LandmarkCleaner, load_landmarks_from_json

cleaner = LandmarkCleaner(window_size=5)
landmarks, conf, _ = load_landmarks_from_json("data/samples/landmarks_sample.json")
cleaned = cleaner.clean_landmarks(landmarks)

print(f"Original shape: {landmarks.shape}")
print(f"Cleaned shape: {cleaned.shape}")
print(f"Mean confidence: {cleaned.mean():.3f}")
```

**Check:**
- Shape is (N, 21, 3) - N frames, 21 points, 3 values (x, y, z)
- Confidence between 0 and 1

---

### **Test Normalization**

```python
from utils.data_cleaning import LandmarkNormalizer

normalizer = LandmarkNormalizer()
normalizer.fit(landmarks)
normalized = normalizer.transform(landmarks)

print(f"Normalized shape: {normalized.shape}")
print(f"X range: {normalized[..., 0].min():.3f} to {normalized[..., 0].max():.3f}")
print(f"Y range: {normalized[..., 1].min():.3f} to {normalized[..., 1].max():.3f}")
```

**Check:**
- X and Y between -1 and 1 (normalized)
- Z between 0 and 1 (confidence)

---

### **Test Validation**

```python
from utils.data_cleaning import LandmarkValidator

validator = LandmarkValidator(confidence_threshold=0.5)
is_valid, score = validator.validate_sequence(landmarks)

print(f"Valid: {is_valid}")
print(f"Quality score: {score:.3f}")
print(f"Min confidence: {validator.check_confidence(landmarks)[1]:.3f}")
```

**Check:**
- `is_valid` is True/False
- `score` between 0 and 1
- All confidence > threshold

---

### **Test Augmentation**

```python
from utils.data_augmentation import LandmarkAugmenter

augmenter = LandmarkAugmenter(seed=42)

# Generate 3 augmented versions
aug_list, conf_list = augmenter.batch_augment(
    landmarks, confidence,
    num_augmentations=3
)

print(f"Original shape: {landmarks.shape}")
print(f"Augmented count: {len(aug_list)}")
print(f"Each augmented shape: {aug_list[0].shape}")
```

**Check:**
- All augmented have same shape as original
- 3 different versions created
- No NaN or Inf values

---

## **Complete Step-by-Step Verification**

Full checklist to verify everything works:

### **Step 1: Environment Setup**

```powershell
# Navigate to project
cd "C:\Users\P C\Desktop\new\camera"

# Activate environment
.\.venv\Scripts\Activate.ps1

# Verify Python
python --version  # Should show Python 3.8+
```

**✓ Check:** Python command works

---

### **Step 2: Run Test Suite**

```powershell
python test_member2.py
```

**✓ Check:** All 8 tests pass

---

### **Step 3: Check Output Files**

```powershell
Get-ChildItem data/cleaned_landmarks/ -Name "*.json"
```

**✓ Check:** See cleaned_*.json files

---

### **Step 4: Run Quick Example**

```powershell
python -c "
from utils.sequence_processor import SequenceProcessor
p = SequenceProcessor()
r = p.process_json_file('data/raw_landmarks/session_1767171443.json', 'test_output.json')
print(f'Success: {r[\"valid\"]}')
print(f'Confidence: {r[\"metrics\"][\"avg_confidence\"]:.3f}')
"
```

**✓ Check:** See `Success: True` and confidence value

---

### **Step 5: Verify Data Quality**

```powershell
python -c "
from utils.data_cleaning import load_landmarks_from_json
landmarks, conf, _ = load_landmarks_from_json('data/cleaned_landmarks/cleaned_session_1767171443.json')
print(f'Shape: {landmarks.shape}')
print(f'Min confidence: {conf.min():.3f}')
print(f'Max confidence: {conf.max():.3f}')
"
```

**✓ Check:**
- Shape is (N, 21, 3)
- Confidence between 0 and 1

---

## **Success Indicators ✅**

Your Member 2 work is complete when:

- [ ] `python test_member2.py` shows all 8 tests PASS
- [ ] No error messages or warnings
- [ ] Cleaned JSON files exist in `data/cleaned_landmarks/`
- [ ] Augmented files created (if augmentation enabled)
- [ ] Output confidence scores between 0 and 1
- [ ] Normalized data between -1 and 1
- [ ] File sizes reasonable (cleaned < original typically)
- [ ] Processing completes in < 1 minute per file

---

## **Error Indicators ❌**

Issues to watch for:

- [ ] Test failures in `test_member2.py`
- [ ] Missing output files
- [ ] Confidence scores outside [0, 1]
- [ ] Normalized data outside [-1, 1]
- [ ] NaN or Inf values in output
- [ ] Very slow processing (> 5 min per file)
- [ ] Import errors (missing packages)
- [ ] File not found errors

---

## **Troubleshooting**

### **Problem: "ModuleNotFoundError: No module named 'utils'"**
```powershell
# Make sure you're in the correct directory
cd "C:\Users\P C\Desktop\new\camera"

# Check that utils folder exists
Get-ChildItem -Path utils -Name "*.py"
```

### **Problem: "No such file or directory: data/raw_landmarks/..."**
```powershell
# Check that input files exist
Get-ChildItem -Path data/raw_landmarks/ -Name "*.json"

# Use correct filename in code
```

### **Problem: "ImportError: numpy/scipy not installed"**
```powershell
# Reinstall packages
.\.venv\Scripts\Activate.ps1
pip install numpy scipy
```

### **Problem: Test hangs or takes too long**
```powershell
# Cancel with Ctrl+C
# Check file sizes - very large files may take time
Get-ChildItem data/raw_landmarks/ | Select-Object Name, Length
```

---

## **Quick Reference: Key Files**

| File | Purpose | How to Use |
|------|---------|-----------|
| `test_member2.py` | Verify all functionality | `python test_member2.py` |
| `QUICK_START_MEMBER2.py` | Copy-paste examples | Reference for code snippets |
| `MEMBER_2_GUIDE.md` | Complete documentation | Read for detailed info |
| `utils/data_cleaning.py` | Core cleaning functions | Import and use |
| `utils/data_augmentation.py` | Augmentation functions | Import and use |
| `utils/sequence_processor.py` | Pipeline orchestration | Main interface |

---

## **What Each Module Does**

### **data_cleaning.py**
- Remove outliers (using Median Absolute Deviation)
- Smooth landmarks (using Savitzky-Golay filter)
- Interpolate missing data
- Normalize coordinates
- Validate quality

### **data_augmentation.py**
- Horizontal and vertical flips
- Rotation transformations
- Scale adjustments
- Add noise for robustness
- Time-warping for temporal variation
- Batch generation of augmented data

### **sequence_processor.py**
- Orchestrate full pipeline
- Process single files or batches
- Create fixed-length sequences
- Create training datasets
- Save/load operations

---

## **Expected Runtime**

| Task | Time |
|------|------|
| Run all tests | 5-10 seconds |
| Process 1 file | 1-2 seconds |
| Process batch (3 files) | 5-10 seconds |
| Create dataset | 2-3 seconds |
| Augment data | 1-2 seconds |

---

## **Next Steps After Verification**

Once Member 2 is verified working:

1. **For Member 3:** Provide cleaned data for AI training
   - Use `MEMBER_3_INTEGRATION_GUIDE.md`
   - Use batch processing to create training dataset

2. **For Production:** Set up real-time processing
   - Use Method 1 or 3 from integration guide
   - Process frames as they come from Member 1

3. **For Quality:** Monitor performance
   - Check confidence scores regularly
   - Validate augmented data quality
   - Monitor processing latency

---

## **Support & Questions**

| Question | Answer |
|----------|--------|
| What's the data format? | (N, 21, 3) - N frames, 21 hand points, 3 values (x,y,z) |
| What's normal confidence? | > 0.7 is good, > 0.8 is excellent |
| How long should processing take? | < 5 seconds per file on most computers |
| Can I augment my data? | Yes, see Example 4 in QUICK_START_MEMBER2.py |
| What if confidence is low? | Check raw data quality, adjust smoothing parameters |

---

**Document Complete ✓**

**For more details, see:**
- [MEMBER_2_GUIDE.md](MEMBER_2_GUIDE.md)
- [QUICK_START_MEMBER2.py](QUICK_START_MEMBER2.py)
- [MEMBER_3_INTEGRATION_GUIDE.md](MEMBER_3_INTEGRATION_GUIDE.md)
