# Claude Code Agent Instructions - LT-AsteroidDetection

> **For developers using Claude Code (claude.ai or GitHub Copilot) to build this project.**

This file provides structured, step-by-step instructions for an AI coding assistant to implement the complete asteroid detection pipeline for Liverpool Telescope data.

---

## Project Overview

Build a machine learning pipeline for detecting asteroid streaks in time-series imagery from the Liverpool Telescope. The project has:
- **Baseline**: ResNet18 classifier + Line Segment Detector (LSD) hybrid pipeline
- **Improvement 1**: EfficientNet classifier + tuned LSD + semi-synthetic data
- **Improvement 2**: YOLOv8 end-to-end streak detector
- **Evaluation**: Systematic comparison on real LT test data

---

## Phase 1: Setup (Start Here)

### 1.1 Create Project Structure

Create the following directory structure from `README.md`:

```
LT-AsteroidDetection/
├── src/                      # Source code packages
│   ├── preprocessing/        # Image preprocessing
│   ├── models/               # Deep learning models
│   ├── localisation/         # Streak localisation
│   ├── simulation/           # Semi-synthetic data
│   ├── evaluation/           # Metrics and evaluation
│   └── utils/                # Utilities
├── notebooks/                # Google Colab notebooks
├── data/                     # Data (not tracked)
├── models/                   # Model weights (not tracked)
└── results/                  # Experiment results (not tracked)
```

### 1.2 Create requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
torchmetrics>=1.0.0
tensorboard>=2.12.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
astropy>=5.3.0
ccdproc>=2.4.0
pylint>=2.17.0
pytest>=7.3.0
black>=23.0.0
flake8>=6.0.0
pre-commit>=3.0.0
mlflow>=2.3.0
optuna>=3.1.0
colab-xterm>=3.5.1
ultralytics>=8.0.0
gradio>=3.32.0
ipywidgets>=8.0.0
tqdm>=4.65.0
joblib>=1.3.0
imageio>=2.31.0
```

---

## Phase 2: Preprocessing Pipeline

### 2.1 Implement `src/preprocessing/gamma_correction.py`

Create a `GammaCorrection` class with:
- Method to estimate optimal gamma from image histogram
- Apply gamma transformation: `output = input^gamma`
- Support for batch processing FITS images

### 2.2 Implement `src/preprocessing/noise_estimation.py`

Create a `NoiseEstimator` class:
- Estimate read noise from image statistics
- Calculate median absolute deviation (MAD)
- Return sigma estimate for adaptive filtering

### 2.3 Implement `src/preprocessing/adaptive_smoothing.py`

Create `AdaptiveSmoother` class:
- Gaussian smoothing with kernel size based on noise estimate
- Median filtering option for cosmic ray removal
- Edge-preserving smoothing (bilateral filter)

### 2.4 Implement `src/preprocessing/ecc_alignment.py`

Create `ECCAligner` class:
- Load frames 1, 2, 3 from triplet
- Use OpenCV `findTransformECC` with `MOTION_EUCLIDEAN`
- Apply transformations to align frames 2 and 3 to frame 1
- Return aligned triplet

### 2.5 Implement `src/preprocessing/temporal_merging.py`

Create `TemporalMerger` class:
- Compute frame differences: `diff1 = frame2 - frame1`, `diff2 = frame3 - frame2`
- Apply threshold based on noise estimate
- Compute OR of thresholded differences
- Apply morphological opening/closing
- Return merged motion map

### 2.6 Create `src/preprocessing/__init__.py`

Expose all classes and create a unified `Preprocessor` pipeline class that chains:
`gamma -> smoothing -> alignment -> temporal_merge`

---

## Phase 3: Baseline Models

### 3.1 Implement `src/models/resnet18_classifier.py`

Create `ResNet18Classifier`:
- Load pretrained ResNet18 from torchvision
- Replace final layer for 2-class output (asteroid/non-asteroid)
- Accept 3-channel merged triplet or single merged image
- Include dropout and batch normalization
- Method `forward(x)` for inference
- Method `predict(image, threshold=0.5)` for binary classification

### 3.2 Implement `src/models/efficientnet_classifier.py`

Create `EfficientNetClassifier`:
- Load EfficientNet-B0 or B1 from torchvision
- Same interface as ResNet18Classifier
- Add squeeze-and-excitation attention blocks
- Method `forward(x)` and `predict(image)`

### 3.3 Implement `src/models/training.py`

Create training utilities:
- `train_model(model, train_loader, val_loader, epochs, device)`
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping (patience=10)
- Model checkpointing (save best by validation loss)
- TensorBoard logging
- K-fold cross-validation wrapper
- Mixed precision training support

---

## Phase 4: Localisation (LSD)

### 4.1 Implement `src/localisation/line_segment_detector.py`

Create `LineSegmentDetector`:
- Use OpenCV `createLineSegmentDetector`
- Detect line segments in merged image
- Filter by minimum length (default: 25 pixels)
- Filter by angle (horizontal/vertical streak exclusion)
- Return list of detected line segments with endpoints

### 4.2 Implement `src/localisation/blob_filtering.py`

Create `BlobFilter` class:
- Connected component analysis on thresholded merged image
- Filter blobs by:
  - Area (min/max pixel count)
  - Elongation ratio (major/minor axis)
  - Orientation (reject near-horizontal/vertical)
- Return filtered blob candidates

### 4.3 Implement `src/localisation/centroid_extraction.py`

Create `CentroidExtractor`:
- Fit line of best fit to blob pixels
- Calculate centroid (weighted mean)
- Return bounding box, centroid coordinates, and streak angle

---

## Phase 5: Semi-Synthetic Data Generation

### 5.1 Implement `src/simulation/psf_model.py`

Create `PSFModel` for Liverpool Telescope:
- Pixel scale: 0.31 arcsec/pixel
- Typical FWHM: 1.5-3.0 arcsec
- Generate 2D Gaussian PSF kernel
- Include astigmatism model (elliptical Gaussian)
- Support for variable seeing conditions

### 5.2 Implement `src/simulation/streak_generator.py`

Create `StreakGenerator`:
- Sample parameters:
  - Streak length: 10-200 pixels (uniform distribution)
  - Streak brightness (SNR): 3-50 (realistic range)
  - Orientation: 0-360 degrees
  - Starting position: random within image
  - Apparent rate of motion: 0.5-10 arcsec/min
- Generate streak as convolved line with PSF
- Add realistic noise (read noise + sky background)

### 5.3 Implement `src/simulation/data_injection.py`

Create `DataInjector`:
- Load real LT background triplet (non-asteroid)
- Inject generated streak into each frame at motion-consistent positions
- Store bounding box labels for each injected streak
- Generate YAML annotation files (YOLO format)
- Create dataset splits (train/val/test)

---

## Phase 6: YOLOv8 Detector

### 6.1 Implement `src/models/yolo_detector.py`

Create `YOLOStreakDetector`:
- Load YOLOv8 model (nano or small variant)
- Train on merged LT images with bounding box labels
- Custom training with:
  - Image size: 512x512
  - Batch size: 16
  - Epochs: 100
  - Augmentation: rotation, scale, brightness
- Methods:
  - `train(data_yaml, epochs)`
  - `predict(image, conf_threshold=0.5)`
  - `evaluate(test_dataset)`
- Optional: LSD verification post-processing

---

## Phase 7: Evaluation

### 7.1 Implement `src/evaluation/metrics.py`

Create evaluation functions:
- `compute_metrics(predictions, ground_truth)`:
  - Image-level: accuracy, precision, recall, F1
  - Streak-level: detection rate, false positive rate
  - Localisation: IoU, centroid distance error
- ROC curve and AUC
- PR curve (important for imbalanced data)
- Confidence calibration curves

### 7.2 Implement `src/evaluation/comparative_analysis.py`

Create comparison utilities:
- `compare_methods(results_dict)` for side-by-side tables
- Performance breakdown by:
  - SNR bins
  - Streak length bins
  - Apparent rate bins
```

## Phase 8: Google Colab Notebooks

### Notebook 1: Project Setup & Data Exploration
- Mount Google Drive
- Install dependencies
- Load sample LT FITS files
- Explore image statistics (histograms, noise distribution)
- Visualize sample triplets (original, aligned, merged)

### Notebook 2: Data Preprocessing Pipeline
- Implement and test each preprocessing step
- Parameter tuning (gamma, smoothing kernel, thresholds)
- Batch processing with progress bar
- Save preprocessed datasets

### Notebook 3: Baseline ResNet18+LSD Pipeline
- Load preprocessed data
- Train ResNet18 with 5-fold cross-validation
- Reproduce thesis results (~76% F1)
- Apply LSD localisation to CNN positives
- Generate annotated output triplets

### Notebook 4: Streak Simulator & Semi-Synthetic Data
- Train PSF model from real LT images
- Generate streaks with varied parameters
- Inject streaks into real backgrounds
- Create YOLO-format annotations
- Build train/val/test splits

### Notebook 5: YOLOv8 Streak Detector
- Train YOLOv8 on semi-synthetic data
- Fine-tune on real LT streaks
- Validate on held-out test set
- Optional LSD verification
- Generate detection outputs

### Notebook 6: Evaluation & Comparison
- Load all method results
- Compute metrics for each pipeline
- Performance vs SNR/length/rate plots
- Ablation study (real vs synthetic vs mixed training)
- Generate publication-quality figures

---

## Phase 9: Utilities & Configuration

### `src/utils/config.py`
Central configuration with:
- LT instrument parameters (pixel scale, FWHM, gain, read noise)
- Preprocessing defaults
- Model hyperparameters
- Training configurations
- Evaluation thresholds
```python
LT_PARAMS = {
    'pixel_scale': 0.31,  # arcsec/pixel
    'typical_fwhm': 2.0,   # arcsec
    'gain': 1.5,           # e-/ADU
    'read_noise': 5.0,     # electrons
    'dark_current': 0.02   # e-/s/pixel
}
```

### `src/utils/fits_io.py`
- Load FITS files with astropy
- Extract header metadata (exposure, filter, date)
- Handle NaN/INF values
- WCS coordinate extraction

---

## Phase 10: Code Quality & Testing

### Testing
- `pytest` for unit tests on each module
- Test FITS loading and preprocessing on sample data
- Test model forward pass
- Test metric computation

### Documentation
- Google-style docstrings for all functions
- Type hints throughout
- README for each subpackage

### Linting & Formatting
- `black` for formatting
- `flake8` for linting
- `pylint` for static analysis
- `pre-commit` hooks

---

## Expected File Order (Build Sequence)

1. `requirements.txt` → Install dependencies
2. `src/utils/config.py` → LT parameters
3. `src/utils/fits_io.py` → FITS utilities
4. `src/preprocessing/` → Gamma, noise, smoothing, ECC, merge
5. `src/localisation/` → LSD, blob filtering, centroid
6. `src/models/resnet18_classifier.py` → Baseline CNN
7. `src/models/training.py` → Training loop
8. `src/simulation/` → PSF, streak gen, injection
9. `src/models/efficientnet_classifier.py` → Improved CNN
10. `src/models/yolo_detector.py` → YOLO detector
11. `src/evaluation/` → Metrics, comparison
12. Notebooks 1-6 → End-to-end demonstration

---

## Notes for Claude Code

- Always check if a file already exists before creating it
- Use `black` formatting for all Python files
- Add type hints to function signatures
- Include `if __name__ == '__main__'` guards in executable scripts
- Use `logging` module instead of `print` statements
- All models should support `device` (CPU/GPU) parameter
- Use `tqdm` for progress bars in training loops
- Save model checkpoints with `torch.save(state_dict, path)`
- Use `astropy.io.fits` for FITS handling
- Test each module individually before integration
- Keep Colab notebooks under 10 cells per major section
- Document all assumptions about LT data format

---

## Build Status Check

Before starting each phase, run:
```bash
# Check project structure
tree src/ notebooks/

# Run quick tests
python -m pytest src/ -v

# Check imports
python -c "from src.preprocessing import Preprocessor; print('OK')"
```

If phase X fails, debug it completely before moving to phase X+1.
