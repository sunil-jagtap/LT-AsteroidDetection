# LT-AsteroidDetection

**Hybrid AI-Vision Framework for Asteroid Detection in Liverpool Telescope Time-Series Imagery**

> MSc Research Project | Liverpool John Moores University
> Based on dissertation: "Hybrid AI-Vision Framework for Asteroid Detection in Time-Series LT Imagery" (August 2025)

## Overview

This repository implements a complete machine learning pipeline for detecting Near-Earth Objects (NEOs) as streaks in time-series imagery from the 2.0m fully robotic Liverpool Telescope (LT). The pipeline addresses the challenge of faint, fast-moving asteroid streaks buried under noise, image artefacts, and background sources in ground-based survey images.

## Research Novelty & Contributions

> **This work adapts established deep-learning streak detection methods to the Liverpool Telescope domain. The novelty lies in the LT-specific implementation, semi-synthetic data generation, and systematic comparison of hybrid vs. end-to-end approaches on real LT data.**

### Key Contributions:
1. **First ML pipeline for NEO streak detection in Liverpool Telescope time-series triplet data**, including a CNN+LSD hybrid baseline and a YOLO-based end-to-end detector.
2. **LT-specific semi-synthetic streak dataset**: realistic streak simulation grounded in LT instrument characteristics (PSF, pixel scale, noise model) injected into real LT backgrounds.
3. **Systematic comparison** of three detection approaches on the same real LT test set:
   - Classical LSD-only baseline
   - ResNet18 + LSD hybrid pipeline
   - YOLOv8 end-to-end streak detector
4. **Ablation study** of training regimes: real-only vs synthetic-only vs mixed data.
5. **Fully reproducible codebase** with Google Colab notebooks and modular Python source code.
## Project Structure

```
LT-AsteroidDetection/
├── README.md                      # This file
├── CLAUDE_CODE_INSTRUCTIONS.md    # Detailed instructions for Claude Code agent
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/             # Image preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── gamma_correction.py
│   │   ├── noise_estimation.py
│   │   ├── adaptive_smoothing.py
│   │   ├── ecc_alignment.py
│   │   └── temporal_merging.py
│   │
│   ├── models/                    # Deep learning models
│   │   ├── __init__.py
│   │   ├── resnet18_classifier.py  # Baseline CNN classifier
│   │   ├── efficientnet_classifier.py  # Improved CNN classifier
│   │   ├── yolo_detector.py        # YOLOv8 streak detector
│   │   └── training.py            # Training loops and utilities
│   │
│   ├── localisation/              # Streak localisation
│   │   ├── __init__.py
│   │   ├── line_segment_detector.py  # LSD implementation
│   │   ├── blob_filtering.py      # Artefact filtering
│   │   └── centroid_extraction.py
│   │
│   ├── simulation/                # Semi-synthetic data generation
│   │   ├── __init__.py
│   │   ├── streak_generator.py    # Realistic streak simulation
│   │   ├── psf_model.py           # Liverpool Telescope PSF
│   │   └── data_injection.py      # Inject streaks into real backgrounds
│   │
│   ├── evaluation/                # Metrics and evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py             # Precision, recall, F1, IoU
│   │   ├── comparative_analysis.py  # Method comparison
│   │   └── visualisation.py       # Result plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuration parameters
│       ├── fits_io.py             # FITS file handling
│       └── metrics_logger.py      # Experiment tracking
│
├── notebooks/
│   ├── 01_project_setup_and_data_exploration.ipynb
│   ├── 02_data_preprocessing_pipeline.ipynb
│   ├── 03_baseline_resnet18_lsd_pipeline.ipynb
│   ├── 04_streak_simulator_semi_synthetic_data.ipynb
│   ├── 05_yolo_streak_detector.ipynb
│   └── 06_evaluation_and_comparison.ipynb
│
├── data/                          # Data directory (not tracked)
│   ├── raw/                       # Raw LT FITS images
│   ├── processed/                 # Preprocessed images
│   ├── synthetic/                 # Semi-synthetic samples
│   └── splits/                    # Train/val/test split files
│
├── models/                        # Saved model weights (not tracked)
│   ├── baseline_resnet18/
│   ├── improved_efficientnet/
│   └── yolo_detector/
│
└── results/                       # Experiment results (not tracked)
    ├── metrics/
    ├── plots/
    └── tables/
```

## Methods

### Pipeline A: Baseline (ResNet18 + LSD Hybrid)
**From original thesis** - A two-stage approach:
1. **Preprocessing**: Gamma correction, adaptive Gaussian/median smoothing, ECC frame alignment
2. **Temporal merging**: Frame differencing + OR operation to highlight motion
3. **Classification**: Modified ResNet18 CNN trained to classify triplets as asteroid/non-asteroid
4. **Localisation**: Line Segment Detector (LSD) applied only to CNN-positive triplets
5. **Post-processing**: Blob filtering + line-of-best-fit for centroid extraction

### Pipeline B: Improved Classifier + LSD
**Research improvement** - Same hybrid architecture with:
- Replaced ResNet18 with EfficientNet-B0/B1 backbone
- Expanded training with semi-synthetic data
- Hyperparameter-tuned LSD thresholds

### Pipeline C: YOLOv8 End-to-End Detector
**Research improvement** - Single-stage detection:
- YOLOv8 trained on merged images with bounding-box labels from semi-synthetic data
- Direct streak localisation without post-hoc geometric processing
- Optional LSD verification step for false-positive reduction

## Installation

```bash
# Clone the repository
git clone https://github.com/sunil-jagtap/LT-AsteroidDetection.git
cd LT-AsteroidDetection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```
## Google Colab Notebooks

All notebooks can be run directly on Google Colab with a T4 GPU runtime:

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| `01_project_setup_and_data_exploration.ipynb` | Environment setup, data exploration, LT image statistics | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/01_project_setup_and_data_exploration.ipynb) |
| `02_data_preprocessing_pipeline.ipynb` | Gamma correction, smoothing, ECC alignment, temporal merging | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/02_data_preprocessing_pipeline.ipynb) |
| `03_baseline_resnet18_lsd_pipeline.ipynb` | Reproduce thesis baseline (ResNet18 + LSD), 5-fold CV | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/03_baseline_resnet18_lsd_pipeline.ipynb) |
| `04_streak_simulator_semi_synthetic_data.ipynb` | LT-specific streak simulator, semi-synthetic dataset generation | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/04_streak_simulator_semi_synthetic_data.ipynb) |
| `05_yolo_streak_detector.ipynb` | YOLOv8 training and inference on semi-synthetic data | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/05_yolo_streak_detector.ipynb) |
| `06_evaluation_and_comparison.ipynb` | Final evaluation, method comparison, ablation study | [Open in Colab](https://colab.research.google.com/github/sunil-jagtap/LT-AsteroidDetection/blob/main/notebooks/06_evaluation_and_comparison.ipynb) |

## Usage Examples

### Run Preprocessing Pipeline
```python
from src.preprocessing import Preprocessor
from src.preprocessing import temporal_merging

preprocessor = Preprocessor(gamma=0.8, smoothing_type='adaptive')
processed_frames = [preprocessor.process(f) for f in triplet_frames]
merged_image = temporal_merging.merge_frames(processed_frames)
```

### Train Baseline Classifier
```python
from src.models import ResNet18Classifier, train_model

model = ResNet18Classifier(num_classes=2)
model = train_model(model, train_loader, val_loader, epochs=50)
```

### Run YOLOv8 Detector
```python
from src.models.yolo_detector import YOLOStreakDetector

detector = YOLOStreakDetector('models/yolo_detector/best.pt')
detections = detector.predict(merged_image, conf_threshold=0.5)
```

### Evaluate on Test Set
```python
from src.evaluation import evaluate_detector, compute_metrics

metrics = compute_metrics(predictions, ground_truth)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```
## Expected Results (Baseline from Thesis)

| Metric | Value |
|--------|-------|
| Accuracy | ~76% |
| Precision | ~76% |
| Recall | ~77% |
| F1 Score | ~0.76 |

Target improvements with new methods:
- **Pipeline B (EfficientNet + LSD)**: F1 >= 0.82
- **Pipeline C (YOLOv8)**: F1 >= 0.85, +20% detection rate for faint streaks (SNR < 5)

## Related Work

| Method | Telescope/Survey | Approach | Key Reference |
|--------|-----------------|----------|---------------|
| DeepStreaks | ZTF | CNN streak classifier | 96-98% TP at <1% FP |
| Nir et al. (2022) | Simulated + Real | EfficientNet-B1 + synthetic streaks | 98.7% accuracy |
| Euclid Pipeline | Euclid | CNN + RNN + XGBoost | 3-stage detection+linking |
| Irureta-Goyena et al. | Wide-field | VGG-16 + Hough Transform | CNN classification + geometry |
| VideoMAE-style | Simulated | Transformer for tracklets | Temporal attention |
| PiNN + YOLOX | Multi-telescope | Physics-informed neural net + object detection | PSF conditioning |
| **This Work** | **Liverpool Telescope** | **ResNet18/EfficientNet/YOLOv8 + LSD** | **-** |

## Development Roadmap

- [x] Phase 1: Repository setup & project structure
- [ ] Phase 2: Data preprocessing pipeline (Notebook 01-02)
- [ ] Phase 3: Baseline ResNet18+LSD reproduction (Notebook 03)
- [ ] Phase 4: Streak simulator & semi-synthetic data (Notebook 04)
- [ ] Phase 5: YOLOv8 end-to-end detector (Notebook 05)
- [ ] Phase 6: Final evaluation & comparative analysis (Notebook 06)
- [ ] Phase 7: Paper writing & submission

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{suryawanshi2025asteroid,
  author = {Tanmayee Suryawanshi},
  title = {Hybrid AI-Vision Framework for Asteroid Detection in Time-Series LT Imagery},
  year = {2025},
  type = {MSc Dissertation},
  institution = {Liverpool John Moores University}
}

@misc{jagtap2026ltasteroid,
  author = {Sunil Jagtap},
  title = {LT-AsteroidDetection: Hybrid AI-Vision Framework for Asteroid Detection in Liverpool Telescope Time-Series Imagery},
  year = {2026},
  url = {https://github.com/sunil-jagtap/LT-AsteroidDetection}
}
```

## Contact

For questions or collaboration, please open an issue on this repository.
