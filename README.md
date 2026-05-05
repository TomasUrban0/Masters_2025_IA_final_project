# Self-Driving Cars Object Detection & Anonymous Web Scraping

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MSc Artificial Intelligence Final Project** — UDIT, 2025

End-to-end final project for the Neural Networks & Computer Vision subjects of the AI Master's programme. Combines a real-time object detection system for autonomous driving scenarios with an anonymous web scraping library for automated data collection through Tor.

📄 **Full project report:** [Trabajo_Final.pdf](Trabajo_Final.pdf)

---

## Table of Contents

- [Overview](#overview)
- [Object Detection Module](#object-detection-module)
- [AutoTor Module](#autotor-module)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualizations](#results--visualizations)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

| Component | Description | Key Tech |
|-----------|-------------|----------|
| **[detection/](detection/)** | YOLOv8-based multi-class detection for self-driving scenarios | PyTorch, Ultralytics, MLflow |
| **[autotor/](autotor/)** | High-level API for anonymous parallel web requests via Tor | requests, stem, fake-useragent |

---

## Object Detection Module

<div align="center">
  <img src="detection/img/sdc.jpg" width="60%" alt="Self-Driving Cars Detection"/>
</div>

### Highlights

- **7-class detection** model for autonomous driving (vehicles, pedestrians, cyclists, traffic signs, etc.)
- **Stratified dataset splitting** with multi-label aware stratification using `scikit-multilearn`
- **Experiment tracking** with MLflow and TensorBoard integration
- **Dataset visualization** via FiftyOne for quality inspection
- **Multiple training iterations** with hyperparameter tuning across 6 runs

### Architecture

The pipeline uses **YOLOv8-nano** (3.1M parameters, 8.9 GFLOPs) with a CSPDarknet backbone and PANet neck:

```
Input (640x640) → CSPDarknet Backbone → PANet Neck → Multi-scale Detection Head (P3/P4/P5)
```

### Training Pipeline

```
Raw Dataset → Stratified Split (70/15/15) → YOLOv8n Training → Validation → MLflow Logging
                                                    ↓
                                        FiftyOne Visualization
```

### Training Performance

| Metric | Value |
|--------|-------|
| mAP@50 | **48.6%** |
| mAP@50-95 | **28.6%** |
| Precision | 64.7% |
| Recall | 42.5% |

---

## AutoTor Module

High-level Python API for parallel anonymous web requests through Tor with:

- Automated Tor download and initialization
- Multi-process Tor circuit management
- Thread pool-based parallel requests
- Automatic IP rotation with randomised user-agents
- Context manager support for resource cleanup

### Quick Example

```python
from autotor import TorRequests
from threading import Lock

LOCK = Lock()

class MyScraper(TorRequests):
    def request(self, elem, n_id):
        session = self.get_tor_session(n_id)
        self.renew_tor_ip(n_id)
        response = session.get(f"https://example.com/page/{elem}")
        with LOCK:
            self.results.append(response.text)

with MyScraper(n_process=5, tor_root=".") as tor:
    tor.threaded_request(range(100))
```

---

## Project Structure

```
.
├── README.md
├── LICENSE
├── Trabajo_Final.pdf              # Final project report
├── Entorno_anaconda.yaml          # Full Conda environment
│
├── docs/                          # Documentation assets
│   ├── images/                    # Result visualizations
│   └── training_results.csv       # Sample training metrics
│
├── detection/                     # Object Detection module
│   ├── scripts/
│   │   ├── train_yolo.py          # YOLO training with CLI args
│   │   ├── val_yolo.py            # Model validation
│   │   ├── split_dataset.py       # Stratified dataset splitting
│   │   ├── visualize_dataset.py   # FiftyOne dataset inspection
│   │   └── visualize_val_predictions.py
│   ├── img/sdc.jpg                # Project banner
│   ├── yolov8.yaml                # Model architecture (7 classes)
│   └── requirements.txt
│
└── autotor/                       # Anonymous scraping module
    ├── src/autotor/
    │   ├── autotor_base.py        # Core TorRequests class
    │   └── autotor_ip.py          # IP rotation example
    ├── setup.py
    └── requirements.txt
```

---

## Installation

### Option 1 — Conda (recommended)

```bash
conda env create -f Entorno_anaconda.yaml
conda activate <env_name>
```

### Option 2 — pip

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# Object Detection
pip install -r detection/requirements.txt

# AutoTor
pip install -r autotor/requirements.txt
```

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (for training)
- Tor Browser (for AutoTor — auto-downloaded on first run)

---

## Usage

### Object Detection

```bash
cd detection

# Stratified dataset split
python scripts/split_dataset.py \
    --input_dir ../dataset_train \
    --output_dir ../dataset_split \
    --obj_names_file ../dataset_train/obj.names \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# Train YOLOv8 (downloads pretrained weights automatically)
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --dataset ../dataset_split/dataset.yaml

# Validate
python scripts/val_yolo.py \
    --model runs/detect/train/weights/best.pt \
    --dataset ../dataset_split/dataset.yaml

# Visualize predictions interactively
python scripts/visualize_val_predictions.py \
    --path ../dataset_split \
    --weights runs/detect/train/weights/best.pt
```

### AutoTor

```bash
cd autotor
pip install .

python src/main.py
```

---

## Results & Visualizations

### Training Batch Sample

A sample batch with ground-truth bounding boxes for the 7-class autonomous driving dataset:

<div align="center">
  <img src="docs/images/training_batch.jpg" width="80%" alt="Training Batch with annotations"/>
</div>

### Class Distribution

Distribution of annotations across the 7 classes in the training set:

<div align="center">
  <img src="docs/images/class_distribution.jpg" width="60%" alt="Class Distribution"/>
</div>

### Label Correlogram

Spatial distribution and correlation between bounding-box centers and dimensions:

<div align="center">
  <img src="docs/images/labels_correlogram.jpg" width="60%" alt="Labels Correlogram"/>
</div>

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n (3.1M params) |
| Image size | 640×640 |
| Batch size | 16 |
| Optimizer | Auto (AdamW) |
| Augmentation | Mosaic (disabled in last 10 epochs) |
| Mixed precision | AMP enabled |
| Hardware | NVIDIA CUDA GPU |

All experiments are tracked via **MLflow** and **TensorBoard** for full reproducibility.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch 2.5, Ultralytics YOLOv8/v11 |
| **Experiment Tracking** | MLflow, TensorBoard |
| **Data Visualization** | FiftyOne, Matplotlib |
| **Dataset Processing** | scikit-multilearn, NumPy, OpenCV |
| **Networking** | Tor, stem, requests[socks] |
| **Environment** | Conda, CUDA 12.1 |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>MSc in Artificial Intelligence — UDIT — 2025</sub>
</div>
