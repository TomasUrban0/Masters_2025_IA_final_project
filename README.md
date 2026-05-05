# Self-Driving Cars Object Detection & Anonymous Web Scraping

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MSc Artificial Intelligence Final Project** — UDIT, 2025

This repository contains the final project for the Neural Networks & Computer Vision subjects of the AI Master's programme. It integrates two complementary components: a real-time object detection system for autonomous driving scenarios, and an anonymous web scraping library for automated data collection through Tor.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Object Detection Module](#object-detection-module)
- [AutoTor Module](#autotor-module)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Project Overview

| Component | Description | Key Tech |
|-----------|-------------|----------|
| **Object Detection** | YOLOv8-based multi-class vehicle/pedestrian detection for self-driving scenarios | PyTorch, Ultralytics, MLflow |
| **AutoTor** | High-level API for anonymous parallel web requests via Tor with circuit renewal | requests, stem, fake-useragent |

---

## Object Detection Module

<div align="center">
  <img src="UDIT-main/UDIT-main/img/sdc.jpg" width="60%" alt="Self-Driving Cars Detection"/>
</div>

### Highlights

- **7-class detection** model for autonomous driving scenarios (vehicles, pedestrians, cyclists, traffic signs, etc.)
- **Stratified dataset splitting** with multi-label aware stratification using `scikit-multilearn`
- **Experiment tracking** with MLflow and TensorBoard integration
- **Dataset visualization** via FiftyOne for quality inspection
- **Multiple training iterations** with hyperparameter tuning

### Architecture

The detection pipeline uses **YOLOv8-nano** (3.1M parameters, 8.9 GFLOPs) with a CSPDarknet backbone and PANet neck:

```
Input (640x640) → CSPDarknet Backbone → PANet Neck → Multi-scale Detection Head (P3/P4/P5)
```

### Training Pipeline

```
Raw Dataset → Stratified Split (70/15/15) → YOLOv8n Training → Validation → MLflow Logging
                                                    ↓
                                        FiftyOne Visualization
```

### Best Results (8 epochs, YOLOv8n)

| Metric | Value |
|--------|-------|
| mAP@50 | **48.6%** |
| mAP@50-95 | **28.6%** |
| Precision | 64.7% |
| Recall | 42.5% |

---

## AutoTor Module

High-level Python API to perform parallel anonymous web requests through Tor with:

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
├── Trabajo_Final.pdf              # Final project report
├── Entorno_anaconda.yaml          # Full Conda environment specification
│
├── UDIT-main/UDIT-main/          # Object Detection module
│   ├── scripts/
│   │   ├── train_yolo.py          # YOLO training with CLI args
│   │   ├── val_yolo.py            # Model validation
│   │   ├── split_dataset.py       # Stratified dataset splitting
│   │   ├── visualize_dataset.py   # FiftyOne dataset inspection
│   │   └── visualize_val_predictions.py
│   ├── runs/                      # Training outputs & MLflow tracking
│   ├── yolov8.yaml               # Model architecture (7 classes)
│   └── requirements.txt
│
└── AutoTor-main/AutoTor-main/    # Anonymous scraping module
    ├── src/autotor/
    │   ├── autotor_base.py        # Core TorRequests class
    │   └── autotor_ip.py          # IP rotation example
    ├── setup.py
    └── requirements.txt
```

---

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f Entorno_anaconda.yaml
conda activate <env_name>
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Object Detection
pip install -r UDIT-main/UDIT-main/requirements.txt

# AutoTor
pip install -r AutoTor-main/AutoTor-main/requirements.txt
```

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (for training)
- Tor Browser (for AutoTor, auto-downloaded on first run)

---

## Usage

### Train Object Detection Model

```bash
cd UDIT-main/UDIT-main

# Split dataset with stratification
python scripts/split_dataset.py \
    --input_dir ../dataset_train \
    --output_dir ../dataset_split \
    --obj_names_file ../dataset_train/obj.names \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# Train YOLOv8
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --dataset ../dataset_split/dataset.yaml

# Validate
python scripts/val_yolo.py \
    --model runs/detect/train/weights/best.pt \
    --dataset ../dataset_split/dataset.yaml

# Visualize predictions with FiftyOne
python scripts/visualize_val_predictions.py \
    --path ../dataset_split \
    --weights runs/detect/train/weights/best.pt
```

### Use AutoTor

```bash
cd AutoTor-main/AutoTor-main
pip install .

python src/main.py
```

---

## Results

Training was performed iteratively across 6 runs, progressively tuning hyperparameters. The model was trained on a custom 7-class dataset for autonomous driving object detection using YOLOv8-nano with:

- **Image size:** 640x640
- **Batch size:** 16
- **Optimizer:** Auto (AdamW)
- **Augmentation:** Mosaic (disabled last 10 epochs)
- **Hardware:** CUDA GPU with AMP (mixed precision)

All experiments are tracked via MLflow for reproducibility.

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built as part of the MSc in Artificial Intelligence at UDIT, 2025</sub>
</div>
