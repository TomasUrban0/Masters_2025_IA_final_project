# Object Detection for Self-Driving Cars

<div align="center">
  <img src="./img/sdc.jpg" width="60%" alt="Self-Driving Cars"/>
</div>

## Overview

Real-time multi-class object detection system for autonomous driving scenarios using YOLOv8/v11. The pipeline includes stratified dataset splitting, model training with experiment tracking, and interactive visualization of predictions.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/train_yolo.py` | Train YOLOv8/v11 model with configurable hyperparameters |
| `scripts/val_yolo.py` | Validate trained model on test/val split |
| `scripts/split_dataset.py` | Stratified multi-label dataset splitting (train/val/test) |
| `scripts/visualize_dataset.py` | Inspect dataset with FiftyOne (YOLO & COCO formats) |
| `scripts/visualize_val_predictions.py` | Visualize model predictions vs ground truth |

## Model Architecture

Uses **YOLOv8-nano** customized for 7 detection classes (see `yolov8.yaml`):

- **Backbone:** CSPDarknet with C2f modules
- **Neck:** PANet with multi-scale feature fusion (P3/P4/P5)
- **Head:** Decoupled detection head
- **Parameters:** 3.1M | **GFLOPs:** 8.9

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Split dataset
python scripts/split_dataset.py \
    --input_dir ../dataset_train \
    --output_dir ../dataset_split \
    --obj_names_file ../dataset_train/obj.names \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# Train
python scripts/train_yolo.py --model yolov8n.pt --dataset ../dataset_split/dataset.yaml

# Validate
python scripts/val_yolo.py --model runs/detect/train/weights/best.pt --dataset ../dataset_split/dataset.yaml

# Visualize
python scripts/visualize_dataset.py --path ../dataset_split --format yolo --split val
```

## Experiment Tracking

All training runs are logged via **MLflow** and **TensorBoard**:

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri runs/mlflow

# Launch TensorBoard
tensorboard --logdir runs/detect
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- See `requirements.txt` for full dependencies
