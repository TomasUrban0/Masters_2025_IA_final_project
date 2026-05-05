"""Train a YOLOv8/v11 model with MLflow + TensorBoard tracking.

Example:
    python scripts/train_yolo.py --model yolov8n.pt \
        --dataset ../dataset_split/dataset.yaml --cfg default.yaml
"""

from argparse import ArgumentParser

from ultralytics import YOLO, settings


def parse_arguments():
    parser = ArgumentParser(description="Train YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model yaml or .pt file")
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to the cfg defining train options")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the root folder of the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    settings.update({"tensorboard": True, "mlflow": True})
    args = parse_arguments()

    model = YOLO(args.model)

    if args.cfg:
        model.train(data=args.dataset, cfg=args.cfg)
    else:
        print("INFO: No --cfg provided. Falling back to defaults.")
        model.train(data=args.dataset, epochs=50, batch=8, imgsz=640)
