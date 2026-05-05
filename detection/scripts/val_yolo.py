"""Validate a trained YOLO model on a dataset split.

Example:
    python scripts/val_yolo.py --model runs/detect/train/weights/best.pt \
        --dataset ../dataset_split/dataset.yaml
"""

from argparse import ArgumentParser

from ultralytics import YOLO


def parse_arguments():
    parser = ArgumentParser(description="Validate YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to the cfg defining validation options")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.yaml file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model = YOLO(args.model)
    model.val(data=args.dataset, cfg=args.cfg)
