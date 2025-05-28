# Nombre del archivo: train_yolo.py

from argparse import ArgumentParser

from ultralytics import YOLO
from ultralytics import settings

def arguments():
    parser = ArgumentParser(
        description="Train YOLO model")
    parser.add_argument("--model", type=str,
                        help="Path to model yaml or .pt file")
    parser.add_argument("--cfg", type=str,
                        help="Path to the cfg definining train options)")
    parser.add_argument("--dataset", type=str,
                        help="Path to the root folder of the database")
    args = parser.parse_args()

    return args
if __name__ == '__main__':
    settings.update({"tensorboard":True,"mlflow":True})
    args = arguments()
    # Load a model
    model = YOLO(args.model)  # build a new model from scratch
    # Use the model
    # En train_yolo.py
    if args.cfg:
        model.train(data=args.dataset, cfg=args.cfg)
    else:
        # Si no se proporciona cfg, Ultralytics YOLO usará hiperparámetros por defecto
        # O puedes especificar aquí argumentos comunes como epochs, batch_size, etc.
        # Ejemplo: model.train(data=args.dataset, epochs=100, batch=16, imgsz=640)
        print("INFO: No se proporcionó archivo de configuración (--cfg). Usando hiperparámetros por defecto de YOLO o los especificados en model.train().")
        model.train(data=args.dataset, epochs=50, batch=8, imgsz=640) # Ejemplo, ajusta según tus necesidades y hardware
        
        
# python scripts\train_yolo.py --model yolov8n.pt --dataset ..\dataset_UDIS_dividido\dataset.yaml --cfg default.yaml   
