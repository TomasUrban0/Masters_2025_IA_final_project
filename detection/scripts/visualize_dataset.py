# Nombre del archivo: visualize_dataset.py

from argparse import ArgumentParser
import fiftyone as fo
from fiftyone import ViewField as F # Para la consulta de F("ground_truth.detections")
import os
import glob # Para buscar archivos
import yaml # Para leer el dataset.yaml

# Comando de ejemplo para ejecutar:
# (Leguajes) C:\Users\ma2mo\Documents\Master\vision\Trabajo-final\UDIT-main\UDIT-main>python .\scripts\visualize_dataset.py --p ..\..\dataset_train\train\ --f yolo --s train

def load_coco_dataset(dataset_dir):
    # Carga del dataset COCO (esta función se mantiene como en tus versiones anteriores)
    print(f"--- Iniciando carga de dataset COCO ---")
    abs_dataset_dir = os.path.abspath(dataset_dir)
    # Asumimos una estructura típica para COCO, ajústala si es necesario
    data_path = os.path.join(abs_dataset_dir, "data") # Donde están las imágenes
    labels_path = os.path.join(abs_dataset_dir, "annotations", "instances_train.json") # Ejemplo de ruta de anotaciones

    print(f"Directorio absoluto del dataset (dataset_dir): {abs_dataset_dir}")
    print(f"Ruta esperada para imágenes COCO: {data_path}")
    print(f"¿Existe la carpeta de imágenes? {os.path.isdir(data_path)}")
    print(f"Ruta esperada para JSON de etiquetas COCO: {labels_path}")
    print(f"¿Existe el JSON de etiquetas? {os.path.exists(labels_path)}")

    if not os.path.isdir(data_path) or not os.path.exists(labels_path):
        print("¡ERROR! Las rutas para el dataset COCO (imágenes o JSON de etiquetas) no son válidas.")
        return None

    dataset_name = "coco_viz_dataset"
    if fo.dataset_exists(dataset_name):
        print(f"Eliminando dataset FiftyOne preexistente: {dataset_name}")
        fo.delete_dataset(dataset_name)
    
    try:
        coco_dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=data_path,
            labels_path=labels_path,
            name=dataset_name,
            include_id=True, 
        )
        coco_dataset.compute_metadata() 
        print(f"--- Carga de dataset COCO finalizada ---")
        print(f"Nombre del dataset en FiftyOne: {coco_dataset.name}")
        print(f"Número de muestras importadas: {len(coco_dataset)}")
        return coco_dataset
    except Exception as e:
        print(f"Ocurrió un error durante la carga del dataset COCO: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_yolo_dataset(dataset_dir, split_name):
    print(f"--- Iniciando carga MANUAL de dataset YOLO para el split '{split_name}' ---")
    abs_dataset_dir = os.path.abspath(dataset_dir) # Directorio raíz pasado con --p
    yaml_path = os.path.join(abs_dataset_dir, "dataset.yaml")
    obj_names_path = os.path.join(abs_dataset_dir, "obj.names")

    # 1. Leer nombres de clases de obj.names
    class_names = []
    if os.path.exists(obj_names_path):
        with open(obj_names_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"Clases leídas de obj.names: {class_names}")
        if not class_names:
            print(f"¡ERROR! obj.names en '{obj_names_path}' está vacío o no se pudo leer correctamente.")
            return None
    else:
        print(f"¡ERROR! obj.names no encontrado en '{obj_names_path}'")
        return None

    # 2. Leer dataset.yaml para obtener la ruta de las imágenes para el split dado
    images_subdir_for_split = None
    labels_subdir_for_split = None

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f_yaml:
            try:
                config_yaml = yaml.safe_load(f_yaml)
                print(f"Contenido de dataset.yaml parseado: {config_yaml}")
                
                # 'path' en yaml es relativo a la ubicación del yaml (abs_dataset_dir)
                base_path_from_yaml = os.path.normpath(os.path.join(abs_dataset_dir, config_yaml.get('path', '.')))
                
                images_rel_path_in_yaml = config_yaml.get(split_name) # ej., 'images/train'
                if images_rel_path_in_yaml:
                    images_subdir_for_split = os.path.normpath(os.path.join(base_path_from_yaml, images_rel_path_in_yaml))
                    print(f"Ruta de imágenes para el split '{split_name}': {images_subdir_for_split}")
                    if not os.path.isdir(images_subdir_for_split):
                        print(f"¡ERROR! La carpeta de imágenes '{images_subdir_for_split}' no existe.")
                        return None
                    
                    # Construir la ruta de etiquetas basada en la de imágenes
                    # Asume que 'images' en la ruta se reemplaza por 'labels'
                    # ej., '.../dataset_root/images/train' -> '.../dataset_root/labels/train'
                    
                    # Dividir la ruta en partes para buscar 'images'
                    # Primero, obtener la parte de la ruta relativa a la base_path_from_yaml
                    rel_img_path_for_split = os.path.relpath(images_subdir_for_split, base_path_from_yaml)

                    if "images" not in rel_img_path_for_split.split(os.sep):
                         print(f"¡ERROR! No se pudo construir la ruta de etiquetas desde '{images_subdir_for_split}'.")
                         print(f"La ruta relativa '{rel_img_path_for_split}' (basada en 'path: {config_yaml.get('path', '.')}' en el YAML y el split '{split_name}')")
                         print("no contiene una subcarpeta 'images' como se esperaba para inferir la carpeta 'labels'.")
                         print("Ejemplo esperado: 'path/images/split_name' -> 'path/labels/split_name'")
                         return None
                    
                    # Reemplazar la primera 'images' por 'labels' en la ruta relativa
                    rel_lbl_path_for_split = rel_img_path_for_split.replace("images", "labels", 1)
                    labels_subdir_for_split = os.path.normpath(os.path.join(base_path_from_yaml, rel_lbl_path_for_split))

                    print(f"Ruta de etiquetas inferida para el split '{split_name}': {labels_subdir_for_split}")
                    if not os.path.isdir(labels_subdir_for_split):
                        print(f"¡ERROR! La carpeta de etiquetas inferida '{labels_subdir_for_split}' no existe.")
                        return None
                else:
                    print(f"¡ERROR! El split '{split_name}' (ej. 'train', 'val') no está definido como clave en dataset.yaml o no tiene una ruta de imágenes asociada.")
                    return None
            except yaml.YAMLError as e:
                print(f"¡ERROR! Al parsear dataset.yaml: {e}")
                return None
    else:
        print(f"¡ERROR! dataset.yaml no encontrado en '{yaml_path}'")
        return None
    
    # 3. Crear o cargar el dataset de FiftyOne
    dataset_name = f"yolo_manual_load_{split_name}"
    if fo.dataset_exists(dataset_name):
        print(f"Eliminando dataset FiftyOne preexistente: {dataset_name}")
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)
    dataset.default_classes = class_names 
    dataset.save() 

    # 4. Recorrer imágenes y cargar etiquetas
    filepaths_in_split = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"): # Añadidas más extensiones comunes
        filepaths_in_split.extend(glob.glob(os.path.join(images_subdir_for_split, ext)))
    
    print(f"Encontradas {len(filepaths_in_split)} imágenes en '{images_subdir_for_split}'")
    
    samples_to_add = []
    count_labels_files_found = 0
    count_detections_loaded = 0

    for image_filepath in filepaths_in_split:
        sample = fo.Sample(filepath=image_filepath)
        
        filename_no_ext, _ = os.path.splitext(os.path.basename(image_filepath))
        label_filepath = os.path.join(labels_subdir_for_split, filename_no_ext + ".txt")
        
        detections_for_sample = []
        if os.path.exists(label_filepath):
            count_labels_files_found += 1
            with open(label_filepath, 'r', encoding='utf-8') as f_label:
                for line_number, line_content in enumerate(f_label):
                    parts = line_content.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            if not (0 <= class_id < len(class_names)):
                                print(f"Advertencia: ID de clase {class_id} fuera de rango (0-{len(class_names)-1}) en '{label_filepath}', línea {line_number+1}. Ignorando detección.")
                                continue
                            
                            if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                                print(f"Advertencia: Coordenadas no normalizadas o inválidas en '{label_filepath}', línea {line_number+1}: {[x_center, y_center, width, height]}. Ignorando detección.")
                                continue
                            if width <= 0 or height <= 0:
                                print(f"Advertencia: Ancho o alto no positivo en '{label_filepath}', línea {line_number+1}: w={width}, h={height}. Ignorando detección.")
                                continue

                            x_min = x_center - width / 2.0
                            y_min = y_center - height / 2.0
                            
                            x_min = max(0.0, x_min)
                            y_min = max(0.0, y_min)
                            width = min(width, 1.0 - x_min)
                            height = min(height, 1.0 - y_min)

                            detections_for_sample.append(
                                fo.Detection(
                                    label=class_names[class_id], 
                                    bounding_box=[x_min, y_min, width, height],
                                    index=class_id 
                                )
                            )
                            count_detections_loaded +=1
                        except ValueError:
                            print(f"Advertencia: Línea mal formateada (no se pudieron convertir números) en '{label_filepath}', línea {line_number+1}: '{line_content.strip()}'")
        
        if detections_for_sample:
             sample["ground_truth"] = fo.Detections(detections=detections_for_sample)
        samples_to_add.append(sample)

    if samples_to_add:
        dataset.add_samples(samples_to_add)
        
    print(f"Procesadas {len(samples_to_add)} imágenes.")
    print(f"Se encontraron y procesaron {count_labels_files_found} archivos de etiquetas correspondientes.")
    print(f"Se cargaron un total de {count_detections_loaded} detecciones.")
    print(f"Número final de muestras en el dataset '{dataset.name}': {len(dataset)}")

    if len(dataset) > 0:
        first_sample_with_gt = dataset.match(F("ground_truth.detections").exists()).first()
        if first_sample_with_gt:
            print("\n--- Información de la primera muestra CON ETIQUETAS (cargada manualmente) ---")
            print(f"Filepath: {first_sample_with_gt.filepath}")
            if first_sample_with_gt.ground_truth and first_sample_with_gt.ground_truth.detections:
                print("Detections en 'ground_truth':")
                for i, det in enumerate(first_sample_with_gt.ground_truth.detections[:5]):
                    print(f"  Detección {i}: Label='{det.label}', BBox={det.bounding_box}, Index (ClassID)={det.index}")
            else:
                print("  No hay detecciones 'ground_truth' para esta muestra.")
        else:
            print("No se encontraron muestras con etiquetas 'ground_truth' en el dataset.")
        print("--------------------------------------------------------")

    return dataset

def arguments():
    parser=ArgumentParser(description="Visualize UDIT NN/CV project dataset")
    parser.add_argument("--path","--p",type=str, required=True, help="Path to the root folder of the dataset (e.g., where dataset.yaml and obj.names are, and where images/labels folders are relative to 'path' in YAML).")
    parser.add_argument("--format","--f",type=str, choices=["yolo","coco"], required=True, help="Format of the dataset")
    parser.add_argument("--split","--s",type=str, default="train", help="Split of the dataset (e.g., 'train', 'val', 'test'. Default: 'train')")
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguments()
    dataset_to_launch = None 

    if args.format =="coco":
        dataset_to_launch = load_coco_dataset(args.path)
    elif args.format =="yolo":
        dataset_to_launch = load_yolo_dataset(args.path, args.split)

    if dataset_to_launch and len(dataset_to_launch) > 0:
        print(f"Lanzando la aplicación FiftyOne para el dataset: {dataset_to_launch.name} con {len(dataset_to_launch)} muestras.")
        session = fo.launch_app(dataset_to_launch, auto=False) # auto=False para que puedas copiar la URL si es necesario
        print(f"FiftyOne App URL: {session.url}")
        session.wait()
    elif dataset_to_launch and len(dataset_to_launch) == 0:
        print("El dataset se cargó pero está vacío. La aplicación FiftyOne no se lanzará. Revisa los logs y la configuración.")
    else:
        print("No se pudo cargar el dataset o ocurrió un error. La aplicación FiftyOne no se lanzará. Revisa los mensajes anteriores.")
        
# python scripts\visualize_dataset.py --p ..\dataset_UDIS_dividido --f yolo --s test