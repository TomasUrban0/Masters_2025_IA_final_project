# Nombre del archivo: split_dataset.py
import os
import glob
import shutil
import argparse
import numpy as np
import yaml # Necesitarás PyYAML: pip install PyYAML
from collections import Counter
# Asegúrate de tener scikit-multilearn instalado: pip install scikit-multilearn
from skmultilearn.model_selection import IterativeStratification


def load_class_names(obj_names_file_path, input_dir_for_relative_path=None):
    path_to_check = obj_names_file_path
    if input_dir_for_relative_path and \
       not os.path.dirname(obj_names_file_path) and \
       not os.path.isabs(obj_names_file_path):
        path_to_check = os.path.join(input_dir_for_relative_path, obj_names_file_path)
    elif not os.path.isabs(path_to_check):
        path_to_check = os.path.abspath(path_to_check)
    print(f"INFO: Intentando cargar nombres de clases desde: {path_to_check}")
    if not os.path.exists(path_to_check):
        print(f"¡ERROR CRÍTICO! Archivo de nombres de clases '{path_to_check}' no encontrado.")
        raise FileNotFoundError(f"Archivo de clases {path_to_check} no encontrado.")
    with open(path_to_check, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        print(f"¡ERROR CRÍTICO! El archivo de clases '{path_to_check}' está vacío.")
        raise ValueError(f"El archivo de clases '{path_to_check}' está vacío.")
    print(f"INFO: Clases cargadas ({len(class_names)}): {class_names}")
    return class_names

def get_image_label_pairs(input_dir, image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
    image_files = []
    images_path = os.path.join(input_dir, 'images')
    labels_path = os.path.join(input_dir, 'labels')
    if not os.path.isdir(images_path):
        print(f"¡ERROR CRÍTICO! Directorio de imágenes no encontrado: {images_path}")
        return [], []
    if not os.path.isdir(labels_path):
        print(f"¡ERROR CRÍTICO! Directorio de etiquetas no encontrado: {labels_path}")
        return [], []
    print(f"INFO: Buscando imágenes en: {images_path}")
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(images_path, f'*{ext.upper()}')))
    image_files = sorted(list(set(image_files)))
    valid_image_files, valid_label_files = [], []
    if not image_files:
        print(f"ADVERTENCIA: No se encontraron archivos de imagen en {images_path} con las extensiones {image_extensions}")
        return [], []
    for img_f_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_f_path))[0]
        label_f_path = os.path.join(labels_path, f'{base_name}.txt')
        if os.path.exists(label_f_path):
            valid_image_files.append(img_f_path)
            valid_label_files.append(label_f_path)
        else:
            print(f"ADVERTENCIA: No se encontró archivo de etiqueta para {img_f_path} (se esperaba en {label_f_path}). Se omitirá.")
    if not valid_image_files:
        print(f"ADVERTENCIA: No se encontraron pares válidos de imagen/etiqueta.")
        return [], []
    print(f"INFO: Encontrados {len(valid_image_files)} pares de imagen/etiqueta.")
    return valid_image_files, valid_label_files

def create_multilabel_matrix(label_files, class_names):
    num_samples, num_classes = len(label_files), len(class_names)
    Y = np.zeros((num_samples, num_classes), dtype=int)
    for i, label_file in enumerate(label_files):
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        class_idx_from_file = int(parts[0])
                        if 0 <= class_idx_from_file < num_classes: Y[i, class_idx_from_file] = 1
                        else: print(f"ADVERTENCIA: ID de clase {class_idx_from_file} fuera de rango en {label_file} (línea {line_num}). Se ignora.")
                    except ValueError: print(f"ADVERTENCIA: Formato incorrecto ID clase en {label_file} (línea {line_num}): '{parts[0]}'. Se ignora.")
        except Exception as e: print(f"ERROR: Procesando {label_file}: {e}.")
    return Y

def copy_files_to_split(original_indices, all_image_files, all_label_files, output_dir_root, split_name):
    img_dest_path, lbl_dest_path = os.path.join(output_dir_root, 'images', split_name), os.path.join(output_dir_root, 'labels', split_name)
    os.makedirs(img_dest_path, exist_ok=True)
    os.makedirs(lbl_dest_path, exist_ok=True)
    split_image_relative_paths = []
    print(f"INFO: Copiando {len(original_indices)} archivos al split '{split_name}'...")
    for original_idx in original_indices:
        img_src, lbl_src = all_image_files[original_idx], all_label_files[original_idx]
        img_basename, lbl_basename = os.path.basename(img_src), os.path.basename(lbl_src)
        try:
            shutil.copy2(img_src, os.path.join(img_dest_path, img_basename))
            shutil.copy2(lbl_src, os.path.join(lbl_dest_path, lbl_basename))
            split_image_relative_paths.append(os.path.join('images', split_name, img_basename).replace(os.sep, '/'))
        except Exception as e: print(f"ERROR: Copiando {img_src} o {lbl_src}: {e}")
    return split_image_relative_paths

def create_yolo_aux_files(output_dir, class_names, train_image_out_paths, val_image_out_paths, test_image_out_paths):
    with open(os.path.join(output_dir, "obj.names"), 'w', encoding='utf-8') as f:
        for name in class_names: f.write(f"{name}\n")
    print(f"INFO: Creado archivo obj.names en: {os.path.join(output_dir, 'obj.names')}")
    for paths, name in [(train_image_out_paths, 'train'), (val_image_out_paths, 'val'), (test_image_out_paths, 'test')]:
        if paths:
            with open(os.path.join(output_dir, f'{name}.txt'), 'w', encoding='utf-8') as f:
                for p in paths: f.write(f"./{p}\n")
            print(f"INFO: Creado archivo {name}.txt en: {output_dir}")
    yaml_data = {'path': '.', 'nc': len(class_names), 'names': class_names}
    if train_image_out_paths: yaml_data['train'] = 'images/train'
    if val_image_out_paths: yaml_data['val'] = 'images/val'
    if test_image_out_paths: yaml_data['test'] = 'images/test'
    yaml_data = {k:v for k,v in yaml_data.items() if v is not None}
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"INFO: Creado archivo dataset.yaml en: {os.path.join(output_dir, 'dataset.yaml')}")

def print_class_distribution_from_files(label_files_in_split, class_names, split_name_display):
    print(f"\n--- Distribución de clases en {split_name_display} ---")
    if not label_files_in_split: print(f"No hay archivos de etiquetas para '{split_name_display}'."); return
    num_classes, class_counts, total_annotations, images_with_annotations_count = len(class_names), Counter(), 0, 0
    for label_file_path in label_files_in_split:
        try:
            has_annotations_in_file = False
            if not os.path.exists(label_file_path): print(f"ADVERTENCIA (Stats): {label_file_path} no existe."); continue
            with open(label_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(); class_idx = int(parts[0])
                    if not parts: continue
                    if 0 <= class_idx < num_classes: class_counts[class_names[class_idx]] += 1; total_annotations += 1; has_annotations_in_file = True
            if has_annotations_in_file: images_with_annotations_count +=1
        except Exception as e: print(f"ERROR: Leyendo {label_file_path} para stats: {e}")
    print(f"Imágenes con anotaciones en {split_name_display}: {images_with_annotations_count} de {len(label_files_in_split)}.")
    print(f"Anotaciones totales (objetos) en {split_name_display}: {total_annotations}")
    if total_annotations > 0:
        for name in class_names: print(f"  Clase '{name}': {class_counts[name]} ({(class_counts[name] / total_annotations) * 100:.2f}%)")
    else: print("No se encontraron anotaciones válidas.")

def main():
    parser = argparse.ArgumentParser(description="Divide un dataset YOLO de forma estratificada.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directorio raíz (con 'images', 'labels').")
    parser.add_argument('--output_dir', type=str, required=True, help="Directorio para dataset dividido.")
    parser.add_argument('--obj_names_file', type=str, required=True, help="Ruta al archivo obj.names.")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Proporción train.")
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Proporción val.")
    parser.add_argument('--test_ratio', type=float, default=0.0, help="Proporción test (default 0.0).")
    parser.add_argument('--random_seed', type=int, default=42, help="Semilla aleatoria (OJO: actualmente no usada por IterativeStratification).")
    args = parser.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0): raise ValueError("Ratios deben sumar 1.0.")
    if not (args.train_ratio > 0): raise ValueError("Train_ratio debe ser > 0.")
    if any(r < 0 for r in [args.train_ratio, args.val_ratio, args.test_ratio]): raise ValueError("Ratios no pueden ser negativos.")

    print("--- Iniciando División del Dataset ---")
    abs_input_dir, abs_output_dir = os.path.abspath(args.input_dir), os.path.abspath(args.output_dir)
    print(f"Entrada: {abs_input_dir}\nSalida: {abs_output_dir}\nRatios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")

    class_names = load_class_names(args.obj_names_file, abs_input_dir)
    all_image_files, all_label_files = get_image_label_pairs(abs_input_dir)
    if not all_image_files: print("¡ERROR CRÍTICO! No se encontraron imágenes/etiquetas. Saliendo."); return

    Y_multilabel = create_multilabel_matrix(all_label_files, class_names)
    valid_samples_mask = Y_multilabel.sum(axis=1) > 0
    if np.sum(~valid_samples_mask) > 0: print(f"ADVERTENCIA: {np.sum(~valid_samples_mask)} muestras sin etiquetas válidas serán excluidas.")

    X_indices_with_labels, Y_stratify = np.arange(len(all_image_files))[valid_samples_mask], Y_multilabel[valid_samples_mask]
    if Y_stratify.shape[0] == 0: print("¡ERROR CRÍTICO! Ninguna muestra tiene etiquetas válidas para estratificación."); return
    print(f"INFO: Muestras con etiquetas válidas para estratificación: {len(X_indices_with_labels)}")

    # (Impresión de análisis previo de etiquetas omitida por brevedad, pero está en la versión anterior)

    train_indices_final, val_indices_final, test_indices_final = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    current_X_indices, current_Y_labels = X_indices_with_labels.copy(), Y_stratify.copy()

    # 1. Separar Entrenamiento
    if args.train_ratio > 0 and len(current_X_indices) > 0:
        if np.isclose(args.train_ratio, 1.0):
            train_indices_final = current_X_indices
            current_X_indices, current_Y_labels = np.array([], dtype=int), np.empty((0, Y_stratify.shape[1]), dtype=int)
            print(f"DEBUG: Todas las {len(train_indices_final)} muestras a train (train_ratio=1.0).")
        else:
            try:
                remainder_ratio = 1.0 - args.train_ratio
                if np.isclose(remainder_ratio, 0): raise ValueError("Remainder_ratio es cero.")
                
                print(f"DEBUG: 1er Split. Ratios para IterativeStratification: [remainder={remainder_ratio:.4f}, train={args.train_ratio:.4f}]")
                stratifier_train = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[remainder_ratio, args.train_ratio])
                
                fold1_rel, fold2_rel = next(stratifier_train.split(current_X_indices.reshape(-1,1), current_Y_labels))
                print(f"DEBUG: Tamaños folds crudos 1er split: fold1 (para ratio {remainder_ratio:.2f})={len(fold1_rel)}, fold2 (para ratio {args.train_ratio:.2f})={len(fold2_rel)}")

                # CORRECCIÓN: Asignar el fold más grande a train, el más pequeño al resto
                if len(fold1_rel) > len(fold2_rel): # fold1 es el más grande
                    indices_for_train_rel = fold1_rel
                    indices_for_remainder_rel = fold2_rel
                    print("DEBUG: fold1 (más grande) asignado a train, fold2 (más pequeño) a remainder.")
                else: # fold2 es el más grande o son iguales
                    indices_for_train_rel = fold2_rel
                    indices_for_remainder_rel = fold1_rel
                    print("DEBUG: fold2 (más grande o igual) asignado a train, fold1 (más pequeño o igual) a remainder.")

                train_indices_final = current_X_indices[indices_for_train_rel]
                temp_current_X_indices = current_X_indices[indices_for_remainder_rel]
                temp_current_Y_labels = current_Y_labels[indices_for_remainder_rel]
                
                print(f"DEBUG: 1er Split OK. Train: {len(train_indices_final)}, Resto: {len(temp_current_X_indices)}")
                current_X_indices, current_Y_labels = temp_current_X_indices, temp_current_Y_labels
            except ValueError as e:
                print(f"!!! ERROR 1er Split: {e}. Asignando todo a train. !!!")
                train_indices_final = np.array(X_indices_with_labels, dtype=int)
                current_X_indices, current_Y_labels = np.array([], dtype=int), np.empty((0, Y_stratify.shape[1]), dtype=int)

    # 2. Separar Validación y Prueba del resto
    if len(current_X_indices) > 0 and (args.val_ratio > 0 or args.test_ratio > 0):
        if np.isclose(args.test_ratio, 0.0) and args.val_ratio > 0 :
            val_indices_final = current_X_indices
            print(f"DEBUG: Resto ({len(current_X_indices)}) a validación (test_ratio=0).")
            current_X_indices, current_Y_labels = np.array([],dtype=int),np.empty((0,Y_stratify.shape[1]),dtype=int)
        elif np.isclose(args.val_ratio, 0.0) and args.test_ratio > 0 :
            test_indices_final = current_X_indices
            print(f"DEBUG: Resto ({len(current_X_indices)}) a prueba (val_ratio=0).")
            current_X_indices, current_Y_labels = np.array([],dtype=int),np.empty((0,Y_stratify.shape[1]),dtype=int)
        elif args.val_ratio > 0 and args.test_ratio > 0:
            try:
                total_prop_for_val_test_in_original = args.val_ratio + args.test_ratio # Denominador
                if np.isclose(total_prop_for_val_test_in_original, 0): raise ValueError("Suma val_ratio y test_ratio es cero en 2do split.")
                val_prop_of_remainder = args.val_ratio / total_prop_for_val_test_in_original
                test_prop_of_remainder = args.test_ratio / total_prop_for_val_test_in_original
                
                print(f"DEBUG: 2do Split. Ratios para IterativeStratification: [test={test_prop_of_remainder:.4f}, val={val_prop_of_remainder:.4f}]")
                stratifier_val_test = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[test_prop_of_remainder, val_prop_of_remainder])
                
                fold1_vt_rel, fold2_vt_rel = next(stratifier_val_test.split(current_X_indices.reshape(-1,1), current_Y_labels))
                print(f"DEBUG: Tamaños folds crudos 2do split: fold1 (para ratio test)={len(fold1_vt_rel)}, fold2 (para ratio val)={len(fold2_vt_rel)}")

                # Asignar basándose en el orden de sample_distribution_per_fold
                indices_for_test_rel = fold1_vt_rel
                indices_for_val_rel = fold2_vt_rel

                val_indices_final = current_X_indices[indices_for_val_rel]
                test_indices_final = current_X_indices[indices_for_test_rel]
                
                print(f"DEBUG: 2do Split OK. Val: {len(val_indices_final)}, Test: {len(test_indices_final)}")
                current_X_indices, current_Y_labels = np.array([],dtype=int),np.empty((0,Y_stratify.shape[1]),dtype=int)
            except ValueError as e:
                print(f"!!! ERROR 2do Split: {e}. Asignando resto a val o test. !!!")
                if args.val_ratio >= args.test_ratio and len(current_X_indices) > 0 : val_indices_final = np.concatenate((val_indices_final, current_X_indices)).astype(int)
                elif args.test_ratio > args.val_ratio and len(current_X_indices) > 0 : test_indices_final = np.concatenate((test_indices_final, current_X_indices)).astype(int)
                elif len(current_X_indices) > 0 : val_indices_final = np.concatenate((val_indices_final, current_X_indices)).astype(int) # Default a val
                current_X_indices = np.array([], dtype=int)
    
    if len(current_X_indices) > 0:
        print(f"INFO: {len(current_X_indices)} muestras restantes finales asignadas a train.")
        train_indices_final = np.concatenate((train_indices_final, current_X_indices)).astype(int)
        train_indices_final = np.unique(train_indices_final)

    print(f"\n--- Resumen de la División Final (Número de Muestras) ---")
    print(f"  Entrenamiento: {len(train_indices_final)}")
    print(f"  Validación: {len(val_indices_final)}")
    print(f"  Prueba: {len(test_indices_final)}")
    total_split_samples = len(train_indices_final) + len(val_indices_final) + len(test_indices_final)
    print(f"  Total en splits: {total_split_samples} (de {len(X_indices_with_labels)} muestras válidas)")

    # (Comprobaciones de duplicados y totales omitidas por brevedad, pero están en la versión anterior)

    if os.path.exists(abs_output_dir): print(f"ADVERTENCIA: Directorio de salida '{abs_output_dir}' ya existe.")
    else: os.makedirs(abs_output_dir, exist_ok=True); print(f"INFO: Creado directorio de salida: {abs_output_dir}")

    train_rel_paths, val_rel_paths, test_rel_paths = [], [], []
    if len(train_indices_final) > 0: train_rel_paths = copy_files_to_split(train_indices_final, all_image_files, all_label_files, abs_output_dir, 'train')
    if len(val_indices_final) > 0: val_rel_paths = copy_files_to_split(val_indices_final, all_image_files, all_label_files, abs_output_dir, 'val')
    if len(test_indices_final) > 0: test_rel_paths = copy_files_to_split(test_indices_final, all_image_files, all_label_files, abs_output_dir, 'test')
    print("\nINFO: Copia de archivos finalizada.")

    create_yolo_aux_files(abs_output_dir, class_names, train_rel_paths, val_rel_paths, test_rel_paths)

    train_labels_out = [os.path.join(abs_output_dir, 'labels', 'train', os.path.basename(all_label_files[i])) for i in train_indices_final] if train_indices_final.size > 0 else []
    val_labels_out = [os.path.join(abs_output_dir, 'labels', 'val', os.path.basename(all_label_files[i])) for i in val_indices_final] if val_indices_final.size > 0 else []
    test_labels_out = [os.path.join(abs_output_dir, 'labels', 'test', os.path.basename(all_label_files[i])) for i in test_indices_final] if test_indices_final.size > 0 else []

    print_class_distribution_from_files(train_labels_out, class_names, 'Train (Nuevo Split)')
    print_class_distribution_from_files(val_labels_out, class_names, 'Validation (Nuevo Split)')
    print_class_distribution_from_files(test_labels_out, class_names, 'Test (Nuevo Split)')
    print_class_distribution_from_files([all_label_files[i] for i in X_indices_with_labels], class_names, 'Original (Solo Muestras Válidas Usadas)')

    print("\n--- ¡Proceso de división completado! ---")
    print(f"Dataset dividido guardado en: {abs_output_dir}")

if __name__ == '__main__':
    main()
    
# python scripts\split_dataset.py --input_dir ..\dataset_train --output_dir ..\dataset_UDIS_dividido --obj_names_file ..\dataset_train\obj.names --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15