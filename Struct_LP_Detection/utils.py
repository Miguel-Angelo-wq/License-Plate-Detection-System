import sys
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from PIL import Image



    
def xywh_to_xyxy(box_xywh):
    """
    Converte uma bounding box do formato [x, y, w, h] para [x1, y1, x2, y2].

    Parâmetros:
        box_xywh (list or tuple): [x, y, w, h]

    Retorna:
        list: [x1, y1, x2, y2]
    """
    x, y, w, h = box_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def extract_position_vehicle(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    match = re.search(r"position_vehicle:\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)", data)
    
    
    if match:
        position = [int(match.group(i)) for i in range(1, 5)]
        return xywh_to_xyxy(position)
    else:
        return []


def crop_image_with_annotations(image, vehicle_position):
    x1, y1, w, h = vehicle_position
    x2 = x1 + w
    y2 = y1 + h
    crop_img = image[y1:y2, x1:x2]

    return crop_img

def crop_image_xyxy(image, vehicle_position):
    x1, y1, x2, y2 = map(round, vehicle_position)
    crop_img = image[y1:y2, x1:x2]

    return crop_img

def deresize_boundingbox_xywh(image, bouding_box, size_ref = (640, 640)):

    h_orig, w_orig = image.shape[:2]

    # 2. Dimensões de referência onde a bbox foi calculada
    h_ref, w_ref = size_ref

    # 3. Calcular os fatores de escala para largura e altura
    escala_w = w_orig / w_ref
    escala_h = h_orig / h_ref

    # 4. Desempacotar a bounding box da referência 640x640
    x_ref, y_ref, w_ref, h_ref = bouding_box

    # 5. Converter as coordenadas da bbox para a escala da imagem original
    x_orig = int(x_ref * escala_w)
    y_orig = int(y_ref * escala_h)
    w_orig = int(w_ref * escala_w)
    h_orig = int(h_ref * escala_h)
    return [x_orig, y_orig, w_orig, h_orig]

def crop_image_xywh(image, bouding_box, size_ref = (640, 640)):
    """
    Recorta uma região de uma imagem original usando uma bounding box
    que foi calculada em uma versão redimensionada da imagem (640x640).

    Args:
        imagem_original (np.array): A imagem original em alta resolução.
        bbox_640 (list): Uma lista Python no formato [x, y, w, h] com as
                         coordenadas da bounding box na escala 640x640.

    Returns:
        np.array: A imagem recortada, ou None se ocorrer um erro.
    """

    x, y, w, h= bouding_box

    croped_img = image[y: y + h, x: x + w]

    return croped_img

def crop_image_with_annotations_from_path(image_path:str, vehicle_position:list):
    image = Image.open(image_path)
    x1, y1, w, h = vehicle_position
    x2 = x1 + w
    y2 = y1 + h
    crop_img = image.crop((x1, y1, x2, y2))
    return crop_img

def crop_car_with_annotations(track: str):
    assert 1 <= int(track) <= 150, "track must be an integer between 1 and 150" 

    track =  "0"*(3 - len(track)) + track

    folder = f"track0{track}"

    if 1 <= (int(track)) <= 60:
        directory = "training"
    elif 61 <= (int(track)) <= 90:
        directory = "validation"
    elif 91 <= (int(track)) <= 150:
        directory = "testing"

    data_path = f"../data/raw/yj4Iu2-UFPR-ALPR/UFPR-ALPR dataset/{directory}"


    print(folder)
    for file_name in os.listdir(data_path + '/' + folder):
        if file_name[-1] == 'g': # if it is an image

            if not os.path.exists(f"../data/processed/track0{track}"):
                os.system(f"mkdir ../data/processed/track0{track}")


            print("file name: ", file_name)
            image = keras.utils.load_img(data_path + '/' + folder + '/' + file_name)
            image = np.array(image)

            annotations_name = file_name[:-3] + "txt"
            print("annotations file:", annotations_name)
            annotations_path = data_path + '/' + folder + '/' + annotations_name

            vehicle_position = extract_position_vehicle(annotations_path)

            cropped = crop_image_with_annotations(image = image, vehicle_position = vehicle_position) 

            plt.imsave(f"../data/processed/track0{track}/{file_name}", cropped)

def calculate_iou(gt_coordinates: tuple, pred_coordinates: tuple) -> float:
        (gt_xmin, gt_ymax), (gt_xmax, gt_ymin) = gt_coordinates  
        (pred_xmin, pred_ymax), (pred_xmax, pred_ymin) = pred_coordinates  

        inter_xmin = max(pred_xmin, gt_xmin)
        inter_xmax = min(pred_xmax, gt_xmax)
        inter_ymin = max(pred_ymin, gt_ymin)
        inter_ymax = min(pred_ymax, gt_ymax)

        iw = np.maximum(inter_xmax - inter_xmin + 1., 0.)
        ih = np.maximum(inter_ymax - inter_ymin + 1., 0.)

        inters = iw * ih

        union = ((pred_xmax - pred_xmin + 1.) * (pred_ymax - pred_ymin + 1.) +
               (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) -
               inters)

        iou = inters / union
        print("iou dentro da funcao", iou)

        return iou


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def nextest_bbox_by_iou(yolo_results, annotation):
    """
    Retorna a bbox dos yolo_results com maior IoU em relação à annotation.

    Parâmetros:
        yolo_results (list): Lista de dicionários com 'bbox' no formato xyxy
        annotation (list): Lista [x1, y1, x2, y2] no formato xyxy

    Retorna:
        dict: Elemento de yolo_results com maior IoU em relação à annotation
    """
    melhor_iou = -1
    melhor_item = yolo_results[0]

    for item in yolo_results:
        iou = calculate_iou(item['bbox'], annotation)
        if iou > melhor_iou:
            melhor_iou = iou
            melhor_item = item

    return melhor_item['bbox']

def nextest_bbox_by_iou_darknet(yolo_results, annotation):
    """
    Retorna a bbox dos yolo_results com maior IoU em relação à annotation.

    Parâmetros:
        yolo_results (list): Lista de dicionários com 'bbox' no formato xyxy
        annotation (list): Lista [x1, y1, x2, y2] no formato xyxy

    Retorna:
        dict: Elemento de yolo_results com maior IoU em relação à annotation
    """
    melhor_iou = -1
    melhor_item = yolo_results[0]

    for item in yolo_results:
        iou = calculate_iou(item[2], annotation)
        if iou > melhor_iou:
            melhor_iou = iou
            melhor_item = item

    return melhor_item[2]

def show_image(image_matrix, title: str):
    if image_matrix is not None and image_matrix.size > 0:
        # Verifica o tipo e normaliza se necessário
        if image_matrix.dtype in [np.float32, np.float64]:
            image_matrix = np.clip(image_matrix, 0.0, 1.0)
        elif not np.issubdtype(image_matrix.dtype, np.uint8):
            image_matrix = np.clip(image_matrix, 0, 255).astype(np.uint8)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_matrix)
        plt.axis("off")
        plt.title(title)
        plt.show()

def extract_track_from_path(impath):
    match = re.search(r'track0*(\d+)\[', impath)
    if match:
        return int(match.group(1))
    return None

def cut_off_extension(name):
    return name.split('.')[0]

def get_plate_annotation(annotations_path):
    with open(annotations_path, 'r') as file:
        data = file.read()
        # i want a sequence of 7 alphanumerical characters
        match = re.search(r"plate:\s*([A-Z0-9]+)", data)
    
    if match:
        return match.group(1)



