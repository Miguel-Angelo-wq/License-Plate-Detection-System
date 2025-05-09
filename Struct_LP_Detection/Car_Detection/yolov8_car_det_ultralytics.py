from ultralytics import YOLO
from argparse import ArgumentParser

import os

def detect_vehicles_yolov8(imagem_path: str, model_path='../models/yolov8n.pt', conf_threshold=0.25) -> list:
    """
    Detecta carros, motos e caminhões em uma imagem usando um modelo YOLOv8 da Ultralytics.

    Parâmetros:
        imagem_path (str): Caminho da imagem.
        model_path (str): Caminho do modelo YOLOv8 (.pt). Pode ser 'yolov8n.pt', 'yolov8m.pt', etc.
        conf_threshold (float): Confiança mínima para considerar a detecção.

    Retorno:
        List[Dict]: Lista de dicionários com bounding boxes e informações dos veículos detectados.
    """
    # Carregar o modelo
    model = YOLO(model_path) #Considerar essa linha na métrica de tempo?

    # Fazer predição
    results = model(imagem_path)[0]

    # Classes que queremos detectar (segundo COCO)
    classes_desejadas = {'car', 'motorcycle', 'truck', 'bus'}

    # Lista para armazenar as detecções filtradas
    veiculos_detectados = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        nome_classe = results.names[class_id]
        confianca = float(box.conf[0])

        if nome_classe in classes_desejadas and confianca >= conf_threshold:
            bbox = box.xyxy[0].tolist()  # formato [x1, y1, x2, y2]
            veiculos_detectados.append({
                'classe': nome_classe,
                'confianca': confianca,
                'bbox': bbox
            })

    return veiculos_detectados




if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--impath', type=str, help='Caminho da imagem', default=os.getenv("TEST_RAW_IMAGE"))
    args.add_argument('--modelpath', type=str, help='Caminho do modelo', default='../models/yolov8n.pt')
    args.add_argument('--conf', type=float, help='Confianca minima', default=0.25)
    args = args.parse_args()
    print(detect_vehicles_yolov8(imagem_path=os.getenv("TEST_RAW_IMAGE")))