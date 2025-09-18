import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "../../models/yolo11n_saved_model/"
CONF_THRESHOLD = 0.4  # Limiar de confiança para uma detecção ser válida
IOU_THRESHOLD = 0.5   # Limiar de IoU para o Non-Maximum Suppression

# Classes de interesse (de acordo com o dataset COCO)
# 2: 'car', 3: 'motorcycle', 7: 'truck'
CLASSES_DE_INTERESSE = {2: 'Carro', 3: 'Moto', 7: 'Caminhão'}
TARGET_CLASSES_IDS = list(CLASSES_DE_INTERESSE.keys())
INPUT_SIZE = 640

def detect_vehicle_from_image(image_original, model_tf, session):

    H_original, W_original, _ = image_original.shape

# --- Preparação da Imagem (Pré-processamento) ---
    INPUT_SIZE = 640
    img_resized = cv2.resize(image_original, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(img_normalized, axis=0)

# --- Inferência com TensorFlow 1.14 ---
    print("Iniciando a sessão do TensorFlow 1.14...")

# Em TF1, tudo roda dentro de um Grafo e uma Sessão

    # Precisamos pegar os tensores de entrada e saída pelo nome
    # Estes nomes são definidos durante a exportação do modelo.
    # Use a ferramenta 'saved_model_cli' se estes nomes não funcionarem.
    input_tensor_name = "images:0" 
    output_tensor_name = "output0:0"

    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    print("Executando a inferência...")
    (detections,) = sess.run([output_tensor], feed_dict={input_tensor: input_data})

    if detections.shape[1] == 6:
        detections = np.transpose(detections, (0, 2, 1))
    
    results = detections[0] # Pega os resultados do primeiro item do batch

    results = results[results[:, 4] > CONF_THRESHOLD]

    class_ids = results[:, 5].astype(int)
    mask_class = np.isin(class_ids, TARGET_CLASSES_IDS)
    results = results[mask_class]
    class_ids = class_ids[mask_class]

    if results.shape[0] == 0:
        print("Nenhum objeto de interesse encontrado com os limiares definidos.")
    else:
        print(f"{results.shape[0]} detecções encontradas antes do NMS.")
        
        # Prepara os dados para o Non-Maximum Suppression do OpenCV
        boxes_center = results[:, :4]
        scores = results[:, 4]
        
        # Converte caixas de (cx, cy, w, h) para (x, y, w, h) para o NMS do OpenCV
        x = (boxes_center[:, 0] - boxes_center[:, 2] / 2).astype(int)
        y = (boxes_center[:, 1] - boxes_center[:, 3] / 2).astype(int)
        w = boxes_center[:, 2].astype(int)
        h = boxes_center[:, 3].astype(int)
        
        # O NMS do OpenCV espera uma lista de retângulos
        boxes_for_nms = list(zip(x, y, w, h))
        
        # Aplica o Non-Maximum Suppression
        selected_indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        
        print(f"{len(selected_indices)} detecções restantes após o NMS.")
        
        # Desenha as caixas finais na imagem original
        for i in selected_indices.flatten():
            box = boxes_for_nms[i]
            score = scores[i]
            class_id = class_ids[i]
            
            x, y, w, h = box
            
            # Reescalar as coordenadas para as dimensões da imagem original
            x1_orig = int(x * W_original / INPUT_SIZE)
            y1_orig = int(y * H_original / INPUT_SIZE)
            x2_orig = int((x + w) * W_original / INPUT_SIZE)
            y2_orig = int((y + h) * H_original / INPUT_SIZE)
            
            # Desenha o retângulo
            cv2.rectangle(image_original, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
            
            # Prepara e desenha o rótulo
            label = f"{CLASSES_DE_INTERESSE[class_id]}: {score:.2f}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_original, (x1_orig, y1_orig - h_text - 5), (x1_orig + w_text, y1_orig), (0, 255, 0), -1)
            cv2.putText(image_original, label, (x1_orig, y1_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite('resultado_deteccao_tf1.jpg', image_original)
        print("\nImagem com as detecções salva como 'resultado_deteccao_tf1.jpg'")
        return image_original




if __name__ == "__main__":


    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # Carrega a estrutura do modelo (o grafo) para a sessão
            tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING], # Tag padrão para modelos de inferência
                MODEL_PATH
            )

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        car_det_model = tf.saved_model.load(MODEL_PATH)
        if ret:
            detect_vehicle_from_image(frame, car_det_model, sess)
        else:
            print("não foi possível capturar uma imagem")
