import time
import cv2
import setup_env
import tensorflow as tf

from utils import crop_image_xywh, deresize_boundingbox_xywh
from utils import show_image

from report import report_results
from report import ExperimentLogger

#from Car_Detection import detect_vehicles_yolov11
from unconstrained_scenarios_ocr import ocr_from_matrix
from unconstrained_scenarios_plate_det import get_license_plate
from tflite_runner import YOLOv11TFLite

from config import FPS, VIDEO
#REMOVER DEPOIS CHAMADAS DE RICH
from rich.console import Console
from rich.traceback import install

console = Console()
install()

detector = YOLOv11TFLite(
    model =  "./models/yolo11n_saved_model/yolo11n_float16.tflite", 
    conf = 0.25, 
    iou = 0.45, 
    metadata = "./models/yolo11n_saved_model/metadata.yaml"
)


def draw_in_image(imagem_a, bounding_box, imagem_b, texto):
    """
    Desenha uma bounding box, uma sub-imagem e um texto na imagem principal.

    Args:
        imagem_a (np.ndarray): A imagem principal carregada com OpenCV (em formato BGR).
        bounding_box (list): Uma lista no formato [x, y, w, h], onde (x, y) é o canto
                             superior esquerdo e (w, h) são a largura e a altura.
        imagem_b (np.ndarray): A sub-imagem a ser desenhada no canto inferior esquerdo.
        texto (str): O conteúdo da string a ser escrita acima da bounding box.

    Returns:
        np.ndarray: A imagem 'A' com as anotações desenhadas.
    """
    imagem_anotada = imagem_a.copy()

    x, y, w, h = bounding_box
    cor_retangulo = (0, 255, 0)  # Verde em BGR
    espessura_linha = 2
    cv2.rectangle(imagem_anotada, (x, y), (x + w, y + h), cor_retangulo, espessura_linha)

    altura_b, largura_b, _ = imagem_b.shape
    altura_a, _, _ = imagem_anotada.shape
    
    roi = imagem_anotada[altura_a - altura_b:altura_a, 0:largura_b]
    
    roi[:] = imagem_b

    posicao_texto = (x, y - 10)  # Posição um pouco acima da caixa
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = 0.7
    cor_texto = (0, 255, 0)  # Verde em BGR
    espessura_texto = 2
    cv2.putText(imagem_anotada, texto, posicao_texto, fonte, escala_fonte, cor_texto, espessura_texto)

    return imagem_anotada

def main(img):
    

    start_time = time.time()

    car_bounding_box = detector.detect(img) #Fazer retornar bounding box do carro
    deresized_bounding_box = deresize_boundingbox_xywh(img, car_bounding_box)
    croped_vehicle = crop_image_xywh(img, deresized_bounding_box)
    #show_image(croped_vehicle, "Veiculo Detectado")

    plate = get_license_plate(croped_vehicle)
    if plate is not None: 
        print("SHAPE DA IMAGEM DA PLACA", plate.shape)
        #show_image(plate, "Placa Detectada")

        characters = ocr_from_matrix(plate)
        print("OUTPUT DE CARACTERES:")
        print(characters)
    else: characters = ""

    end_time = time.time()
    console.print("Time: ", end_time - start_time, style = "magenta")
    console.print("Characters:", characters, style = "magenta")

    img = draw_in_image(img, deresized_bounding_box, plate, characters)

    return img

def main_loop():

    cap = cv2.VideoCapture(VIDEO)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(VIDEO)
            continue

        frame = main(frame)

        cv2.imshow("MAIN", frame)

        if cv2.waitKey(FPS) & 0xff == ord("q"):
            break




if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--image', type=str, help='Image path', default=os.getenv("TEST_RAW_IMAGE"))
    #parser.add_argument('--annotations_path', type=str, help='Image annotations path', default= os.getenv("TEST_ANNOTATIONS_PATH"))
    #parser.add_argument('--modelname', type=str, help='Car detector YOLO path\n yolov8@.pt for @ in { n m s l x }', default="yolov4-p6")
    #parser.add_argument('--thresh', type=float, help='minimum threshold for car detection ', default=0.25)
    #parser.add_argument('--output_path', type=str, help='csv output file, if needed to spefify a single file for process output', default=( os.getenv("GRID_SEARCH_RESULTS_DIR") + 'results_yolov4-p6.csv' ))
    #args = parser.parse_args()

    #main(args.image, args.annotations_path, args.modelname, args.thresh, args.output_path)
    main_loop()



