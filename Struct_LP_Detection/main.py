import time
import cv2
import tensorflow as tf

from utils import crop_image_xywh, deresize_boundingbox_xywh

from unconstrained_scenarios_ocr import ocr_from_matrix
from unconstrained_scenarios_plate_det import get_license_plate
from tflite_runner import YOLOv11TFLite

from draw import draw_in_image
from config import ( FPS, VIDEO, OCR_THRESHOLD, LP_THRESHOLD )

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


def main(img):
    

    start_time = time.time()

    car_bounding_box = detector.detect(img) #Fazer retornar bounding box do carro
    deresized_bounding_box = deresize_boundingbox_xywh(img, car_bounding_box)
    croped_vehicle = crop_image_xywh(img, deresized_bounding_box)
    #show_image(croped_vehicle, "Veiculo Detectado")

    plate = get_license_plate(croped_vehicle, lp_threshold=LP_THRESHOLD)
    print("Plate is None:", plate is None)
    if plate is not None: 
        print("SHAPE DA IMAGEM DA PLACA", plate.shape)
        #show_image(plate, "Placa Detectada")

        characters = ocr_from_matrix(plate, ocr_threshold=OCR_THRESHOLD)
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



