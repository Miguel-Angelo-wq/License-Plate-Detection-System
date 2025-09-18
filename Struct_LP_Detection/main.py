import sys
import os
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import setup_env
import tensorflow as tf

from utils import crop_image_with_annotations
from utils import extract_position_vehicle
from utils import nextest_bbox_by_iou, nextest_bbox_by_iou_darknet
from utils import crop_image_xyxy
from utils import show_image
from utils import extract_track_from_path
from utils import cut_off_extension
from utils import get_plate_annotation

from report import report_results
from report import ExperimentLogger

#from Car_Detection import detect_vehicles_yolov11
from unconstrained_scenarios_ocr import ocr_from_matrix
from unconstrained_scenarios_plate_det import get_license_plate
from tflite_runner import YOLOv11TFLite

from config import FPS, VIDEO
from rich.console import Console
console = Console()

detector = YOLOv11TFLite(
    model =  "./models/yolo11n_saved_model/yolo11n_float16.tflite", 
    conf = 0.25, 
    iou = 0.45, 
    metadata = "./models/yolo11n_saved_model/metadata.yaml"
)

def main(img, annotatios_path: str, modelname: str, thresh: float, output_path: str):
    
    #track = extract_track_from_path(image)
    #model = cut_off_extension(modelname)
    #logger = ExperimentLogger(experiment_name=f"track_{track}_model_{model}", model = model)

    #if isinstance(image, str):
    #    image_matrix = cv2.imread(image)
    #    #show_image(image_matrix, "Imagem Original")
    #elif image is None: 
    #    raise TypeError
    #else:
    #    image_matrix = image
    ##logger.save_chart(image_matrix, "original_image.png")

    start_time = time.time()
    #vehicles = detect_vehicles_yolov4(image, modelname=modelname)
    #vehicle_position = extract_position_vehicle(annotatios_path) #ground truth
    #car_bounding_box = nextest_bbox_by_iou_darknet(vehicles, vehicle_position)

    car_bounding_box = detector.detect(img) #Fazer retornar bounding box do carro
    croped_vehicle = crop_image_xyxy(img, car_bounding_box)
    show_image(croped_vehicle, "Veiculo Detectado")
    #logger.save_chart(croped_vehicle, "croped_vehicle.png")


    #plate = detect_plate(croped_vehicle)
    plate = get_license_plate(croped_vehicle)
    if plate is not None: 
        print("SHAPE DA IMAGEM DA PLACA", plate.shape)
        show_image(plate, "Placa Detectada")
        #logger.save_chart(plate, "plate.png")

        characters = ocr_from_matrix(plate)
    else: characters = ""

    end_time = time.time()
    console.print("Time: ", end_time - start_time, style = "magenta")
    console.print("Characters:", characters, style = "magenta")

    #logger.save_config({
    #        "image_path": image,
    #        "annotations_path": annotatios_path,
    #        "model_name": modelname,
    #        "threshold": thresh,
    #    })

    #original_characters = get_plate_annotation(annotatios_path)
    #logger.save_metrics({
    #    "detected_characters": characters, 
    #    "original_characters": original_characters,
    #    "detected_characters_matched": characters == original_characters,
    #    "time_(s)": round( end_time - start_time, 3 )
    #    })



    #report_results(characters, end_time - start_time, image, modelname, annotatios_path, output_path)
    return characters

# função main:       
# imagem -> main() -> ?? 
# main deve desenhar no frame as boungind boxes e os caracteres.
#
#
def main_loop():

    cap = cv2.VideoCapture(VIDEO)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        #characters = main(image=frame)

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



