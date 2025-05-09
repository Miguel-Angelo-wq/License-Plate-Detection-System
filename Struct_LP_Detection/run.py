import sys
import os
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import setup_env

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

from Car_Detection import detect_vehicles_yolov4
from WPODNet_Pytorch import detect_plate
from unconstrained_scenarios_ocr import ocr_from_matrix
from unconstrained_scenarios_plate_det import get_license_plate

from rich.console import Console
console = Console()

def main(impath: str, annotatios_path: str, modelname: str, thresh: float, output_path: str):
    
    track = extract_track_from_path(impath)
    model = cut_off_extension(modelname)
    #logger = ExperimentLogger(experiment_name=f"track_{track}_model_{model}", model = model)

    image_matrix = cv2.imread(impath)
    show_image(image_matrix, "Imagem Original")
    #logger.save_chart(image_matrix, "original_image.png")

    start_time = time.time()
    vehicles = detect_vehicles_yolov4(impath, modelname=modelname)
    vehicle_position = extract_position_vehicle(annotatios_path)

    car_bounding_box = nextest_bbox_by_iou_darknet(vehicles, vehicle_position)
    croped_vehicle = crop_image_xyxy(image_matrix, car_bounding_box)
    show_image(croped_vehicle, "Veiculo Detectado")
    #logger.save_chart(croped_vehicle, "croped_vehicle.png")


    #plate = detect_plate(croped_vehicle)
    plate = get_license_plate(croped_vehicle)
    print("SHAPE DA IMAGEM DA PLACA", plate.shape)
    show_image(plate, "Placa Detectada")
    #logger.save_chart(plate, "plate.png")

    characters = ocr_from_matrix(plate)
    end_time = time.time()

    #logger.save_config({
    #        "image_path": impath,
    #        "annotations_path": annotatios_path,
    #        "model_name": modelname,
    #        "threshold": thresh,
    #    })

    original_characters = get_plate_annotation(annotatios_path)
    #logger.save_metrics({
    #    "detected_characters": characters, 
    #    "original_characters": original_characters,
    #    "detected_characters_matched": characters == original_characters,
    #    "time_(s)": round( end_time - start_time, 3 )
    #    })

    console.print("Time: ", end_time - start_time, style = "magenta")
    console.print("Characters:", characters, style = "magenta")


    #report_results(characters, end_time - start_time, impath, modelname, annotatios_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--impath', type=str, help='Image path', default=os.getenv("TEST_RAW_IMAGE"))
    parser.add_argument('--annotations_path', type=str, help='Image annotations path', default= os.getenv("TEST_ANNOTATIONS_PATH"))
    parser.add_argument('--modelname', type=str, help='Car detector YOLO path\n yolov8@.pt for @ in { n m s l x }', default="yolov4-p6")
    parser.add_argument('--thresh', type=float, help='minimum threshold for car detection ', default=0.25)
    parser.add_argument('--output_path', type=str, help='csv output file, if needed to spefify a single file for process output', default=( os.getenv("GRID_SEARCH_RESULTS_DIR") + 'results_yolov4-p6.csv' ))
    args = parser.parse_args()

    main(args.impath, args.annotations_path, args.modelname, args.thresh, args.output_path)

