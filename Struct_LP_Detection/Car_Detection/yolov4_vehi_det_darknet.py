import cv2
import sys
import os

import setup_env

import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt

from darknet.python import darknet
from darknet.python.darknet import detect_image, make_image, network_width, network_height, array_to_image, load_network 

from cv2 import resize, copyMakeBorder, BORDER_CONSTANT
#"./darknet/cfg/yolov4.cfg",  # ou yolov4-custom.cfg

print(os.getenv("TEST_RAW_IMAGE"))


def letterbox_image(img, target_size):

    ih, iw = img.shape[:2]
    h, w = target_size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized_image = resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left

    padded_image = copyMakeBorder(resized_image, top, bottom, left, right, BORDER_CONSTANT, value=(128,128,128))
    return padded_image, scale, left, top

def detect_vehicles_yolov4(image_path, modelname = 'yolov4-p6.cfg', class_names_filepath = "./darknet/cfg/coco.names", vehicle_classes=["car", "motorbike", "bus", "truck"]):

    """

    Detect vehicles in an image using YOLOv4. Also draw the bounding boxes on the image and show the image.
    Loads the network with pre-defined global paths to the required files.
    
    Args: 
        image_path (str): The path to the image file.
        net (darknet.network): The YOLOv4 network.
        class_names (list): A list of class names.
        vehicle_classes (list, optional): A list of class names. Defaults to ["car", "motorbike", "bus", "truck"].

    Returns:
        list: A list of bounding boxes for detected vehicles compatible with the original image 
        in the format (x1, y1, x2, y2), top left and bottom right corners of the rectangular bounding box.
    """
    #TODO put the paths bellow in the .env and load env.
    net, class_names, class_colors = load_network(
        f"/home/dialog/Documentos/Struct_LP_Detection/Struct_LP_Detection/Car_Detection/darknet_weights/{modelname}.cfg",
        "/home/dialog/Documentos/Struct_LP_Detection/Struct_LP_Detection/Car_Detection/darknet/cfg/coco.data",   # ou seu próprio .data com as classes que quer
        f"/home/dialog/Documentos/Struct_LP_Detection/Struct_LP_Detection/Car_Detection/darknet_weights/{modelname}.weights",  # ou yolov4.conv.137 se for treinar
        batch_size=1
    )
    #TODO eliminate the with open statement and test if it changes somenthing. If dont, keep it out.

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # a posição dos canais de fato influencia?
    
    network_size = (network_width(net), network_height(net))
    image_resized, scale, pad_x, pad_y = letterbox_image(image, network_size)

    im = array_to_image(image_resized)
    detections = darknet.detect_image(net, class_names, im, thresh=0.25)

    detection_original_im = []
    for label, conf, (xc, yc, w, h) in detections:
        xc = (xc - pad_x) / scale
        yc = (yc - pad_y) / scale
        w = w / scale
        h = h / scale

        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)

        detection_original_im.append((label, conf, (x1, y1, x2, y2)))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    filtered = [det for det in detection_original_im if det[0] in vehicle_classes]
    print(f"Filtered Vehicles: {filtered}")

    return filtered


if __name__ == "__main__":
    import sys
    import os
    sys.path.append("../")
    import setup_env


    image_path = os.getenv("TEST_RAW_IMAGE")
    results = detect_vehicles(image_path)

    print("RESULTADOS: ")
    print(results)

    #draw the results in the image, please:
    #image = cv2.imread(image_path)
    ##image_resized = cv2.resize(image, (network_width(net), network_height(net)))
    #network_size = (network_width(net), network_height(net))
    #image_resized, scale, pad_x, pad_y = letterbox_image(image, network_size)

    #for result in results:
    #    center_x, center_y, width, height = result[2]

    #    x1 = int(center_x - width / 2)
    #    y1 = int(center_y - height / 2)
    #    x2 = int(center_x + width / 2)
    #    y2 = int(center_y + height / 2)

    #    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #cv2.imshow("image", image_resized)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


