import tensorflow as tf
import keras_cv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras_cv 
import tensorflow as tf
from keras_cv import visualization, bounding_box
from tensorflow import keras


def yolov8_car_detection(filepath: str):
    pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_m_pascalvoc", bounding_box_format="xyxy"
    )

    image = keras.utils.load_img(filepath)
    image = np.array(image)

    """ visualization.plot_image_gallery(
        np.array([image]),
        value_range=(0, 255),
        rows=1,
        cols=1,
        scale=5,
    )
    """
    inference_resizing = keras_cv.layers.Resizing(
      640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
    )


if __name__ == "__main__":
    yolov8_car_detection()