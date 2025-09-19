import sys
import os
import keras
import cv2
import traceback
from rich.traceback import install
from pathlib import Path

from glob 						import glob
from os.path 					import splitext, basename
from .utils 	    		import im2single
from .keras_utils 			import load_model, detect_lp
from .label 				import Shape, writeShapes

import numpy as np

install()


BASE_DIR = Path(__file__).resolve().parent


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


	
def get_license_plate(img_array):

    try:
        
        lp_threshold = .5
        wpod_net_path  = BASE_DIR/"../../data/lp-detector/wpod-net_update1.json"

        wpod_net = load_model(wpod_net_path)


        print ('Searching for license plates using WPOD-NET')


        ratio = float(max(img_array.shape[:2]))/min(img_array.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(img_array),bound_dim,2**4,(240,80),lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            Ilp = Ilp*255.

            s = Shape(Llp[0].pts)
            print("IM_SHAPE:", Ilp.shape)
            Ilp = Ilp.astype(np.uint8)
            cv2.imwrite('lp.png',Ilp)
            writeShapes('lp.txt',[s])
            return Ilp

    except:
        traceback.print_exc()
        sys.exit(1)

	
if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../")
    import setup_env

    #impath = "/home/dialog/Documentos/Vehicular_Plates_Detection/keras_approach/data/processed/track0143/track0143[01].png"
    impath = "/home/dialog/Documentos/Vehicular_Plates_Detection/keras_approach/data/processed/track0135/track0135[05].png"
    image = cv2.imread(impath)
    plate = get_license_plate(image)
    print(type(plate))
    print(plate.shape)
