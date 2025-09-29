import sys
import cv2
import traceback
from pathlib import Path

from .utils 	    		import load_tflite_model
from .keras_utils 			import detect_lp
from .label 				import Shape, writeShapes

BASE_DIR = Path(__file__).resolve().parent

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

wpod_net_path  = str( BASE_DIR/"cfg/wpod_net.tflite" )
wpod_net_tflite = load_tflite_model( wpod_net_path   )
	
def get_license_plate(img_array, lp_threshold=.4):
    try:

        Llp, LlpImgs, _ = detect_lp(
            interpreter=wpod_net_tflite,
            I=img_array,  
            out_size=(240, 80),
            threshold=lp_threshold
        )

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            s = Shape(Llp[0].pts)
            print("IM_SHAPE:", Ilp.shape)
            cv2.imwrite('lp.png', Ilp)
            writeShapes('lp.txt', [s])
            return Ilp
        else:
            print("Nenhuma placa encontrada.")
            return None 

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

	
if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("../")

    #impath = "/home/dialog/Documentos/Vehicular_Plates_Detection/keras_approach/data/processed/track0143/track0143[01].png"
    impath = "/home/dialog/Documentos/Vehicular_Plates_Detection/keras_approach/data/processed/track0135/track0135[05].png"
    image = cv2.imread(impath)
    plate = get_license_plate(image)
    print(type(plate))
    print(plate.shape)
