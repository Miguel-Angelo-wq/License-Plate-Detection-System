import sys
import cv2
import traceback
import os


from os.path 				import splitext, basename
from glob					import glob
from .src.label				import dknet_label_conversion
from .src.utils 				import nms

from darknet.python import darknet as dn
from darknet.python.darknet import detect_from_array, detect, array_to_image
from pathlib import Path

# must receive a matrix of image and return the extracted character as a list of strings. 
SCRIPT_DIR = Path(__file__).parent

def ocr():
	try:
		#track = sys.argv[1]
		track = 1

		output_dir = f"../data/characters/track0{track}"

		output_dir = "output_dir/"

		ocr_threshold = .4

		ocr_weights = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.weights'
		ocr_netcfg  = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.cfg'
		ocr_dataset = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.jpg' % output_dir))

		print('Performing OCR...')
		print(imgs_paths)
		
		detected_license_plates = []

		for i,img_path in enumerate(imgs_paths):

			print ('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print ('\t\tLP: %s' % lp_str)
				detected_license_plates.append(lp_str)
		

			else:

				print ('No characters found')
		print(detected_license_plates)
		return detected_license_plates

			
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


def ocr_from_matrix(image, ocr_threshold = .4) -> str:
    print('Performing OCR...')
    print('These are the path to reach the network configuration')

    ocr_weights = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.weights'
    ocr_netcfg  = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.cfg'
    ocr_dataset = SCRIPT_DIR / 'data' / 'ocr' / 'ocr-net.data'

    print("ocr_weights ", ocr_weights)
    print("ocr_netcfg ", type( ocr_netcfg ))
    print("ocr_dataset ", ocr_dataset)

    ocr_net  = dn.load_net(str( ocr_netcfg ).encode('utf-8'), str( ocr_weights ).encode('utf-8'), 0)
    print("consegui carregar o ocr-net do lado do python")
    ocr_meta = dn.load_meta(str( ocr_dataset ).encode('utf-8'))

    print ('\tScanning image')

    #bname = basename(splitext(image)[0]) # remove file extension

    detected_license_plates = ""

    print("Imagem recebida para OCR:", type(image), image.shape)
    print("ocr_net:", ocr_net)
    print("ocr_meta:", ocr_meta)
	

    darknet_image = array_to_image(image)
    #R,(width,height) = detect_from_array(ocr_net, ocr_meta, darknet_image  ,thresh=ocr_threshold, nms=None)
    R = detect_from_array(ocr_net, ocr_meta, darknet_image  ,thresh=ocr_threshold, nms=None)

    if len(R):

        L = dknet_label_conversion(R,image.shape[0],image.shape[1])
        L = nms(L,.45)

        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])

        """ with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
            f.write(lp_str + '\n') """

        print('\t\tLP: %s' % lp_str)
        detected_license_plates += lp_str

    return detected_license_plates



if __name__ == '__main__':
    import os
    sys.path.append("../")
    image = cv2.imread(os.getenv("TEST_PLATE_IMAGE"))
    print("IMAGEM: ", image.shape)

    print(ocr_from_matrix(image))


