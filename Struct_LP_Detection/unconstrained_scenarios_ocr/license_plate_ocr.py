import sys
import cv2
import numpy as np
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
    import setup_env
    image = cv2.imread(os.getenv("TEST_PLATE_IMAGE"))
    print("IMAGEM: ", image.shape)

    print(ocr_from_matrix(image))



#[(b'L', 0.9172979593276978, 
#  (61.837547302246094, 42.549251556396484, 28.743051528930664, 43.27523422241211)), 
# (b'L', 0.8989779949188232, 
#  (61.73507308959961, 41.95447540283203, 27.477752685546875, 40.8448371887207)), 
# (b'S', 0.8974624276161194, (88.7516098022461, 43.70576858520508, 33.046791076660156, 41.87464904785156)), 
# (b'5', 0.8891560435295105, (158.0615997314453, 43.674598693847656, 29.173625946044922, 39.938323974609375)), 
# (b'5', 0.8814458250999451, (158.10589599609375, 42.886199951171875, 32.19983673095703, 41.61393356323242)),
# (b'S', 0.8800897002220154, (88.9622573852539, 44.786643981933594, 29.7872371673584, 39.43619155883789)),
# (b'5', 0.8768128752708435, (130.34132385253906, 42.43988800048828, 34.30813217163086, 43.64461898803711)), 
# (b'5', 0.8563768267631531, (131.1571044921875, 43.74018096923828, 30.459903717041016, 38.93483352661133)), 
# (b'M', 0.847510814666748, (33.30636978149414, 42.33253479003906, 26.956628799438477, 40.80732345581055)), 
# (b'5', 0.8030648827552795, (160.63893127441406, 43.46636962890625, 33.57871627807617, 40.58162307739258)),
# (b'S', 0.7977339029312134, (87.5255126953125, 43.32847595214844, 32.23238754272461, 43.289031982421875)), 
# (b'I', 0.7953614592552185, (211.51443481445312, 41.900691986083984, 18.85240936279297, 40.57880783081055)), 
# (b'I', 0.7722547054290771, (184.76651000976562, 42.680240631103516, 16.615633010864258, 38.76629638671875)), 
# (b'L', 0.6887409090995789, (64.3211898803711, 42.75608444213867, 30.478029251098633, 43.87327575683594)), 
# (b'L', 0.6881091594696045, (61.74455261230469, 39.78516387939453, 30.124277114868164, 45.66709518432617)), 
# (b'S', 0.6764365434646606, (86.98538208007812, 44.15228271484375, 29.739974975585938, 37.88175582885742)), 
# (b'M', 0.6634621024131775, (33.218353271484375, 42.26689147949219, 31.13768768310547, 42.350440979003906)), 
# (b'I', 0.645679235458374, (211.21482849121094, 43.351768493652344, 22.474538803100586, 48.22338104248047)), (b'5', 0.6455262899398804, (160.61181640625, 43.726966857910156, 29.35112190246582, 39.962730407714844)), (b'I', 0.6053329706192017, (182.79981994628906, 42.26918029785156, 20.442312240600586, 40.93043518066406)), (b'L', 0.6022198796272278, (64.52677154541016, 43.11088562011719, 29.011377334594727, 36.12821960449219)), (b'L', 0.5320494174957275, (61.149295806884766, 39.763099670410156, 27.644804000854492, 37.56142807006836)), (b'M', 0.506396472454071, (34.9155387878418, 39.348350524902344, 33.310577392578125, 43.31012725830078))]






""" 
if __name__ == "__main__":
	import os, sys
	sys.path.append("../")
	import setup_env

	placa = Image.open(os.getenv("TEST_PLATE_IMAGE"))  # ou uma imagem retornada do .warp()
	texto = ocr_placa(placa)
	print(texto) """
