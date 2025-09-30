from .src.label				import dknet_label_conversion
from .src.utils 			import nms

from darknet.python import darknet as dn
from darknet.python.darknet import detect_from_array, array_to_image
from pathlib import Path
"../models/ocr/"

SCRIPT_DIR = Path(__file__).parent
ocr_weights = SCRIPT_DIR / '../models/' / 'ocr' / 'ocr-net.weights'
ocr_netcfg  = SCRIPT_DIR / '../models/' / 'ocr' / 'ocr-net.cfg'
ocr_dataset = SCRIPT_DIR / '../models/' / 'ocr' / 'ocr-net.data'

print("ocr_weights ", ocr_weights)
print("ocr_netcfg ", type( ocr_netcfg ))
print("ocr_dataset ", ocr_dataset)

ocr_net  = dn.load_net(str( ocr_netcfg ).encode('utf-8'), str( ocr_weights ).encode('utf-8'), 0)
ocr_meta = dn.load_meta(str( ocr_dataset ).encode('utf-8'))

def ocr_from_matrix(image, ocr_threshold = .4) -> str:
    detected_license_plates = ""

    print("Imagem recebida para OCR:", type(image), image.shape)
    print("ocr_net:", ocr_net)
    print("ocr_meta:", ocr_meta)
	
    darknet_image = array_to_image(image)
    R = detect_from_array(ocr_net, ocr_meta, darknet_image  ,thresh=ocr_threshold, nms=None)

    if len(R):

        L = dknet_label_conversion(R,image.shape[0],image.shape[1])
        L = nms(L,.45)

        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])

        print('\t\tLP: %s' % lp_str)
        detected_license_plates += lp_str

    return detected_license_plates

