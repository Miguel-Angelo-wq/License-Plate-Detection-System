import os
import re
import json
import setup_env
from datetime import datetime
from run import main

data_folder = os.getenv("TEST_RAW_DATA_DIR")      
    
fixed_args = {
    'impath': os.getenv("TEST_RAW_IMAGE"),
    'annotatios_path': os.getenv("TEST_ANNOTATIONS_PATH")
}

args = {
    "impath": os.getenv("TEST_RAW_IMAGE"), 
    "annotatios_path": os.getenv("TEST_ANNOTATIONS_PATH"),
    "vehicle_detect_file_name": "yolov4_darknet", 
    "thresh": 0.25, 
    }

for track_folder in os.listdir(data_folder):
    for image in os.listdir(os.path.join(data_folder, track_folder)):
        if image.endswith('t'):
            continue

        args['impath'] = os.path.join(data_folder, track_folder, image)
        match = re.search(r"track(\d+)", fixed_args['impath'])
        if match:
            track = match.group(1)
            main(**args)

        break
        print(fixed_args['impath'])

#impath = fixed_args['impath']
#annotatios_path = fixed_args['annotatios_path']
#
