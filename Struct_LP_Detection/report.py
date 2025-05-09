import os
import re
import cv2
import csv
import json
import yaml
import setup_env
from pathlib import Path

from datetime import datetime




def report_results(recognized_char, total_time, impath, car_det_yolo, annotations_path, output_path):
    """
    Reports the results of a single experiment run in a JSON file with the format:
    {
        "recognized_char": <string with recognized characters>,
        "original_plate": <string with the original plate>,
        "total_time": <float with the total running time>,
        "timestamp": <string with the timestamp>,
        "image_path": <string with the path to the image>,
        "car_det_yolo": <boolean indicating if the car was detected with YOLO>
    }
    The filename is in the format 'results_<timestamp>.json', where <timestamp> is the
    current date and time in the format 'YYYYMMDD_HHMMSS'.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(annotations_path, 'r') as file:
        data = file.read()
        # i want a sequence of 7 alphanumerical characters
        match = re.search(r"plate:\s*([A-Z0-9]+)", data)
    
    
    if match:
        original = match.group(1)
        results = {
            "recognized_char": recognized_char,
            "original_plate": original,
            "are_they_equal": recognized_char == original,
            "total_time": total_time,
            "timestamp": timestamp,
            "image_path": impath,
            "car_det_yolo": car_det_yolo
        }


        with open(output_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            writer.writerow(results)
    
class ExperimentLogger:
    def __init__(self, base_dir = None, experiment_name = None, model = 'unscpecificed'):
        timestamp = datetime.now().strftime("%Y-%m-%d")

        if base_dir is None:
            base_dir = os.path.join(
                os.getenv("REPORTS_DIR"),
                f"experiments_{ model }/"
            )
        if experiment_name is None:
            experiment_name = "unnamed_experiment"

        self.experiment_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.charts_dir = os.path.join(self.experiment_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)

    def save_config(self, config_dict):
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

    def save_metrics(self, metrics_dict):
        metrics_path = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

    def save_log(self, log_text):
        log_path = os.path.join(self.experiment_dir, "training_log.txt")
        with open(log_path, "w") as f:
            f.write(log_text)

    def save_model_summary(self, model_summary_text):
        summary_path = os.path.join(self.experiment_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(model_summary_text)

    def save_chart(self, figure, filename):
        figure_path = os.path.join(self.charts_dir, filename)
        #figure.savefig(figure_path)
        cv2.imwrite(figure_path, figure)

    def save_notes(self, notes_text):
        notes_path = os.path.join(self.experiment_dir, "notes.md")
        with open(notes_path, "w") as f:
            f.write(notes_text)

    def get_experiment_path(self):
        return self.experiment_dir



