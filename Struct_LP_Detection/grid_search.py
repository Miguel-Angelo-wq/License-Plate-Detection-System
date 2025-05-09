import os
import json
import itertools
import subprocess
from pathlib import Path
from datetime import datetime

import setup_env

# ================== GRID CONFIGURATION ==================
param_grid = {
    'thresh': [0.25],
    'modelpath': [
        'yolov8n.pt',
        'yolov8m.pt',
        'yolov8s.pt',
        'yolov8l.pt',
        'yolov8x.pt'
    ]
}

fixed_args = {
    'impath': os.getenv("TEST_RAW_IMAGE"),
    'annotatios_path': os.getenv("TEST_ANNOTATIONS_PATH")
}
def backup_function():
    # ================== EXPERIMENTS SETUP ==================
    output_dir = os.getenv("GRID_SEARCH_RESULTS_DIR")
    os.makedirs(output_dir, exist_ok=True)

    # Combinações de hiperparâmetros
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # ================== EXECUTION ==================
    for idx, params in enumerate(experiments):

        all_args = {**fixed_args, **params}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%:M%:S")

        exp_name = f"exp_{idx:02d}_{timestamp}"

        log_file = os.path.join(output_dir, f"{exp_name}.log")
        meta_file = os.path.join(output_dir, f"{exp_name}.json")

        args_str = ' '.join([f"--{k} \"{v}\"" for k, v in all_args.items()])
        print(args_str)

        command = f"python run.py {args_str}"
        print(f"🔍 Executing experiment {idx + 1}/{len(experiments)}: {command}")

        with open(meta_file, "w") as f:
            json.dump(all_args, f, indent=4)

        with open(log_file, "w") as f:
            subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

        print(f"✅ Experiment {idx + 1}/{len(experiments)} completed.")

if __name__ == "__main__":
    raw_data_dir = os.getenv("TEST_RAW_DATA_DIR")
    output_dir = os.getenv("GRID_SEARCH_RESULTS_DIR")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%:M%:S")
    exp_name = f"exp_{timestamp}"
    log_file = os.path.join(output_dir, f"{exp_name}.log")
    count = 0

    for impath in os.listdir(raw_data_dir):
        subdir = Path(raw_data_dir) / impath
        if subdir.is_dir():
            for imagem in subdir.glob("*.png"):
                anotacao = imagem.with_suffix('.txt')
                print('rodando experimento:', imagem)
                print('anotacao:', anotacao)
                command = f"python run.py --impath '{imagem}' --annotations_path '{anotacao}'"
                count += 1
                if count > 45:
                    with open(log_file, "w") as f:
                        subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

