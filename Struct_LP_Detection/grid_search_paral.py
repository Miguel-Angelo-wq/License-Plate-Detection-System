import os
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import setup_env

def run_experiment(imagem_path, anotacao_path, log_file_path, output_path):
    command = f"python run.py --impath '{imagem_path}' --annotations_path '{anotacao_path}' --output_path '{output_path}'"
    with open(log_file_path, "w") as f:
        subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

# Diretórios e setup
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

raw_data_dir = os.getenv("TEST_RAW_DATA_DIR")
output_dir = Path(os.getenv("GRID_SEARCH_RESULTS_DIR"))
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_name = f"exp_{timestamp}"

max_workers = 3
futures = []
result_id = 0

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for impath in os.listdir(raw_data_dir):
        subdir = Path(raw_data_dir) / impath
        if subdir.is_dir():
            for imagem in subdir.glob("*.png"):
                anotacao = imagem.with_suffix('.txt')
                log_file_path = log_dir / f"log_{result_id}.txt"
                output_path = output_dir / f"results_{result_id}.csv"
                result_id += 1
                if result_id > 49:

                    futures.append(
                        executor.submit(run_experiment, str(imagem), str(anotacao), str(log_file_path), str(output_path))
                    )

