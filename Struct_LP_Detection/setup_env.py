import warnings
from dotenv import load_dotenv
from rich.traceback import install

import sys

sys.path.append("../Car_Detection/")
warnings.filterwarnings("ignore")
load_dotenv()
install()
