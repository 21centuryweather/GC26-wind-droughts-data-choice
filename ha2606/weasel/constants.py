import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

NCI_USERNAME = os.environ.get('NCI_USERNAME', None)
NCI_PASSWORD = os.environ.get('NCI_PASSWORD', None)

# Data paths
BARRA_PATH   = '/g/data/ob53/catalog/v2/esm/barra2-ob53.csv.gz'
BARPA_PATH   = '/g/data/py18/catalog/v2/esm/barpa-py18.csv.gz'

VARIABLES = [
    'evspsbl',
    'hurs',
    'pr',
    'uas',
    'vas',
    'tasmax',
    'tas',
    'tasmin',
    'mrsos',
    'mrros',
    'rsds',
    'rlds',
    'ps',
    'huss',
    'hfls',
    'hfss',
    'mrro',
]

# Training data paths
PARENT_DIR   = '/g/data/x77/ha2606/barpa'
VAETRAIN_RUN = False
EXPERIMENT_NO = 1

# Variable order mapping (for encoding)
VAR_ORDER = {var: i for i, var in enumerate(VARIABLES)}

# Model architecture constants
CHANNELS = len(VARIABLES)
LATENTS = 16
NEURONS = 160
DOWN_FACTOR = 8
MODE = "kl"
