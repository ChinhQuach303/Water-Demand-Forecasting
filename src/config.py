import os
import random
import numpy as np
import logging

# ============= CONFIGURATION =============
FILE_PATH = 'data/CaRDS.csv'
ARTIFACTS_DIR = 'artifacts'
DATA_DIR = os.path.join(ARTIFACTS_DIR, 'data')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_SEED = 42

SUMMER_MONTHS = [6, 7, 8]

# ============= LOGGING SETUP =============
def setup_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

# ============= SEEDING FUNCTION =============
def set_seed(seed=RANDOM_SEED):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 Global Seed set to: {seed}")
