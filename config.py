import os
import torch
import random
import numpy as np
from datetime import datetime

class Config:
    # --- Paths ---
    RAW_DATA_ROOT = 'data/ICHBI'
    AUDIO_DIR = os.path.join(RAW_DATA_ROOT, 'ICBHI_final_database')
    DEMO_FILE = os.path.join(RAW_DATA_ROOT, 'ICBHI_Challenge_demographic_information.txt')
    DIAG_FILE = os.path.join(RAW_DATA_ROOT, 'patient_diagnosis.csv')

    # Output
    PROCESSED_DIR = 'data/processed_audio/'
    FULL_CSV = 'data/full_dataset_with_folds.csv'
    
    # --- Model and Report Paths ---
    SAVED_MODELS_DIR = 'results/saved_models'
    REPORT_DIR = 'results/classification_report'
    
    # --- 3-CLASS CONFIGURATION ---
    # The model will now output 3 logits.
    CLASSES = ['COPD', 'Healthy', 'Other']
    NUM_CLASSES = len(CLASSES)

    # Aggregation Map: Raw Diagnosis -> Target Class
    CLASS_MAPPING = {
        'COPD': 'COPD',
        'Healthy': 'Healthy',
        'Asthma': 'Other',
        'Bronchiectasis': 'Other',
        'Bronchiolitis': 'Other',
        'LRTI': 'Other',
        'Pneumonia': 'Other',
        'URTI': 'Other'
    }

    # --- Audio Parameters ---
    SAMPLE_RATE = 22050
    DURATION = 6
    OVERLAP = 0.5 
    STEP_SIZE = int(DURATION * (1 - OVERLAP) * SAMPLE_RATE) 
    TARGET_LEN = SAMPLE_RATE * DURATION
    
    # Feature Config
    N_MELS = 64
    N_MFCC = 40
    N_FFT = 1024
    HOP_LEN = 512

    # --- Training Config ---
    SEED = 42
    NUM_FOLDS = 5 
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- REPRODUCIBILITY UTILS ---
    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"✅ Random seed set to {seed}")

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    @staticmethod
    def get_generator(seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    @staticmethod
    def check_paths():
        required = [Config.AUDIO_DIR, Config.DEMO_FILE, Config.DIAG_FILE]
        for p in required:
            if not os.path.exists(p):
                raise FileNotFoundError(f"❌ Missing required path: {p}")
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)
        os.makedirs(Config.REPORT_DIR, exist_ok=True)


class OmicsConfig:
    # --- Omics Specifics (Teacher remains Binary: Healthy vs Severe) ---
    CLINICAL_PATH = "data/proteomic/raw/data_1_10_22/clinical_data/COPDGene_P1P2P3_Flat_SM_NS_Mar20.txt"
    TRAIN_CSV_PATH = "data/proteomic/processed/original/x_train.csv"
    TEST_CSV_PATH = "data/proteomic/processed/original/x_test.csv"
    TARGET_COL = 'finalGold_P1' 
    INPUT_DIM = 599