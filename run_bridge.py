import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config, OmicsConfig
from dataset import OmicsTextDataset
from models import OmicsEncoder
import os

# --- CONFIGURATION ---
AGE_TOLERANCE = 5 
OMICS_MODEL_PATH = "omics_encoder_best.pth"
OUTPUT_TARGETS_PATH = "data/audio_omics_distillation_targets.npy"

def normalize_sex(val):
    val = str(val).lower()
    if 'f' in val or '2' in val: return 'Female'
    if 'm' in val or '1' in val: return 'Male'
    return None

def normalize_omics_disease(row):
    """
    Teacher Data (COPDGene) only knows Healthy vs COPD.
    """
    try:
        gold = row.get('finalGold_P1')
        if gold is None: return None
        g = int(float(gold))
        if g == 0: return 'Healthy'
        if g >= 3: return 'COPD' 
    except:
        pass
    return None

def run_value_based_bridge():
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    
    print(f"--- 🌉 Initiating Clinical Bridge (3-Class Adaptation) ---")
    
    # 1. Load Omics Encoder
    omics_encoder = OmicsEncoder(input_dim=OmicsConfig.INPUT_DIM).to(device)
    if os.path.exists(OMICS_MODEL_PATH):
        omics_encoder.load_state_dict(torch.load(OMICS_MODEL_PATH, map_location=device))
        print(f"   ✅ Loaded weights from {OMICS_MODEL_PATH}")
    else:
        print(f"   ⚠️ {OMICS_MODEL_PATH} not found. Ensure you ran train_omics_neural.py first.")
        return 
    omics_encoder.eval()

    # 2. Build Omics Database
    print("   Building Omics Patient Database...")
    omics_ds = OmicsTextDataset()
    omics_db = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(omics_ds), batch_size):
            batch_feats = omics_ds.features[i : i+batch_size].to(device)
            teacher_vecs = omics_encoder(batch_feats).cpu()
            batch_meta = omics_ds.meta_rows[i : i+batch_size]
            
            for j, row in enumerate(batch_meta):
                age = row.get('Age_P1') or row.get('Age')
                sex = normalize_sex(row.get('gender') or row.get('Sex'))
                disease = normalize_omics_disease(row)
                
                if age and sex and disease:
                    omics_db.append({'vec': teacher_vecs[j], 'age': float(age), 'sex': sex, 'disease': disease})
    
    df_omics = pd.DataFrame(omics_db)
    print(f"   Teacher Database: {len(df_omics)} profiles (Healthy/COPD only).")

    # 3. Process Audio Patients
    full_df = pd.read_csv(Config.FULL_CSV)
    
    # MAPPING FOR DISTILLATION:
    # We only have teachers for COPD and Healthy.
    # 'Other' (Asthma, etc.) MUST be mapped to None so they get Masked=0 in training.
    TARGET_MAPPING = {
        'COPD': 'COPD',      
        'Healthy': 'Healthy',
        'Other': None 
    }
    
    audio_to_omics_map = {}
    matched_count = 0
    skipped_count = 0
    
    print(f"   Bridging {len(full_df)} Audio Samples...")
    
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        filename = row['filename']
        audio_disease = row['disease'] # This is now 'COPD', 'Healthy', or 'Other'
        
        target_disease = TARGET_MAPPING.get(audio_disease)
        
        # If Other, or unknown, skip immediately
        if not target_disease:
            audio_to_omics_map[filename] = None
            skipped_count += 1
            continue
            
        audio_sex = normalize_sex(row.get('sex'))
        try: audio_age = float(row.get('age'))
        except: 
            audio_to_omics_map[filename] = None
            continue
            
        # Find Matches
        matches = df_omics[
            (df_omics['sex'] == audio_sex) & 
            (df_omics['disease'] == target_disease) & 
            (df_omics['age'] >= audio_age - AGE_TOLERANCE) &
            (df_omics['age'] <= audio_age + AGE_TOLERANCE)
        ]
        
        if len(matches) == 0:
            audio_to_omics_map[filename] = None
        else:
            vectors = torch.stack(matches['vec'].tolist())
            avg_target = torch.mean(vectors, dim=0).numpy()
            audio_to_omics_map[filename] = avg_target
            matched_count += 1

    np.save(OUTPUT_TARGETS_PATH, audio_to_omics_map)
    print(f"✅ Bridge Complete.")
    print(f"   Distillable Samples (Mask=1): {matched_count}")
    print(f"   Non-Distillable Samples (Mask=0, mainly 'Other'): {skipped_count}")

if __name__ == "__main__":
    run_value_based_bridge()