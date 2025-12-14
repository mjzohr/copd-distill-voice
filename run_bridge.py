import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config, OmicsConfig
from dataset import OmicsTextDataset
from models import OmicsEncoder
import os

# --- CONFIGURATION ---
AGE_TOLERANCE = 5  # Years (+/-)
OMICS_MODEL_PATH = "omics_encoder_best.pth"
OUTPUT_TARGETS_PATH = "data/audio_omics_distillation_targets.npy"

def normalize_sex(val):
    """
    Normalize sex to 'Male', 'Female', or None.
    Handles both string ('F', 'M') and numeric (1=Male, 2=Female) formats.
    """
    val = str(val).lower()
    if 'f' in val or '2' in val: return 'Female'
    if 'm' in val or '1' in val: return 'Male'
    return None

def normalize_omics_disease(row):
    """
    Map Omics GOLD Stage to our classes.
    GOLD 0 -> 'Healthy'
    GOLD >= 3 -> 'COPD' (Severe)
    Others -> None (We don't use them for training targets)
    """
    try:
        # Check keys based on your specific clinical file headers
        # Adjust 'finalGold_P1' if your CSV header is different
        gold = row.get('finalGold_P1')
        if gold is None: return None
        
        g = int(float(gold))
        if g == 0: return 'Healthy'
        if g >= 3: return 'COPD' 
    except:
        pass
    return None

def run_value_based_bridge():
    # Ensure reproducible environment
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    
    print(f"--- 🌉 Initiating Clinical Value-Based Bridge ---")
    print(f"   Criteria: Sex=Exact, Disease=Exact, Age=+/-{AGE_TOLERANCE} yrs")

    # ---------------------------------------------------------
    # 1. Load Omics Encoder (The Teacher)
    # ---------------------------------------------------------
    print("   Loading Omics Encoder...")
    omics_encoder = OmicsEncoder(input_dim=OmicsConfig.INPUT_DIM).to(device)
    
    if os.path.exists(OMICS_MODEL_PATH):
        # Load weights
        state_dict = torch.load(OMICS_MODEL_PATH, map_location=device)
        omics_encoder.load_state_dict(state_dict)
        print(f"   ✅ Loaded weights from {OMICS_MODEL_PATH}")
    else:
        print(f"   ⚠️ WARNING: {OMICS_MODEL_PATH} not found! Using random weights.")
    
    omics_encoder.eval()

    # ---------------------------------------------------------
    # 2. Build Omics "Database"
    # ---------------------------------------------------------
    print("   Building Omics Patient Database...")
    # We use the dataset class to load data, but access internals directly
    omics_ds = OmicsTextDataset()
    
    omics_db = []
    
    # Process in batches to get vectors efficiently
    batch_size = 64
    num_samples = len(omics_ds)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Slice features and move to GPU
            batch_feats = omics_ds.features[i : i+batch_size].to(device) # [B, InputDim]
            
            # Generate Teacher Vectors (Forward pass)
            teacher_vecs = omics_encoder(batch_feats).cpu() # [B, 512]
            
            # Get corresponding metadata rows
            batch_meta = omics_ds.meta_rows[i : i+batch_size]
            
            for j, row in enumerate(batch_meta):
                age = row.get('Age_P1') or row.get('Age')
                sex = normalize_sex(row.get('gender') or row.get('Sex'))
                disease = normalize_omics_disease(row)
                
                # Only add if we have all necessary clinical fields
                if age is not None and sex is not None and disease is not None:
                    omics_db.append({
                        'vec': teacher_vecs[j],
                        'age': float(age),
                        'sex': sex,
                        'disease': disease
                    })
    
    df_omics = pd.DataFrame(omics_db)
    print(f"   Valid Omics Candidates Found: {len(df_omics)}")
    if len(df_omics) > 0:
        print(f"   [Example Omics Entry]: {df_omics.iloc[0].drop('vec').to_dict()}")

    # ---------------------------------------------------------
    # 3. Process Audio Patients (The Students)
    # ---------------------------------------------------------
    full_df = pd.read_csv(Config.FULL_CSV)
    audio_to_omics_map = {}
    
    # Define which Audio Classes map to which Omics Classes
    TARGET_MAPPING = {
        'COPD': 'COPD',      # Audio COPD maps to Omics COPD
        'Healthy': 'Healthy' # Audio Healthy maps to Omics Healthy
    }
    
    matched_count = 0
    
    print(f"   Bridging {len(full_df)} Audio Samples...")
    
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df)):
        filename = row['filename']
        audio_disease = row['disease']
        
        # A. Check if class is mappable (Asthma/URTI will be skipped here)
        target_disease = TARGET_MAPPING.get(audio_disease)
        if not target_disease:
            audio_to_omics_map[filename] = None
            continue
            
        # B. Get Audio Metadata
        audio_sex = normalize_sex(row.get('sex'))
        try:
            audio_age = float(row.get('age'))
        except:
            # If age is missing/invalid, we cannot match reliably
            audio_to_omics_map[filename] = None
            continue
            
        # C. FIND MATCHES
        # Logic: Exact Sex AND Exact Disease Class AND Age within Tolerance
        matches = df_omics[
            (df_omics['sex'] == audio_sex) & 
            (df_omics['disease'] == target_disease) & 
            (df_omics['age'] >= audio_age - AGE_TOLERANCE) &
            (df_omics['age'] <= audio_age + AGE_TOLERANCE)
        ]
        
        if len(matches) == 0:
            # No matching omics profile found
            audio_to_omics_map[filename] = None
        else:
            # D. AVERAGE VECTORS
            # Stack all matching vectors and compute the mean
            vectors = torch.stack(matches['vec'].tolist())
            avg_target = torch.mean(vectors, dim=0).numpy()
            
            audio_to_omics_map[filename] = avg_target
            matched_count += 1

    # ---------------------------------------------------------
    # 4. Save Results
    # ---------------------------------------------------------
    np.save(OUTPUT_TARGETS_PATH, audio_to_omics_map)
    
    print(f"✅ Bridge Complete.")
    print(f"   Matched Audio Files: {matched_count} / {len(full_df)}")
    print(f"   Targets saved to: {OUTPUT_TARGETS_PATH}")

if __name__ == "__main__":
    run_value_based_bridge()