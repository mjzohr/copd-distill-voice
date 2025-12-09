import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config, OmicsConfig
from dataset import OmicsTextDataset
from models import TextEncoder, OmicsEncoder
from clinical_bridge import ClinicalTextBridge
import os

def run_bridge_masked():
    device = Config.DEVICE
    TOP_K = 5  # Robust averaging
    
    print(f"--- 🌉 Initiating Class-Consistent Bridge (Healthy/COPD Only) ---")

    # 1. Load Models
    text_encoder = TextEncoder(device).to(device)
    omics_encoder = OmicsEncoder(input_dim=OmicsConfig.INPUT_DIM).to(device)
    
    if os.path.exists("omics_encoder_best.pth"):
        omics_encoder.load_state_dict(torch.load("omics_encoder_best.pth"))
    else:
        print("⚠️ Warning: omics_encoder_best.pth not found. Using random weights.")
    omics_encoder.eval()

    # 2. Index Omics Data (Healthy vs COPD)
    omics_ds = OmicsTextDataset()
    loader = torch.utils.data.DataLoader(omics_ds, batch_size=32, shuffle=False)
    
    omics_data = {'Healthy': {'vec': [], 'text': []}, 'COPD': {'vec': [], 'text': []}}
    
    print("--- Indexing Omics ---")
    with torch.no_grad():
        for vec, text_list in tqdm(loader):
            vec = vec.to(device)
            visual_rep = omics_encoder(vec)
            text_emb = text_encoder(list(text_list))
            
            for i, txt in enumerate(text_list):
                if "Healthy" in txt:
                    omics_data['Healthy']['vec'].append(visual_rep[i].cpu())
                    omics_data['Healthy']['text'].append(text_emb[i].cpu())
                elif "COPD" in txt:
                    omics_data['COPD']['vec'].append(visual_rep[i].cpu())
                    omics_data['COPD']['text'].append(text_emb[i].cpu())

    # Stack
    for k in omics_data:
        if len(omics_data[k]['vec']) > 0:
            omics_data[k]['vec'] = torch.stack(omics_data[k]['vec'])
            omics_data[k]['text'] = F.normalize(torch.stack(omics_data[k]['text']), p=2, dim=1)

    # 3. Match Audio Patients
    full_df = pd.read_csv(Config.FULL_CSV)
    audio_to_omics_map = {} 
    bridge = ClinicalTextBridge()
    
    # Strictly Map Audio Class -> Omics Class
    class_map = {'COPD': 'COPD', 'Healthy': 'Healthy'}
    
    for i in tqdm(range(len(full_df)), desc="Bridging"):
        row = full_df.iloc[i]
        filename = row['filename']
        disease = row['disease']
        
        target_class = class_map.get(disease, None)
        
        # If class is Asthma/URTI/etc OR no omics data for class, set None
        if target_class is None or len(omics_data[target_class]['vec']) == 0:
            audio_to_omics_map[filename] = None
            continue
            
        # Find Neighbors within SAME class
        search_text = omics_data[target_class]['text']
        search_vec  = omics_data[target_class]['vec']
        
        text_str = bridge.generate_text(row, source_type='icbhi')
        with torch.no_grad():
            query = F.normalize(text_encoder([text_str]).cpu(), p=2, dim=1)
            
        scores = torch.mm(query, search_text.t())
        k = min(TOP_K, len(search_text))
        _, indices = torch.topk(scores, k=k, dim=1)
        
        centroid = torch.mean(search_vec[indices.squeeze()], dim=0).numpy()
        audio_to_omics_map[filename] = centroid

    np.save("data/audio_omics_distillation_targets.npy", audio_to_omics_map)
    print("✅ Bridge Complete. Targets Saved.")

if __name__ == "__main__":
    run_bridge_masked()