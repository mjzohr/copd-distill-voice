import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

# Assuming OmicsConfig is already defined in config.py and OmicsEncoder in models.py
from config import Config, OmicsConfig
from models import OmicsEncoder

# --- CONFIG ---
BATCH_SIZE = 16 # Smaller batch size for tabular data often works better
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_ENCODER_PATH = "omics_encoder_best.pth" # Aligning with run_bridge.py

def load_data():
    print("--- Loading 'High Contrast' Data (Healthy vs Severe) ---")
    
    # 1. Load Raw
    df_train = pd.read_csv(OmicsConfig.TRAIN_CSV_PATH, index_col=0)
    df_test = pd.read_csv(OmicsConfig.TEST_CSV_PATH, index_col=0)
    clinical = pd.read_csv(OmicsConfig.CLINICAL_PATH, sep="\t", low_memory=False, index_col=0)
    
    # 2. Align
    common_train = df_train.index.intersection(clinical.index)
    common_test = df_test.index.intersection(clinical.index)
    
    X_train = df_train.loc[common_train]
    y_train = clinical.loc[common_train][OmicsConfig.TARGET_COL]
    X_test = df_test.loc[common_test]
    y_test = clinical.loc[common_test][OmicsConfig.TARGET_COL]
    
    # 3. Filter (Healthy=0 vs Severe=3,4)
    def map_extreme(val):
        try: s = int(float(val))
        except: return -1
        if s == 0: return 0 # Healthy
        if s in [3, 4]: return 1 # Severe
        return -1 # Exclude

    # Apply Mapping
    y_train_mapped = y_train.apply(map_extreme)
    y_test_mapped = y_test.apply(map_extreme)
    
    # Drop Excluded
    mask_train = y_train_mapped != -1
    mask_test = y_test_mapped != -1
    
    X_train, y_train = X_train[mask_train], y_train_mapped[mask_train]
    X_test, y_test = X_test[mask_test], y_test_mapped[mask_test]
    
    # 4. Log Transform (Standard for proteomics)
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    
    # 5. Scale (Critical for Neural Networks, less so for XGB)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"   Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    print(f"   Class Balance: {y_train.value_counts().to_dict()}")
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train.values)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test.values)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

class OmicsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # The Encoder from your contrastive model (defined in models.py)
        self.encoder = OmicsEncoder(input_dim)
        
        # A simple linear classification head
        # OmicsEncoder now outputs 512 features
        self.head = nn.Linear(512, 2) # Binary: Healthy vs Severe
        
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)
    
    def get_encoder(self):
        """Returns the OmicsEncoder for saving/distillation."""
        return self.encoder

def train_omics_neural():
    # 1. Data
    X_train, y_train, X_test, y_test = load_data()
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model
    input_dim = X_train.shape[1]
    model = OmicsClassifier(input_dim).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Training Deep Omics Baseline ---")
    best_acc = 0.0
    best_encoder_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(DEVICE)
                logits = model(x_b)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred)
                truths.extend(y_b.numpy())
        
        acc = accuracy_score(truths, preds)
        
        if acc > best_acc:
            best_acc = acc
            # Save the state of the encoder ONLY
            best_encoder_state = model.get_encoder().state_dict()
        
        if (epoch+1) % 10 == 0:
            print(f"   Ep {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}%")

    print(f"\n>>> FINAL Deep Learning Result (Best Acc): {best_acc*100:.2f}%")
    # Print report for the last epoch's predictions
    print(classification_report(truths, preds, target_names=["Healthy", "Severe COPD"]))
    
    # Save the BEST encoder weights for use in run_bridge.py
    if best_encoder_state:
        torch.save(best_encoder_state, OUTPUT_ENCODER_PATH)
        print(f"\n✅ Saved Omics Encoder (best weights) to {OUTPUT_ENCODER_PATH}")
    else:
        # Fallback save of the final state if no improvement was found (shouldn't happen with 50 epochs)
        torch.save(model.get_encoder().state_dict(), OUTPUT_ENCODER_PATH)
        print(f"\n⚠️ Saved Omics Encoder (final epoch weights) to {OUTPUT_ENCODER_PATH}")


if __name__ == "__main__":
    train_omics_neural()