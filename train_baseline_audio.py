import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from config import Config
from dataset import AudioClassifierDataset
from models import AudioEncoder, ClassifierHead
from utils import EarlyStopping 

def get_weighted_sampler(df, generator):
    class_counts = df['label_idx'].value_counts().sort_index()
    class_weights = 1. / class_counts
    class_weights = class_weights.fillna(0)
    sample_weights = df['label_idx'].map(class_weights).values
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )

def evaluate_patient_level(model, loader):
    model.eval()
    patient_probs, patient_true = {}, {}
    with torch.no_grad():
        for spec, label, pid in loader:
            spec = spec.to(Config.DEVICE)
            logits = model(spec)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for i in range(len(pid)):
                p_id = pid[i].item()
                if p_id not in patient_probs:
                    patient_probs[p_id] = []
                    patient_true[p_id] = label[i].item()
                patient_probs[p_id].append(probs[i])
    preds, truths = [], []
    for p_id, prob_list in patient_probs.items():
        avg_prob = np.mean(prob_list, axis=0)
        preds.append(np.argmax(avg_prob))
        truths.append(patient_true[p_id])
    return truths, preds

def train_kfold():
    Config.set_seed(Config.SEED)
    print(f"--- STARTING BASELINE EXPERIMENT (With Early Stopping) ---")
    
    if not os.path.exists(Config.FULL_CSV):
        print("❌ Data not found.")
        return
        
    full_df = pd.read_csv(Config.FULL_CSV)
    g = Config.get_generator(Config.SEED)
    fold_metrics = {'accuracy': [], 'macro_f1': [], 'weighted_f1': []}
    
    # --- SETUP CM DIRECTORY ---
    cm_save_dir = "results/cm/audio_only"
    os.makedirs(cm_save_dir, exist_ok=True)
    print(f"📂 Saving Confusion Matrices to: {cm_save_dir}")
    
    for fold in range(Config.NUM_FOLDS):
        print(f"\n{'='*20} Baseline Fold {fold} {'='*20}")
        train_df = full_df[full_df['fold'] != fold]
        val_df = full_df[full_df['fold'] == fold]
        
        train_ds = AudioClassifierDataset(train_df)
        val_ds = AudioClassifierDataset(val_df)
        sampler = get_weighted_sampler(train_df, generator=g)
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler, shuffle=False, worker_init_fn=Config.seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, worker_init_fn=Config.seed_worker, generator=g)
        
        # FRESH INIT (Train from scratch)
        encoder = AudioEncoder()
        model = ClassifierHead(encoder, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        for param in model.encoder.parameters(): param.requires_grad = True
            
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.head.parameters(), 'lr': 1e-3}
        ])
        criterion = nn.CrossEntropyLoss()
        
        # EARLY STOPPING SETUP
        checkpoint_path = f"checkpoint_baseline_fold_{fold}.pth"
        early_stopping = EarlyStopping(patience=5, verbose=True, path=checkpoint_path, maximize=True)
        
        EPOCHS = 30
        
        for epoch in range(EPOCHS):
            model.train()
            for spec, label, _ in train_loader:
                spec, label = spec.to(Config.DEVICE), label.to(Config.DEVICE)
                optimizer.zero_grad()
                out = model(spec)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                
            y_true, y_pred = evaluate_patient_level(model, val_loader)
            val_f1 = f1_score(y_true, y_pred, average='weighted')
            
            print(f"   Ep {epoch+1} | Val Weighted F1: {val_f1:.4f}")
            
            early_stopping(val_f1, model)
            if early_stopping.early_stop:
                print("   ⏹️ Early stopping triggered")
                break
        
        # RELOAD BEST MODEL
        model.load_state_dict(torch.load(checkpoint_path))
        best_y_true, best_y_pred = evaluate_patient_level(model, val_loader)

        print(f"\n--- 📊 Report for Fold {fold} ---")
        print(classification_report(
            best_y_true, best_y_pred, 
            labels=range(Config.NUM_CLASSES), 
            target_names=Config.CLASSES, 
            zero_division=0
        ))
        
        # --- PLOT AND SAVE CONFUSION MATRIX ---
        try:
            cm = confusion_matrix(best_y_true, best_y_pred, labels=range(Config.NUM_CLASSES))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=Config.CLASSES, 
                        yticklabels=Config.CLASSES)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix (Audio Only) - Fold {fold}')
            
            save_path = os.path.join(cm_save_dir, f'cm_fold_{fold}.png')
            plt.savefig(save_path)
            plt.close() 
            print(f"   📊 Saved Confusion Matrix: {save_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save confusion matrix: {e}")
        # --------------------------------------
        
        fold_metrics['accuracy'].append(accuracy_score(best_y_true, best_y_pred))
        fold_metrics['macro_f1'].append(f1_score(best_y_true, best_y_pred, average='macro'))
        fold_metrics['weighted_f1'].append(f1_score(best_y_true, best_y_pred, average='weighted'))

    print("\n" + "="*40)
    print("        FINAL BASELINE RESULTS")
    print("="*40)
    for metric_name, values in fold_metrics.items():
        print(f"{metric_name.replace('_', ' ').title():<15}: {np.mean(values)*100:.2f}% (+/- {np.std(values)*100:.2f})")
    print("="*40)

if __name__ == "__main__":
    train_kfold()