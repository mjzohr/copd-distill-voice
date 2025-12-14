import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from datetime import datetime

from config import Config
from dataset import DistilledAudioDataset
from models import AudioEncoder, ClassifierHead, DistillationProjector
from utils import EarlyStopping 

def create_report_entry_distilled(fold_metrics, max_lambda, rampup_epochs):
    """Generates a single-row DataFrame for the final distilled report."""
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M")
    
    # Hyperparameters & Setup
    data = {
        'run_type': ['distilled_audio'],
        'timestamp': [timestamp],
        'seed': [Config.SEED],
        'folds': [Config.NUM_FOLDS],
        'epochs_max': [40], # Defined in script
        'patience': [8],    # Defined in script
        'batch_size': [Config.BATCH_SIZE],
        'lr_model': ['1e-4'],
        'lr_projector': ['1e-3'],
        'distill_lambda': [max_lambda],
        'rampup_epochs': [rampup_epochs],
        'device': [str(Config.DEVICE)],
        'audio_mels': [Config.N_MELS],
        'audio_mfcc': [Config.N_MFCC]
    }
    
    # Results (Mean +/- Std)
    for metric_name, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        data[metric_name] = [f"{mean_val*100:.2f} (+/- {std_val*100:.2f})"]

    return pd.DataFrame(data)

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

def evaluate_patient_level_distilled(model, loader):
    model.eval()
    patient_probs, patient_true = {}, {}
    with torch.no_grad():
        for spec, label, _, _, pid in loader:
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

def train_distilled_kfold():
    Config.set_seed(Config.SEED)
    Config.check_paths() # Ensure new directories exist
    print(f"--- STARTING CORRECTED MASKED DISTILLATION ---")
    
    if not os.path.exists(Config.FULL_CSV): return
    target_path = "data/audio_omics_distillation_targets.npy"
    if not os.path.exists(target_path): 
        print("Run run_bridge.py first.")
        return
        
    full_df = pd.read_csv(Config.FULL_CSV)
    g = Config.get_generator(Config.SEED)
    fold_metrics = {'accuracy': [], 'macro_f1': [], 'weighted_f1': []}
    
    cm_save_dir = "results/cm/distilled_corrected"
    os.makedirs(cm_save_dir, exist_ok=True)
    
    # Distillation Weight
    # Since we normalized correctly, the signal is stronger. 
    # We set a reasonable Lambda.
    MAX_LAMBDA = 2.0 
    RAMPUP_EPOCHS = 8 
    
    for fold in range(Config.NUM_FOLDS):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        train_df = full_df[full_df['fold'] != fold]
        val_df = full_df[full_df['fold'] == fold]
        
        train_ds = DistilledAudioDataset(train_df, target_path)
        val_ds = DistilledAudioDataset(val_df, target_path)
        
        sampler = get_weighted_sampler(train_df, generator=g)
        
        # --- FIX: ADDED drop_last=True TO AVOID BATCH NORM CRASH ON SIZE 1 BATCHES ---
        train_loader = DataLoader(
            train_ds, 
            batch_size=Config.BATCH_SIZE, 
            sampler=sampler, 
            shuffle=False, 
            worker_init_fn=Config.seed_worker, 
            generator=g,
            drop_last=True 
        )
        
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, worker_init_fn=Config.seed_worker, generator=g)
        
        encoder = AudioEncoder()
        model = ClassifierHead(encoder, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        projector = DistillationProjector(input_dim=512, output_dim=512).to(Config.DEVICE)
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': 1e-4},
            {'params': projector.parameters(), 'lr': 1e-3}
        ])
        
        ce_loss_fn = nn.CrossEntropyLoss()
        distill_loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='none') # REDUCTION NONE IS KEY
        
        # EARLY STOPPING SETUP: Use new directory
        checkpoint_path = os.path.join(Config.SAVED_MODELS_DIR, f"checkpoint_distilled_fold_{fold}.pth")
        early_stopping = EarlyStopping(patience=6, verbose=True, path=checkpoint_path, maximize=True)
        
        EPOCHS = 40
        
        for epoch in range(EPOCHS):
            model.train()
            projector.train()
            running_loss = 0.0
            
            # Linear Warmup
            current_lambda = MAX_LAMBDA * min(1.0, epoch / RAMPUP_EPOCHS)
            
            for spec, label, target_omics, mask, _ in train_loader:
                spec = spec.to(Config.DEVICE)
                label = label.to(Config.DEVICE)
                target_omics = target_omics.to(Config.DEVICE)
                mask = mask.to(Config.DEVICE) # 0 or 1
                
                optimizer.zero_grad()
                
                logits, audio_feat = model(spec, return_features=True)
                projected_audio = projector(audio_feat)
                
                # 1. Classification Loss (All samples)
                loss_ce = ce_loss_fn(logits, label)
                
                # 2. Distillation Loss (Active Only)
                target_ones = torch.ones(spec.size(0)).to(Config.DEVICE)
                raw_distill = distill_loss_fn(projected_audio, target_omics, target_ones)
                
                # --- THE FIX IS HERE ---
                masked_distill = raw_distill * mask
                num_active = mask.sum()
                
                if num_active > 0:
                    # Normalize by number of ACTIVE samples, not batch size
                    loss_distill_final = masked_distill.sum() / num_active
                else:
                    loss_distill_final = torch.tensor(0.0).to(Config.DEVICE)
                
                # Total
                loss = loss_ce + (current_lambda * loss_distill_final)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            y_true, y_pred = evaluate_patient_level_distilled(model, val_loader)
            val_f1 = f1_score(y_true, y_pred, average='weighted')
            
            print(f"   Ep {epoch+1} (λ={current_lambda:.2f}) | Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
            
            early_stopping(val_f1, model)
            if early_stopping.early_stop:
                print("   ⏹️ Early stopping triggered")
                break
        
        model.load_state_dict(torch.load(checkpoint_path))
        best_y_true, best_y_pred = evaluate_patient_level_distilled(model, val_loader)

        print(f"\n--- 📊 Report for Fold {fold} ---")
        report = classification_report(best_y_true, best_y_pred, 
                                    labels=range(Config.NUM_CLASSES), 
                                    target_names=Config.CLASSES, zero_division=0,
                                    output_dict=True)
        print(classification_report(best_y_true, best_y_pred, 
                                    labels=range(Config.NUM_CLASSES), 
                                    target_names=Config.CLASSES, zero_division=0))
        
        # Save CM
        try:
            cm = confusion_matrix(best_y_true, best_y_pred, labels=range(Config.NUM_CLASSES))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
            save_path = os.path.join(cm_save_dir, f'cm_fold_{fold}.png')
            plt.savefig(save_path)
            plt.close()
        except: pass
        
        # Capture metrics for final report
        fold_metrics['accuracy'].append(report['accuracy'])
        fold_metrics['macro_f1'].append(report['macro avg']['f1-score'])
        fold_metrics['weighted_f1'].append(report['weighted avg']['f1-score'])
        
        # Save final model for the fold (not just best checkpoint)
        final_model_path = os.path.join(Config.SAVED_MODELS_DIR, f"final_distilled_fold_{fold}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"   ✅ Saved Final Model to: {final_model_path}")

    print("\n" + "="*40)
    print("      FINAL CORRECTED RESULTS")
    print("="*40)
    
    # Generate and Save CSV Report
    report_df = create_report_entry_distilled(fold_metrics, MAX_LAMBDA, RAMPUP_EPOCHS)
    timestamp = report_df['timestamp'][0]
    csv_path = os.path.join(Config.REPORT_DIR, f"distilled_report_{timestamp}.csv")
    report_df.to_csv(csv_path, index=False)

    for metric_name, values in fold_metrics.items():
        print(f"{metric_name.replace('_', ' ').title():<15}: {np.mean(values)*100:.2f}% (+/- {np.std(values)*100:.2f})")
    print(f"\n✅ Saved final report to: {csv_path}")
    print("="*40)

if __name__ == "__main__":
    train_distilled_kfold()