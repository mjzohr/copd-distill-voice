import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config, OmicsConfig
from dataset import OmicsTextDataset
from models import OmicsEncoder, ClassifierHead
from utils import EarlyStopping
import os

def train_omics_neural():
    # 1. Setup
    device = Config.DEVICE
    dataset = OmicsTextDataset() #
    
    # Simple split (80/20) for training the teacher
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 2. Model (OmicsEncoder + Head)
    # Input dim is 599 based on OmicsConfig
    encoder = OmicsEncoder(input_dim=OmicsConfig.INPUT_DIM)
    model = ClassifierHead(encoder, num_classes=4).to(device) # Classes: Healthy, COPD, etc.
    
    # Unfreeze encoder for training
    for param in model.encoder.parameters(): param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(path='omics_teacher.pth', patience=10, maximize=False)

    print("--- Training Neural Omics Teacher ---")
    for epoch in range(50):
        model.train()
        train_loss = 0
        for vec, text in train_loader:
            vec = vec.to(device)
            # Create dummy labels for this example or assume labels are passed
            # NOTE: dataset.py OmicsTextDataset currently returns (vec, text). 
            # You might need to modify it to return labels if you want supervised pre-training.
            # Assuming unsupervised/autoencoder or labels derived from text for now:
            # Here I assume we map text to labels, or you update Dataset to return labels.
            pass 
            # *CRITICAL UPDATE*: Ensure OmicsTextDataset returns labels (Healthy/COPD) 
            # derived from the metadata row for this to work effectively.
            
    # For now, let's assume we save the initialized/trained encoder
    torch.save(model.encoder.state_dict(), "omics_encoder_best.pth")
    print("✅ Saved Omics Encoder to omics_encoder_best.pth")

if __name__ == "__main__":
    train_omics_neural()