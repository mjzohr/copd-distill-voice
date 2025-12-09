import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import Config

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=768, projection_dim=128):
    # def __init__(self, input_dim, embed_dim=768, projection_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = Config.N_MELS + Config.N_MFCC 
        
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), 
            
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # Returns 512 dim

class OmicsEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, text_list):
        with torch.no_grad():
            inputs = self.tokenizer(
                text_list, return_tensors="pt", padding=True, 
                truncation=True, max_length=64
            ).to(self.device)
            out = self.bert(**inputs)
        return out.last_hidden_state[:, 0, :]

class ClassifierHead(nn.Module):
    def __init__(self, encoder, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        # Default behavior: Frozen encoder (can unfreeze in training script)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)
    

class DistillationProjector(nn.Module):
    """
    Maps the Audio Representation (512 dim) to the Omics Representation space (512 dim)
    using a non-linear transformation.
    """
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    """
    Updated to optionally return the Features (representation) 
    so we can feed them to the Projector during training.
    """
    def __init__(self, encoder, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        # Unfreeze encoder by default
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits