import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from config import Config, OmicsConfig
from clinical_bridge import ClinicalTextBridge 

class AudioAugmentations:
    def __init__(self):
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    
    def __call__(self, spec):
        aug_spec = self.time_mask(spec)
        aug_spec = self.freq_mask(aug_spec)
        noise = torch.randn_like(aug_spec) * 0.05
        return aug_spec + noise

# --- FOR CONTRASTIVE PRE-TRAINING ---
class AudioTextDataset(Dataset):
    def __init__(self, data_source, augment=False):
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source.reset_index(drop=True)
            
        self.bridge = ClinicalTextBridge()
        self.augment = augment
        self.augmenter = AudioAugmentations()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
            n_fft=Config.N_FFT, hop_length=Config.HOP_LEN
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC,
            melkwargs={'n_fft': Config.N_FFT, 'n_mels': 128, 'hop_length': Config.HOP_LEN, 'mel_scale': 'htk'}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(Config.PROCESSED_DIR, row['filename'])
        wav_numpy, sr = sf.read(path)
        waveform = torch.from_numpy(wav_numpy).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            
        mel = torch.log(self.mel_transform(waveform).squeeze(0) + 1e-9)
        mfcc = self.mfcc_transform(waveform).squeeze(0)
        audio_features = torch.cat([mel, mfcc], dim=0)
        
        if self.augment:
            audio_features = self.augmenter(audio_features)
            
        text = self.bridge.generate_text(row, source_type='icbhi')
        return audio_features, text

class OmicsTextDataset(Dataset):
    def __init__(self):
        self.bridge = ClinicalTextBridge()
        self.features, self.meta_rows = self._load_data()

    def _load_data(self):
        df = pd.read_csv(OmicsConfig.TRAIN_CSV_PATH, index_col=0)
        clinical = pd.read_csv(OmicsConfig.CLINICAL_PATH, sep="\t", low_memory=False, index_col=0)
        common = df.index.intersection(clinical.index)
        df = df.loc[common]
        clinical = clinical.loc[common]
        
        valid_indices = []
        for idx, row in clinical.iterrows():
            try:
                g = int(float(row[OmicsConfig.TARGET_COL]))
                if g == 0 or g >= 3: valid_indices.append(idx)
            except: pass
        
        df = df.loc[valid_indices]
        clinical = clinical.loc[valid_indices]
        X = np.log1p(df.values).astype(np.float32)
        meta = clinical.to_dict('records')
        return torch.tensor(X), meta

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        omics_vec = self.features[idx]
        row = self.meta_rows[idx]
        text = self.bridge.generate_text(row, source_type='omics')
        return omics_vec, text

# --- FOR STANDARD CLASSIFICATION ---
class AudioClassifierDataset(Dataset):
    def __init__(self, data_source):
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source.reset_index(drop=True)
            
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
            n_fft=Config.N_FFT, hop_length=Config.HOP_LEN
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC,
            melkwargs={'n_fft': Config.N_FFT, 'n_mels': 128, 'hop_length': Config.HOP_LEN, 'mel_scale': 'htk'}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(Config.PROCESSED_DIR, row['filename'])
        wav_numpy, sr = sf.read(path)
        waveform = torch.from_numpy(wav_numpy).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            
        mel = torch.log(self.mel_transform(waveform).squeeze(0) + 1e-9)
        mfcc = self.mfcc_transform(waveform).squeeze(0)
        audio_features = torch.cat([mel, mfcc], dim=0)

        label = int(row['label_idx'])
        pid = row['pid']
        
        return audio_features, label, pid

# --- NEW: DISTILLED DATASET WITH MASKING ---
class DistilledAudioDataset(Dataset):
    def __init__(self, data_source, target_map_path):
        """
        Returns: Audio, Label, Target_Vector, Mask, PID
        Mask is 1.0 if target exists, 0.0 otherwise.
        """
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source.reset_index(drop=True)
            
        self.target_map = np.load(target_map_path, allow_pickle=True).item()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
            n_fft=Config.N_FFT, hop_length=Config.HOP_LEN
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC,
            melkwargs={'n_fft': Config.N_FFT, 'n_mels': 128, 'hop_length': Config.HOP_LEN, 'mel_scale': 'htk'}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Load Audio
        path = os.path.join(Config.PROCESSED_DIR, row['filename'])
        wav_numpy, sr = sf.read(path)
        waveform = torch.from_numpy(wav_numpy).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            
        mel = torch.log(self.mel_transform(waveform).squeeze(0) + 1e-9)
        mfcc = self.mfcc_transform(waveform).squeeze(0)
        audio_features = torch.cat([mel, mfcc], dim=0)

        # 2. Get Label and PID
        label = int(row['label_idx'])
        pid = row['pid']
        
        # 3. Get Distillation Target (Logic for Masking)
        target_entry = self.target_map.get(row['filename'])
        
        if target_entry is None:
            # Case: Asthma, URTI (No matching Omics class)
            target_vec = torch.zeros(512).float() # Dummy zero vector
            mask = 0.0 # Tells loss function to IGNORE this sample
        else:
            # Case: COPD, Healthy (Matching Omics class exists)
            target_vec = torch.tensor(target_entry).float()
            mask = 1.0 # Tells loss function to USE this sample

        # Return 5 items
        return audio_features, label, target_vec, mask, pid