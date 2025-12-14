#!/bin/bash

# ==========================================
# Cross-Modal Knowledge Distillation Pipeline
# ==========================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 STARTING PIPELINE..."
echo "---------------------------------"

# --- Step 1: Data Preprocessing ---
# This slices the audio files and creates the master CSV with folds.
# Uses: config.py, preprocessing.py
if [ ! -f "data/full_dataset_with_folds.csv" ]; then
    echo "[Step 1/4] Processing Raw Audio Data..."
    python preprocessing.py
else
    echo "[Step 1/4] ✅ Audio Data already processed. Skipping..."
fi

python preprocessing.py

# --- Step 2: Train Omics Teacher ---
# Trains the Neural Omics Encoder (not the XGBoost baseline) to learn protein representations.
# Output: omics_encoder_best.pth
echo "[Step 2/4] Training Omics Teacher Model..."
python train_omics_neural.py

# --- Step 3: Run Clinical Bridge ---
# 1. Matches Audio patients to Omics patients using Text Encoders.
# 2. Uses the matched Omics Encoder to generate target vectors.
# Output: data/audio_omics_distillation_targets.npy
echo "[Step 3/4] Running Clinical Bridge & Generating Targets..."
python run_bridge.py

# --- Step 4: Train Distilled Audio Model ---
# Trains the audio model using the combined Loss = CrossEntropy + MSE(Omics_Target).
echo "[Step 4/4] Training Distilled Audio Model..."
python train_distilled_kfold.py

echo "---------------------------------"
echo "🎉 PIPELINE COMPLETE!"