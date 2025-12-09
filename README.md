# Multimodal Respiratory Disease Classification using Contrastive Learning

This repository implements a multimodal framework (Audio + Text + Synthetic Omics) to classify respiratory diseases using the **ICBHI 2017 Challenge Database**.

It addresses a key challenge in medical AI: while audio data is easy to collect, it is often noisy and insufficient for modeling complex biological heterogeneity. This framework uses `Clinical Text` and `Proteomics (Omics)` as "anchors" during pre-training to force the audio model to learn biologically grounded features.

Key Innovation:

* Clinical-Text-Based Contrastive Bridge: We convert clinical metadata into natural language (e.g., "A 75-year-old male with COPD...") encoded by Bio_ClinicalBERT, to create a clinical text-based bridge between forcing the audio model to learn biologically relevant features.

* Cross-Modal Alignment: The Audio model aligns with these text prompts in a shared latent space (via Bio_ClinicalBERT), indirectly capturing the structure of the Omics data.

* Robustness Improvement: The contrastive pre-trained model demonstrates superior stability on "hard" data splits, maintaining 64% accuracy on outlier folds where traditional supervised baselines drop to 56%.

## 📂 Data & Preprocessing

### 1. Expected Input Data
The expected ICBHI 2017 dataset structure includes:

- **Audio:** `.wav` lung sound recordings  
- **Metadata:**  
  - `ICBHI_Challenge_demographic_information.txt`  
  - `patient_diagnosis.csv`
- **Omics (Optional):** Proteomic data files defined in `OmicsConfig`

---

### 2. Preprocessing Pipeline (`preprocessing.py`)

- **Metadata Merging:** Connects demographics & diagnosis labels to audio files  
- **Slicing:** Converts raw audio into 6-second segments (optionally centered on crackles/wheezes)  
- **Stratified Group K-Fold:** Ensures all audio slices from a single patient remain in the same fold, preventing leakage  

---

### 3. Output Files
```
data/processed_audio/              # Normalized .wav segments
data/full_dataset_with_folds.csv   # Filenames, labels, folds, text descriptions
```


---

## 🧠 Model Architecture

The framework uses a **CLIP-style contrastive learning** approach.

### Encoders

#### 1. Audio Encoder
- Custom 1D CNN  
- **Input:** Mel-spectrograms + MFCC  
- **Output:** 512-dimensional embedding  

#### 2. Text Encoder
- Frozen **Bio_ClinicalBERT**  
- **Input:** Natural language prompts  
- **Role:** Provides a stable semantic anchor  

#### 3. Omics Encoder
- MLP consuming proteomic vectors  
- **Role:** Aligns biological ground truth with the text bridge  

---

## Training Phases

### 1. Contrastive Pre-training
Optimizes symmetric contrastive loss to maximize similarity for:

- **Audio ↔ Text**  
- **Omics ↔ Text**  

**Result:** The audio encoder learns biologically meaningful structure.

---

### 2. Fine-Tuning (Differential Learning Rates)

| Component | Learning Rate |
|----------|----------------|
| Audio Backbone | 1e-4 |
| Classifier Head | 1e-3 |

Additional features:
- Unfreezing audio backbone  
- BatchNorm frozen for small-batch stability  

---

## 📊 Results & Assumptions

### Evaluation Strategy
This project evaluates performance at the **Patient Level**, not the Slice Level.

1.  **Slice Prediction**: The model predicts probabilities for every 6-second slice of a patient's recording.
2.  **Aggregation (Soft Voting)**:
    * All slice probability vectors for a single patient are averaged.
    * The class with the highest average probability is the final predicted label.

5-fold cross-validation reveals:

- Unimodal baseline achieves slightly higher peak accuracy  
- But collapses on hard folds  
- Contrastive model remains stable and robust  

| Model | Mean Accuracy | Fold 1 (Hard Split) | Stability (SD) |
|-------|--------------|---------------------|----------------|
| Unimodal Baseline | 71.32% | 56.00% | ± 7.94 |
| Contrastive Fine-Tuned | 69.78% | 64.00% | ± 6.57 |

**Conclusion:** Multimodal alignment prevents collapse on outlier data, offering a more reliable real-world deployment strategy.

---

# 🚀 How to Run

## Step 0: Install Dependencies
```bash
pip install torch torchaudio transformers pandas numpy scikit-learn soundfile librosa tqdm seaborn matplotlib
```

## Step 1: ETL & Preprocessing
```bash
python preprocessing.py
```
**Output:** `data/full_dataset_with_folds.csv`

---

## Step 2: Establish Baseline (Audio-Only)
```bash
python train_baseline_audio.py
```
**Output:** `baseline_audio_fold_X.pth`

---

## Step 3: Contrastive Pre-training
```bash
python train_contrastive.py
```
**Output:** `audio_encoder_fold_X.pth`

---

## Step 4: Fine-Tuning Experiment
```bash
python fine_tuning.py
```
**Output:** `finetuned_model_fold_X.pth`
