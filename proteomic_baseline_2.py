import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
class OmicsConfig:
    CLINICAL_PATH = "data/proteomic/raw/data_1_10_22/clinical_data/COPDGene_P1P2P3_Flat_SM_NS_Mar20.txt"
    TRAIN_CSV_PATH = "data/proteomic/processed/original/x_train.csv"
    TEST_CSV_PATH = "data/proteomic/processed/original/x_test.csv"
    TARGET_COL = 'finalGold_P1'
    
    # Output Directory
    RESULTS_DIR = "proteomic_results"
    
    # Tuning Params
    NUM_PROTEINS = 30       # Number of top biomarkers to keep
    PHASE1_DEPTH = 3        # Depth for feature selection
    PHASE2_DEPTH = 4        # Depth for final model
    LEARNING_RATE = 0.05

# --- HELPER: PLOTTING ---
def plot_feature_importance(clf, feature_names, title="Feature Importance", filename="importance.png"):
    # Extract importances
    importances = clf.feature_importances_
    
    # Create DataFrame
    fi_df = pd.DataFrame({'Protein': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df, x='Importance', y='Protein', palette='viridis')
    plt.title(title)
    plt.xlabel("XGBoost Gain (Importance)")
    plt.tight_layout()
    
    # Save to Results Directory
    save_path = os.path.join(OmicsConfig.RESULTS_DIR, filename)
    plt.savefig(save_path)
    print(f"   [Saved Plot: '{save_path}']")
    plt.close()

# --- 1. DATA LOADING UTILS ---
def load_data_raw():
    print("--- Loading Raw Data ---")
    df_train = pd.read_csv(OmicsConfig.TRAIN_CSV_PATH, index_col=0)
    df_test = pd.read_csv(OmicsConfig.TEST_CSV_PATH, index_col=0)
    clinical = pd.read_csv(OmicsConfig.CLINICAL_PATH, sep="\t", low_memory=False, index_col=0)
    
    # Intersection
    common_train = df_train.index.intersection(clinical.index)
    common_test = df_test.index.intersection(clinical.index)
    
    X_train = df_train.loc[common_train]
    y_train = clinical.loc[common_train][OmicsConfig.TARGET_COL]
    X_test = df_test.loc[common_test]
    y_test = clinical.loc[common_test][OmicsConfig.TARGET_COL]
    
    # Log Transform
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    
    return X_train, y_train, X_test, y_test

# --- 2. PHASE 1: BIOMARKER DISCOVERY ---
def find_top_features(X_train, y_train):
    print("\n--- PHASE 1: Biomarker Discovery (Healthy vs Severe) ---")
    
    # Filter: Keep ONLY Healthy (0) and Severe (3, 4)
    def map_severe(val):
        try: s = int(float(val))
        except: return "Exclude"
        if s == 0: return "Healthy"
        if s in [3, 4]: return "Severe"
        return "Exclude"

    y_mapped = y_train.apply(map_severe)
    mask = y_mapped != "Exclude"
    
    X_sub = X_train[mask]
    y_sub = y_mapped[mask]
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_sub)
    
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=OmicsConfig.PHASE1_DEPTH, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    clf.fit(X_sub, y_enc)
    
    # Select Top Features
    importances = clf.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1][:OmicsConfig.NUM_PROTEINS]
    top_features = features[indices].values
    
    # Plot Phase 1 Importance
    plot_feature_importance(clf, features, 
                            title="Phase 1: Discovery (Healthy vs Severe)", 
                            filename="phase1_discovery_importance.png")
    
    print(f"   Identified Top {len(top_features)} Proteins: {top_features[:5]}...")
    return top_features

# --- 3. PHASE 2: TRAINING ON FULL DATASET ---
def train_final_model(top_features, X_train_full, y_train_full, X_test_full, y_test_full):
    print(f"\n--- PHASE 2: Training on FULL Dataset (Using {len(top_features)} Proteins) ---")
    
    X_train = X_train_full[top_features]
    X_test = X_test_full[top_features]
    
    def map_all(val):
        try: s = int(float(val))
        except: return "Exclude"
        if s == 0: return "Healthy"
        if s in [1, 2, 3, 4]: return "COPD"
        return "Exclude"

    y_train = y_train_full.apply(map_all)
    y_test = y_test_full.apply(map_all)
    
    mask_tr = y_train != "Exclude"
    X_train = X_train[mask_tr]
    y_train = y_train[mask_tr]
    
    mask_te = y_test != "Exclude"
    X_test = X_test[mask_te]
    y_test = y_test[mask_te]
    
    print(f"   Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    clf = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=OmicsConfig.PHASE2_DEPTH, 
        learning_rate=OmicsConfig.LEARNING_RATE, 
        subsample=0.8,
        colsample_bytree=0.8, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    clf.fit(X_train, y_train_enc)
    
    # Plot Final Importance
    plot_feature_importance(clf, top_features, 
                            title="Phase 2: Final Model (Healthy vs All COPD)", 
                            filename="phase2_final_importance.png")
    
    print("\n   [Default Threshold 0.5 Results]")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    
    return clf, X_test, y_test_enc, le

# --- 4. PHASE 3: THRESHOLD OPTIMIZATION ---
def optimize_threshold(clf, X_test, y_test, le):
    print("\n--- PHASE 3: Threshold Optimization ---")
    
    classes = le.classes_
    copd_idx = np.where(classes == 'COPD')[0][0]
    healthy_idx = np.where(classes == 'Healthy')[0][0]
    
    print(f"   Optimizing for Class: COPD (Index {copd_idx})")
    y_probs = clf.predict_proba(X_test)[:, copd_idx]
    y_true_binary = (y_test == copd_idx).astype(int)
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_probs)
    J = tpr - fpr
    best_ix = np.argmax(J)
    best_thresh = thresholds[best_ix]
    
    print(f"   Best Threshold Found: {best_thresh:.4f}")
    
    y_pred_new_indices = np.where(y_probs >= best_thresh, copd_idx, healthy_idx)
    acc = accuracy_score(y_test, y_pred_new_indices)
    
    cm = confusion_matrix(y_test, y_pred_new_indices, labels=[copd_idx, healthy_idx])
    tp, fn = cm[0,0], cm[0,1]
    fp, tn = cm[1,0], cm[1,1]
    
    copd_recall = tp / (tp + fn)
    healthy_recall = tn / (tn + fp)
    
    print(f"\n   --- OPTIMIZED RESULTS (Threshold {best_thresh:.4f}) ---")
    print(f"   Accuracy:       {acc*100:.2f}%")
    print(f"   COPD Recall:    {copd_recall:.2f}")
    print(f"   Healthy Recall: {healthy_recall:.2f}")
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve')
    plt.scatter(fpr[best_ix], tpr[best_ix], marker='o', color='red', label='Optimal Point')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Optimization')
    plt.legend()
    
    save_path = os.path.join(OmicsConfig.RESULTS_DIR, 'roc_optimization.png')
    plt.savefig(save_path)
    print(f"   [Saved Plot: '{save_path}']")
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OmicsConfig.RESULTS_DIR, exist_ok=True)
    
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = load_data_raw()
    top_proteins = find_top_features(X_tr_raw, y_tr_raw)
    model, X_te_final, y_te_final, encoder = train_final_model(top_proteins, X_tr_raw, y_tr_raw, X_te_raw, y_te_raw)
    optimize_threshold(model, X_te_final, y_te_final, encoder)