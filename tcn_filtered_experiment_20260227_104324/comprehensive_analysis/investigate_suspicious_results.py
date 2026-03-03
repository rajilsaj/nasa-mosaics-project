#!/usr/bin/env python3
"""
Investigate Suspicious Training Results
======================================

This script investigates why the extended model shows suspicious results:
- Perfect training performance (ROC AUC = 1.0)
- All positive samples get exactly the same probability (99.74%)
- Dramatic drop from training to validation

Checks for:
1. Data leakage (temporal, feature, label leakage)
2. Overfitting indicators
3. Training data quality issues
4. Feature importance analysis
5. Model behavior analysis
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# =============================================================================
# LOAD DATA AND MODEL
# =============================================================================

def load_data_and_model():
    """Load training data and extended model."""
    print("=" * 70)
    print("LOADING DATA AND MODEL")
    print("=" * 70)
    
    # Load model
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    if not autoencoder_files:
        print("[ERROR] Extended model not found!")
        return None, None, None, None
    
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    print(f"Loaded model: {os.path.basename(max(autoencoder_files, key=os.path.getctime))}")
    
    # Load training data
    train_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
    if not os.path.exists(train_file):
        print(f"[ERROR] Training file not found: {train_file}")
        return None, None, None, None
    
    train_df = pd.read_csv(train_file)
    print(f"Loaded {len(train_df):,} training samples")
    
    # Load validation data
    val_file = os.path.join(FEATURES_DIR, "val_sliding_features_step10.csv")
    val_df = pd.read_csv(val_file) if os.path.exists(val_file) else None
    if val_df is not None:
        print(f"Loaded {len(val_df):,} validation samples")
    
    # Get features
    label_cols = ['label']
    feature_cols = [col for col in train_df.columns if col not in label_cols]
    
    # Load model metadata
    autoencoder_metadata_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_metadata_*.json"))
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = feature_cols
    
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    X_train = train_df[autoencoder_feature_cols].values
    y_train = train_df['label'].values
    
    if val_df is not None:
        val_label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                          'sliding_start_sclk', 'sliding_end_sclk']
        val_feature_cols = [col for col in val_df.columns if col not in val_label_cols]
        val_autoencoder_feature_cols = [f for f in autoencoder_features if f in val_feature_cols]
        X_val = val_df[val_autoencoder_feature_cols].values
        y_val = val_df['label'].values
    else:
        X_val, y_val = None, None
    
    return autoencoder_model, train_df, X_train, y_train, X_val, y_val, autoencoder_feature_cols

# =============================================================================
# CHECK 1: DATA LEAKAGE
# =============================================================================

def check_data_leakage(train_df, feature_cols):
    """Check for data leakage in features."""
    print("\n" + "=" * 70)
    print("CHECK 1: DATA LEAKAGE INVESTIGATION")
    print("=" * 70)
    
    issues = []
    
    # Check for label leakage in feature names
    print("\n1. Checking for label leakage in feature names...")
    suspicious_features = []
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['label', 'target', 'gt_', 'ground_truth', 'vortex', 'event']):
            suspicious_features.append(col)
            issues.append(f"Feature '{col}' may contain label information")
    
    if suspicious_features:
        print(f"  [WARNING] Found {len(suspicious_features)} suspicious features:")
        for feat in suspicious_features:
            print(f"    - {feat}")
    else:
        print("  [OK] No obvious label leakage in feature names")
    
    # Check for perfect correlation with labels
    print("\n2. Checking for perfect feature-label correlation...")
    perfect_correlations = []
    for col in feature_cols:
        if col in train_df.columns:
            corr = abs(train_df[col].corr(train_df['label']))
            if corr > 0.99:
                perfect_correlations.append((col, corr))
                issues.append(f"Feature '{col}' has perfect correlation ({corr:.4f}) with label")
    
    if perfect_correlations:
        print(f"  [WARNING] Found {len(perfect_correlations)} features with near-perfect correlation:")
        for feat, corr in perfect_correlations:
            print(f"    - {feat}: {corr:.4f}")
    else:
        print("  [OK] No perfect correlations found")
    
    # Check for constant features
    print("\n3. Checking for constant or near-constant features...")
    constant_features = []
    for col in feature_cols:
        if col in train_df.columns:
            unique_vals = train_df[col].nunique()
            if unique_vals <= 1:
                constant_features.append(col)
                issues.append(f"Feature '{col}' is constant (only {unique_vals} unique values)")
    
    if constant_features:
        print(f"  [WARNING] Found {len(constant_features)} constant features:")
        for feat in constant_features:
            print(f"    - {feat}")
    else:
        print("  [OK] No constant features found")
    
    # Check for duplicate features
    print("\n4. Checking for duplicate features...")
    feature_matrix = train_df[feature_cols].values
    duplicate_pairs = []
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols[i+1:], start=i+1):
            if np.allclose(feature_matrix[:, i], feature_matrix[:, j], rtol=1e-10):
                duplicate_pairs.append((col1, col2))
                issues.append(f"Features '{col1}' and '{col2}' are identical")
    
    if duplicate_pairs:
        print(f"  [WARNING] Found {len(duplicate_pairs)} duplicate feature pairs:")
        for feat1, feat2 in duplicate_pairs:
            print(f"    - {feat1} == {feat2}")
    else:
        print("  [OK] No duplicate features found")
    
    return issues

# =============================================================================
# CHECK 2: TRAINING DATA QUALITY
# =============================================================================

def check_training_data_quality(train_df, X_train, y_train):
    """Check training data quality."""
    print("\n" + "=" * 70)
    print("CHECK 2: TRAINING DATA QUALITY")
    print("=" * 70)
    
    issues = []
    
    # Check sample size
    print(f"\n1. Sample size: {len(train_df):,} samples")
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    print(f"   Positive: {pos_count} ({pos_count/len(train_df)*100:.1f}%)")
    print(f"   Negative: {neg_count} ({neg_count/len(train_df)*100:.1f}%)")
    
    if len(train_df) < 500:
        issues.append(f"Small training set ({len(train_df)} samples) may lead to overfitting")
        print(f"  [WARNING] Small training set may lead to overfitting")
    
    # Check for duplicate samples
    print("\n2. Checking for duplicate samples...")
    feature_matrix = X_train
    unique_samples = len(np.unique(feature_matrix, axis=0))
    duplicate_count = len(train_df) - unique_samples
    
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate samples in training data")
        print(f"  [WARNING] Found {duplicate_count} duplicate samples ({duplicate_count/len(train_df)*100:.1f}%)")
    else:
        print("  [OK] No duplicate samples found")
    
    # Check feature variance
    print("\n3. Checking feature variance...")
    low_variance_features = []
    for i in range(feature_matrix.shape[1]):
        var = np.var(feature_matrix[:, i])
        if var < 1e-10:
            low_variance_features.append(i)
    
    if low_variance_features:
        issues.append(f"Found {len(low_variance_features)} features with near-zero variance")
        print(f"  [WARNING] Found {len(low_variance_features)} features with near-zero variance")
    else:
        print("  [OK] All features have sufficient variance")
    
    # Check class separation in feature space
    print("\n4. Checking class separation in feature space...")
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        pos_mean = feature_matrix[pos_indices].mean(axis=0)
        neg_mean = feature_matrix[neg_indices].mean(axis=0)
        separation = np.abs(pos_mean - neg_mean)
        
        max_separation_idx = np.argmax(separation)
        max_separation = separation[max_separation_idx]
        
        print(f"   Maximum class separation: {max_separation:.4f}")
        print(f"   Feature with max separation: Feature {max_separation_idx}")
        
        if max_separation > 100:  # Arbitrary threshold
            issues.append(f"Extremely high class separation ({max_separation:.2f}) may indicate leakage")
            print(f"  [WARNING] Extremely high class separation may indicate data leakage")
    
    return issues

# =============================================================================
# CHECK 3: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(model, feature_cols, X_train, y_train):
    """Analyze feature importance."""
    print("\n" + "=" * 70)
    print("CHECK 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    issues = []
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Check for dominant features
        top_feature_importance = feature_importance_df.iloc[0]['importance']
        if top_feature_importance > 0.5:
            issues.append(f"Single feature dominates ({top_feature_importance:.2%}) - may indicate leakage")
            print(f"\n  [WARNING] Top feature has {top_feature_importance:.2%} importance (suspicious!)")
        
        # Check for zero importance features
        zero_importance = (importances == 0).sum()
        if zero_importance > 0:
            print(f"\n  [INFO] {zero_importance} features have zero importance")
    
    return issues, feature_importance_df if 'feature_importance_df' in locals() else None

# =============================================================================
# CHECK 4: MODEL BEHAVIOR ANALYSIS
# =============================================================================

def analyze_model_behavior(model, X_train, y_train, X_val, y_val):
    """Analyze model behavior in detail."""
    print("\n" + "=" * 70)
    print("CHECK 4: MODEL BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    issues = []
    
    # Get training probabilities
    train_proba = model.predict_proba(X_train)[:, 1]
    
    # Check probability distribution for positive samples
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    
    if len(pos_indices) > 0:
        pos_proba = train_proba[pos_indices]
        unique_pos_proba = np.unique(pos_proba)
        
        print(f"\n1. Positive sample probabilities:")
        print(f"   Count: {len(pos_indices)}")
        print(f"   Unique values: {len(unique_pos_proba)}")
        print(f"   Min: {pos_proba.min():.6f}")
        print(f"   Max: {pos_proba.max():.6f}")
        print(f"   Mean: {pos_proba.mean():.6f}")
        print(f"   Std: {pos_proba.std():.6f}")
        
        if len(unique_pos_proba) <= 3:
            issues.append(f"All positive samples have only {len(unique_pos_proba)} unique probability values (suspicious!)")
            print(f"  [WARNING] Only {len(unique_pos_proba)} unique probability values for all positive samples!")
            print(f"            Values: {unique_pos_proba}")
        
        # Check if all positive samples are in same leaf
        if hasattr(model, 'estimators_'):
            print(f"\n2. Checking tree structure...")
            # Sample a few positive samples and check which leaves they fall into
            sample_pos_indices = pos_indices[:min(5, len(pos_indices))]
            for tree in model.estimators_[:3]:  # Check first 3 trees
                leaves = []
                for idx in sample_pos_indices:
                    leaf = tree.apply(X_train[idx:idx+1])[0]
                    leaves.append(leaf)
                if len(set(leaves)) == 1:
                    print(f"  [WARNING] All sampled positive samples fall into same leaf in tree (overfitting!)")
                    issues.append("Positive samples fall into same leaves in trees (overfitting)")
    
    # Check training vs validation performance
    if X_val is not None and y_val is not None:
        val_proba = model.predict_proba(X_val)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        
        print(f"\n3. Training vs Validation Performance:")
        print(f"   Training ROC AUC: {train_auc:.4f}")
        print(f"   Validation ROC AUC: {val_auc:.4f}")
        print(f"   Drop: {train_auc - val_auc:.4f}")
        
        if train_auc - val_auc > 0.2:
            issues.append(f"Large performance drop ({train_auc - val_auc:.2f}) indicates overfitting")
            print(f"  [WARNING] Large performance drop indicates overfitting")
    
    return issues

# =============================================================================
# CHECK 5: AUTOENCODER FEATURES INVESTIGATION
# =============================================================================

def investigate_autoencoder_features(train_df, feature_cols):
    """Investigate autoencoder features specifically."""
    print("\n" + "=" * 70)
    print("CHECK 5: AUTOENCODER FEATURES INVESTIGATION")
    print("=" * 70)
    
    issues = []
    
    # Find autoencoder features
    ae_features = [f for f in feature_cols if 'autoencoder' in f.lower()]
    
    if ae_features:
        print(f"\nFound {len(ae_features)} autoencoder features:")
        for feat in ae_features:
            print(f"  - {feat}")
        
        # Check correlation with labels
        print(f"\nCorrelation with labels:")
        for feat in ae_features:
            if feat in train_df.columns:
                corr = train_df[feat].corr(train_df['label'])
                print(f"  {feat}: {corr:.4f}")
                if abs(corr) > 0.9:
                    issues.append(f"Autoencoder feature '{feat}' has very high correlation ({corr:.4f}) with label")
        
        # Check value distribution
        print(f"\nValue distribution:")
        for feat in ae_features:
            if feat in train_df.columns:
                pos_vals = train_df[train_df['label'] == 1][feat]
                neg_vals = train_df[train_df['label'] == 0][feat]
                print(f"  {feat}:")
                print(f"    Positive mean: {pos_vals.mean():.4f}, std: {pos_vals.std():.4f}")
                print(f"    Negative mean: {neg_vals.mean():.4f}, std: {neg_vals.std():.4f}")
                print(f"    Separation: {abs(pos_vals.mean() - neg_vals.mean()):.4f}")
    else:
        print("\nNo autoencoder features found in feature list")
    
    return issues

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main investigation pipeline."""
    print("=" * 70)
    print("INVESTIGATION: SUSPICIOUS TRAINING RESULTS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    result = load_data_and_model()
    if result[0] is None:
        return 1
    
    model, train_df, X_train, y_train, X_val, y_val, feature_cols = result
    
    all_issues = []
    
    # Run all checks
    issues1 = check_data_leakage(train_df, feature_cols)
    all_issues.extend(issues1)
    
    issues2 = check_training_data_quality(train_df, X_train, y_train)
    all_issues.extend(issues2)
    
    issues3, feature_importance_df = analyze_feature_importance(model, feature_cols, X_train, y_train)
    all_issues.extend(issues3)
    
    issues4 = analyze_model_behavior(model, X_train, y_train, X_val, y_val)
    all_issues.extend(issues4)
    
    issues5 = investigate_autoencoder_features(train_df, feature_cols)
    all_issues.extend(issues5)
    
    # Summary
    print("\n" + "=" * 70)
    print("INVESTIGATION SUMMARY")
    print("=" * 70)
    
    if all_issues:
        print(f"\n[WARNING] Found {len(all_issues)} potential issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n[OK] No obvious issues found")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'issues_found': len(all_issues),
        'issues': all_issues,
        'training_samples': len(train_df),
        'positive_samples': int((y_train == 1).sum()),
        'negative_samples': int((y_train == 0).sum()),
        'num_features': len(feature_cols)
    }
    
    if feature_importance_df is not None:
        results['top_features'] = feature_importance_df.head(10).to_dict('records')
    
    results_file = os.path.join(RESULTS_DIR, f"investigation_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to: {results_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())




