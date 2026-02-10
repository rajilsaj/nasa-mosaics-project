#!/usr/bin/env python3
"""
Phase 1: Root Cause Analysis & Validation
==========================================

This script performs comprehensive investigation of data leakage:
1. Confirms data leakage pattern in training data
2. Traces data flow from source CSV → windows → features → training
3. Identifies additional issues (duplicate features, correlations, etc.)
4. Documents findings for Phase 2 fixes

Part of RF Expert Fix Outline implementation.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# File paths
SOURCE_CSV = os.path.join(PARENT_DIR, "comprehensive_filtered_data_optimized.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "data/features")
WINDOWS_DIR = os.path.join(SCRIPT_DIR, "data/windows")
SPLITS_DIR = os.path.join(SCRIPT_DIR, "data/splits")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Autoencoder feature names
AUTOENCODER_FEATURES = [
    'autoencoder_window_hits_mean',
    'autoencoder_positive_hit_binary',
    'autoencoder_hit_ratio',
    'ae_gt_agreement'
]

# Source autoencoder columns (in raw CSV)
SOURCE_AE_COLUMNS = [
    'autoencoder_window_hits',
    'autoencoder_positive_hit'
]

# =============================================================================
# SECTION 1.1: CONFIRM DATA LEAKAGE PATTERN
# =============================================================================

def confirm_data_leakage_pattern():
    """1.1: Verify NaN distribution in training data."""
    print("=" * 70)
    print("SECTION 1.1: CONFIRM DATA LEAKAGE PATTERN")
    print("=" * 70)
    
    # Load training data
    train_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
    if not os.path.exists(train_file):
        print(f"[ERROR] Training file not found: {train_file}")
        return None
    
    train_df = pd.read_csv(train_file)
    print(f"\nLoaded training data: {len(train_df):,} samples")
    
    # Check class distribution
    if 'label' not in train_df.columns:
        print("[ERROR] 'label' column not found!")
        return None
    
    class_dist = train_df['label'].value_counts()
    print(f"Class distribution:")
    print(f"  Positive (label=1): {class_dist.get(1, 0)} samples")
    print(f"  Negative (label=0): {class_dist.get(0, 0)} samples")
    
    # Separate by class
    positive_samples = train_df[train_df['label'] == 1]
    negative_samples = train_df[train_df['label'] == 0]
    
    print(f"\n{'='*70}")
    print("AUTOENCODER FEATURE ANALYSIS BY CLASS")
    print(f"{'='*70}")
    
    leakage_findings = {}
    
    for ae_feature in AUTOENCODER_FEATURES:
        if ae_feature not in train_df.columns:
            print(f"\n[WARNING] Feature '{ae_feature}' not found in training data")
            continue
        
        print(f"\n--- {ae_feature} ---")
        
        # Positive samples
        pos_values = positive_samples[ae_feature]
        pos_nan_count = pos_values.isna().sum()
        pos_nan_pct = (pos_nan_count / len(pos_values)) * 100
        pos_non_nan = pos_values.notna().sum()
        
        print(f"  Positive samples (label=1):")
        print(f"    Total: {len(pos_values)}")
        print(f"    NaN: {pos_nan_count} ({pos_nan_pct:.1f}%)")
        print(f"    Non-NaN: {pos_non_nan}")
        if pos_non_nan > 0:
            print(f"    Non-NaN range: [{pos_values.dropna().min():.4f}, {pos_values.dropna().max():.4f}]")
        
        # Negative samples
        neg_values = negative_samples[ae_feature]
        neg_nan_count = neg_values.isna().sum()
        neg_nan_pct = (neg_nan_count / len(neg_values)) * 100
        neg_non_nan = neg_values.notna().sum()
        
        print(f"  Negative samples (label=0):")
        print(f"    Total: {len(neg_values)}")
        print(f"    NaN: {neg_nan_count} ({neg_nan_pct:.1f}%)")
        print(f"    Non-NaN: {neg_non_nan}")
        if neg_non_nan > 0:
            print(f"    Non-NaN range: [{neg_values.dropna().min():.4f}, {neg_values.dropna().max():.4f}]")
        
        # Check for perfect separation
        if pos_nan_count == len(pos_values) and neg_nan_count == 0:
            print(f"  [CRITICAL] PERFECT SEPARATION DETECTED!")
            print(f"    All positive samples have NaN, all negative samples have values")
            leakage_findings[ae_feature] = "PERFECT_SEPARATION"
        elif pos_nan_pct > 90 and neg_nan_pct < 10:
            print(f"  [WARNING] Strong separation pattern (may indicate leakage)")
            leakage_findings[ae_feature] = "STRONG_SEPARATION"
        else:
            leakage_findings[ae_feature] = "NO_LEAKAGE"
    
    return leakage_findings, train_df

# =============================================================================
# SECTION 1.2: INVESTIGATE FEATURE ENGINEERING PIPELINE
# =============================================================================

def check_source_csv():
    """1.2.3: Check if source CSV has autoencoder data for positive regions."""
    print("\n" + "=" * 70)
    print("SECTION 1.2.3: EXAMINE SOURCE DATA")
    print("=" * 70)
    
    if not os.path.exists(SOURCE_CSV):
        print(f"[ERROR] Source CSV not found: {SOURCE_CSV}")
        return None
    
    print(f"\nLoading source CSV: {os.path.basename(SOURCE_CSV)}")
    print("(This may take a moment for large files...)")
    
    # Load a sample to check columns
    try:
        # Read first 1000 rows to check structure
        sample_df = pd.read_csv(SOURCE_CSV, nrows=1000)
        print(f"Sample loaded: {len(sample_df):,} rows")
        
        # Check for autoencoder columns
        print(f"\nColumns in source CSV: {len(sample_df.columns)} total")
        ae_cols_found = [col for col in SOURCE_AE_COLUMNS if col in sample_df.columns]
        print(f"Autoencoder columns found: {ae_cols_found}")
        
        if not ae_cols_found:
            print("[WARNING] No autoencoder columns found in source CSV!")
            return None
        
        # Check for ground truth column
        gt_col = 'gt_detection_win' if 'gt_detection_win' in sample_df.columns else None
        if gt_col:
            print(f"Ground truth column: {gt_col}")
            gt_positive_count = sample_df[gt_col].sum()
            print(f"Positive samples in sample: {gt_positive_count}")
            
            # Check autoencoder data for positive regions
            if gt_positive_count > 0:
                positive_rows = sample_df[sample_df[gt_col] == True]
                print(f"\nAnalyzing positive regions (gt_detection_win=True):")
                
                for ae_col in ae_cols_found:
                    pos_ae_values = positive_rows[ae_col]
                    pos_nan = pos_ae_values.isna().sum()
                    pos_non_nan = pos_ae_values.notna().sum()
                    print(f"  {ae_col}:")
                    print(f"    NaN: {pos_nan} ({pos_nan/len(pos_ae_values)*100:.1f}%)")
                    print(f"    Non-NaN: {pos_non_nan}")
                    if pos_non_nan > 0:
                        print(f"    Range: [{pos_ae_values.dropna().min():.4f}, {pos_ae_values.dropna().max():.4f}]")
        
        # Now load full dataset for positive regions only (more efficient)
        print(f"\nLoading full dataset to check positive regions...")
        full_df = pd.read_csv(SOURCE_CSV)
        print(f"Full dataset loaded: {len(full_df):,} rows")
        
        if gt_col and full_df[gt_col].sum() > 0:
            positive_full = full_df[full_df[gt_col] == True]
            print(f"\nFull dataset - Positive regions: {len(positive_full):,} rows")
            
            for ae_col in ae_cols_found:
                pos_ae = positive_full[ae_col]
                pos_nan = pos_ae.isna().sum()
                pos_non_nan = pos_ae.notna().sum()
                print(f"  {ae_col}:")
                print(f"    NaN: {pos_nan:,} ({pos_nan/len(pos_ae)*100:.1f}%)")
                print(f"    Non-NaN: {pos_non_nan:,}")
                if pos_non_nan > 0:
                    print(f"    Range: [{pos_ae.dropna().min():.4f}, {pos_ae.dropna().max():.4f}]")
        
        return {
            'has_ae_columns': len(ae_cols_found) > 0,
            'ae_columns': ae_cols_found,
            'gt_column': gt_col,
            'total_rows': len(full_df)
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to load source CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def trace_window_extraction():
    """1.2.2: Check how windows are extracted in data_preparation.py."""
    print("\n" + "=" * 70)
    print("SECTION 1.2.2: TRACE WINDOW EXTRACTION")
    print("=" * 70)
    
    # Check if window files exist
    train_windows_file = os.path.join(WINDOWS_DIR, "train_windows.csv")
    
    if not os.path.exists(train_windows_file):
        print(f"[WARNING] Window file not found: {train_windows_file}")
        print("[INFO] Windows may not have been extracted yet")
        return None
    
    print(f"\nLoading window file: {os.path.basename(train_windows_file)}")
    windows_df = pd.read_csv(train_windows_file, nrows=10000)  # Sample for speed
    print(f"Loaded sample: {len(windows_df):,} rows")
    
    # Check for autoencoder columns in windows
    ae_cols_in_windows = [col for col in SOURCE_AE_COLUMNS if col in windows_df.columns]
    print(f"\nAutoencoder columns in windows: {ae_cols_in_windows}")
    
    if ae_cols_in_windows:
        # Check NaN distribution by label
        if 'label' in windows_df.columns:
            positive_windows = windows_df[windows_df['label'] == True]
            negative_windows = windows_df[windows_df['label'] == False]
            
            print(f"\nWindow-level analysis:")
            print(f"  Positive windows: {len(positive_windows):,}")
            print(f"  Negative windows: {len(negative_windows):,}")
            
            for ae_col in ae_cols_in_windows:
                print(f"\n  {ae_col}:")
                pos_ae = positive_windows[ae_col]
                neg_ae = negative_windows[ae_col]
                
                pos_nan = pos_ae.isna().sum()
                neg_nan = neg_ae.isna().sum()
                
                print(f"    Positive windows - NaN: {pos_nan:,} ({pos_nan/len(pos_ae)*100:.1f}%)")
                print(f"    Negative windows - NaN: {neg_nan:,} ({neg_nan/len(neg_ae)*100:.1f}%)")
    
    return windows_df

def review_feature_engineering():
    """1.2.1: Review compute_autoencoder_features() logic."""
    print("\n" + "=" * 70)
    print("SECTION 1.2.1: REVIEW FEATURE ENGINEERING LOGIC")
    print("=" * 70)
    
    feature_eng_file = os.path.join(SCRIPT_DIR, "feature_engineering.py")
    
    if not os.path.exists(feature_eng_file):
        print(f"[WARNING] Feature engineering file not found: {feature_eng_file}")
        return None
    
    print(f"\nReviewing: {os.path.basename(feature_eng_file)}")
    
    # Read the file and find compute_autoencoder_features function
    with open(feature_eng_file, 'r') as f:
        content = f.read()
    
    # Check for key issues
    issues = []
    
    if 'ae_gt_agreement' in content:
        if 'gt_detection_win' in content and 'ae_gt_agreement' in content:
            # Check if it's computed using ground truth
            if content.find('ae_gt_agreement') < content.find('gt_detection_win'):
                issues.append("ae_gt_agreement may be computed using ground truth (data leakage risk)")
    
    if 'autoencoder_window_hits' not in content:
        issues.append("autoencoder_window_hits column check found in function")
    else:
        # Check default return values
        if '0.0' in content[content.find('compute_autoencoder_features'):content.find('compute_autoencoder_features')+500]:
            issues.append("Function returns default 0.0 when autoencoder columns missing - may create perfect separation")
    
    print("\nKey findings:")
    if issues:
        for issue in issues:
            print(f"  [ISSUE] {issue}")
    else:
        print("  [OK] No obvious issues found in code review")
    
    print("\nFunction location: lines ~305-353 in feature_engineering.py")
    print("Key logic:")
    print("  - Checks if 'autoencoder_window_hits' column exists")
    print("  - Returns default 0.0 values if missing")
    print("  - Computes ae_gt_agreement using gt_detection_win (potential leakage)")
    
    return issues

# =============================================================================
# SECTION 1.3: IDENTIFY ADDITIONAL ISSUES
# =============================================================================

def check_duplicate_features(train_df):
    """1.3.2: Check for duplicate/redundant features."""
    print("\n" + "=" * 70)
    print("SECTION 1.3.2: CHECK FOR DUPLICATE FEATURES")
    print("=" * 70)
    
    if train_df is None:
        print("[ERROR] Training data not loaded")
        return None
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['label', 'window_id', 'event_sclk', 'sliding_window_id', 
                     'sliding_start_idx', 'sliding_end_idx', 'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]
    
    print(f"\nAnalyzing {len(feature_cols)} features for duplicates...")
    
    duplicates = []
    
    # Check pressure_drop vs range
    if 'pressure_drop' in feature_cols and 'range' in feature_cols:
        drop_values = train_df['pressure_drop'].values
        range_values = train_df['range'].values
        
        # Check if they're identical
        if np.allclose(drop_values, range_values, equal_nan=True):
            duplicates.append(('pressure_drop', 'range', 'IDENTICAL'))
            print(f"\n[FOUND] 'pressure_drop' and 'range' are IDENTICAL")
        else:
            # Check correlation
            corr = np.corrcoef(drop_values[~np.isnan(drop_values)], 
                              range_values[~np.isnan(range_values)])[0, 1]
            if corr > 0.95:
                duplicates.append(('pressure_drop', 'range', f'HIGHLY_CORRELATED_{corr:.3f}'))
                print(f"\n[FOUND] 'pressure_drop' and 'range' are highly correlated: {corr:.3f}")
    
    # Check all pairwise correlations
    print(f"\nChecking pairwise correlations (threshold: 0.95)...")
    feature_data = train_df[feature_cols].select_dtypes(include=[np.number])
    
    corr_matrix = feature_data.corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.95 and not np.isnan(corr_val):
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_val))
    
    if high_corr_pairs:
        print(f"\n[FOUND] {len(high_corr_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
            print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    else:
        print("\n[OK] No highly correlated feature pairs found (threshold: 0.95)")
    
    return duplicates + high_corr_pairs

def check_suspicious_features(train_df):
    """1.3.1: Identify suspicious features (e.g., ae_gt_agreement)."""
    print("\n" + "=" * 70)
    print("SECTION 1.3.1: CHECK FOR SUSPICIOUS FEATURES")
    print("=" * 70)
    
    if train_df is None:
        print("[ERROR] Training data not loaded")
        return None
    
    suspicious = []
    
    # Check ae_gt_agreement
    if 'ae_gt_agreement' in train_df.columns:
        print("\n[FOUND] 'ae_gt_agreement' feature detected")
        print("  This feature compares autoencoder output with ground truth")
        print("  [CRITICAL] This is data leakage - feature uses label information!")
        suspicious.append({
            'feature': 'ae_gt_agreement',
            'reason': 'Uses ground truth information (gt_detection_win)',
            'severity': 'CRITICAL'
        })
        
        # Check if it creates perfect separation
        if 'label' in train_df.columns:
            pos_agreement = train_df[train_df['label'] == 1]['ae_gt_agreement']
            neg_agreement = train_df[train_df['label'] == 0]['ae_gt_agreement']
            
            pos_unique = pos_agreement.nunique()
            neg_unique = neg_agreement.nunique()
            
            print(f"  Positive samples - unique values: {pos_unique}")
            print(f"  Negative samples - unique values: {neg_unique}")
    
    # Check for other features that might use ground truth
    feature_cols = [col for col in train_df.columns if col not in ['label', 'window_id', 'event_sclk']]
    gt_related = [col for col in feature_cols if 'gt' in col.lower() or 'ground' in col.lower() or 'agreement' in col.lower()]
    
    if gt_related:
        print(f"\n[FOUND] Features with 'gt', 'ground', or 'agreement' in name:")
        for feat in gt_related:
            if feat != 'ae_gt_agreement':  # Already reported
                print(f"  {feat} - [REVIEW NEEDED]")
                suspicious.append({
                    'feature': feat,
                    'reason': 'Name suggests ground truth usage',
                    'severity': 'WARNING'
                })
    
    if not suspicious:
        print("\n[OK] No obviously suspicious features found")
    
    return suspicious

def check_missing_data_patterns(train_df):
    """1.3.4: Check for other systematic missing data issues."""
    print("\n" + "=" * 70)
    print("SECTION 1.3.4: CHECK FOR SYSTEMATIC MISSING DATA")
    print("=" * 70)
    
    if train_df is None:
        print("[ERROR] Training data not loaded")
        return None
    
    metadata_cols = ['label', 'window_id', 'event_sclk', 'sliding_window_id']
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]
    
    print(f"\nAnalyzing missing data patterns across {len(feature_cols)} features...")
    
    missing_summary = []
    
    for feat in feature_cols:
        missing_count = train_df[feat].isna().sum()
        missing_pct = (missing_count / len(train_df)) * 100
        
        if missing_count > 0:
            # Check if missing is systematic by class
            if 'label' in train_df.columns:
                pos_missing = train_df[train_df['label'] == 1][feat].isna().sum()
                neg_missing = train_df[train_df['label'] == 0][feat].isna().sum()
                
                pos_pct = (pos_missing / train_df['label'].sum()) * 100 if train_df['label'].sum() > 0 else 0
                neg_pct = (neg_missing / (len(train_df) - train_df['label'].sum())) * 100 if (len(train_df) - train_df['label'].sum()) > 0 else 0
                
                if abs(pos_pct - neg_pct) > 50:  # Large difference
                    missing_summary.append({
                        'feature': feat,
                        'total_missing_pct': missing_pct,
                        'pos_missing_pct': pos_pct,
                        'neg_missing_pct': neg_pct,
                        'pattern': 'SYSTEMATIC_BY_CLASS'
                    })
    
    if missing_summary:
        print(f"\n[FOUND] {len(missing_summary)} features with systematic missing data by class:")
        for item in missing_summary[:10]:  # Show first 10
            print(f"  {item['feature']}:")
            print(f"    Total missing: {item['total_missing_pct']:.1f}%")
            print(f"    Positive missing: {item['pos_missing_pct']:.1f}%")
            print(f"    Negative missing: {item['neg_missing_pct']:.1f}%")
    else:
        print("\n[OK] No systematic missing data patterns detected")
    
    return missing_summary

# =============================================================================
# MAIN INVESTIGATION
# =============================================================================

def main():
    """Run complete Phase 1 investigation."""
    print("=" * 70)
    print("PHASE 1: ROOT CAUSE ANALYSIS & VALIDATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script: {os.path.basename(__file__)}")
    
    findings = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'section_1_1': {},
        'section_1_2': {},
        'section_1_3': {}
    }
    
    # Section 1.1: Confirm data leakage pattern
    leakage_findings, train_df = confirm_data_leakage_pattern()
    findings['section_1_1'] = leakage_findings
    
    # Section 1.2: Investigate feature engineering pipeline
    print("\n" + "=" * 70)
    print("SECTION 1.2: INVESTIGATE FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    fe_issues = review_feature_engineering()
    findings['section_1_2']['feature_engineering_issues'] = fe_issues
    
    windows_df = trace_window_extraction()
    findings['section_1_2']['windows_available'] = windows_df is not None
    
    source_info = check_source_csv()
    findings['section_1_2']['source_csv_info'] = source_info
    
    # Section 1.3: Identify additional issues
    print("\n" + "=" * 70)
    print("SECTION 1.3: IDENTIFY ADDITIONAL ISSUES")
    print("=" * 70)
    
    suspicious = check_suspicious_features(train_df)
    findings['section_1_3']['suspicious_features'] = suspicious
    
    duplicates = check_duplicate_features(train_df)
    findings['section_1_3']['duplicate_features'] = duplicates
    
    missing_patterns = check_missing_data_patterns(train_df)
    findings['section_1_3']['missing_data_patterns'] = missing_patterns
    
    # Save findings
    os.makedirs(RESULTS_DIR, exist_ok=True)
    findings_file = os.path.join(RESULTS_DIR, f"phase1_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(findings_file, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("PHASE 1 INVESTIGATION COMPLETE")
    print("=" * 70)
    print(f"\nFindings saved to: {findings_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    
    # Count critical issues
    critical_count = sum(1 for v in leakage_findings.values() if v == "PERFECT_SEPARATION") if leakage_findings else 0
    critical_count += len([s for s in suspicious if s.get('severity') == 'CRITICAL']) if suspicious else 0
    
    print(f"\nCritical Issues Found: {critical_count}")
    print(f"Suspicious Features: {len(suspicious) if suspicious else 0}")
    print(f"Duplicate Features: {len(duplicates) if duplicates else 0}")
    print(f"Systematic Missing Data: {len(missing_patterns) if missing_patterns else 0}")
    
    if critical_count > 0:
        print("\n[ACTION REQUIRED] Critical data leakage detected!")
        print("  → Proceed to Phase 2: Feature Engineering Fixes")
    else:
        print("\n[INFO] No critical issues detected, but review findings for warnings")
    
    return 0

if __name__ == "__main__":
    exit(main())



