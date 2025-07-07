"""
feature_importance_rf.py

Modular Random Forest feature importance analysis for LSTM features.
Uses test data as-is (no sampling) to match LSTM evaluation approach.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


def compute_rf_feature_importance(X, y, feature_names=None, selected_features=None, n_estimators=100, 
                                random_state=42, plot=True, balance_classes=False, original_data=None):
    """
    Compute and display Random Forest feature importances using test data as-is.
    
    Args:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        feature_names: list of str, names for each feature (optional)
        selected_features: list of str or int, features to include (by name or index, optional)
        n_estimators: int, number of trees in the forest
        random_state: int, random seed
        plot: bool, whether to plot the importances
        balance_classes: bool, whether to balance classes (default: False to match LSTM test evaluation)
        original_data: pd.DataFrame, original data (not used when balance_classes=False)
    Returns:
        importances: np.ndarray, feature importances
        used_feature_names: list of str, names of features used
        rf_performance: dict, RF performance metrics
    """
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    
    # Subset features if requested
    if selected_features is not None:
        if feature_names is not None and isinstance(selected_features[0], str):
            indices = [feature_names.index(f) for f in selected_features]
        else:
            indices = selected_features
        X = X[:, indices]
        used_feature_names = [feature_names[i] for i in indices] if feature_names is not None else [str(i) for i in indices]
    else:
        used_feature_names = feature_names if feature_names is not None else [str(i) for i in range(X.shape[1])]

    # Use data as-is (like LSTM test evaluation with apply_sampling=False)
    if not balance_classes:
        print("\nUsing test data as-is (matching LSTM test evaluation approach)...")
        print("Note: This preserves the natural class distribution for realistic evaluation")
        
        # Split the data for RF training/validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    else:
        # Balance classes using temporal sampling (like LSTM training)
        if original_data is None:
            raise ValueError("original_data must be provided when balance_classes=True")
        
        print("\nBalancing classes using temporal sampling (matching LSTM training approach)...")
        
        # Get ground truth information
        gt_detection = original_data['gt_detection_win'].values
        gt_fwhm = original_data['gt_fwhm'].values
        
        # Find vortex indices (positive samples)
        vortex_indices = np.where(np.logical_or(gt_detection == 1, gt_fwhm == 1))[0]
        
        # Find non-vortex indices (negative samples)
        non_vortex_indices = np.where(np.logical_and(gt_detection == 0, gt_fwhm == 0))[0]
        
        print(f"Found {len(vortex_indices)} vortex events and {len(non_vortex_indices)} non-vortex points")
        
        # Sample equal numbers of positive and negative examples
        n_positive = len(vortex_indices)
        n_negative = len(non_vortex_indices)
        n_samples_per_class = min(n_positive, n_negative)
        
        # For positive samples: use actual vortex indices (temporal order preserved)
        balanced_positive_indices = vortex_indices[:n_samples_per_class]
        
        # For negative samples: randomly sample from non-vortex areas (like LSTM)
        np.random.seed(random_state)
        balanced_negative_indices = np.random.choice(non_vortex_indices, size=n_samples_per_class, replace=False)
        
        # Combine balanced samples (positive first, then negative, like LSTM)
        balanced_indices = np.concatenate([balanced_positive_indices, balanced_negative_indices])
        
        # Get corresponding features and labels
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"Balanced data shape: {X_balanced.shape}")
        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        print(f"Positive samples from actual vortex events: {sum(y_balanced[:n_samples_per_class])}")
        print(f"Negative samples from non-vortex areas: {sum(y_balanced[n_samples_per_class:])}")
        
        # Use balanced data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=random_state, stratify=y_balanced
        )

    # Train RF
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state, 
        verbose=0,
        class_weight='balanced' if not balance_classes else None  # Use class weights for imbalanced data
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # Evaluate RF performance
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    rf_performance = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'test_samples': len(y_test),
        'balanced': balance_classes,
        'temporal_sampling': balance_classes
    }
    
    print(f"\nRandom Forest Performance:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Test samples: {len(y_test)}")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-vortex', 'Vortex']))

    # Print importances
    print(f"\nRandom Forest Feature Importances:")
    for name, imp in sorted(zip(used_feature_names, importances), key=lambda x: -x[1]):
        print(f"{name:30s}: {imp:.4f}")

    # Plot importances
    if plot:
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importances)[::-1]
        
        # Create bar plot
        plt.subplot(1, 2, 1)
        plt.bar(range(len(importances)), importances[sorted_idx], 
                tick_label=np.array(used_feature_names)[sorted_idx])
        plt.ylabel("Importance")
        plt.title("Random Forest Feature Importances")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Create cumulative importance plot
        plt.subplot(1, 2, 2)
        cumulative_importance = np.cumsum(importances[sorted_idx])
        plt.plot(range(1, len(importances) + 1), cumulative_importance, 'bo-')
        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Importance")
        plt.title("Cumulative Feature Importance")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    return importances, used_feature_names, rf_performance


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(0)
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # All features with test data as-is (default)
    compute_rf_feature_importance(X, y, feature_names=feature_names, balance_classes=False)

    # Subset of features by name
    compute_rf_feature_importance(X, y, feature_names=feature_names, 
                                selected_features=["feature_0", "feature_2"], balance_classes=False) 