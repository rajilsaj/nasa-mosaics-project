# Random Forest Classifier for Mars Vortex Detection: Summary

## Methodology

This work trains and evaluates a **Random Forest classifier** for binary vortex detection on Mars. The classifier uses scikit-learn's `RandomForestClassifier` with 100 decision trees, maximum depth of 12-15, minimum samples split of 15, and custom class weights calculated from inverse frequency to handle extreme imbalance (99.4% negative samples). The model processes 60-sample pressure windows represented by 15 features: trend features (overall slope, first/second half slopes, trend consistency), pressure drop features (magnitude, rate, position), statistical features (mean, std, range, half-window means), and anomaly features (z-scores, deviations). The Random Forest is trained on balanced data (1:1 positive-to-negative ratio) using custom class weights `{0: weight_negative, 1: weight_positive}` where weights are inversely proportional to class frequency. For inference, the model outputs class probabilities, with decision thresholds varied (0.45-0.90) to explore precision-recall trade-offs. The classifier is evaluated on both fixed windows (242 aligned precursor windows) and sliding windows (85,925 continuous windows with step=10) to assess performance under different deployment scenarios.

## Results

**Fixed-Window Evaluation (Test Set, 242 windows, threshold=0.5):**
- F1-Score: 0.8000 | Precision: 0.7143 | Recall: 0.9091 | ROC AUC: 0.9849

**Sliding-Window Evaluation (Test Set, 85,925 windows, 380 positive, 85,545 negative):**

| Threshold | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-----------|----------|-----------|--------|-----------|---------|
| 0.45 | 88.52% | 1.65% | 42.63% | 3.18% | 0.7457 |
| 0.60 | 95.65% | 2.35% | 21.84% | 4.25% | 0.7457 |
| 0.75 | 97.60% | 2.86% | 13.42% | 4.72% | 0.7457 |
| 0.90 | 98.85% | 3.78% | 6.58% | 4.80% | 0.7457 |

The Random Forest classifier demonstrates strong discrimination capability (ROC AUC=0.75-0.98) but suffers from distribution shift: trained on balanced data, it achieves F1=0.80 on fixed windows but only F1=0.03-0.05 on naturally imbalanced sliding windows. Feature importance analysis shows `second_half_slope` (21.8%), `pressure_drop` (13.5%), and `range` (13.1%) as the most important features, indicating the classifier relies primarily on late-window pressure trends and drop magnitude. The low precision (1.65-3.78%) on sliding windows indicates the Random Forest produces too many false positives when deployed on imbalanced data, despite reasonable discrimination ability.

## Future Work

**Random Forest improvements:** (1) Tune class weights and decision thresholds on validation sliding windows to optimize for mission-specific precision/recall trade-offs; (2) Implement cost-sensitive learning by adjusting class weights to penalize false positives more heavily during Random Forest training; (3) Experiment with Random Forest hyperparameters (n_estimators, max_depth, min_samples_split) optimized specifically for imbalanced deployment scenarios; (4) Develop post-processing filters (temporal voting, probability calibration) to reduce false positives from Random Forest predictions; (5) Compare Random Forest performance with alternative classifiers (Gradient Boosting, XGBoost) to assess if different ensemble methods better handle the imbalanced deployment scenario.

