# Class Prior Integration Strategy for Comprehensive Dataset
## Incorporating Deployment Priors into Random Forest Pipeline

---

## 🎯 **Overview: Why Class Priors Matter**

Class priors (the true distribution of classes in deployment) often differ from training data. Your model was trained on balanced data (1:1) but deployed on natural imbalance (94:1 or 225:1). This mismatch causes poor precision.

**Solution**: Adjust model outputs to match deployment priors.

---

## 📊 **Phase 1: Class Prior Analysis**

### **1.1 Calculate Priors for Each Split**

```python
def analyze_class_priors():
    """
    Calculate class priors for train/val/test splits.
    """
    # Load splits
    train_df = pd.read_csv('data/splits/ml_train.csv')
    val_df = pd.read_csv('data/splits/ml_val.csv')
    test_df = pd.read_csv('data/splits/ml_test.csv')
    
    # Calculate priors
    priors = {}
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        pos_count = df['gt_detection_win'].sum()
        neg_count = (~df['gt_detection_win']).sum()
        total = len(df)
        
        priors[split_name] = {
            'prior_positive': pos_count / total,
            'prior_negative': neg_count / total,
            'ratio': neg_count / pos_count,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'total': total
        }
    
    return priors
```

### **1.2 Identify Deployment Prior**

**Key Question**: What is the expected class distribution in deployment?

**Options**:
- **Option A**: Use validation set prior (most common)
- **Option B**: Use test set prior (if known)
- **Option C**: Use historical deployment data (if available)
- **Option D**: Use comprehensive dataset prior (1.05% positive)

**Recommendation**: Use validation set prior as proxy for deployment.

---

## 🔧 **Phase 2: Class Prior Integration Methods**

### **Method 1: Class Weighting During Training (Pre-Training)**

**When to Use**: If you know deployment prior before training.

```python
def calculate_class_weights_from_prior(deployment_prior_pos, deployment_prior_neg):
    """
    Calculate class weights based on deployment priors.
    
    Args:
        deployment_prior_pos: Expected positive class prior in deployment
        deployment_prior_neg: Expected negative class prior in deployment
    
    Returns:
        Dictionary of class weights
    """
    # Inverse frequency weighting adjusted for deployment priors
    weight_positive = 1.0 / deployment_prior_pos
    weight_negative = 1.0 / deployment_prior_neg
    
    # Normalize so they sum to 2 (for two classes)
    total = weight_positive + weight_negative
    weight_positive = 2 * weight_positive / total
    weight_negative = 2 * weight_negative / total
    
    return {0: weight_negative, 1: weight_positive}

# Example usage:
deployment_prior_pos = 0.0105  # 1.05% from comprehensive dataset
deployment_prior_neg = 0.9895  # 98.95%

class_weights = calculate_class_weights_from_prior(
    deployment_prior_pos, 
    deployment_prior_neg
)

rf_model = RandomForestClassifier(
    class_weight=class_weights,  # Use deployment priors
    ...
)
```

**Pros**: 
- Adjusts model during training
- Learns decision boundaries for deployment distribution

**Cons**:
- Requires knowing deployment prior before training
- May overfit to specific prior

---

### **Method 2: Probability Adjustment Post-Training (Post-Inference)**

**When to Use**: If deployment prior differs from training, or you want flexibility.

**Based on Latinne et al. (2001) EM Algorithm**:

```python
def adjust_probabilities_em(y_proba, training_prior_pos, training_prior_neg,
                           deployment_prior_pos, deployment_prior_neg,
                           max_iter=100, tol=1e-6):
    """
    Adjust classifier probabilities to new a priori probabilities using EM algorithm.
    
    Based on: Latinne et al. (2001) "Adjusting the Outputs of a Classifier to 
    New a Priori Probabilities"
    
    Args:
        y_proba: Original probabilities from classifier
        training_prior_pos: Positive class prior in training data
        training_prior_neg: Negative class prior in training data
        deployment_prior_pos: Expected positive class prior in deployment
        deployment_prior_neg: Expected negative class prior in deployment
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
    
    Returns:
        Adjusted probabilities
    """
    from scipy.optimize import minimize_scalar
    
    # Initial estimate of adjustment factor
    # Using Bayes' theorem: P(y|x) = P(x|y) * P(y) / P(x)
    # Adjustment: P_new(y|x) = P_old(y|x) * (P_new(y) / P_old(y)) / normalization
    
    # Simple adjustment (first approximation)
    adjustment_factor = deployment_prior_pos / training_prior_pos
    
    # EM algorithm for refinement
    adjusted_proba = y_proba.copy()
    
    for iteration in range(max_iter):
        # E-step: Estimate class membership
        # P(y=1|x) adjusted
        old_proba = adjusted_proba.copy()
        
        # M-step: Update probabilities
        # Using Bayes' theorem with new priors
        numerator = old_proba * deployment_prior_pos
        denominator = (old_proba * deployment_prior_pos + 
                      (1 - old_proba) * deployment_prior_neg)
        
        adjusted_proba = numerator / denominator
        
        # Check convergence
        if np.abs(adjusted_proba - old_proba).max() < tol:
            break
    
    return adjusted_proba

# Simpler version (direct Bayes adjustment):
def adjust_probabilities_bayes(y_proba, training_prior_pos, deployment_prior_pos):
    """
    Simpler Bayes-based probability adjustment.
    
    P_new(y=1|x) = P_old(y=1|x) * (P_new(y=1) / P_old(y=1)) / 
                   [P_old(y=1|x) * (P_new(y=1) / P_old(y=1)) + 
                    P_old(y=0|x) * (P_new(y=0) / P_old(y=0))]
    """
    # Calculate adjustment ratios
    ratio_pos = deployment_prior_pos / training_prior_pos
    ratio_neg = (1 - deployment_prior_pos) / (1 - training_prior_pos)
    
    # Adjust probabilities
    numerator = y_proba * ratio_pos
    denominator = (y_proba * ratio_pos + (1 - y_proba) * ratio_neg)
    
    adjusted_proba = numerator / denominator
    
    return adjusted_proba
```

**Usage**:
```python
# After model training
y_proba_train = model.predict_proba(X_train)[:, 1]
y_proba_val = model.predict_proba(X_val)[:, 1]
y_proba_test = model.predict_proba(X_test)[:, 1]

# Get priors
training_prior_pos = 0.5  # Balanced training (1:1)
deployment_prior_pos = 0.0105  # From comprehensive dataset (1.05%)

# Adjust probabilities
y_proba_val_adjusted = adjust_probabilities_bayes(
    y_proba_val, 
    training_prior_pos, 
    deployment_prior_pos
)

y_proba_test_adjusted = adjust_probabilities_bayes(
    y_proba_test,
    training_prior_pos,
    deployment_prior_pos
)
```

**Pros**:
- Flexible - can adjust for any deployment prior
- No retraining needed
- Can test multiple deployment scenarios

**Cons**:
- Assumes classifier is well-calibrated
- May not work if training and deployment distributions are very different

---

### **Method 3: Threshold Optimization Based on Priors**

**When to Use**: Quick fix, doesn't require probability adjustment.

```python
def optimize_threshold_for_prior(y_true, y_proba, deployment_prior_pos, 
                                 cost_fp=1.0, cost_fn=10.0):
    """
    Optimize threshold based on deployment prior and misclassification costs.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        deployment_prior_pos: Expected positive class prior
        cost_fp: Cost of false positive
        cost_fn: Cost of false negative
    
    Returns:
        Optimal threshold
    """
    from sklearn.metrics import precision_recall_curve
    
    # Calculate expected cost for each threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Expected cost = P(FP) * cost_fp + P(FN) * cost_fn
    # P(FP) = (1 - precision) * deployment_prior_pos
    # P(FN) = (1 - recall) * deployment_prior_pos
    
    costs = []
    for i, thresh in enumerate(thresholds):
        # Approximate FPR and FNR at this threshold
        # This is simplified - in practice, calculate from confusion matrix
        fpr = 1 - precision[i] if i < len(precision) else 0
        fnr = 1 - recall[i] if i < len(recall) else 0
        
        # Expected cost
        cost = (fpr * (1 - deployment_prior_pos) * cost_fp + 
                fnr * deployment_prior_pos * cost_fn)
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
```

---

## 🔄 **Phase 3: Integrated Pipeline**

### **Complete Workflow**

```python
def train_with_prior_adjustment():
    """
    Complete pipeline with class prior integration.
    """
    # 1. Load data
    train_df = pd.read_csv('train_features.csv')
    val_df = pd.read_csv('val_features.csv')
    test_df = pd.read_csv('test_features.csv')
    
    # 2. Calculate priors
    train_prior_pos = train_df['label'].mean()
    val_prior_pos = val_df['label'].mean()
    test_prior_pos = test_df['label'].mean()
    
    print(f"Training prior (positive): {train_prior_pos:.4f}")
    print(f"Validation prior (positive): {val_prior_pos:.4f}")
    print(f"Test prior (positive): {test_prior_pos:.4f}")
    
    # 3. Train model with deployment prior weighting
    # Option A: Use deployment prior for class weights
    deployment_prior_pos = val_prior_pos  # Use validation as proxy
    class_weights = calculate_class_weights_from_prior(
        deployment_prior_pos, 
        1 - deployment_prior_pos
    )
    
    model = RandomForestClassifier(
        class_weight=class_weights,
        ...
    )
    model.fit(X_train, y_train)
    
    # 4. Get predictions
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # 5. Adjust probabilities (if needed)
    # If training prior differs significantly from deployment
    if abs(train_prior_pos - deployment_prior_pos) > 0.01:
        y_proba_val_adjusted = adjust_probabilities_bayes(
            y_proba_val,
            train_prior_pos,
            deployment_prior_pos
        )
        y_proba_test_adjusted = adjust_probabilities_bayes(
            y_proba_test,
            train_prior_pos,
            deployment_prior_pos
        )
    else:
        y_proba_val_adjusted = y_proba_val
        y_proba_test_adjusted = y_proba_test
    
    # 6. Optimize threshold
    optimal_threshold = optimize_threshold_for_prior(
        y_val,
        y_proba_val_adjusted,
        deployment_prior_pos
    )
    
    # 7. Evaluate
    evaluate_with_threshold(
        y_test,
        y_proba_test_adjusted,
        optimal_threshold
    )
    
    return model, optimal_threshold
```

---

## 📋 **Recommended Approach for Comprehensive Dataset**

### **Step-by-Step Implementation**

**Step 1: Calculate Priors**
```python
# From comprehensive dataset analysis:
comprehensive_prior_pos = 0.0105  # 1.05%
comprehensive_prior_neg = 0.9895  # 98.95%
```

**Step 2: Training Strategy**
- **Option A** (Recommended): Train with balanced data (1:1), adjust probabilities post-training
- **Option B**: Train with deployment prior weights (if prior is stable)

**Step 3: Probability Adjustment**
- Use Bayes adjustment for validation/test sets
- Adjust from training prior (0.5) to deployment prior (0.0105)

**Step 4: Threshold Optimization**
- Optimize threshold on validation set (with adjusted probabilities)
- Use optimized threshold on test set

**Step 5: Evaluation**
- Report metrics at optimized threshold
- Compare adjusted vs unadjusted performance

---

## 🎯 **Expected Improvements**

**Without Prior Adjustment**:
- Precision: ~3.78% (at threshold 0.90)
- Recall: ~6.58%
- F1: ~4.80%

**With Prior Adjustment** (estimated):
- Precision: **5-8%** (improved by 30-100%)
- Recall: **8-12%** (improved by 20-80%)
- F1: **6-9%** (improved by 25-90%)

**Key**: Prior adjustment should improve precision significantly while maintaining or improving recall.

---

## 💡 **Implementation Priority**

1. **High Priority**: Probability adjustment post-training (Method 2)
   - Easy to implement
   - No retraining needed
   - Can test immediately

2. **Medium Priority**: Threshold optimization (Method 3)
   - Quick win
   - Improves metrics immediately

3. **Low Priority**: Class weighting during training (Method 1)
   - Requires retraining
   - Less flexible
   - Do this if prior adjustment doesn't help enough

---

## 🔬 **Validation Strategy**

1. Train model on balanced data (baseline)
2. Adjust probabilities to deployment prior
3. Compare metrics:
   - Unadjusted vs Adjusted
   - Different threshold strategies
4. Choose best approach based on validation performance
5. Apply to test set (final evaluation)

---

## 📊 **Code Integration**

I'll create a script that:
1. Calculates class priors from comprehensive dataset
2. Trains model (balanced or prior-weighted)
3. Adjusts probabilities post-training
4. Optimizes threshold
5. Evaluates performance

**Ready to implement?**

