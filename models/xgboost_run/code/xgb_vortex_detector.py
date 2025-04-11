import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from pathlib import Path
from model_utils import visualize_model_metrics, create_model_report


import sys

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
# from visualize_metrics import visualize_model_metrics, create_model_report
# from analyze_features import analyze_feature_importance




# 1. Load your data (adjust this path as needed)
data = pd.read_csv('../../../data/vortex_data.csv')  # Update if path is different

# 2. Define your features and target
X = data.drop(columns=['target'])  # Replace 'target' with your actual target column
y = data['target']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Define XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,  # ← Adjust based on class imbalance
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 5. Fit the model
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)

# 6. Predict
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# 7. Visualize
save_dir = Path('../../results/xgboost')
visualize_model_metrics(
    model=xgb_model,
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    model_name='XGBoost Vortex Detector',
    save_dir=save_dir
)

# 8. HTML report
create_model_report(model_name='XGBoost Vortex Detector', results_dir=save_dir)

# 9. (Optional) Print classification report
print(classification_report(y_test, y_pred))

print("\n🚀 XGBoost script completed successfully! Report saved to:", save_dir)
