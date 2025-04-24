import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path

def visualize_model_metrics(model, X_test, y_test, y_pred, y_pred_proba, model_name, save_dir):
    """
    Create and save a complete set of model evaluation visualizations.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('default')

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    probs = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png')
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, lw=2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curve.png')
    plt.close()

    # 4. Feature Importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        labels = [f'Feature {i+1}' for i in range(len(importances))]
        plt.bar(labels, importances)
        plt.xticks(rotation=45)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(save_dir / 'feature_importance.png')
        plt.close()

    # 5. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, alpha=0.75, color='skyblue')
    plt.title(f'Prediction Probability Distribution - {model_name}')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_dir / 'probability_distribution.png')
    plt.close()

    # 6. Combined Metrics Summary
    summary = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC": roc_auc
    }

    plt.figure(figsize=(8, 6))
    plt.bar(summary.keys(), summary.values())
    plt.title(f'Model Metrics - {model_name}')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_metrics.png')
    plt.close()

    pd.DataFrame([summary]).to_csv(save_dir / 'metrics_summary.csv', index=False)
    print(f" Visual metrics saved in: {save_dir}")


def create_model_report(model_name: str, results_dir: str):
    """
    Build an HTML report with embedded metrics and saved graphs.
    """
    results_dir = Path(results_dir)
    metrics_df = pd.read_csv(results_dir / 'metrics_summary.csv')

    html_content = f"""
    <html>
    <head>
        <title>{model_name} Model Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
            .metric-block {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>{model_name} Evaluation Report</h1>
        <h2> Metrics</h2>
        <ul>
            {''.join([f"<li><strong>{col}:</strong> {val:.4f}</li>" for col, val in metrics_df.iloc[0].items()])}
        </ul>
        <div class="metric-block">
            <h2> Confusion Matrix</h2>
            <img src="confusion_matrix.png">
        </div>
        <div class="metric-block">
            <h2> ROC Curve</h2>
            <img src="roc_curve.png">
        </div>
        <div class="metric-block">
            <h2> Precision-Recall Curve</h2>
            <img src="precision_recall_curve.png">
        </div>
        <div class="metric-block">
            <h2> Feature Importance</h2>
            <img src="feature_importance.png">
        </div>
        <div class="metric-block">
            <h2> Prediction Probability Distribution</h2>
            <img src="probability_distribution.png">
        </div>
        <div class="metric-block">
            <h2> Combined Metrics</h2>
            <img src="combined_metrics.png">
        </div>
    </body>
    </html>
    """

    report_path = results_dir / "model_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f" HTML report saved to {report_path}")
