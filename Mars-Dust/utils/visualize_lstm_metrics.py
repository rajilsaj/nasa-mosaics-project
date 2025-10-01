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

def visualize_lstm_metrics(model, X_test, y_test, y_pred, y_pred_proba, model_name, save_dir):
    """
    Create comprehensive visualizations for LSTM model metrics and save them to the specified directory.
    
    Parameters:
    -----------
    model : tensorflow model
        The trained LSTM model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Prediction probabilities (1D array for LSTM)
    model_name : str
        Name of the model for plot titles
    save_dir : str or Path
        Directory to save the visualizations
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for all plots
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
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png')
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curve.png')
    plt.close()
    
    # 4. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba, bins=50, alpha=0.75)
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_dir / 'probability_distribution.png')
    plt.close()
    
    # 5. Combined Metrics Plot
    metrics = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': auc(fpr, tpr)
    }
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Metrics')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_metrics.png')
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / 'metrics_summary.csv', index=False)
    
    print(f"Visualizations saved to {save_dir}")

def create_lstm_report(model_name, results_dir):
    """
    Create a comprehensive HTML report for the LSTM model using the saved metrics and visualizations.
    """
    results_dir = Path(results_dir)
    
    # Read metrics
    metrics_df = pd.read_csv(results_dir / 'metrics_summary.csv')
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>{model_name} LSTM Model Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .visualization {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{model_name} LSTM Model Report</h1>
            
            <h2>Model Metrics</h2>
            <div class="metrics">
                {''.join([f'<div class="metric-card"><h3>{col}</h3><p>{val:.4f}</p></div>' for col, val in metrics_df.iloc[0].items()])}
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualization">
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="visualization">
                <h3>ROC Curve</h3>
                <img src="roc_curve.png" alt="ROC Curve">
            </div>
            <div class="visualization">
                <h3>Precision-Recall Curve</h3>
                <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
            </div>
            <div class="visualization">
                <h3>Prediction Probability Distribution</h3>
                <img src="probability_distribution.png" alt="Probability Distribution">
            </div>
            <div class="visualization">
                <h3>Combined Metrics</h3>
                <img src="combined_metrics.png" alt="Combined Metrics">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(results_dir / 'model_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {results_dir / 'model_report.html'}") 