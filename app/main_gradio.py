# main
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# === Feature Extraction ===
def extract_features(df, window_size):
    df['pressure_diff'] = df['PRESSURE'].diff().shift(1)
    df = df.dropna()
    X, y = [], []
    for i in range(window_size, len(df)):
        window = df['pressure_diff'].iloc[i - window_size:i].values
        label = df['gt_detection_win'].iloc[i]
        features = [
            np.mean(window),
            np.std(window),
            np.sum(window < 0),
            np.mean(window[-10:]),
            np.polyfit(range(len(window)), window, 1)[0]  # slope
        ]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y), df.iloc[window:].reset_index(drop=True)

# === Training and Evaluation ===
def run_model(df, model_name, window_size, threshold):
    X, y, trimmed_df = extract_features(df, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42) if model_name == "Random Forest" else \
            XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    return fig, {"Precision": precision, "Recall": recall, "F1 Score": f1, "Threshold": threshold}

# === Gradio App ===
def gradio_app(data_file, model_choice, window_size, threshold):
    df = pd.read_csv(data_file.name if hasattr(data_file, 'name') else data_file)
    return run_model(df, model_choice, window_size, threshold)

with gr.Blocks(title="Vortex Detection Tuner") as demo:
    gr.Markdown("## Vortex Detection Model Explorer")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            model_choice = gr.Radio(["Random Forest", "XGBoost"], value="XGBoost", label="Model Type")
            window_slider = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="Sliding Window Size")
            threshold_slider = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Threshold")
            run_button = gr.Button("Run Model")

        with gr.Column(scale=2):
            roc_output = gr.Plot(label="ROC Curve")
            metrics_output = gr.JSON(label="Model Metrics")

    run_button.click(fn=gradio_app, 
                     inputs=[file_input, model_choice, window_slider, threshold_slider],
                     outputs=[roc_output, metrics_output])

demo.launch()
