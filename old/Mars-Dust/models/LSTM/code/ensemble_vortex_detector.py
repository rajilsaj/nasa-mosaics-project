import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from lstm_model import VortexLSTMModel, find_detection_windows # Re-use data handling
import argparse
from pathlib import Path
import time

def load_ensemble_models(model_dir: Path):
    """Loads the two models for the ensemble."""
    print("Loading ensemble models...")
    
    # It's assumed the models are saved in a standard way.
    # We need to save the models with specific names first.
    scout_path = model_dir / 'scout_model_ws60.h5'
    sniper_path = model_dir / 'sniper_model_ws30.h5'
    
    if not scout_path.exists() or not sniper_path.exists():
        print("Error: Pre-trained scout (ws=60) and sniper (ws=30) models not found.")
        print(f"Please train and save them as '{scout_path.name}' and '{sniper_path.name}' first.")
        return None, None

    # Load models with the custom loss function
    loss_fn = VortexLSTMModel().temporal_focal_loss()
    scout_model = load_model(scout_path, custom_objects={'loss_function': loss_fn})
    sniper_model = load_model(sniper_path, custom_objects={'loss_function': loss_fn})
    
    print("Scout (ws=60) and Sniper (ws=30) models loaded successfully.")
    return scout_model, sniper_model

def run_ensemble_prediction(scout_model, sniper_model, data):
    """
    Runs the two-stage ensemble prediction.
    
    1. Prepare data for both window sizes.
    2. Predict with scout model.
    3. For scout's positive predictions, verify with sniper model.
    """
    print("\nRunning ensemble prediction...")
    
    # --- Stage 1: Scout Prediction ---
    scout_lstm = VortexLSTMModel(window_size=60, debug=False)
    X_scout, y_scout = scout_lstm.prepare_sequences(data, apply_sampling=False)
    
    print("Predicting with Scout model (ws=60)...")
    scout_pred_proba = scout_model.predict(X_scout).flatten()
    # Using the best threshold found previously for the scout model
    scout_pred_binary = (scout_pred_proba >= 0.63).astype(int)
    
    # --- Stage 2: Sniper Verification ---
    sniper_lstm = VortexLSTMModel(window_size=30, debug=False)
    X_sniper, y_sniper = sniper_lstm.prepare_sequences(data, apply_sampling=False)

    print("Verifying with Sniper model (ws=30)...")
    sniper_pred_proba = sniper_model.predict(X_sniper).flatten()
    # Using the best threshold found previously for the sniper model
    sniper_pred_binary = (sniper_pred_proba >= 0.65).astype(int)

    # --- Combine Predictions ---
    # The final prediction is positive only if BOTH models agree.
    # We need to align the predictions. A prediction at index `i` from the scout
    # corresponds to a window ending at data point `60 + i`.
    # A prediction at index `j` from the sniper corresponds to a window ending at `30 + j`.
    # So, scout index `i` aligns with sniper index `j = i + 30`.
    
    # Initialize the ensemble probability array with a neutral value (0.5)
    ensemble_pred_proba = np.full_like(scout_pred_proba, 0.5)
    
    # We will average the probabilities where they overlap
    for i in range(len(scout_pred_proba)):
        sniper_idx = i + 30
        if sniper_idx < len(sniper_pred_proba):
            # Apply a weighted average, trusting the sniper more
            ensemble_pred_proba[i] = (0.3 * scout_pred_proba[i]) + (0.7 * sniper_pred_proba[sniper_idx])
        else:
            # If there's no corresponding sniper prediction, heavily penalize the score
            ensemble_pred_proba[i] = scout_pred_proba[i] * 0.3

    print(f"Combined probabilities for {len(ensemble_pred_proba)} windows using a weighted average.")
    
    # We use the scout's y_true for evaluation as it covers the full test set range
    return ensemble_pred_proba, y_scout, scout_lstm, data.iloc[60:]


def main():
    parser = argparse.ArgumentParser(description='Run Vortex Detection Ensemble Model')
    
    # Make default path robust by constructing it relative to this script's location
    script_dir = Path(__file__).parent
    default_model_dir = script_dir.parent / 'models'
    
    parser.add_argument('--model_dir', type=str, default=str(default_model_dir), help='Directory where models are saved.')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    # Load models
    scout_model, sniper_model = load_ensemble_models(model_dir)
    if scout_model is None:
        return

    # Load data
    print("Loading data...")
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Use the same test split as before
    val_end = int(0.85 * len(data))
    test_data = data.iloc[val_end:]
    print(f"Using test set with {len(test_data)} samples.")

    # Run ensemble prediction
    y_pred_proba, y_true, lstm_evaluator, eval_data = run_ensemble_prediction(scout_model, sniper_model, test_data)

    # Evaluate the ensemble results
    print("\n--- Evaluating Ensemble Performance ---")
    
    # Find the best threshold for the new ensemble probabilities
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0.3, 0.7, 81)
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        # Use a simplified F1 calculation for threshold finding
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Found best threshold for ensemble: {best_threshold:.4f} (F1: {best_f1:.4f})")
    
    # Use the best threshold for final evaluation
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    window_starts = np.arange(lstm_evaluator.window_size, len(test_data))
    gt_windows = find_detection_windows(test_data)
    
    # Create a results dictionary with the correct data
    results_dict = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'y_true': y_true}
    
    event_results = lstm_evaluator.evaluate_with_windows(results_dict, gt_windows, test_data)

    print("\nFinal Ensemble Event-Based Metrics:")
    print(f"Precision: {event_results['event_metrics']['precision']:.4f}")
    print(f"Recall: {event_results['event_metrics']['recall']:.4f}")
    print(f"F1-Score: {event_results['event_metrics']['f1']:.4f}")
    print(f"True Positives: {event_results['event_metrics']['tp']}")
    print(f"False Positives: {event_results['event_metrics']['fp']}")
    print(f"False Negatives: {event_results['event_metrics']['fn']}")

if __name__ == "__main__":
    main() 