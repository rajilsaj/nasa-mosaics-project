import numpy as np
import pandas as pd
from pathlib import Path
from lstm_model import VortexLSTMModel
import matplotlib.pyplot as plt

def test_real_time_prediction():
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Initialize model
    model = VortexLSTMModel(window_size=60)
    
    # Prepare sequences for training
    X_train, y_train, _ = model.prepare_sequences_from_windows(data, [])
    
    # Fit scaler on training data
    model.fit_scaler(X_train)
    
    # Build and train model
    model.build_model((60, 1))
    model.train(X_train, y_train, X_train, y_train, epochs=1)  # Just one epoch for testing
    
    # Simulate real-time predictions
    pressure_readings = data['PRESSURE'].values
    predictions = []
    actual_labels = []
    
    # Slide through the data making predictions
    for i in range(60, len(pressure_readings)):
        current_readings = pressure_readings[i-60:i]
        prediction = model.predict_real_time(current_readings)
        predictions.append(prediction)
        actual_labels.append(data.iloc[i-1]['gt_detection_win'])
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot pressure
    plt.subplot(2, 1, 1)
    plt.plot(pressure_readings[60:], label='Pressure')
    plt.title('Pressure Readings')
    plt.legend()
    
    # Plot predictions and actual labels
    plt.subplot(2, 1, 2)
    plt.plot(predictions, label='Predicted Probability')
    plt.plot(actual_labels, label='Actual Label', alpha=0.5)
    plt.title('Predictions vs Actual Labels')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nPrediction Statistics:")
    print(f"Mean prediction: {np.mean(predictions):.4f}")
    print(f"Std prediction: {np.std(predictions):.4f}")
    print(f"Max prediction: {np.max(predictions):.4f}")
    print(f"Min prediction: {np.min(predictions):.4f}")
    
    # Count predictions above threshold
    threshold = 0.5
    high_confidence = sum(p > threshold for p in predictions)
    print(f"\nPredictions above {threshold}: {high_confidence} ({high_confidence/len(predictions)*100:.2f}%)")

if __name__ == "__main__":
    test_real_time_prediction() 