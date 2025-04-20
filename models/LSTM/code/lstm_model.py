import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import argparse
import matplotlib.pyplot as plt
import joblib
print("Using GPU:", tf.config.list_physical_devices('GPU'))
mixed_precision.set_global_policy("mixed_float16")



# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from visualize_lstm_metrics import visualize_lstm_metrics, create_lstm_report

class VortexLSTMModel:
    """LSTM model for vortex prediction."""
    
    def __init__(self, window_size: int = 100):
        """Initialize the LSTM model."""
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
        
    def save_model_and_scaler(self, model_path: Path):
        """Save both the model and scaler."""
        # Save the model
        self.model.save(model_path)
        # Save the scaler
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
    def load_model_and_scaler(self, model_path: Path):
        """Load both the model and scaler."""
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        # Load the scaler
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        
    def prepare_sequences_from_windows(self, data: pd.DataFrame, windows: list) -> tuple:
        """Prepare sequences from detection windows."""
        sequences = []
        labels = []
        sequence_info = []  # Store diagnostic information
        
        # Get pressure differences
        pressure_values = data['PRESSURE'].values
        pressure_diff = np.diff(pressure_values, prepend=pressure_values[0])
        
        # Process detection windows
        for start, end in windows:
            # Get the detection window data
            window_data = data.iloc[start:end]
            
            # Create one sequence per window, using fixed window_size
            if end - start >= self.window_size:
                sequence = pressure_diff[end-self.window_size:end]
                sequences.append(sequence.reshape(-1, 1))
                labels.append(1)  # This is a vortex sequence
                
                # Store diagnostic info for vortex sequence
                sequence_info.append({
                    'type': 'vortex',
                    'start': end-self.window_size,
                    'end': end,
                    'pressure_range': (window_data['PRESSURE'].min(), window_data['PRESSURE'].max()),
                    'pressure_diff_range': (sequence.min(), sequence.max())
                })
                
                # Get non-vortex sequence from before this window
                non_vortex_start = max(0, start - self.window_size)
                non_vortex_sequence = pressure_diff[non_vortex_start:start]
                
                if len(non_vortex_sequence) == self.window_size:
                    sequences.append(non_vortex_sequence.reshape(-1, 1))
                    labels.append(0)  # This is a non-vortex sequence
                    
                    # Store diagnostic info for non-vortex sequence
                    sequence_info.append({
                        'type': 'non_vortex',
                        'start': non_vortex_start,
                        'end': start,
                        'pressure_range': (data.iloc[non_vortex_start:start]['PRESSURE'].min(), 
                                         data.iloc[non_vortex_start:start]['PRESSURE'].max()),
                        'pressure_diff_range': (non_vortex_sequence.min(), non_vortex_sequence.max())
                    })
        
        return np.array(sequences), np.array(labels), sequence_info
    
    def build_model(self, input_shape: tuple):
        """Build the LSTM model."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model."""
        # Scale each sequence individually to maintain temporal relationships
        X_train_scaled = np.zeros_like(X_train)
        for i in range(X_train.shape[0]):
            sequence = X_train[i].reshape(-1, 1)
            scaled_sequence = self.scaler.fit_transform(sequence)
            X_train_scaled[i] = scaled_sequence.reshape(X_train[i].shape)
        
        X_val_scaled = np.zeros_like(X_val)
        for i in range(X_val.shape[0]):
            sequence = X_val[i].reshape(-1, 1)
            scaled_sequence = self.scaler.transform(sequence)
            X_val_scaled[i] = scaled_sequence.reshape(X_val[i].shape)
        
        # Build model
        self.model = self.build_model((self.window_size, 1))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('models/lstm/models/lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train with class weights
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight={0: 1.0, 1: np.sum(y_train == 0) / np.sum(y_train == 1)}
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with the trained model."""
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = average_precision_score(y_test, y_pred_proba)  # PR AUC instead of ROC AUC
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

def find_detection_windows(data: pd.DataFrame) -> list:
    """Find all detection windows (where gt_detection_win == 1)."""
    windows = []
    in_window = False
    start_idx = None
    
    for idx, row in data.iterrows():
        if row['gt_detection_win'] == 1 and not in_window:
            in_window = True
            start_idx = idx
        elif row['gt_detection_win'] == 0 and in_window:
            in_window = False
            windows.append((start_idx, idx))
    
    if in_window:
        windows.append((start_idx, len(data)))
    
    return windows

def normalize_pressure(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize pressure values using min-max scaling."""
    data = data.copy()
    min_pressure = data['PRESSURE'].min()
    max_pressure = data['PRESSURE'].max()
    data['PRESSURE'] = (data['PRESSURE'] - min_pressure) / (max_pressure - min_pressure)
    return data

def plot_confidence_distribution(y_true, y_pred_proba, save_path='confidence_distribution.png'):
    """Plot distribution of confidence values for each class."""
    plt.figure(figsize=(12, 6))
    
    # Get confidence values for each class
    vortex_conf = y_pred_proba[y_true == 1]
    non_vortex_conf = y_pred_proba[y_true == 0]
    
    # Plot histograms
    plt.hist(vortex_conf, bins=50, alpha=0.5, label='Vortex', color='red')
    plt.hist(non_vortex_conf, bins=50, alpha=0.5, label='Non-Vortex', color='blue')
    
    plt.title('Distribution of Confidence Values')
    plt.xlabel('Confidence Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confidence_timeline(data, y_pred_proba, detection_windows, save_path='confidence_timeline.png'):
    """Plot confidence values over time with vortex events marked."""
    plt.figure(figsize=(15, 6))
    
    # Create time index
    time_index = range(len(y_pred_proba))
    
    # Plot confidence values
    plt.plot(time_index, y_pred_proba, label='Confidence', color='blue', alpha=0.7)
    
    # Mark detection windows
    for start, end in detection_windows:
        plt.axvspan(start, end, color='red', alpha=0.2, label='Detection Window' if start == detection_windows[0][0] else "")
    
    plt.title('Confidence Values Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Confidence Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function to train and evaluate the LSTM model."""
    parser = argparse.ArgumentParser(description='Train or evaluate LSTM model')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    args = parser.parse_args()
    
    print("Starting LSTM model training...")
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Normalize pressure values
    print("\nNormalizing pressure values...")
    data = normalize_pressure(data)
    print(f"Pressure range after normalization: {data['PRESSURE'].min():.4f} to {data['PRESSURE'].max():.4f}")
    
    # Find all detection windows
    detection_windows = find_detection_windows(data)
    print(f"Found {len(detection_windows)} detection windows")
    
    # Analyze window lengths
    window_lengths = [end - start for start, end in detection_windows]
    print("\nDetection window length statistics:")
    print(f"Min length: {min(window_lengths)}")
    print(f"Max length: {max(window_lengths)}")
    print(f"Mean length: {np.mean(window_lengths):.2f}")
    print(f"Median length: {np.median(window_lengths)}")
    print(f"Standard deviation: {np.std(window_lengths):.2f}")
    
    # Plot window length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(window_lengths, bins=50)
    plt.title('Distribution of Detection Window Lengths')
    plt.xlabel('Window Length (points)')
    plt.ylabel('Frequency')
    plt.savefig('detection_window_lengths.png')
    plt.close()
    
    # Split windows into train/val/test
    train_windows = detection_windows[:int(0.7 * len(detection_windows))]
    val_windows = detection_windows[int(0.7 * len(detection_windows)):int(0.85 * len(detection_windows))]
    test_windows = detection_windows[int(0.85 * len(detection_windows)):]
    
    # Initialize model with window size matching median detection window length
    model = VortexLSTMModel(window_size=60)
    
    # Prepare sequences for each split
    print("\nPreparing training sequences...")
    X_train, y_train, train_info = model.prepare_sequences_from_windows(data, train_windows)
    
    print("\nPreparing validation sequences...")
    X_val, y_val, val_info = model.prepare_sequences_from_windows(data, val_windows)
    
    print("\nPreparing test sequences...")
    X_test, y_test, test_info = model.prepare_sequences_from_windows(data, test_windows)
    
    # Print diagnostic information
    print("\nTraining set characteristics:")
    train_vortex = [s for s in train_info if s['type'] == 'vortex']
    train_non_vortex = [s for s in train_info if s['type'] == 'non_vortex']
    print(f"Vortex sequences: {len(train_vortex)}")
    print(f"Non-vortex sequences: {len(train_non_vortex)}")
    print("\nPressure ranges in training set:")
    print(f"Vortex: {[s['pressure_range'] for s in train_vortex[:5]]}")
    print(f"Non-vortex: {[s['pressure_range'] for s in train_non_vortex[:5]]}")
    
    print("\nValidation set characteristics:")
    val_vortex = [s for s in val_info if s['type'] == 'vortex']
    val_non_vortex = [s for s in val_info if s['type'] == 'non_vortex']
    print(f"Vortex sequences: {len(val_vortex)}")
    print(f"Non-vortex sequences: {len(val_non_vortex)}")
    print("\nPressure ranges in validation set:")
    print(f"Vortex: {[s['pressure_range'] for s in val_vortex[:5]]}")
    print(f"Non-vortex: {[s['pressure_range'] for s in val_non_vortex[:5]]}")
    
    # Print class distribution
    print("\nClass distribution in sets:")
    print(f"Training - Vortex: {sum(y_train)}, Non-vortex: {len(y_train) - sum(y_train)}")
    print(f"Validation - Vortex: {sum(y_val)}, Non-vortex: {len(y_val) - sum(y_val)}")
    print(f"Test - Vortex: {sum(y_test)}, Non-vortex: {len(y_test) - sum(y_test)}")
    
    # Model path
    model_path = Path(__file__).parent.parent / 'models' / 'lstm_model.h5'
    
    # Train or load model
    if model_path.exists() and not args.retrain:
        print("\nLoading existing model and scaler...")
        model.load_model_and_scaler(model_path)
        print("Model and scaler loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new model...")
        else:
            print("\nNo model found. Training new model...")
        
        # Train model
        print("\nTraining LSTM model...")
        history = model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)
        
        # Save model and scaler
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model_and_scaler(model_path)
        print(f"\nModel and scaler saved to: {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test, y_test)
    
    # Get predictions and confidence values
    y_pred_proba = model.model.predict(X_test)
    
    # Plot confidence distributions
    plot_confidence_distribution(y_test, y_pred_proba, 'confidence_distribution.png')
    
    # Plot confidence timeline
    plot_confidence_timeline(data, y_pred_proba, test_windows, 'confidence_timeline.png')
    
    # Print results
    print("\nModel Performance:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_lstm_metrics(
        model=model.model,
        X_test=X_test,
        y_test=y_test,
        y_pred=results['y_pred'],
        y_pred_proba=results['y_pred_proba'],
        model_name='LSTM Model',
        save_dir=results_dir
    )
    
    create_lstm_report('LSTM Model', results_dir)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nModel training and analysis complete!")

if __name__ == "__main__":
    main() 
