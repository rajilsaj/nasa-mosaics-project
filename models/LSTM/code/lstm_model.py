import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import argparse
print("Using GPU:", tf.config.list_physical_devices('GPU'))
mixed_precision.set_global_policy("mixed_float16")



# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from visualize_lstm_metrics import visualize_lstm_metrics, create_lstm_report

class VortexLSTMModel:
    """LSTM model for vortex prediction."""
    
    def __init__(self, window_size=50, sequence_length=5):
        """Initialize the LSTM model."""
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_sequences(self, data: pd.DataFrame) -> tuple:
        """Prepare sequences for LSTM input."""
        # Get raw pressure values and calculate difference
        pressure_values = data['PRESSURE']
        pressure_diff = np.diff(pressure_values, prepend=pressure_values[0])
        
        # Create sequences
        sequences = []
        labels = []
        
        # Create a mask for valid samples (where gt_fwhm is False)
        valid_mask = data['gt_fwhm'] == 0
        
        # Calculate the maximum valid index
        max_valid_idx = len(pressure_values) - self.sequence_length
        
        for i in range(self.window_size, max_valid_idx):
            # Skip if any of the samples in our window or lookahead are in a vortex period
            if not all(valid_mask.iloc[i-self.window_size:i+self.sequence_length]):
                continue
                
            # Get sequence of raw differences
            sequence = pressure_diff[i-self.window_size:i]
            sequences.append(sequence.reshape(-1, 1))  # Ensure correct shape
            
            # Label is 1 if any of the next sequence_length samples contain a vortex
            label = 1 if any(data['gt_detection_win'].iloc[i:i+self.sequence_length] == 1) else 0
            labels.append(label)
            
        return np.array(sequences), np.array(labels)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(32, input_shape=input_shape),  # input_shape is now (window_size, 1)
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model."""
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # Build model
        self.model = self.build_model((self.window_size, 1))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(
                'models/lstm/models/lstm_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight={0: 1.0, 1: np.sum(y_train == 0) / np.sum(y_train == 1)}  # Balance classes
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

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate LSTM model')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    args = parser.parse_args()
    
    print("Starting LSTM model training...")
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    df = pd.read_csv(data_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Initialize LSTM model
    lstm_model = VortexLSTMModel()
    
    # Prepare sequences
    print("Preparing sequences...")
    X, y = lstm_model.prepare_sequences(df)
    
    # Split data
    train_size = int(len(X) * 0.8)
    val_size = int(train_size * 0.8)
    
    X_train = X[:val_size]
    y_train = y[:val_size]
    
    X_val = X[val_size:train_size]
    y_val = y[val_size:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Model path
    model_path = Path(__file__).parent.parent / 'models' / 'lstm_model.h5'
    
    # Train or load model
    if model_path.exists() and not args.retrain:
        print("\nLoading existing model...")
        lstm_model.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new model...")
        else:
            print("\nNo model found. Training new model...")
        
        # Train model
        print("\nTraining LSTM model...")
        history = lstm_model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)
        
        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        lstm_model.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = lstm_model.evaluate(X_test, y_test)
    
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
        model=lstm_model.model,
        X_test=X_test,
        y_test=y_test,
        y_pred=results['y_pred'],
        y_pred_proba=results['y_pred_proba'],
        model_name='LSTM Model',
        save_dir=results_dir
    )
    
    create_lstm_report('LSTM Model', results_dir)
    
    print("\nModel training and analysis complete!")

if __name__ == "__main__":
    main() 
