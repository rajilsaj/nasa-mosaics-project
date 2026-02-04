import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

class VortexLSTMModelROC:
    """LSTM model for vortex prediction using rate of change features."""
    
    def __init__(self, window_size: int = 100):
        """Initialize the LSTM model."""
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
        self.scaler_fitted = False
        
    def calculate_rate_of_change(self, values, window_size=11):
        """Calculate rate of change using a centered window."""
        rate = np.zeros_like(values, dtype=float)
        half_window = window_size // 2
        
        for i in range(len(values)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(values), i + half_window + 1)
            if end_idx - start_idx < 2:
                continue
            time_points = np.arange(end_idx - start_idx)
            coeffs = np.polyfit(time_points, values[start_idx:end_idx], 1)
            rate[i] = coeffs[0]
        
        return rate
        
    def fit_scaler(self, data: np.ndarray):
        """Fit scaler on all sequences at once."""
        all_sequences = data.reshape(-1, data.shape[-1])
        self.scaler.fit(all_sequences)
        self.scaler_fitted = True
        
    def save_model_and_scaler(self, model_path: Path):
        """Save both the model and scaler."""
        self.model.save(model_path)
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
    def load_model_and_scaler(self, model_path: Path):
        """Load both the model and scaler."""
        self.model = tf.keras.models.load_model(model_path)
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        self.scaler_fitted = True
        
    def prepare_sequences(self, data: pd.DataFrame) -> tuple:
        """Prepare sequences using rate of change features."""
        sequences = []
        labels = []
        sequence_info = []
        
        # Calculate rate of change
        pressure_values = data['PRESSURE'].values
        pressure_roc = self.calculate_rate_of_change(pressure_values)
        
        # Slide window across entire dataset
        for i in range(len(data) - self.window_size + 1):
            # Get window of pressure and rate of change
            sequence_pressure = pressure_values[i:i + self.window_size]
            sequence_roc = pressure_roc[i:i + self.window_size]
            
            # Combine features
            sequence = np.column_stack((sequence_pressure, sequence_roc))
            sequences.append(sequence)
            
            # Label is the vortex state at the end of the window
            labels.append(data.iloc[i + self.window_size - 1]['gt_detection_win'])
            
            # Store diagnostic info
            sequence_info.append({
                'type': 'vortex' if data.iloc[i + self.window_size - 1]['gt_detection_win'] == 1 else 'non_vortex',
                'start': i,
                'end': i + self.window_size,
                'pressure_range': (sequence_pressure.min(), sequence_pressure.max()),
                'roc_range': (sequence_roc.min(), sequence_roc.max())
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
        if not self.scaler_fitted:
            self.fit_scaler(X_train)
            
        # Scale all sequences using the same scaler
        X_train_scaled = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        # Build model
        self.model = self.build_model((self.window_size, 2))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(
                Path(__file__).parent.parent / 'models' / 'lstm_roc_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
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
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return self.model.predict(X_scaled)
    
    def predict_real_time(self, pressure_readings: np.ndarray) -> float:
        """Make real-time predictions on new pressure readings."""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before making predictions")
            
        if len(pressure_readings) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} pressure readings")
            
        # Get last window_size readings
        current_readings = pressure_readings[-self.window_size:]
        
        # Calculate rate of change
        pressure_roc = self.calculate_rate_of_change(current_readings)
        
        # Combine features
        sequence = np.column_stack((current_readings, pressure_roc))
        
        # Reshape for model input
        sequence = sequence.reshape(1, self.window_size, 2)
        
        # Scale and predict
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 2)).reshape(sequence.shape)
        return self.model.predict(sequence_scaled)[0][0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before evaluation")
            
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = average_precision_score(y_test, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

def main():
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Initialize model
    model = VortexLSTMModelROC(window_size=100)
    
    # Prepare sequences
    X, y, sequence_info = model.prepare_sequences(data)
    
    # Split into train, validation, and test sets (60/20/20)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("\nEvaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"PR AUC: {results['auc']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    
    plt.tight_layout()
    save_path = Path(__file__).parent.parent / 'results' / 'training_history_roc.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")

if __name__ == "__main__":
    main() 