import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def create_temporal_features(sequence):
    """Create features from pressure difference sequence."""
    return np.array([
        np.mean(sequence),           # average pressure change
        np.std(sequence),            # variability
        np.min(sequence),            # maximum pressure drop
        np.sum(sequence < 0),        # number of pressure drops
        np.mean(sequence[-10:]),     # recent pressure trend
        np.polyfit(range(len(sequence)), sequence, 1)[0]  # slope
    ])

def load_and_prepare_data(file_path, window_size=50):
    """Load data and prepare sequences for vortex detection."""
    print("Loading data...")
    start_time = time.time()
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Calculate pressure differences
    df['pressure_diff'] = df['PRESSURE'].diff()
    
    # Get vortex events
    vortex_events = df[df['gt_fwhm'] > 0].index
    
    # Prepare features and labels
    X = []
    y = []
    
    # Process vortex events
    print("Processing vortex events...")
    for idx in tqdm(vortex_events):
        if idx >= window_size:
            sequence = df['pressure_diff'].iloc[idx-window_size:idx].values
            features = create_temporal_features(sequence)
            X.append(features)
            y.append(1)
    
    # Process non-vortex events (equal number)
    print("Processing non-vortex events...")
    non_vortex_indices = df[df['gt_fwhm'] == 0].index.tolist()  # Convert to list
    np.random.shuffle(non_vortex_indices)
    
    count = 0
    for idx in non_vortex_indices:
        if count >= len(vortex_events):
            break
        if idx >= window_size:
            sequence = df['pressure_diff'].iloc[idx-window_size:idx].values
            features = create_temporal_features(sequence)
            X.append(features)
            y.append(0)
            count += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
    print(f"Total sequences: {len(X)}")
    print(f"Vortex events: {sum(y == 1)}")
    print(f"Non-vortex events: {sum(y == 0)}")
    
    return X, y

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, threshold=0.5):
    """Train and evaluate a model with given threshold."""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate energy efficiency metrics
    total_samples = len(y_test)
    true_vortices = sum(y_test == 1)
    predicted_vortices = sum(y_pred == 1)
    true_positives = sum((y_pred == 1) & (y_test == 1))
    false_positives = sum((y_pred == 1) & (y_test == 0))
    
    # Energy efficiency metrics
    energy_saved = (total_samples - predicted_vortices) / total_samples  # % of time in low energy mode
    data_quality = true_positives / true_vortices  # % of vortices caught
    
    print(f"\n{model_name} Results:")
    print(f"Precision: {report['1']['precision']:.3f}")
    print(f"Recall: {report['1']['recall']:.3f}")
    print(f"F1-score: {report['1']['f1-score']:.3f}")
    print(f"Energy Saved: {energy_saved:.2%}")
    print(f"Data Quality: {data_quality:.2%}")
    
    return model, y_pred, y_pred_proba, report

def plot_comparison(balanced_results, conservative_results):
    """Plot comparison of both models' performance."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. ROC Curve
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(balanced_results['fpr'], balanced_results['tpr'], label='Balanced')
    plt.plot(conservative_results['fpr'], conservative_results['tpr'], label='Conservative')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 2. Precision-Recall Curve
    plt.subplot(2, 2, 2)
    plt.plot(balanced_results['recall'], balanced_results['precision'], label='Balanced')
    plt.plot(conservative_results['recall'], conservative_results['precision'], label='Conservative')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # 3. Energy Efficiency Comparison
    plt.subplot(2, 2, 3)
    models = ['Balanced', 'Conservative']
    energy_saved = [balanced_results['energy_saved'], conservative_results['energy_saved']]
    data_quality = [balanced_results['data_quality'], conservative_results['data_quality']]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, energy_saved, width, label='Energy Saved')
    plt.bar(x + width/2, data_quality, width, label='Data Quality')
    plt.xlabel('Model')
    plt.ylabel('Percentage')
    plt.title('Energy Efficiency vs Data Quality')
    plt.xticks(x, models)
    plt.legend()
    
    # 4. Confusion Matrices
    plt.subplot(2, 2, 4)
    cm_balanced = balanced_results['confusion_matrix']
    cm_conservative = conservative_results['confusion_matrix']
    
    plt.imshow(np.vstack([cm_balanced, cm_conservative]), cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrices (Balanced above, Conservative below)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Parameters
    window_size = 50
    
    # Load and prepare data
    X, y = load_and_prepare_data('ml_ready_vortex_data.csv', window_size=window_size)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate balanced model
    balanced_model, y_pred_balanced, y_pred_proba_balanced, balanced_report = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test, "Balanced Model", threshold=0.5
    )
    
    # Find threshold for conservative model (target precision > 0.90)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_balanced)
    conservative_threshold = thresholds[np.argmax(precision >= 0.90)]
    
    # Train and evaluate conservative model
    conservative_model, y_pred_conservative, y_pred_proba_conservative, conservative_report = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test, "Energy-Conservative Model", 
        threshold=conservative_threshold
    )
    
    # Calculate ROC and PR curves for both models
    balanced_results = {
        'fpr': balanced_report['1']['fpr'],
        'tpr': balanced_report['1']['tpr'],
        'precision': balanced_report['1']['precision'],
        'recall': balanced_report['1']['recall'],
        'energy_saved': 1 - sum(y_pred_balanced == 1) / len(y_test),
        'data_quality': balanced_report['1']['recall'],
        'confusion_matrix': confusion_matrix(y_test, y_pred_balanced)
    }
    
    conservative_results = {
        'fpr': conservative_report['1']['fpr'],
        'tpr': conservative_report['1']['tpr'],
        'precision': conservative_report['1']['precision'],
        'recall': conservative_report['1']['recall'],
        'energy_saved': 1 - sum(y_pred_conservative == 1) / len(y_test),
        'data_quality': conservative_report['1']['recall'],
        'confusion_matrix': confusion_matrix(y_test, y_pred_conservative)
    }
    
    # Plot comparison
    plot_comparison(balanced_results, conservative_results)
    
    # Plot feature importances for both models
    feature_names = ['Mean Pressure Change', 'Pressure Variability', 
                    'Max Pressure Drop', 'Number of Drops',
                    'Recent Trend', 'Overall Slope']
    
    for model_name, model in [('Balanced', balanced_model), ('Conservative', conservative_model)]:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name} Model")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'feature_importances_{model_name.lower()}.png')
        plt.close()

if __name__ == "__main__":
    main() 