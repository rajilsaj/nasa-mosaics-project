# TCN Model for Bow Shock Crossing Event Detection

This directory contains a Temporal Convolutional Network (TCN) model for detecting bow shock crossing events in Cassini spacecraft plasma spectrometer data.

## Overview

The model uses a **hybrid labeling approach** to learn to detect bow shock events:

1. **true_processed data**: Files that contain bow shock events
   - Uses YAML label files to mark specific time points where events occur
   - Only sequences around YAML-labeled event times are positive (1)
   - Other sequences in these files are negative (0)

2. **false_processed data**: Files that do NOT contain bow shock events
   - All sequences are labeled as negative (0)
   - Provides negative examples to help the model learn what "no event" looks like

The model takes spectrogram data (time × energy bins) as input and predicts whether a bow shock crossing event occurs at each time step. Bow shock events are identified as the second change point in the YAML label files (index 1).

## Files

- `tcn_model.py`: TCN model architecture
- `data_loader.py`: Data loading and preprocessing utilities
- `train.py`: Training script
- `inference.py`: Inference script for prediction and visualization

## Installation

### Option 1: Install from requirements.txt

**Windows (PowerShell):**
```powershell
cd tcn_model
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
cd tcn_model
pip install -r requirements.txt
```

### Option 2: Install PyTorch manually

If you have network issues, install PyTorch first:

**Windows (PowerShell/Command Prompt):**
```cmd
# For CPU only (recommended for most users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA GPU support (if you have an NVIDIA GPU)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install other dependencies:

```cmd
pip install numpy pandas xarray matplotlib scikit-learn
```

### Option 3: Using a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
cd tcn_model

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
cd tcn_model

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

**Note:** If you're using a virtual environment, make sure to activate it before running training or inference scripts.

### Troubleshooting Installation Issues

#### Network/DNS Issues

If you get errors like "nodename nor servname provided" or "Unknown host":
1. **Check internet connection** - Make sure you're connected to the internet
2. **Run network diagnostic script** - Diagnose network issues:
   ```cmd
   REM Windows Command Prompt
   diagnose_network.bat
   
   REM Windows PowerShell
   .\diagnose_network.ps1
   ```
3. **Try a different network** - Switch Wi-Fi or use mobile hotspot
4. **Use conda instead** - Conda may work even if pip doesn't:
   ```cmd
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   conda install numpy pandas matplotlib scikit-learn
   pip install xarray
   ```
5. **Or run the conda install script:**
   ```cmd
   REM Windows Command Prompt
   install_with_conda.bat
   
   REM Windows PowerShell
   .\install_with_conda.ps1
   ```

#### Python Version Issues

Python 3.14.0 is very new and PyTorch may not have wheels for it yet. PyTorch supports Python 3.8-3.12. If you have issues:
1. Use Python 3.11 or 3.12 (recommended)
2. Or use conda which manages Python versions automatically

#### Verify Installation

After installing, verify it works:
```cmd
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
```

## Usage

### Training

The model uses a hybrid approach:
- **true_processed**: Uses YAML labels to mark specific time points where bow shock events occur
- **false_processed**: All sequences are treated as negative (no bow shock events)

Train the model:

**Windows (PowerShell):**
```powershell
python train.py `
    --true-data-dir ..\data\true_processed `
    --false-data-dir ..\data\false_processed `
    --labels-dir ..\data\zenodo-3946033\crossings\labels\all `
    --output-dir checkpoints `
    --sequence-length 100 `
    --batch-size 32 `
    --epochs 50 `
    --lr 0.001 `
    --window-size 5
```

**Windows (Command Prompt):**
```cmd
python train.py ^
    --true-data-dir ..\data\true_processed ^
    --false-data-dir ..\data\false_processed ^
    --labels-dir ..\data\zenodo-3946033\crossings\labels\all ^
    --output-dir checkpoints ^
    --sequence-length 100 ^
    --batch-size 32 ^
    --epochs 50 ^
    --lr 0.001 ^
    --window-size 5
```

**Note:** If you don't specify paths, the script will use relative paths based on the project structure.

**Key arguments:**
- `--true-data-dir`: Directory containing true processed NetCDF files (with bow shock events)
- `--false-data-dir`: Directory containing false processed NetCDF files (no bow shock events)
- `--labels-dir`: Directory containing YAML label files (for true_processed files)
- `--output-dir`: Directory to save model checkpoints
- `--sequence-length`: Length of input sequences (default: 100)
- `--stride`: Stride for sliding window (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--num-channels`: Number of channels in each TCN layer (default: 64 128 256 128)
- `--dropout`: Dropout probability (default: 0.2)
- `--window-size`: Number of time steps around YAML event to label as positive (default: 5)

### Inference

Run inference on new data:

**Windows (PowerShell):**
```powershell
python inference.py `
    --model checkpoints\best_model.pt `
    --input ..\data\true_processed\2004\153_182_JUN\ELS_200418000_V01.nc `
    --output predictions `
    --threshold 0.5
```

**Windows (Command Prompt):**
```cmd
python inference.py ^
    --model checkpoints\best_model.pt ^
    --input ..\data\true_processed\2004\153_182_JUN\ELS_200418000_V01.nc ^
    --output predictions ^
    --threshold 0.5
```

Or process a directory of files:

**Windows (PowerShell):**
```powershell
python inference.py `
    --model checkpoints\best_model.pt `
    --input ..\data\true_processed `
    --output predictions
```

**Windows (Command Prompt):**
```cmd
python inference.py ^
    --model checkpoints\best_model.pt ^
    --input ..\data\true_processed ^
    --output predictions
```

Key arguments:
- `--model`: Path to model checkpoint
- `--input`: Path to input NetCDF file or directory
- `--output`: Output directory for plots (optional, if not provided plots are displayed)
- `--threshold`: Classification threshold (default: 0.5)
- `--sequence-length`: Length of input sequences (should match training)
- `--stride`: Stride for sliding window (should match training)

## Model Architecture

The TCN model uses:
- Dilated causal convolutions for temporal modeling
- Residual connections for stable training
- Dropout for regularization
- Binary classification output (sigmoid activation)

Input shape: `(batch, sequence_length, num_energy_bins)`
Output shape: `(batch, sequence_length, 1)`

## Data Format

The model expects NetCDF files with:
- `count_rate`: Array of shape `(time, energy, anode)`
- `time`: Array of timestamps
- `energy`: Array of energy bin values

The data loader:
1. Averages over anode dimension: `(time, energy, anode) → (time, energy)`
2. Interpolates NaN values
3. Normalizes features
4. Creates sliding windows of specified length

## Label Format

### YAML Labels (for true_processed files)

Bow shock events are extracted from YAML label files:
- File format: `{filename}.yaml` in the labels directory
- Bow shock is the **second change_point** (index 1) in the `change_points` list
- The first change_point (index 0) is the Magnetopause
- If a file only has 1 change_point, it's treated as bow shock

Example YAML:
```yaml
change_points:
- 10-10-2005/06:10:00  # Magnetopause (index 0)
- 10-10-2005/06:40:00  # Bow Shock (index 1) ← This is what we detect
- 10-10-2005/11:05:00
```

The model labels `window_size` time steps around each bow shock event time as positive (1).

### Directory-Based Labels (for false_processed files)

All sequences from `false_processed` files are labeled as negative (0), providing examples of periods without bow shock events.

## Output

### Training

The training script saves:
- `best_model.pt`: Best model based on validation F1 score
- `final_model.pt`: Final model after all epochs
- `training_history.json`: Training and validation metrics

### Inference

The inference script generates:
- Visualization plots showing:
  - Spectrogram with ground truth bow shock events (if available)
  - Predicted probabilities and binary predictions
  - Threshold line

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

These metrics are computed at the sequence level (per time step).

## Notes

- The model uses a sliding window approach, so predictions may overlap
- Predictions are aggregated using majority voting across overlapping windows
- The window_size parameter controls how many time steps around a true event are labeled as positive
- Normalization statistics are computed from the training data

