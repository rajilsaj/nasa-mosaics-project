# TCN Model 2.0 for Bow Shock Crossing Event Detection

This is the **full version** of the TCN model based on `small_tcn_model`, but **without constraints** for faster processing. It uses all available data and full model defaults for production-quality training.

## Differences from Small Model

- **No data limits**: Uses all files from each directory (vs. limited to 10 files)
- **No sample limits**: Uses all sequences from each file (vs. limited to 100 sequences per file)
- **Full sequences**: Default sequence length 100 (vs. 50)
- **Smaller stride**: Default stride 10 (vs. 20) - creates more overlapping windows for better coverage
- **Larger batch size**: Default 32 (vs. 16)
- **Full model**: Default channels [64, 128, 256, 128] (vs. [32, 64, 128, 64])
- **Full training**: Default 50 epochs (vs. 5)
- **Production quality**: Designed for final model training

## Features

This model includes all improvements from `small_tcn_model`:
- **Per-energy-bin normalization**: Normalizes each energy channel independently for better feature scaling
- **Log scaling**: Applies `log10(counts + 1)` to handle log-normally distributed count rates
- **Directory-based labeling**: Uses `true_processed` (positive) and `false_processed` (negative) directories for labels
- **No YAML dependency**: Simplified labeling based on directory structure
- **No date filtering**: Uses all available data from all time periods

## Quick Start

### Training (Full Model)

**Windows (PowerShell):**
```powershell
cd tcn_model_2.0
python train.py
```

**Windows (Command Prompt):**
```cmd
cd tcn_model_2.0
python train.py
```

This will use full model defaults:
- All true_processed files and false_processed files
- All sequences from each file
- Sequence length 100
- Stride 10 (more overlapping windows)
- Batch size 32
- Full model architecture [64, 128, 256, 128]
- Train for 50 epochs
- Complete training for production use

### Adjusting Training Parameters

You can still customize training if needed:

```powershell
# Use fewer epochs for testing
python train.py --epochs 10

# Use smaller batch size if memory is limited
python train.py --batch-size 16

# Use longer sequences
python train.py --sequence-length 150

# Limit files if needed (though not recommended for production)
python train.py --max-files 50 --max-samples-per-file 200
```

## All Arguments

The training script supports all standard arguments:

- `--true-data-dir`: Directory with true processed files (default: `../data/true_processed`)
- `--false-data-dir`: Directory with false processed files (default: `../data/false_processed`)
- `--output-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--sequence-length`: Sequence length (default: 100)
- `--stride`: Stride for sliding window (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout probability (default: 0.2)
- `--num-channels`: Model channels (default: `64 128 256 128`)
- `--train-split`: Fraction for training (default: 0.7)
- `--val-split`: Fraction for validation (default: 0.15)
- `--max-files`: Limit number of files (default: None, uses all)
- `--max-samples-per-file`: Limit sequences per file (default: None, uses all)
- `--device`: Device to use (default: auto)

## When to Use This vs. Small Model

**Use `tcn_model_2.0` when:**
- Training the final production model
- Need best possible performance
- Running full dataset evaluation
- Have sufficient computational resources
- Want to use all available data

**Use `small_tcn_model` when:**
- Testing code changes quickly
- Debugging data loading issues
- Quick validation of model architecture
- Need results in minutes, not hours
- Limited computational resources

## Notes

- The model architecture is the full production version
- Checkpoints are saved to `tcn_model_2.0/checkpoints/`
- All improvements from `small_tcn_model` are included (per-energy-bin normalization, log scaling, directory-based labeling)
- Training will take significantly longer than `small_tcn_model` but will produce better results
- Uses all available data from all time periods (no date filtering)
