# Small TCN Model for Bow Shock Crossing Event Detection

This is a **lightweight version** of the TCN model designed for **fast testing and troubleshooting**. It uses a small subset of data and optimized defaults to allow quick iteration during development.

## Differences from Full Model

- **Limited data**: Uses only 5 files from each directory by default (vs. all files)
- **Limited samples**: Maximum 100 sequences per file (vs. unlimited)
- **Smaller sequences**: Default sequence length 50 (vs. 100)
- **Larger stride**: Default stride 20 (vs. 10) - creates fewer overlapping windows
- **Smaller batch size**: Default 16 (vs. 32)
- **Smaller model**: Default channels [32, 64, 128, 64] (vs. [64, 128, 256, 128])
- **Fewer epochs**: Default 5 epochs (vs. 50)
- **Faster training**: Complete in minutes instead of hours

## Quick Start

### Training (Fast Test Run)

**Windows (PowerShell):**
```powershell
cd small_tcn_model
python train.py
```

**Windows (Command Prompt):**
```cmd
cd small_tcn_model
python train.py
```

This will use optimized defaults:
- Only 5 true_processed files and 50 false_processed files
- Maximum 100 sequences per file
- Sequence length 50 (vs. 100)
- Stride 20 (vs. 10) - fewer overlapping windows
- Batch size 16 (vs. 32)
- Smaller model architecture
- Train for 5 epochs
- Complete in minutes instead of hours

### Adjusting Data Size

You can control how much data to use:

```powershell
# Use even fewer files (ultra-fast)
python train.py --max-files 3 --max-samples-per-file 50

# Use 10 files (still fast, more data)
python train.py --max-files 10 --max-samples-per-file 200 --epochs 5

# Use 20 files (slower but more representative)
python train.py --max-files 20 --max-samples-per-file 200 --epochs 10

# Use all files (same as full model)
python train.py --max-files None --max-samples-per-file None --sequence-length 100 --stride 10 --batch-size 32 --epochs 50
```

## All Arguments

The training script supports all the same arguments as the full model, with optimized defaults:

- `--true-data-dir`: Directory with true processed files (default: `../data/true_processed`)
- `--false-data-dir`: Directory with false processed files (default: `../data/false_processed`)
- `--labels-dir`: Directory with YAML labels (default: `../data/zenodo-3946033/crossings/labels/all`)
- `--max-files`: Limit number of files (default: 5 for fast testing)
- `--max-samples-per-file`: Limit sequences per file (default: 100 for fast testing)
- `--epochs`: Number of epochs (default: 5 for fast testing)
- `--batch-size`: Batch size (default: 16 for fast testing, vs. 32 for full)
- `--sequence-length`: Sequence length (default: 50 for fast testing, vs. 100 for full)
- `--stride`: Stride for sliding window (default: 20 for fast testing, vs. 10 for full)
- `--num-channels`: Model channels (default: `32 64 128 64` for fast testing, vs. `64 128 256 128` for full)
- `--lr`: Learning rate (default: 0.001)
- `--window-size`: Window size around events (default: 5)

## When to Use This vs. Full Model

**Use `small_tcn_model` when:**
- Testing code changes
- Debugging data loading issues
- Quick validation of model architecture
- Troubleshooting YAML label matching
- Learning how the training works
- Need results in minutes, not hours

**Use `tcn_model` when:**
- Training the final model
- Need best performance
- Running production training
- Full dataset evaluation

## Notes

- The model architecture is smaller but similar to the full model
- Checkpoints are saved to `small_tcn_model/checkpoints/`
- All other functionality (inference, etc.) works the same way
- Defaults are optimized for speed, not accuracy - use full model for production
