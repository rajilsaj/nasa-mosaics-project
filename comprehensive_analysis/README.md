# Comprehensive Dataset Analysis Workspace

This folder contains analysis and modeling work using `comprehensive_filtered_data_optimized.csv`.

## Dataset Overview

- **Source**: `comprehensive_filtered_data_optimized.csv`
- **Size**: 1,694,934 samples
- **Time Span**: Sols 1-89 (88 sols)
- **Class Distribution**: 94.4:1 imbalance (1.05% vortex events)

## Unique Features

This dataset includes features not in the temporal splits:
- `autoencoder_window_hits`: Counter (0-60)
- `autoencoder_positive_hit`: Binary prediction (3.41% positive)
- `PRESSURE_MA_500`: 500-sample moving average

## Goals

1. Use comprehensive dataset as primary data source
2. Leverage autoencoder features for improved RF performance
3. Create temporal splits from comprehensive data
4. Build and evaluate Random Forest model

## Next Steps

- [ ] Create temporal splits from comprehensive data
- [ ] Extract windows with autoencoder features
- [ ] Engineer features including autoencoder signals
- [ ] Train Random Forest model
- [ ] Evaluate on natural imbalance

