# Core Pipeline Order

Use this order for all new runs.

1. Split only  
   `python "core pipeline scripts/split_data.py"`
2. Extract positive windows only  
   `python "core pipeline scripts/extract_windows.py" --split all --window_size 60`
3. Generate negatives / balanced datasets  
   `python "core pipeline scripts/negative_sampling.py" --split train --ratio 1.0 --window_size 60 --buffer 50`
4. Engineer features  
   `python "core pipeline scripts/feature_engineering.py" --split train --window_size 60`
5. Train/evaluate model scripts

## Notes

- `temporal_splits.py` is deprecated for new experiments; use `split_data.py`.
- Do not mix split generators across experiments.
- Keep one split policy per experiment for fair validation/test comparison.
