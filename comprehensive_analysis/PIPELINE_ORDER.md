# Comprehensive Pipeline Order

Use this order for all new runs.

1. Split only  
   `python "comprehensive_analysis/split_data.py"`
2. Extract positive windows only  
   `python "comprehensive_analysis/extract_windows.py" --split all --window_size 60`
3. Engineer positive-window features  
   `python "comprehensive_analysis/feature_engineering.py" --split train`
4. Add negative samples and create balanced feature set  
   `python "comprehensive_analysis/negative_sampling.py" --split train --ratio 1.0 --window_size 60 --buffer 50`
5. Train/evaluate model scripts

## Notes

- Keep this order: in this folder, negative sampling expects feature files.
- Use one split policy consistently across train/val/test.
- Re-run split stage only when intentionally changing split policy.
