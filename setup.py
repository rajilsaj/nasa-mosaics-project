# Setup.py

import pandas as pd 
df = pd.read_csv("data/ml_ready_vortex_data.csv")
# print(df.info())
# print(df.head())

df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
# print(df['pressure_diff'].head())
# Drop missing values 
df = df.dropna(subset=["pressure_diff"])  # Clean up
df["diff_std_50"] = df["pressure_diff"].rolling(50).std()

# Define X and Y
X = df[["pressure_diff"]]  # Add other features if desired
y = df["gt_detection_win"]  # Binary target


