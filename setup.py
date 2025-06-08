# Setup.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd 
import matplotlib.pyplot as plt

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


# Train-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=False)
# Model
model = XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Evaluet the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Visual Debugging

plt.figure(figsize=(12, 5))
plt.plot(df["pressure_diff"].values, label="Pressure Diff")
plt.scatter(df.index[df["gt_detection_win"]], df["pressure_diff"][df["gt_detection_win"]], color="red", label="Actual Vortex")
plt.scatter(df.index[-len(y_pred):][y_pred == 1], df["pressure_diff"].iloc[-len(y_pred):][y_pred == 1], color="green", label="Predicted Vortex")
plt.legend()
plt.show()

