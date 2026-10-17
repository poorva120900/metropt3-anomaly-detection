from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

project_folder = Path(__file__).resolve().parent.parent
csv_path = (
    project_folder
    / "data"
    / "raw"
    / "metropt+3+dataset"
    / "MetroPT3(AirCompressor).csv"
)

data = pd.read_csv(csv_path)
data["timestamp"] = pd.to_datetime(data["timestamp"])

print("Number of rows:", len(data))
print("Date range:", data["timestamp"].min(), "to", data["timestamp"].max())
print("Dataset size (rows, columns):", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nMissing values per column:")
print(data.isna().sum())
sensor_data = data.drop(columns=["Unnamed: 0", "timestamp"])
print(sensor_data.describe().T[["min", "max", "mean", "std"]])
print("Sorted by time:", data["timestamp"].is_monotonic_increasing)
print("\nMost common gaps between readings:")
print(data["timestamp"].diff().value_counts().head())

split_date = pd.Timestamp("2020-03-01")

train_data = data[data["timestamp"] < split_date]
test_data = data[data["timestamp"] >= split_date]

print(
    "Training period:",
    train_data["timestamp"].min(),
    "to",
    train_data["timestamp"].max(),
)
print("Training rows:", len(train_data))
print(
    "Testing period:", test_data["timestamp"].min(), "to", test_data["timestamp"].max()
)
print("Testing rows:", len(test_data))

gap_report = pd.DataFrame(
    {"timestamp": data["timestamp"], "gap": data["timestamp"].diff()}
)

print("\nLargest gaps between readings:")
print(gap_report.nlargest(10, "gap").to_string(index=False))

excluded_columns = ["Unnamed: 0", "timestamp"]
feature_columns = [column for column in data.columns if column not in excluded_columns]

X_train = train_data[feature_columns].to_numpy(dtype="float32")
X_test = test_data[feature_columns].to_numpy(dtype="float32")

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype("float32")
X_test_scaled = scaler.transform(X_test).astype("float32")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)
