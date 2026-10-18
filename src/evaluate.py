from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from model import CompressorAutoencoder


project_folder = Path(__file__).resolve().parent.parent
csv_path = (
    project_folder
    / "data"
    / "raw"
    / "metropt+3+dataset"
    / "MetroPT3(AirCompressor).csv"
)
output_folder = project_folder / "outputs"

data = pd.read_csv(csv_path, parse_dates=["timestamp"])

# Load the preprocessing settings saved during training.
preprocessor = joblib.load(output_folder / "preprocessor_cpu.joblib")
continuous_columns = preprocessor["continuous_columns"]
binary_columns = preprocessor["binary_columns"]
feature_columns = continuous_columns + binary_columns
scaler = preprocessor["scaler"]

# Keep February for training-error comparison; later months are the test set.
train_mask = data["timestamp"] < pd.Timestamp("2020-03-01")
test_mask = data["timestamp"] >= pd.Timestamp("2020-03-01")

X_train_cont_raw = data.loc[train_mask, continuous_columns].to_numpy(dtype="float32")
X_train_binary = data.loc[train_mask, binary_columns].to_numpy(dtype="float32")

X_test_cont_raw = data.loc[test_mask, continuous_columns].to_numpy(dtype="float32")
X_test_binary = data.loc[test_mask, binary_columns].to_numpy(dtype="float32")

# Scale continuous sensors using February's settings; leave binary signals as 0/1.
X_train_cont = scaler.transform(X_train_cont_raw).astype("float32")
X_test_cont = scaler.transform(X_test_cont_raw).astype("float32")

X_train = np.concatenate([X_train_cont, X_train_binary], axis=1)
X_test = np.concatenate([X_test_cont, X_test_binary], axis=1)

# Load the updated model trained with this preprocessing.
model = CompressorAutoencoder()
model.load_state_dict(
    torch.load(output_folder / "autoencoder_cpu.pth", map_location="cpu")
)
model.eval()


def get_errors(values):
    """Calculate one reconstruction-error score for each reading."""
    errors = np.empty(len(values), dtype="float32")
    batch_size = 8192

    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            batch = torch.from_numpy(values[start:end])
            reconstructed = model(batch)
            errors[start:end] = ((batch - reconstructed) ** 2).mean(dim=1).numpy()

    return errors


train_errors = get_errors(X_train)

start_time = time.perf_counter()
test_errors = get_errors(X_test)
test_scoring_time = time.perf_counter() - start_time

# Start with the top 1% of February training errors as the cutoff.
threshold = np.quantile(train_errors, 0.99)
flagged = test_errors > threshold

print(f"February 99th-percentile threshold: {threshold:.6f}")
print(f"Later readings flagged: {flagged.sum()} out of {len(test_errors)}")
print(f"Percentage flagged: {flagged.mean() * 100:.2f}%")
print(f"CPU test-scoring time: {test_scoring_time:.2f} seconds")

# Show the original sensor values for the 10 highest-error later readings.
test_rows = data.loc[test_mask, ["timestamp"] + feature_columns].copy()
test_rows = test_rows.reset_index(drop=True)
test_rows["reconstruction_error"] = test_errors

top_rows = test_rows.nlargest(10, "reconstruction_error")
print("\nRaw sensor values for the 10 highest-error readings:")
print(top_rows.to_string(index=False))

# Inspect which features contributed most to the single highest-error reading.
worst_position = int(np.argmax(test_errors))
worst_input = torch.from_numpy(X_test[worst_position : worst_position + 1])

with torch.no_grad():
    worst_reconstruction = model(worst_input).numpy()[0]

worst_row = data.loc[test_mask].iloc[worst_position]

details = pd.DataFrame(
    {
        "sensor": feature_columns,
        "raw_value": worst_row[feature_columns].to_numpy(dtype="float32"),
        "scaled_value": X_test[worst_position],
        "reconstructed_scaled": worst_reconstruction,
        "squared_error": (X_test[worst_position] - worst_reconstruction) ** 2,
    }
)

print("\nFeatures contributing most to the worst reading's error:")
print(details.nlargest(5, "squared_error").to_string(index=False))

# Match timestamps to the test errors
test_timestamps = data.loc[data["timestamp"] >= "2020-03-01", "timestamp"].reset_index(
    drop=True
)

failure_windows = [
    ("April 18 air leak", "2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("May 29–30 air leak", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("June 5–7 air leak", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("July 15 air leak", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]

print("\nAlerts during documented air-leak periods:")

for name, start, end in failure_windows:
    in_window = (test_timestamps >= start) & (test_timestamps <= end)
    errors_in_window = test_errors[in_window.to_numpy()]

    flagged = errors_in_window > threshold
    count = len(errors_in_window)
    flagged_count = flagged.sum()
    percentage = 100 * flagged_count / count if count else 0

    print(f"{name}: {flagged_count} of {count} readings flagged ({percentage:.2f}%)")

monthly_results = pd.DataFrame(
    {
        "month": test_timestamps.dt.to_period("M"),
        "flagged": test_errors > threshold,
    }
)

monthly_summary = monthly_results.groupby("month").agg(
    readings=("flagged", "size"),
    flagged_readings=("flagged", "sum"),
)

monthly_summary["flagged_percentage"] = (
    monthly_summary["flagged_readings"] / monthly_summary["readings"] * 100
)

print("\nAlerts by month:")
print(monthly_summary.round(2))

continuous_sensors = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

data["month"] = data["timestamp"].dt.to_period("M")

print("\nMonthly average sensor values:")
print(data.groupby("month")[continuous_sensors].mean().round(2))

continuous_sensors = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

data["month"] = data["timestamp"].dt.to_period("M")

print("\nMonthly sensor variation (standard deviation):")
print(data.groupby("month")[continuous_sensors].std().round(2))
