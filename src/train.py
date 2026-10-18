from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
output_folder.mkdir(exist_ok=True)

data = pd.read_csv(csv_path, parse_dates=["timestamp"])

continuous_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

binary_columns = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

train_mask = data["timestamp"] < pd.Timestamp("2020-03-01")

X_train_cont_raw = data.loc[train_mask, continuous_columns].to_numpy(dtype="float32")
X_train_binary = data.loc[train_mask, binary_columns].to_numpy(dtype="float32")

scaler = StandardScaler()
X_train_cont = scaler.fit_transform(X_train_cont_raw).astype("float32")

# Keep the on/off signals in their original 0/1 form.
X_train = np.concatenate([X_train_cont, X_train_binary], axis=1)

train_tensor = torch.from_numpy(X_train)
train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=1024,
    shuffle=True,
)

model = CompressorAutoencoder()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
start_time = time.perf_counter()

for epoch in range(epochs):
    total_loss = 0.0

    for (batch,) in train_loader:
        reconstructed = model(batch)
        loss = loss_function(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.6f}")

training_time = time.perf_counter() - start_time

torch.save(model.state_dict(), output_folder / "autoencoder_cpu.pth")
joblib.dump(
    {
        "scaler": scaler,
        "continuous_columns": continuous_columns,
        "binary_columns": binary_columns,
    },
    output_folder / "preprocessor_cpu.joblib",
)

print(f"\nCPU training time: {training_time:.2f} seconds")
print("Saved updated model and preprocessing settings.")
