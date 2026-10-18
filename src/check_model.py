import torch
from model import CompressorAutoencoder

model = CompressorAutoencoder()
sample = torch.randn(3, 15)
reconstructed = model(sample)

print("Input shape:", sample.shape)
print("Output shape:", reconstructed.shape)
