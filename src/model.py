from torch import nn


class CompressorAutoencoder(nn.Module):
    def __init__(self, input_size=15):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

    def forward(self, values):
        compressed = self.encoder(values)
        reconstructed = self.decoder(compressed)
        return reconstructed
