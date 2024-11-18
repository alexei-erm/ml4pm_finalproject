import torch
import torch.nn as nn
from itertools import chain


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels: int, input_length: int) -> None:
        super(ConvolutionalAutoencoder, self).__init__()

        kernel_size = 7
        channels = [32, 32, 64, 64]
        latent_features = 128
        padding = kernel_size // 2

        def conv_layer(in_channels, out_channels):
            return (
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                # nn.Dropout(0.1),
            )

        def conv_transpose_layer(in_channels, out_channels):
            return (
                # nn.Dropout(0.1),
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

        self.encoder = nn.Sequential(
            *conv_layer(input_channels, channels[0]),
            *chain.from_iterable(conv_layer(channels[i - 1], channels[i]) for i in range(1, len(channels))),
            nn.Flatten(),
            nn.Linear(channels[-1] * input_length, latent_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, channels[2] * input_length),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(channels[2], input_length)),
            *chain.from_iterable(
                conv_transpose_layer(channels[i], channels[i - 1]) for i in range(len(channels) - 1, 0, -1)
            ),
            nn.ConvTranspose1d(channels[0], input_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def get_latent(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)
