from config import Config

import torch
import torch.nn as nn
from itertools import chain


class SimpleAE(nn.Module):
    def __init__(self, input_channels: int, cfg: Config) -> None:
        super(SimpleAE, self).__init__()

        sizes = [input_channels, 64, 32, 16]

        self.encoder = nn.Sequential(
            *chain.from_iterable(
                [
                    (nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1d(sizes[i + 1]), nn.ELU(), nn.Dropout(0.1))
                    for i in range(len(sizes) - 2)
                ]
            ),
            nn.Linear(sizes[-2], sizes[-1]),
        )

        self.decoder = nn.Sequential(
            *chain.from_iterable(
                [
                    (nn.Linear(sizes[i], sizes[i - 1]), nn.BatchNorm1d(sizes[i - 1]), nn.ELU())
                    for i in range(len(sizes) - 1, 1, -1)
                ]
            ),
            nn.Linear(sizes[1], sizes[0]),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x.squeeze(-1))
        output = self.decoder(latent).unsqueeze(-1)
        return output, latent


class ConvAE(nn.Module):
    def __init__(self, input_channels: int, cfg: Config) -> None:
        super(ConvAE, self).__init__()

        kernel_size = 5
        channels = [16, 32, 64, 64]
        latent_features = 128
        padding = kernel_size // 2

        def conv_layer(in_channels, out_channels):
            return (
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

        def conv_transpose_layer(in_channels, out_channels):
            return (
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

        self.encoder = nn.Sequential(
            *conv_layer(input_channels, channels[0]),
            *chain.from_iterable(conv_layer(channels[i - 1], channels[i]) for i in range(1, len(channels))),
            nn.Flatten(),
            nn.Linear(channels[-1] * cfg.window_size, latent_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, channels[2] * cfg.window_size),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(channels[2], cfg.window_size)),
            *chain.from_iterable(
                conv_transpose_layer(channels[i], channels[i - 1]) for i in range(len(channels) - 1, 0, -1)
            ),
            nn.ConvTranspose1d(channels[0], input_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent
