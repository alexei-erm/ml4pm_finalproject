from config import *

import torch
import torch.nn as nn
from itertools import chain


class SingleSampleAE(nn.Module):
    def __init__(self, input_channels: int, cfg: Config) -> None:
        super(SingleSampleAE, self).__init__()

        sizes = [input_channels, 16, 8, 4]

        self.encoder = nn.Sequential(
            *chain.from_iterable(
                [
                    (nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1d(sizes[i + 1]), nn.ReLU())
                    for i in range(len(sizes) - 2)
                ]
            ),
            nn.Linear(sizes[-2], sizes[-1]),
        )

        self.decoder = nn.Sequential(
            *chain.from_iterable(
                [
                    (nn.Linear(sizes[i], sizes[i - 1]), nn.BatchNorm1d(sizes[i - 1]), nn.ReLU())
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


class SingleChannelAE(nn.Module):
    def __init__(self, input_channels: int, cfg: Config) -> None:
        super(SingleChannelAE, self).__init__()

        kernel_size = 5
        max_pool_size = 2
        # channels = [input_channels, 4, 8, 16, 32, 64, 128]
        channels = [input_channels, 32, 64, 128, 256, 512]
        latent_features = 256

        padding = kernel_size // 2
        conv_output_size = cfg.window_size // (max_pool_size ** (len(channels) - 1))

        self.encoder = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=max_pool_size, stride=max_pool_size),
                    nn.Dropout(0.1),
                )
                for i in range(0, len(channels) - 1)
            ),
            nn.Flatten(),
            nn.Linear(conv_output_size * channels[-1], latent_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, conv_output_size * channels[-1]),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(channels[-1], conv_output_size)),
            *chain.from_iterable(
                (
                    nn.Upsample(scale_factor=max_pool_size),
                    nn.ConvTranspose1d(channels[i], channels[i - 1], kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels[i - 1]),
                    nn.ReLU(),
                )
                for i in range(len(channels) - 1, 1, -1)
            ),
            nn.Upsample(scale_factor=max_pool_size),
            nn.ConvTranspose1d(channels[1], channels[0], kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class LSTMAE(nn.Module):
    def __init__(self, input_channels: int, cfg: Config) -> None:
        super(LSTMAE, self).__init__()

        hidden_size = 16
        num_layers = 2

        self.encoder_lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True)
        self.encoder_dropout = nn.Dropout(0.2)
        self.encoder_fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Sigmoid()

        self.decoder_fc = nn.Linear(hidden_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder_dropout = nn.Dropout(0.2)
        self.decoder_out_fc = nn.Linear(hidden_size, input_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.activation(self.encoder_fc(self.encoder_dropout(hidden[-1, ...])))

        decoder_input = self.decoder_fc(latent)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, x.shape[1], 1)
        reconstruction, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.decoder_out_fc(self.decoder_dropout(reconstruction))
        return reconstruction.permute(0, 2, 1), latent
