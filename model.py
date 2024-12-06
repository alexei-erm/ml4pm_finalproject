from config import FullyConnectedAEConfig, ConvolutionalAEConfig, LSTMAEConfig

import torch
import torch.nn as nn
from itertools import chain


class FullyConnectedAE(nn.Module):
    def __init__(self, input_channels: int, window_size: int, cfg: FullyConnectedAEConfig) -> None:
        super(FullyConnectedAE, self).__init__()

        sizes = [input_channels * window_size] + cfg.hidden_sizes

        self.encoder = nn.Sequential(
            *chain.from_iterable(
                [
                    (nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1d(sizes[i + 1]), nn.ReLU())
                    + ((nn.Dropout(cfg.dropout),) if cfg.dropout > 0.0 else ())
                    for i in range(len(sizes) - 2)
                ]
            ),
            nn.Linear(sizes[-2], sizes[-1]),
            *((nn.Sigmoid(),) if cfg.latent_sigmoid else ()),
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


class ConvolutionalAE(nn.Module):
    def __init__(self, input_channels: int, window_size: int, cfg: ConvolutionalAEConfig) -> None:
        super(ConvolutionalAE, self).__init__()

        channels = [input_channels] + cfg.channels

        padding = cfg.kernel_size // 2
        conv_output_size = window_size // (cfg.max_pool_size ** len(cfg.channels))

        self.encoder = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=cfg.kernel_size, padding=padding),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=cfg.max_pool_size, stride=cfg.max_pool_size),
                )
                + ((nn.Dropout(cfg.dropout),) if cfg.dropout > 0.0 else ())
                for i in range(0, len(channels) - 1)
            ),
            nn.Flatten(),
            nn.Linear(conv_output_size * channels[-1], cfg.hidden_sizes[0]),
            *chain.from_iterable(
                (
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_sizes[i], cfg.hidden_sizes[i + 1]),
                )
                for i in range(0, len(cfg.hidden_sizes) - 1)
            ),
            *((nn.Sigmoid(),) if cfg.latent_sigmoid else ()),
        )

        self.decoder = nn.Sequential(
            *chain.from_iterable(
                (
                    nn.Linear(cfg.hidden_sizes[i], cfg.hidden_sizes[i - 1]),
                    nn.ReLU(),
                )
                for i in range(len(cfg.hidden_sizes) - 1, 0, -1)
            ),
            nn.Linear(cfg.hidden_sizes[0], conv_output_size * channels[-1]),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(channels[-1], conv_output_size)),
            *chain.from_iterable(
                (
                    nn.Upsample(scale_factor=cfg.max_pool_size),
                    nn.ConvTranspose1d(channels[i], channels[i - 1], kernel_size=cfg.kernel_size, padding=padding),
                    nn.BatchNorm1d(channels[i - 1]),
                    nn.ReLU(),
                )
                for i in range(len(channels) - 1, 1, -1)
            ),
            nn.Upsample(scale_factor=cfg.max_pool_size),
            nn.ConvTranspose1d(channels[1], channels[0], kernel_size=cfg.kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class LSTMAE(nn.Module):
    def __init__(self, input_channels: int, window_size: int, cfg: LSTMAEConfig) -> None:
        super(LSTMAE, self).__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_channels, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True
        )

        encoder_fc = [
            *((nn.Dropout(cfg.dropout),) if cfg.dropout > 0.0 else ()),
            *((nn.Linear(cfg.hidden_size, cfg.fc_hidden_sizes[0]),) if len(cfg.fc_hidden_sizes) > 0 else ()),
            *chain.from_iterable(
                (nn.ReLU(), nn.Linear(cfg.fc_hidden_sizes[i], cfg.fc_hidden_sizes[i + 1]))
                for i in range(0, len(cfg.fc_hidden_sizes) - 1)
            ),
            *((nn.Sigmoid(),) if cfg.latent_sigmoid else ()),
        ]
        if len(encoder_fc) > 0:
            self.encoder_fc = nn.Sequential(*encoder_fc)
        else:
            self.encoder_fc = None

        if len(cfg.fc_hidden_sizes) > 0:
            self.decoder_fc = nn.Sequential(
                *chain.from_iterable(
                    (nn.Linear(cfg.fc_hidden_sizes[i], cfg.fc_hidden_sizes[i - 1]), nn.ReLU())
                    for i in range(len(cfg.fc_hidden_sizes) - 1, 0, -1)
                ),
                nn.Linear(cfg.fc_hidden_sizes[0], cfg.hidden_size),
                nn.ReLU(),
            )
        else:
            self.decoder_fc = None

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.hidden_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True
        )
        self.decoder_out_fc = nn.Sequential(
            *((nn.Dropout(cfg.dropout),) if cfg.dropout > 0.0 else ()), nn.Linear(cfg.hidden_size, input_channels)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden[-1, ...]) if self.encoder_fc is not None else hidden[-1, ...]

        decoder_input = self.decoder_fc(latent) if self.decoder_fc is not None else latent
        decoder_input = decoder_input.unsqueeze(1).repeat(1, x.shape[1], 1)
        reconstruction, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.decoder_out_fc(reconstruction)
        return reconstruction.permute(0, 2, 1), latent
