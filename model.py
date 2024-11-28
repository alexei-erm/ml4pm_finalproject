import torch
import torch.nn as nn
from itertools import chain


import torch
import torch.nn as nn
from itertools import chain

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels: int, input_length: int) -> None:
        super(ConvolutionalAutoencoder, self).__init__()
        
        kernel_size = 5
        channels = [128, 200, 256]
        latent_features = 512
        padding = kernel_size // 2

        def conv_block(in_channels, out_channels):
            return (
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )

        def conv_transpose_block(in_channels, out_channels, is_last=False):
            return (
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2) if not is_last else nn.Identity()
            )

        # Calculate compressed length after maxpooling
        compressed_length = input_length // (2 ** len(channels))
        
        self.encoder = nn.Sequential(
            *conv_block(input_channels, channels[0]),
            *chain.from_iterable(conv_block(channels[i-1], channels[i]) 
                               for i in range(1, len(channels))),
            nn.Flatten(),
            nn.Linear(channels[-1] * compressed_length, latent_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, channels[-1] * compressed_length),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(channels[-1], compressed_length)),
            *chain.from_iterable(conv_transpose_block(channels[i], channels[i-1])
                               for i in range(len(channels)-1, 0, -1)),
            *conv_transpose_block(channels[0], input_channels, is_last=True)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def get_latent(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

def correlation_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Computes correlation matrix difference loss"""
    def compute_corr_matrix(x):
        x_centered = x - x.mean(dim=2, keepdim=True)
        std = torch.sqrt(torch.sum(x_centered ** 2, dim=2, keepdim=True))
        x_normalized = x_centered / (std + 1e-8)
        corr = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        return corr / x.size(2)

    target_corr = compute_corr_matrix(target)
    pred_corr = compute_corr_matrix(pred)
    loss = torch.norm(target_corr - pred_corr, p='fro', dim=(1,2))
    return loss.mean()

def combined_loss(target: torch.Tensor, pred: torch.Tensor, lambda_corr: float = 0.1) -> torch.Tensor:
    """Combines MSE and correlation loss"""
    mse = nn.MSELoss()(target, pred)
    corr = correlation_loss(target, pred)
    return mse + lambda_corr * corr
