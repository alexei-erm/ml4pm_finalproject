from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    model: str = "ConvAE"
    unit: str = "VG5"
    operating_mode: str = "turbine"
    equilibrium: bool = True
    window_size: int = 64
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-3
    validation_split: float = 0.2


@dataclass
class SimpleAEConfig(Config):
    model: str = "SimpleAE"
    window_size: int = 1

@dataclass
class LSTM_VAEConfig(Config):
    model: str = "LSTM_VAE"
    seq_len: int = 64  # Sequence length for LSTM
    latent_dim: int = 16  # Latent space dimensions
    hidden_dim: int = 128  # Hidden size of LSTM
    num_layers: int = 2  # Number of LSTM layers