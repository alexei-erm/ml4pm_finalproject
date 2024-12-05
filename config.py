from dataclasses import dataclass


@dataclass
class FullyConnectedAEConfig:
    hidden_sizes: list[int]
    dropout: float
    latent_sigmoid: bool


@dataclass
class ConvolutionalAEConfig:
    window_size: int
    channels: list[int]
    hidden_sizes: list[int]
    kernel_size: int
    max_pool_size: int
    dropout: float
    latent_sigmoid: bool


@dataclass
class LSTMAEConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    fc_hidden_sizes: list[int]
    latent_sigmoid: bool


@dataclass
class Config:
    model: str = "NONAME"
    seed: int = 42
    unit: str = "VG5"
    operating_mode: str = "turbine"
    equilibrium: bool = True
    features: list[str] | None = None
    window_size: int = 64
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    subsampling: int = 1
