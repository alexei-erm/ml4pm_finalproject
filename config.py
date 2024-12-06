from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class FullyConnectedAEConfig:
    hidden_sizes: list[int]
    dropout: float
    latent_sigmoid: bool


@dataclass
class ConvolutionalAEConfig:
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
    model: Literal["FullyConnectedAE", "ConvolutionalAE", "LSTMAE"]
    model_cfg: FullyConnectedAEConfig | ConvolutionalAEConfig | LSTMAEConfig
    features: list[str]
    seed: int = 42
    unit: str = "VG5"
    operating_mode: str = "turbine"
    transient: bool = False
    window_size: int = 256
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-3
    latent_kl_divergence: float = 0.0
    latent_l1: float = 0.0
    validation_split: float = 0.2
    subsampling: int = 1


def inherit(base_config: Any, **overrides: Any) -> Any:
    """Clone a base configuration and override specific parameters."""
    return base_config.__class__(**{**base_config.__dict__, **overrides})


CFG = {}
CFG["OneSample"] = Config(
    model="FullyConnectedAE",
    model_cfg=FullyConnectedAEConfig(hidden_sizes=[64, 32, 16, 8], dropout=0.2, latent_sigmoid=False),
    features=[],
    window_size=1,
)
CFG["OneSampleSparse"] = inherit(CFG["OneSample"], latent_l1=0.01)
