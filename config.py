from dataclasses import dataclass
from typing import Any
from enum import Enum


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
class LSTMForecasterConfig:
    hidden_size: int
    num_layers: int
    target_feature: str


class ModelType(Enum):
    FullyConnectedAE = "FullyConnectedAE"
    ConvolutionalAE = "ConvolutionalAE"
    LSTMAE = "LSTMAE"
    LSTMForecaster = "LSTMForecaster"
    SPC = "SPC"
    KPCA = "KPCA"


@dataclass
class Config:
    model: ModelType
    model_cfg: FullyConnectedAEConfig | ConvolutionalAEConfig | LSTMAEConfig | LSTMForecasterConfig | None
    features: list[str]
    seed: int = 42
    unit: str = "VG5"
    operating_mode: str = "turbine"
    transient: bool = False
    window_size: int = 256
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-4
    kl_divergence_weight: float = 0.0
    kl_divergence_rho: float = 0.05
    l1_weight: float = 0.0
    validation_split: float = 0.2
    subsampling: int = 1
    measurement_downsampling: int = 1


def inherit(base_config: Any, **overrides: Any) -> Any:
    """Clone a base configuration and override specific parameters."""
    return base_config.__class__(**{**base_config.__dict__, **overrides})


CFG: dict[str, Config] = {}

CFG["OneSample"] = Config(
    model=ModelType.FullyConnectedAE,
    model_cfg=FullyConnectedAEConfig(hidden_sizes=[64, 32, 16], dropout=0.1, latent_sigmoid=False),
    features=[],
    window_size=1,
    measurement_downsampling=16,
)
CFG["OneSampleSparse"] = inherit(CFG["OneSample"], l1_weight=0.1)


CFG["Conv"] = Config(
    model=ModelType.ConvolutionalAE,
    model_cfg=ConvolutionalAEConfig(
        channels=[4, 8, 16, 32],
        hidden_sizes=[16],
        kernel_size=7,
        max_pool_size=2,
        dropout=0.0,
        latent_sigmoid=False,
    ),
    features=["stat_coil_ph01_01_tmp"],
    window_size=32,
    measurement_downsampling=32,
)


CFG["LSTM"] = Config(
    model=ModelType.LSTMAE,
    model_cfg=LSTMAEConfig(hidden_size=64, num_layers=2, dropout=0.0, fc_hidden_sizes=[32], latent_sigmoid=False),
    features=["stat_coil_ph01_01_tmp"],
    window_size=32,
    measurement_downsampling=32,
)
CFG["LSTMSparse"] = inherit(
    CFG["LSTM"],
    l1_weight=2.0,
)
CFG["LSTMSimple"] = Config(
    model=ModelType.LSTMAE,
    model_cfg=LSTMAEConfig(hidden_size=16, num_layers=1, dropout=0.0, fc_hidden_sizes=[], latent_sigmoid=False),
    features=["stat_coil_ph01_01_tmp"],
    window_size=32,
    measurement_downsampling=32,
)


CFG["SPC"] = Config(
    model=ModelType.SPC,
    model_cfg=None,
    features=[],
    operating_mode="all",
    window_size=1,
    measurement_downsampling=1,
)


CFG["KPCAVG5"] = Config(
    model=ModelType.KPCA,
    model_cfg=None,
    features=[],
    operating_mode="all",
    window_size=1,
    measurement_downsampling=32,
    subsampling=10,
)
CFG["KPCAVG6"] = inherit(CFG["KPCAVG5"], unit="VG6")
CFG["KPCAVG4"] = inherit(CFG["KPCAVG5"], unit="VG4")
