from dataclasses import dataclass


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


@dataclass
class SingleSampleAEConfig(Config):
    window_size: int = 1


@dataclass
class SingleChannelAEConfig(Config):
    window_size: int = 1024


@dataclass
class LSTMAEConfig(Config):
    window_size: int = 128
