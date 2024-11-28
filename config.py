from dataclasses import dataclass, MISSING


@dataclass
class Config:
    model: str = MISSING
    seed: int = 42
    unit: str = "VG5"
    operating_mode: str = "turbine"
    equilibrium: bool = True
    window_size: int = 64
    batch_size: int = 256
    epochs: int = 500
    learning_rate: float = 1e-3
    validation_split: float = 0.2


@dataclass
class ConvAEConfig(Config):
    model: str = "ConvAE"


@dataclass
class SimpleAEConfig(Config):
    model: str = "SimpleAE"
    window_size: int = 1
