from dataclasses import dataclass, MISSING


@dataclass
class BaseConfig:
    name: str = MISSING
    seed: int = 42
    window_size: int = 50
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-4


@dataclass
class SimpleAEConfig(BaseConfig):
    name: str = "SimpleAE"
