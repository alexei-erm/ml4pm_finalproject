from dataclasses import dataclass


@dataclass
class ConvAEConfig:
    seed: int = 42
    window_size: int = 64
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-3


@dataclass
class SimpleAEConfig(ConvAEConfig):
    window_size: int = 1
