from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler


class SlidingDataset(Dataset):
    def __init__(
        self,
        unit: Literal["VG4", "VG5", "VG6"],
        dataset_type: Literal["training", "testing_synthetic_01", "testing_synthetic_02", "testing_real"],
        operating_mode: Literal["turbine", "pump", "short_circuit"],
        equilibrium: bool = True,
        dataset_folder: str = "Dataset",
        window_size: int = 50,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        assert window_size >= 1
        self.window_size = window_size
        self.device = device

        # Load file
        pq_path = f"{dataset_folder}/{unit}_generator_data_{dataset_type}_measurements.parquet"
        df = pd.read_parquet(pq_path)

        # Filter operating mode
        if equilibrium:
            df = df[df[f"equilibrium_{operating_mode}_mode"] & ~df["dyn_only_on"]]
        else:
            df = df[df[f"{operating_mode}_mode"]]

        # Remove operating mode variables from data
        operating_mode_vars = [
            "machine_on",
            "machine_off",
            "turbine_mode",
            "equilibrium_turbine_mode",
            "pump_mode",
            "equilibrium_pump_mode",
            "short_circuit_mode",
            "equilibrium_short_circuit_mode",
            "dyn_only_on",
            "all",
        ]
        df.drop(columns=operating_mode_vars, inplace=True)

        # Aggregate injector openings
        injector_columns_mask = df.columns.to_series().str.match("injector_0[1-9]_opening")
        total_injector_opening = df.loc[:, injector_columns_mask].sum(axis=1)
        df.drop(columns=df.columns[injector_columns_mask], inplace=True)
        df["total_injector_opening"] = total_injector_opening

        # Compute subsequence indices
        valid_end_indices = (
            df.index.to_series()
            .diff(periods=self.window_size - 1)
            .eq((self.window_size - 1) * pd.Timedelta(seconds=30))
        )
        start_indices = np.nonzero(valid_end_indices)[0] - self.window_size + 1
        self.start_indices = torch.from_numpy(start_indices)

        # Convert to tensor and normalize
        self.measurements = torch.from_numpy(df.to_numpy().astype(np.float32)).to(device)
        self.measurements = self.measurements.T
        mean = self.measurements.mean(dim=1, keepdim=True)
        std = self.measurements.std(dim=1, keepdim=True)
        self.measurements = torch.where(std > 0, (self.measurements - mean) / std, self.measurements)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> torch.tensor:
        start_index = self.start_indices[idx]
        return self.measurements[:, start_index : start_index + self.window_size]


def create_dataloaders(
    dataset: Dataset, batch_size: int = 256, validation_split: float = 0.2
) -> tuple[DataLoader, DataLoader]:
    """Creates train/validation DataLoaders"""

    num_samples = len(dataset)
    validation_size = int(validation_split * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    # FIXME
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(validation_indices))

    return train_loader, validation_loader
