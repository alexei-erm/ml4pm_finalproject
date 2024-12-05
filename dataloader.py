from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class SlidingDataset(Dataset):
    def __init__(
        self,
        parquet_file: str,
        operating_mode: Literal["turbine", "pump", "short_circuit"],
        equilibrium: bool,
        features: list[str] | None,
        window_size: int,
        device: torch.device,
    ) -> None:
        assert window_size >= 1
        self.window_size = window_size
        self.device = device

        # Load file
        df = pd.read_parquet(parquet_file)

        # Filter operating mode
        if equilibrium:
            df = df[df[f"equilibrium_{operating_mode}_mode"] & ~df["dyn_only_on"]]
        else:
            df = df[df[f"{operating_mode}_mode"]]

        # Remove operating mode variables from data
        operating_mode_vars = [var for var in df.columns if "mode" in var] + [
            "machine_on",
            "machine_off",
            "dyn_only_on",
            "all",
        ]
        df.drop(columns=operating_mode_vars, inplace=True)

        # Handle features selection
        if features is not None:
            for feature in features:
                if feature not in df.columns:
                    raise ValueError(f"Feature '{feature}' not found. Available: {df.columns.tolist()}")
            df = df[features]

        # Handle injector openings
        injector_columns_mask = df.columns.to_series().str.match("injector_0[1-9]_opening")
        if any(injector_columns_mask):
            total_opening = df.loc[:, injector_columns_mask].sum(axis=1)
            df.drop(columns=df.columns[injector_columns_mask], inplace=True)
            df["total_injector_opening"] = total_opening

        assert (df.dtypes == float).all()

        # Handle ground truth if present
        if "ground_truth" in df.columns:
            self.ground_truth = torch.from_numpy(df["ground_truth"].to_numpy()).to(device)
            df.drop(columns="ground_truth", inplace=True)

        self.df = df

        # Compute indices
        valid_end_indices = (
            df.index.to_series()
            .diff(periods=self.window_size - 1)
            .eq((self.window_size - 1) * pd.Timedelta(seconds=30))
        )
        start_indices = np.nonzero(valid_end_indices)[0] - self.window_size + 1
        self.start_indices = torch.from_numpy(start_indices)

        # Normalize
        self.measurements = torch.from_numpy(df.to_numpy().astype(np.float32)).to(device)
        mean = self.measurements.mean(dim=0, keepdim=True)
        std = self.measurements.std(dim=0, keepdim=True)
        self.measurements = torch.where(std > 0, (self.measurements - mean) / std, self.measurements)
        self.measurements = self.measurements.T

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        start_index = self.start_indices[index]
        end_index = start_index + self.window_size
        return self.measurements[:, start_index:end_index]


class SlidingLabeledDataset(SlidingDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_index = self.start_indices[index]
        end_index = start_index + self.window_size
        return (
            self.measurements[:, start_index:end_index],
            self.ground_truth[start_index:end_index],
        )


def create_dataloaders(dataset: Dataset, batch_size: int, validation_split: float) -> tuple[DataLoader, DataLoader]:
    """Creates train/validation DataLoaders"""

    num_samples = len(dataset)
    validation_size = int(validation_split * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_indices))

    return train_loader, validation_loader
