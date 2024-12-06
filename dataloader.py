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
        transient: bool,
        window_size: int,
        device: torch.device,
        features: list[str] | None = None,
    ) -> None:

        assert window_size >= 1
        self.window_size = window_size
        self.device = device

        # Load file
        df = pd.read_parquet(parquet_file)

        # Filter operating mode
        if transient:
            df = df[df[f"{operating_mode}_mode"]]
        else:
            df = df[df[f"equilibrium_{operating_mode}_mode"] & ~df["dyn_only_on"]]

        # Remove operating mode variables from data
        operating_mode_vars = [var for var in df.columns if "mode" in var] + [
            "machine_on",
            "machine_off",
            "dyn_only_on",
            "all",
        ]
        df.drop(columns=operating_mode_vars, inplace=True)
        assert (df.dtypes == float).all()

        # If the dataset contains labels, separate them from the measurement data
        if "ground_truth" in df.columns:
            self.ground_truth = torch.from_numpy(df["ground_truth"].to_numpy()).to(device)
            df.drop(columns="ground_truth", inplace=True)

        # Aggregate injector openings
        injector_columns_mask = df.columns.to_series().str.match("injector_0[1-9]_opening")
        total_injector_opening = df.loc[:, injector_columns_mask].sum(axis=1)
        df.drop(columns=df.columns[injector_columns_mask], inplace=True)
        df["total_injector_opening"] = total_injector_opening

        # Save filtered measurements DataFrame
        self.df = df
        self.index = df.index.values

        # Compute subsequence indices according to window size
        valid_end_indices = (
            df.index.to_series()
            .diff(periods=self.window_size - 1)
            .eq((self.window_size - 1) * pd.Timedelta(seconds=30))
        )
        start_indices = np.nonzero(valid_end_indices)[0] - self.window_size + 1
        self.start_indices = torch.from_numpy(start_indices)

        # Convert to tensor
        if features is not None:
            columns = []
            for feature in features:
                matching_columns = df.columns[df.columns.to_series().str.match(feature)].to_list()
                if len(matching_columns) == 0:
                    print(f"Selected feature '{feature}' does not match any feature in the dataset.")
                    exit()
                columns += matching_columns
            self.measurements = torch.from_numpy(df[columns].to_numpy(dtype=np.float32)).to(device)
        else:
            self.measurements = torch.from_numpy(df.to_numpy(dtype=np.float32)).to(device)

        # Normalize
        mean = self.measurements.mean(dim=0, keepdim=True)
        std = self.measurements.std(dim=0, keepdim=True)
        self.measurements = torch.where(std > 0, (self.measurements - mean) / std, self.measurements)
        self.measurements = self.measurements.T

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray]:
        start_index = self.start_indices[index]
        end_index = start_index + self.window_size
        ground_truth = self.ground_truth[start_index:end_index] if hasattr(self, "ground_truth") else None
        return (
            self.measurements[:, start_index:end_index],
            ground_truth,
            self.index[start_index:end_index],
        )


def collate_fn(batch):
    x, label, index = zip(*batch)
    x = torch.stack(x)
    label = torch.stack(label) if label[0] is not None else None
    index = np.stack(index)
    return x, label, index


def create_train_val_dataloaders(
    dataset: Dataset, batch_size: int, validation_split: float, subsampling: int = 1
) -> tuple[DataLoader, DataLoader]:
    """Creates train/validation DataLoaders"""

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[::subsampling]
    validation_size = int(validation_split * len(indices))
    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=collate_fn
    )

    validation_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_indices), collate_fn=collate_fn
    )

    return train_loader, validation_loader


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Creates non-shuffled DataLoader"""

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
