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
        features: list[str],
        downsampling: int,
        device: torch.device,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:

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

        # Aggregate injector openings
        injector_columns_mask = df.columns.to_series().str.match("injector_0[1-9]_opening")
        total_injector_opening = df.loc[:, injector_columns_mask].sum(axis=1)
        df.drop(columns=df.columns[injector_columns_mask], inplace=True)
        df["total_injector_opening"] = total_injector_opening

        # Downsample
        if downsampling > 1:
            df = downsample(df, period=downsampling * pd.Timedelta(seconds=30))

        # If the dataset contains labels, separate them from the measurement data
        if "ground_truth" in df.columns:
            df.loc[df["ground_truth"] > 0, "ground_truth"] = 1
            self.ground_truth = torch.from_numpy(df["ground_truth"].to_numpy(dtype=bool)).to(device)
            df.drop(columns="ground_truth", inplace=True)

        # Select features
        if len(features) > 0:
            columns = []
            for feature in features:
                matching_columns = df.columns[df.columns.to_series().str.match(feature)].to_list()
                if len(matching_columns) == 0:
                    print(f"Selected feature '{feature}' does not match any feature in the dataset.")
                    exit()
                columns += matching_columns
            df = df[columns]

        # FIXME
        # Reduce features
        def aggregate(name: str, cols: str) -> pd.DataFrame:
            matching_columns = df.columns[df.columns.str.match(cols)]
            df[name] = df[matching_columns].max(axis=1)
            df.drop(columns=matching_columns, inplace=True)
            return df

        df = aggregate("stat_coil_agg", "stat_coil_.*")
        df = aggregate("stat_magn_agg", "stat_magn_.*")
        df = aggregate("current_agg", ".*_current")
        df = aggregate("voltage_agg", ".*_voltage")
        df = aggregate("air_circ_cold_agg", "air_circ_cold_.*")
        df = aggregate("air_circ_hot_agg", "air_circ_hot_.*")
        df = aggregate("water_circ_cold_agg", "water_circ_cold_.*")
        df = aggregate("water_circ_hot_agg", "water_circ_hot_.*")
        df.drop(columns=df.columns[df.columns.str.match("air_gap_.*")], inplace=True)

        # Save filtered dataframe
        self.df = df.copy()
        self.index = df.index.values.copy()

        # Compute subsequence indices according to window size
        valid_end_indices = df.index.to_series().diff(periods=window_size - 1) == (
            (window_size - 1) * downsampling * pd.Timedelta(seconds=30)
        )
        start_indices = np.nonzero(valid_end_indices)[0] - window_size + 1
        self.start_indices = torch.from_numpy(start_indices)

        # Convert to tensor and normalize
        self.measurements = torch.from_numpy(df.to_numpy(dtype=np.float32)).to(device)
        self.mean = mean if mean is not None else self.measurements.mean(dim=0)
        self.std = std if std is not None else self.measurements.std(dim=0)
        self.measurements = torch.where(
            self.std.reshape(1, -1) > 0,
            (self.measurements - self.mean.reshape(1, -1)) / self.std.reshape(1, -1),
            self.measurements,
        )

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray]:
        start_index = self.start_indices[index]
        end_index = start_index + self.window_size
        ground_truth = self.ground_truth[start_index:end_index] if hasattr(self, "ground_truth") else None
        return (
            self.measurements[start_index:end_index, :],
            ground_truth,
            self.index[start_index:end_index],
        )


def downsample(df: pd.DataFrame, period: pd.Timedelta) -> pd.DataFrame:
    """Downsamples a dataframe with timestamp index, handling gaps."""

    # Identify large gaps
    large_gaps = df.index.to_series().diff() > period
    group = large_gaps.cumsum()

    # Downsample each group separately
    downsampled = []
    for group_id, group_data in df.groupby(group):
        downsampled_group = group_data.resample(period, closed="right", label="right").mean()
        downsampled.append(downsampled_group)

    # Combine all downsampled groups
    return pd.concat(downsampled)


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

    indices = np.arange(len(dataset))
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
