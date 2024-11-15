from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class RawDataset(Dataset):
    def __init__(
        self,
        unit: Literal["VG4", "VG5", "VG6"],
        operating_mode: Literal["turbine", "pump", "short_circuit"],
        equilibrium: bool = True,
        dataset_folder: str = "Dataset",
        window: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        self.window = window
        self.device = device

        # Load files
        pq_path = f"{dataset_folder}/{unit}_generator_data_training_measurements.parquet"
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

        # FIXME: this whole thing does not work

        # Find contiguous sequences
        df = df.reset_index()
        index_diff = df.index.diff()
        sequence_breaks = index_diff != 1
        sequence_ids = sequence_breaks.cumsum()
        print(sequence_ids)

        # Get valid sequences using pandas Series
        sequence_counts = pd.Series(sequence_ids).value_counts()
        valid_sequences = sequence_counts[sequence_counts >= window]

        # Create indices for continuous sequences
        seq_indices_list = []
        print(len(valid_sequences.index))
        exit()
        for seq_id in valid_sequences.index:
            seq_mask = sequence_ids == seq_id
            seq_indices = df[seq_mask].index.values
            starts = np.arange(0, len(seq_indices) - window + 1)
            ends = starts + window - 1
            seq_indices_list.append(np.column_stack((seq_indices[starts], seq_indices[ends])))

        self.indices = torch.tensor(np.vstack(seq_indices_list)).to(device)

        # Convert to tensors
        self.measurements = torch.from_numpy(df.to_numpy().astype(np.float32)).to(device)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        x_seq = self.X[start_idx : end_idx + 1]  # [window, features]
        y_seq = self.Y[start_idx : end_idx + 1]
        # Return shape: [features, window]
        return x_seq, y_seq  # No transpose here since model handles it


def create_dataloaders(dataset, batch_size=32, train_split=0.7, val_split=0.15, seed=42):
    """Creates train/validation/test DataLoaders with random splits"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples = len(dataset)
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)

    indices = list(range(n_samples))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    return train_loader, val_loader, test_loader
