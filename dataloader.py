import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class Dataloader(Dataset):
    def __init__(self, parquet_path, info_csv_path, unit_id, operating_mode, window=50, stride=1, device='cpu'):
        self.window = window
        self.stride = stride
        self.device = device
        
        # Load measurements
        df_filtered = pd.read_parquet(parquet_path)
        
        # Filter for operating mode
        mode_filter = f"equilibrium_{operating_mode}"
        df_filtered = df_filtered[df_filtered[mode_filter] == True]
        
        # Define variable groups
        control_vars = ['tot_activepower', 'ext_tmp', 'plant_tmp', 'pump_rotspeed',
                        'turbine_rotspeed', 'turbine_pressure']
        
        generator_vars = [col for col in df_filtered.columns if any(x in col for x in 
                        ['stat_coil', 'stat_magn', 'air_circ', 'water_circ'])]
        
        # Handle missing columns
        control_vars = [var for var in control_vars if var in df_filtered.columns]
        
        # Prepare X and Y data
        self.X = np.array(df_filtered[control_vars].values).astype(np.float32)
        self.Y = np.array(df_filtered[generator_vars].values).astype(np.float32)
        
        # Find continuous sequences
        df_filtered = df_filtered.reset_index()
        index_diff = df_filtered.index.diff()
        sequence_breaks = index_diff != 1
        sequence_ids = sequence_breaks.cumsum()
        
        # Get valid sequences using pandas Series
        sequence_counts = pd.Series(sequence_ids).value_counts()
        valid_sequences = sequence_counts[sequence_counts >= window]
        
        # Create indices for continuous sequences
        seq_indices_list = []
        for seq_id in valid_sequences.index:
            seq_mask = sequence_ids == seq_id
            seq_indices = df_filtered[seq_mask].index.values
            starts = np.arange(0, len(seq_indices) - window + 1, stride)
            ends = starts + window - 1
            seq_indices_list.append(np.column_stack((seq_indices[starts], seq_indices[ends])))
        
        self.indices = torch.tensor(np.vstack(seq_indices_list)).to(device)
        
        # Convert to tensors
        self.X = torch.from_numpy(self.X).to(device)
        self.Y = torch.from_numpy(self.Y).to(device)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        x_seq = self.X[start_idx:end_idx+1]  # [window, features]
        y_seq = self.Y[start_idx:end_idx+1]
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
    val_indices = indices[train_size:train_size+val_size] 
    test_indices = indices[train_size+val_size:]
    
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size,
                           sampler=SubsetRandomSampler(test_indices))
    
    return train_loader, val_loader, test_loader

# Example usage:
"""
dataset_vg5_turbine = AlpiqDataset(
    parquet_path='VG5_generator_data_training_measurements.parquet',
    info_csv_path='VG5_generator_data_training_info.csv',
    unit_id='VG5',
    operating_mode='turbine_mode'
)

train_loader, val_loader, test_loader = create_dataloaders(dataset_vg5_turbine)
"""