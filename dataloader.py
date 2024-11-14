import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class Dataloader(Dataset):
    def __init__(self, parquet_path, info_csv_path, unit_id, operating_mode, window=50, stride=1, device='cpu'):
        """
        Args:
            parquet_path: Path to measurements parquet file
            info_csv_path: Path to info CSV file
            unit_id: VG4, VG5, or VG6 
            operating_mode: 'turbine_mode', 'pump_mode', or 'short_circuit_mode'
            window: Sequence window length
            stride: Window stride length
            device: Computing device
        """
        self.window = window
        self.stride = stride
        self.device = device
        
        # Load data
        df_measurements = pd.read_parquet(parquet_path)
        df_info = pd.read_csv(info_csv_path)
        df = pd.merge(df_measurements, df_info, on='timestamp')
        
        # Filter data for specific unit and operating mode
        mode_filter = f"equilibrium_{operating_mode}"
        df_filtered = df[
            (df['unit'] == unit_id) & 
            (df[mode_filter] == True)
        ]
        
        # Separate control variables (X) and generator variables (Y)
        control_vars = ['tot_activepower', 'ext_tmp', 'plant_tmp', 'pump_rotspeed',
                        'turbine_rotspeed', 'turbine_pressure']
        generator_vars = ['stat_coil_tmp', 'stat_magn_tmp', 'air_circ_hot_tmp',
                        'air_circ_cold_tmp', 'water_circ_flow', 'water_circ_hot_tmp',
                        'water_circ_cold_tmp']
        
        # Handle missing columns for different units
        control_vars = [var for var in control_vars if var in df_filtered.columns]
        generator_vars = [var for var in generator_vars if var in df_filtered.columns]
        
        self.X = np.array(df_filtered[control_vars].values).astype(np.float32)
        self.Y = np.array(df_filtered[generator_vars].values).astype(np.float32)
        
        # Create sliding window indices
        self.indices = self._get_sliding_indices()
        
        # Convert to tensors
        self.X = torch.from_numpy(self.X).to(device)
        self.Y = torch.from_numpy(self.Y).to(device)
        
        # [Previous initialization code remains the same until df_filtered]

        # Convert timestamp to datetime and sort
        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
        df_filtered = df_filtered.sort_values('timestamp')
        
        # Find continuous sequences
        time_diff = df_filtered['timestamp'].diff()
        expected_diff = pd.Timedelta(seconds=30)  # 30-second intervals
        sequence_breaks = time_diff != expected_diff
        sequence_ids = sequence_breaks.cumsum()
        
        # Get valid sequences that are long enough for window
        valid_sequences = sequence_ids.value_counts()
        valid_sequences = valid_sequences[valid_sequences >= window]
        
        # Create indices only for valid continuous sequences
        self.indices = []
        for seq_id in valid_sequences.index:
            seq_mask = sequence_ids == seq_id
            seq_indices = df_filtered[seq_mask].index
            
            for i in range(0, len(seq_indices) - window + 1, stride):
                start_idx = seq_indices[i]
                end_idx = seq_indices[i + window - 1]
                self.indices.append((start_idx, end_idx))
        
        self.indices = torch.tensor(self.indices).to(device)
        
        # Prepare X and Y data
        self.X = torch.from_numpy(self.X).to(device)
        self.Y = torch.from_numpy(self.Y).to(device)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        x_seq = self.X[start_idx:end_idx+1]
        y_seq = self.Y[start_idx:end_idx+1]
        return x_seq, y_seq

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