import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class AlpiqDataset(Dataset):
    def __init__(self, dataframe, unit_id, operating_mode, window=50, stride=1, device='cpu'):
        """
        Args:
            dataframe: Input DataFrame
            unit_id: VG4, VG5, or VG6
            operating_mode: 'turbine_mode', 'pump_mode', or 'short_circuit_mode'
            window: Sequence window length
            stride: Window stride length
            device: Computing device
        """
        self.window = window
        self.stride = stride
        self.device = device
        
        # Filter data for specific unit and operating mode
        mode_filter = f"equilibrium_{operating_mode}"
        df_filtered = dataframe[
            (dataframe['unit'] == unit_id) & 
            (dataframe[mode_filter] == True)
        ]
        
        # Separate control variables (X) and generator variables (Y)
        control_vars = ['tot_activepower', 'ext_tmp', 'plant_tmp', 'pump_rotspeed',
                       'turbine_rotspeed', 'turbine_pressure']  
        generator_vars = ['stat_coil_tmp', 'stat_magn_tmp', 'air_circ_hot_tmp',
                         'air_circ_cold_tmp', 'water_circ_flow', 'water_circ_hot_tmp',
                         'water_circ_cold_tmp']
        
        self.X = np.array(df_filtered[control_vars].values).astype(np.float32)
        self.Y = np.array(df_filtered[generator_vars].values).astype(np.float32)
        
        # Create sliding window indices
        self.indices = self._get_sliding_indices()
        
        # Convert to tensors
        self.X = torch.from_numpy(self.X).to(device)
        self.Y = torch.from_numpy(self.Y).to(device)
        
    def _get_sliding_indices(self):
        n_samples = len(self.X)
        valid_indices = []
        
        for start_idx in range(0, n_samples - self.window + 1, self.stride):
            if start_idx + self.window <= n_samples:
                valid_indices.append((start_idx, start_idx + self.window))
                
        return torch.tensor(valid_indices).to(self.device)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        x_seq = self.X[start_idx:end_idx]
        y_seq = self.Y[start_idx:end_idx]
        return x_seq, y_seq

def create_dataloaders(dataset, batch_size=32, train_split=0.7, val_split=0.15, seed=42):
    """
    Creates train/validation/test DataLoaders with random splits
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate split sizes
    n_samples = len(dataset)
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    test_size = n_samples - train_size - val_size
    
    # Create indices for splits
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader

# Example usage:
"""
# For VG5 turbine mode
dataset_vg5_turbine = AlpiqDataset(
    dataframe=df,
    unit_id='VG5',
    operating_mode='turbine_mode',
    window=50,
    stride=1
)

train_loader, val_loader, test_loader = create_dataloaders(
    dataset_vg5_turbine,
    batch_size=32
)
"""