import torch
import torch.nn as nn
from tqdm.auto import tqdm


class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dim=64):
        super(CNNAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, input_channels, kernel_size=3, stride=1, padding='same')
        )
        
    def forward(self, x):
        if x.shape[1] == self.sequence_length:
            x = x.transpose(1, 2)
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        return reconstruction.transpose(1, 2)
    
    def get_latent(self, x):
        x = x.transpose(1, 2)
        return self.encoder(x)

def train_autoencoder(model, train_loader, val_loader, n_epochs=100, learning_rate=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        from tqdm.auto import tqdm
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstruction = model(batch_x)
                val_loss += criterion(reconstruction, batch_x).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses