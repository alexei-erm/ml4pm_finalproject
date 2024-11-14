import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dim=64):
        super(CNNAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Flatten(),
            nn.Linear(128 * (sequence_length // 8), latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (sequence_length // 8)),
            nn.Unflatten(1, (128, sequence_length // 8)),
            
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_channels]
        x = x.transpose(1, 2)  # [batch_size, input_channels, sequence_length]
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.transpose(1, 2)  # Back to [batch_size, sequence_length, input_channels]
        
        return reconstruction
    
    def get_latent(self, x):
        x = x.transpose(1, 2)
        return self.encoder(x)

# Training function
def train_autoencoder(model, train_loader, val_loader, n_epochs=100, learning_rate=1e-3, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch_x, _ in train_loader:  # We only need X for autoencoder
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstruction = model(batch_x)
                val_loss += criterion(reconstruction, batch_x).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Example usage:
"""
# Initialize model
input_channels = len(control_vars)  # Number of input features
sequence_length = 50  # Window size
model = CNNAutoencoder(input_channels, sequence_length).to(device)

# Train model
train_losses, val_losses = train_autoencoder(model, train_loader, val_loader)
"""