from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    patience: int = 5,
    min_delta: float = 1e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    writer = SummaryWriter(log_dir="runs/conv", flush_secs=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for x in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            reconstruction = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        writer.add_scalar("Loss/Training", train_loss, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                reconstruction = model(x)
                val_loss += criterion(reconstruction, x).item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "models/best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                model.load_state_dict(best_model_state)
                return train_losses, val_losses

        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    return train_losses, val_losses