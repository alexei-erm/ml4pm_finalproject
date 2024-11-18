from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir="runs/conv", flush_secs=10)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        best_val_loss = torch.inf

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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                reconstruction = model(x)
                val_loss += criterion(reconstruction, x).item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/Validation", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")

        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
