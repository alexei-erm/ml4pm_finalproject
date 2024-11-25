from config import Config
from dataloader import SlidingDataset, SlidingLabeledDataset, create_dataloaders

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model: nn.Module, cfg: Config, dataset_root: str, log_dir: str, device: torch.device) -> None:
    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        equilibrium=cfg.equilibrium,
        window_size=cfg.window_size,
        device=device,
    )

    train_loader, val_loader = create_dataloaders(
        dataset, batch_size=cfg.batch_size, validation_split=cfg.validation_split
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    best_val_loss = torch.inf

    for epoch in range(cfg.epochs):

        model.train()
        train_loss = 0.0
        for x in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            optimizer.zero_grad(set_to_none=True)

            reconstruction, latent = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                reconstruction, latent = model(x)
                loss = criterion(reconstruction, x)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

        writer.add_scalar("Loss/Training", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")


def compute_spes(loader: DataLoader, model: nn.Module) -> np.ndarray:
    spes = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(loader):
            reconstruction, latent = model(x)
            spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
            spes.append(spe.cpu().numpy())

    return np.concatenate(spes)


def compute_spes_and_labels(model: nn.Module, parquet_file: str, window_size: int, device: torch.device) -> np.ndarray:
    dataset = SlidingLabeledDataset(
        parquet_file=parquet_file,
        operating_mode="turbine",
        window_size=window_size,
        device=device,
    )

    loader = DataLoader(dataset, batch_size=256)
    spes = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            reconstruction, latent = model(x)
            spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
            spes.append(spe.cpu().numpy())
            labels.append(y.squeeze().cpu().numpy())

    return np.concatenate(spes), np.concatenate(labels)
