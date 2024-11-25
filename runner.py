from config import Config
from dataloader import SlidingDataset, SlidingLabeledDataset, create_dataloaders
from model import *  # noqa F401
from utils import dump_yaml, class_to_dict

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, cfg: Config, dataset_root: str, log_dir: str, device=torch.device) -> None:
        self.cfg = cfg
        self.dataset_root = dataset_root
        self.log_dir = log_dir
        self.device = device

        parquet_file = os.path.abspath(
            os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
        )
        self.dataset = SlidingDataset(
            parquet_file=parquet_file,
            operating_mode=cfg.operating_mode,
            equilibrium=cfg.equilibrium,
            window_size=cfg.window_size,
            device=device,
        )

        model_type = eval(cfg.model)
        self.model = model_type(input_channels=self.dataset.measurements.size(0), cfg=cfg).to(device)

    def train(self) -> None:
        dump_yaml(os.path.join(self.log_dir, "config.yaml"), self.cfg)

        train_loader, val_loader = create_dataloaders(
            self.dataset, batch_size=self.cfg.batch_size, validation_split=self.cfg.validation_split
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        best_val_loss = torch.inf

        for epoch in range(self.cfg.epochs):

            self.model.train()
            train_loss = 0.0
            for x in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}"):
                optimizer.zero_grad(set_to_none=True)

                reconstruction, latent = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x in val_loader:
                    reconstruction, latent = self.model(x)
                    loss = criterion(reconstruction, x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "model.pt"))

            writer.add_scalar("Loss/Training", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}"
            )

    def test(self) -> None:
        if self.cfg.unit == "VG4":
            print("ROC evaluation is not possible with VG4")
            return

        model_path = os.path.join(self.log_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        self.model.eval()

        fig, ax = plt.subplots()

        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            dataset = SlidingLabeledDataset(
                parquet_file=os.path.join(
                    self.dataset_root, "synthetic_anomalies", f"{self.cfg.unit}_anomaly_{name}.parquet"
                ),
                operating_mode=self.cfg.operating_mode,
                equilibrium=self.cfg.equilibrium,
                window_size=self.cfg.window_size,
                device=self.device,
            )
            loader = DataLoader(dataset, batch_size=self.cfg.batch_size)
            spes = []
            labels = []
            with torch.no_grad():
                for x, y in tqdm(loader):
                    reconstruction, latent = self.model(x)
                    spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
                    spes.append(spe.cpu().numpy())
                    labels.append(y.squeeze().cpu().numpy())

            spes = np.concatenate(spes)
            labels = np.concatenate(labels)

            RocCurveDisplay.from_predictions(y_true=labels, y_pred=spes, ax=ax, name=name)

        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        plt.legend()
        plt.show()
