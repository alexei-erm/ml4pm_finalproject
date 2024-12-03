from config import Config
from dataloader import *
from model import *  # noqa F401
from utils import dump_yaml, dump_pickle

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import torch
import torch.nn as nn
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
        self.training_dataset = SlidingDataset(
            parquet_file=parquet_file,
            operating_mode=cfg.operating_mode,
            equilibrium=cfg.equilibrium,
            window_size=cfg.window_size,
            device=device,
            features=cfg.features,
        )

        model_type = eval(cfg.model)
        self.model = model_type(input_channels=self.training_dataset.measurements.shape[0], cfg=cfg).to(device)

    def train_autoencoder(self) -> None:
        dump_yaml(os.path.join(self.log_dir, "config.yaml"), self.cfg)
        dump_pickle(os.path.join(self.log_dir, "model.pkl"), self.model)

        train_loader, val_loader = create_train_val_dataloaders(
            self.training_dataset, batch_size=self.cfg.batch_size, validation_split=self.cfg.validation_split
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        best_val_loss = torch.inf

        for epoch in range(self.cfg.epochs):

            self.model.train()

            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_sparsity_loss = 0.0

            for x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}"):
                optimizer.zero_grad(set_to_none=True)

                reconstruction, latent = self.model(x)

                reconstruction_loss = criterion(reconstruction, x)
                sparsity_loss = self.kl_divergence(latent, rho=0.05)
                beta = 0.01
                loss = reconstruction_loss + beta * sparsity_loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_reconstruction_loss += reconstruction_loss.item()
                train_sparsity_loss += sparsity_loss.item()

            train_loss /= len(train_loader)
            train_reconstruction_loss /= len(train_loader)
            train_sparsity_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, _, _ in val_loader:
                    reconstruction, latent = self.model(x)
                    loss = criterion(reconstruction, x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "model.pt"))

            writer.add_scalar("Loss/Training", train_loss, epoch + 1)
            writer.add_scalar("Loss/Training_reconstruction", train_reconstruction_loss, epoch + 1)
            writer.add_scalar("Loss/Training_sparsity", train_sparsity_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs}, Loss: Train={train_loss:.7f}, "
                f"Rec.={train_reconstruction_loss:.7f}, Spars.={train_sparsity_loss:.7f}, "
                f"Val.={val_loss:.7f}"
            )

    def test_autoencoder(self) -> None:
        if self.cfg.unit == "VG4":
            print("Evaluation is not possible with VG4")
            return

        model_path = os.path.join(self.log_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))

        self.model.eval()

        train_latent = self.get_training_latent_features()

        from sklearn.svm import OneClassSVM

        ocsvm = OneClassSVM(nu=0.002)
        ocsvm.fit(train_latent.T.cpu().numpy())

        latent_mean, latent_covariance = self.get_training_latent_statistics(train_latent)
        inv_latent_covariance = torch.linalg.inv(latent_covariance)

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()

        for i, name in enumerate(["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]):
            ax = axes[i]

            dataset = SlidingDataset(
                parquet_file=os.path.join(
                    self.dataset_root, "synthetic_anomalies", f"{self.cfg.unit}_anomaly_{name}.parquet"
                ),
                operating_mode=self.cfg.operating_mode,
                equilibrium=self.cfg.equilibrium,
                window_size=self.cfg.window_size,
                features=self.cfg.features,
                device=self.device,
            )
            loader = create_dataloader(dataset, batch_size=self.cfg.batch_size)

            indices = []
            preds = []
            xs = []
            labels = []
            spes = []
            t2s = []
            svm = []

            with torch.no_grad():
                for x, y, index in tqdm(loader):
                    # x[(y == 1).unsqueeze(1).repeat(1, x.shape[1], 1)] += 1.0

                    xs.append(x)
                    labels.append(y)
                    indices.append(index)

                    reconstruction, latent = self.model(x)
                    preds.append(reconstruction)

                    spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
                    spes.append(spe)

                    latent_diff = latent - latent_mean.unsqueeze(0)
                    t2 = torch.einsum("bi,ij,bj->b", latent_diff, inv_latent_covariance, latent_diff)
                    t2s.append(t2)

                    svm.append(ocsvm.predict(latent.cpu().numpy()))

            xs = torch.concatenate(xs).cpu().numpy()
            preds = torch.concatenate(preds).cpu().numpy()
            indices = np.concatenate(indices)
            labels = torch.concatenate(labels).cpu().numpy()
            spes = torch.concatenate(spes).cpu().numpy()
            t2s = torch.concatenate(t2s).cpu().numpy()
            svm = np.concatenate(svm)

            labels = labels[..., -1]
            xs = xs[..., 0, -1]
            preds = preds[..., 0, -1]

            # spes = spes.clip(max=spes.mean() + 3.0 * spes.std())
            # t2s = t2s.clip(max=t2s.mean() + 3.0 * t2s.std())
            spes = (spes - spes.min()) / (spes.max() - spes.min())
            t2s = (t2s - t2s.min()) / (t2s.max() - t2s.min())

            ax.plot(xs, label="x")
            ax.plot(preds, label="pred")
            ax.plot(spes, label="SPE")
            ax.plot(t2s, label="T2")
            ax.plot(svm, label="OCSVM")

            for start, end in zip(
                np.where(np.diff(labels, prepend=0) == 1)[0], np.where(np.diff(labels, append=0) == -1)[0]
            ):
                ax.axvspan(start, end, color="red", alpha=0.3)

            ax.set_title(name)
            ax.legend()

        fig.tight_layout()
        plt.show()

    def fit_spc(self) -> None:
        x_healthy = self.training_dataset.measurements

        mean = x_healthy.mean(dim=1, keepdim=True)
        centered = x_healthy - mean
        num_samples = x_healthy.shape[1]
        covariance = (centered @ centered.T) / (num_samples - 1)

        inv_covariance = torch.linalg.inv(covariance)

        fig, ax = plt.subplots()
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            dataset = SlidingDataset(
                parquet_file=os.path.join(
                    self.dataset_root, "synthetic_anomalies", f"{self.cfg.unit}_anomaly_{name}.parquet"
                ),
                operating_mode=self.cfg.operating_mode,
                equilibrium=self.cfg.equilibrium,
                window_size=self.cfg.window_size,
                features=self.cfg.features,
                device=self.device,
            )

            x_test = dataset.measurements
            diff = x_test - mean
            t2 = torch.einsum("ib,ij,jb->b", diff, inv_covariance, diff)

            labels = dataset.ground_truth.cpu().numpy()

            window = 1
            tpr = []
            fpr = []
            for threshold in np.linspace(t2.min().item(), t2.max().item(), 1000):
                fault = t2 > threshold
                windowed = fault.unfold(0, window, 1)
                consecutive_above_threshold = windowed.sum(dim=-1) == window
                pred = torch.zeros_like(fault)
                pred[window - 1 :] = consecutive_above_threshold
                pred = pred.cpu().numpy()

                tp = np.sum((pred == 1) & (labels == 1))
                fp = np.sum((pred == 1) & (labels == 0))
                tn = np.sum((pred == 0) & (labels == 0))
                fn = np.sum((pred == 0) & (labels == 1))
                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

            tpr = np.array(tpr)
            fpr = np.array(fpr)
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot(name=name, ax=ax)
            # t2 = t2.cpu().numpy()
            # RocCurveDisplay.from_predictions(y_true=labels, y_pred=t2, ax=ax, name=name)

            t2 = t2.cpu().numpy()
            fig, ax2 = plt.subplots()
            ax2.plot(labels)
            t2 = t2.clip(max=t2.mean() + 4 * t2.std())
            ax2.plot((t2 - t2.min()) / (t2.max() - t2.min()))
            ax2.set_title(name)

        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        plt.legend()
        plt.show()

    def fit_if(self) -> None:
        from sklearn.ensemble import IsolationForest

        x = self.training_dataset.measurements

        isolation_forest = IsolationForest(n_estimators=500, max_samples=1.0, max_features=0.5, n_jobs=-1)
        isolation_forest.fit(x.T.cpu().numpy())

        fig, ax = plt.subplots()
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            dataset = SlidingDataset(
                parquet_file=os.path.join(
                    self.dataset_root, "synthetic_anomalies", f"{self.cfg.unit}_anomaly_{name}.parquet"
                ),
                operating_mode=self.cfg.operating_mode,
                equilibrium=self.cfg.equilibrium,
                window_size=self.cfg.window_size,
                features=self.cfg.features,
                device=self.device,
            )

            pred = isolation_forest.score_samples(dataset.measurements.T.cpu().numpy())
            labels = dataset.ground_truth.cpu().numpy()

            RocCurveDisplay.from_predictions(y_true=labels, y_pred=pred, ax=ax, name=name)

        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        ax.legend()
        plt.show()

    def get_training_latent_features(self) -> torch.Tensor:
        self.model.eval()

        loader = create_dataloader(self.training_dataset, batch_size=self.cfg.batch_size)

        all_latent = []

        with torch.no_grad():
            for x, _, _ in tqdm(loader):
                reconstruction, latent = self.model(x)
                all_latent.append(latent)

        return torch.concatenate(all_latent).T

    def get_training_latent_statistics(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and covariance of the latent features commputed on the training set."""

        mean = latent.mean(dim=1, keepdim=True)
        centered = latent - mean
        num_samples = latent.shape[1]
        covariance = (centered @ centered.T) / (num_samples - 1)

        return mean.squeeze(), covariance

    def kl_divergence(self, latent: torch.Tensor, rho: float) -> torch.Tensor:
        """Computes the KL-divergence of the batch. Input shape (batch_size, n_features)."""

        rho_hat = torch.mean(latent, dim=0)
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return torch.sum(kl)
