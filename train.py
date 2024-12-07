from config import Config
from dataloader import *
from model import *  # noqa F401
from utils import *

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def train_autoencoder(cfg: Config, dataset_root: str, log_dir: str, device: torch.device) -> None:
    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    training_dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        device=device,
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
    )

    model_type = eval(cfg.model.value)
    model = model_type(
        input_channels=training_dataset.measurements.shape[-1], window_size=cfg.window_size, cfg=cfg.model_cfg
    ).to(device)

    dump_yaml(os.path.join(log_dir, "config.yaml"), cfg)
    dump_pickle(os.path.join(log_dir, "config.pkl"), cfg)
    dump_pickle(os.path.join(log_dir, "model.pkl"), model)

    train_loader, val_loader = create_train_val_dataloaders(
        training_dataset,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        subsampling=cfg.training_subsampling,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    best_val_loss = torch.inf

    for epoch in range(cfg.epochs):

        model.train()

        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_l1_loss = 0.0

        for x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            optimizer.zero_grad(set_to_none=True)

            reconstruction, latent = model(x)

            reconstruction_loss = criterion(reconstruction, x)
            loss = reconstruction_loss
            total_reconstruction_loss += reconstruction_loss.item()

            if cfg.kl_divergence_weight > 0.0:
                kl_loss = cfg.kl_divergence_weight * kl_divergence(latent, rho=cfg.kl_divergence_rho)
                loss += kl_loss
                total_kl_loss += kl_loss.item()

            if cfg.l1_weight > 0.0:
                l1_loss = cfg.l1_weight * torch.mean(torch.abs(latent))
                loss += l1_loss
                total_l1_loss += l1_loss.item()

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        total_loss /= len(train_loader)
        total_reconstruction_loss /= len(train_loader)
        total_kl_loss /= len(train_loader)
        total_l1_loss /= len(train_loader)

        model.eval()

        total_val_loss = 0.0
        with torch.no_grad():
            for x, _, _ in val_loader:
                reconstruction, latent = model(x)
                loss = criterion(reconstruction, x)
                total_val_loss += loss.item()

        total_val_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))

        writer.add_scalar("Loss/Training", total_loss, epoch + 1)
        writer.add_scalar("Loss/Training_reconstruction", total_reconstruction_loss, epoch + 1)
        writer.add_scalar("Loss/Training_kl", total_kl_loss, epoch + 1)
        writer.add_scalar("Loss/Training_l1", total_l1_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", total_val_loss, epoch + 1)
        print(
            f"Epoch {epoch + 1}/{cfg.epochs}, "
            f"Loss: Train={total_loss:.6f}, "
            f"Rec.={total_reconstruction_loss:.6f}, "
            f"KL={total_kl_loss:.6f}, "
            f"L1={total_l1_loss:.6f}, "
            f"Val.={total_val_loss:.6f}"
        )


def test_autoencoder(cfg: Config, dataset_root: str, log_dir: str, load_best: bool, device: torch.device) -> None:
    if cfg.unit == "VG4":
        print("Evaluation is not possible with VG4")
        return

    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    training_dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        device=device,
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
    )

    model_type = eval(cfg.model.value)
    model = model_type(
        input_channels=training_dataset.measurements.shape[-1], window_size=cfg.window_size, cfg=cfg.model_cfg
    ).to(device)

    model_state_path = os.path.join(log_dir, "best_model.pt" if load_best else "model.pt")
    model.load_state_dict(torch.load(model_state_path, weights_only=True, map_location=device))

    model.eval()

    train_loader = create_dataloader(training_dataset, batch_size=cfg.batch_size)
    train_latent = get_latent_features(model, train_loader)

    from sklearn.svm import OneClassSVM

    ocsvm = OneClassSVM(nu=0.002)
    ocsvm.fit(train_latent.cpu().numpy())

    latent_mean, latent_covariance, inv_latent_covariance = get_statistics(train_latent)

    figs = []
    axes = []
    for _ in range(6):
        fig, ax = plt.subplots()
        figs.append(fig)
        axes.append(ax)

    for i, name in enumerate(["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]):
        ax = axes[i]

        dataset = SlidingDataset(
            parquet_file=os.path.join(dataset_root, "synthetic_anomalies", f"{cfg.unit}_anomaly_{name}.parquet"),
            operating_mode=cfg.operating_mode,
            transient=cfg.transient,
            window_size=cfg.window_size,
            features=cfg.features,
            device=device,
            downsampling=cfg.measurement_downsampling,
        )
        if len(dataset) == 0:
            continue

        loader = create_dataloader(dataset, batch_size=cfg.batch_size)

        indices = []
        preds = []
        xs = []
        labels = []
        spes = []
        t2s = []
        svm = []

        with torch.no_grad():
            for x, y, index in tqdm(loader):
                x[y == 1] += 0.5

                xs.append(x)
                labels.append(y)
                indices.append(index)

                reconstruction, latent = model(x)
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

        # df = pd.DataFrame(xs.flatten(), index=indices.flatten())
        # df.sort_index(inplace=True)
        # df = df.iloc[: len(df) // 10, :]
        # sns.lineplot(df)
        # plt.show()
        # exit()

        indices = indices[:, -1]
        labels = labels[:, -1]
        xs = xs[:, -1, 0]
        preds = preds[:, -1, 0]

        # spes = spes.clip(max=spes.mean() + 3.0 * spes.std())
        # t2s = t2s.clip(max=t2s.mean() + 3.0 * t2s.std())
        spes = (spes - spes.min()) / (spes.max() - spes.min())
        t2s = (t2s - t2s.min()) / (t2s.max() - t2s.min())

        # Find gaps in timestamps
        # gaps = np.diff(indices) > np.timedelta64(30, "s") * cfg.measurement_downsampling
        # gap_indices = np.nonzero(gaps)[0] + 1

        # for idx in gap_indices:
        #    ax.axvline(x=idx, color="k", linestyle="--", label="Gap" if idx == gap_indices[0] else None)

        ax.plot(xs, label="x")
        ax.tick_params(axis="x", labelrotation=45)
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

    for fig in figs:
        fig.tight_layout()
    plt.show()


def fit_spc(cfg: Config, dataset_root: str, device: torch.device) -> None:
    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    training_dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        device=device,
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
    )

    x_healthy = training_dataset.measurements

    mean, covariance, inv_covariance = get_statistics(x_healthy)

    fig, ax = plt.subplots()
    for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
        dataset = SlidingDataset(
            parquet_file=os.path.join(dataset_root, "synthetic_anomalies", f"{cfg.unit}_anomaly_{name}.parquet"),
            operating_mode=cfg.operating_mode,
            transient=cfg.transient,
            window_size=cfg.window_size,
            features=cfg.features,
            device=device,
            downsampling=cfg.measurement_downsampling,
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
        t2 = t2.clip(max=t2.mean() + 3 * t2.std())
        ax2.plot((t2 - t2.min()) / (t2.max() - t2.min()))
        ax2.set_title(name)

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.legend()
    plt.show()


def get_latent_features(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Returns a tensor of shape (n_samples, n_features) of all the latent features for the given model and data."""

    model.eval()
    all_latent = []
    with torch.no_grad():
        for x, _, _ in tqdm(dataloader):
            _, latent = model(x)
            all_latent.append(latent)

    return torch.concatenate(all_latent)


def get_statistics(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the mean, covariance and inverse covariance of the input tensor, of shape (n_samples, n_features)."""

    mean = input.mean(dim=0, keepdim=True)
    centered = input - mean
    covariance = (centered.T @ centered) / (input.shape[0] - 1)
    inv_covariance = torch.linalg.inv(covariance)
    return mean.squeeze(), covariance, inv_covariance


def kl_divergence(latent: torch.Tensor, rho: float) -> torch.Tensor:
    """Computes the KL-divergence of the batch. Input shape (n_samples, n_features)."""

    rho_hat = torch.mean(latent, dim=0)
    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return torch.sum(kl)
