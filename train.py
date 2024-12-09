from config import Config
from dataloader import *
from model import *  # noqa F401
from utils import *

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.svm import OneClassSVM
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

    summer = np.isin(training_dataset.index.astype("datetime64[M]").astype(int) % 12 + 1, [6, 7])
    winter = np.isin(training_dataset.index.astype("datetime64[M]").astype(int) % 12 + 1, [11, 12])

    x_healthy_summer = training_dataset.measurements[summer, :]
    x_healthy_winter = training_dataset.measurements[winter, :]

    # gamma, n_components = optimize_kpca(x_healthy_summer.cpu().numpy())
    gamma = 0.0235
    n_components = 43

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        eigen_solver="randomized",
        n_jobs=-1,
    )
    x_healthy_pca = kpca.fit_transform(x_healthy_summer.cpu().numpy())

    fig, ax = plt.subplots()
    fig2, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    for i, name in enumerate(["01_type_a", "01_type_b", "01_type_c"]):  # , "02_type_a", "02_type_b", "02_type_c"]):
        print(f"Testing on synthetic anomalies {name}")
        dataset = SlidingDataset(
            parquet_file=os.path.join(dataset_root, "synthetic_anomalies", f"{cfg.unit}_anomaly_{name}.parquet"),
            operating_mode=cfg.operating_mode,
            transient=cfg.transient,
            window_size=cfg.window_size,
            features=cfg.features,
            device=device,
            downsampling=cfg.measurement_downsampling,
        )
        dataset.measurements = x_healthy_winter
        dataset.ground_truth = torch.zeros(dataset.measurements.shape[0], dtype=torch.bool, device=device)
        dataset.ground_truth[1000:2000] = 1
        dataset.measurements[dataset.ground_truth, :] += 0.5

        # TODO: normalize test data using statistics from training set

        x_test = dataset.measurements.cpu().numpy()
        x_test_pca = kpca.transform(x_test)

        # ocsvm = OneClassSVM(kernel="rbf", nu=0.02)
        # ocsvm.fit(x_healthy_pca)
        # svm_pred = ocsvm.score_samples(x_test_pca)

        mean, covariance, inv_covariance = get_statistics(torch.from_numpy(x_healthy_pca))
        diff = torch.from_numpy(x_test_pca) - mean
        # mean, covariance, inv_covariance = get_statistics(x_healthy)
        # diff = dataset.measurements - mean
        t2 = torch.einsum("bi,ij,bj->b", diff, inv_covariance, diff)

        x_test_reconstructed = kpca.inverse_transform(x_test_pca)
        msre = np.mean(np.square(x_test_reconstructed - x_test), axis=1)

        labels = dataset.ground_truth.cpu().numpy()
        t2 = t2.cpu().numpy()
        RocCurveDisplay.from_predictions(y_true=labels, y_pred=t2, ax=ax, name=name)

        axes[i].plot(labels, label="label")
        t2 = t2.clip(max=4 * t2.mean())
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())
        msre = (msre - msre.min()) / (msre.max() - msre.min())
        axes[i].plot(t2, label="T2")
        axes[i].plot(msre, label="MSRE")
        # axes[i].plot(x_test[:, 0], label="x")
        # axes[i].plot(x_test_reconstructed[:, 0], label="Rec")
        # svm_pred = (svm_pred - svm_pred.min()) / (svm_pred.max() - svm_pred.min())
        # axes[i].plot(svm_pred - 2, label="SVM")
        axes[i].legend()
        axes[i].set_title(name)

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
    """Returns the mean, covariance and inverse covariance of the input tensor.
    Input shape: (n_samples, n_features)"""

    mean = input.mean(dim=0, keepdim=True)
    centered = input - mean
    covariance = (centered.T @ centered) / input.shape[0]
    inv_covariance = torch.linalg.inv(covariance)
    return mean.squeeze(), covariance, inv_covariance


def kl_divergence(latent: torch.Tensor, rho: float) -> torch.Tensor:
    """Computes the KL-divergence of the batch. Input shape (n_samples, n_features)."""

    rho_hat = torch.mean(latent, dim=0)
    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return torch.sum(kl)


def optimize_kpca(input: np.ndarray) -> tuple[float, int]:

    data_train, data_val = train_test_split(input, test_size=0.2)

    num_gamma = 11
    gamma_values = 1.0 / input.shape[1] * np.logspace(-1.5, 1.5, num=num_gamma)
    errors_gamma = []
    components = []
    errors_components = []
    elbows = []

    for gamma in gamma_values:
        kpca = KernelPCA(
            n_components=None,
            kernel="rbf",
            gamma=gamma,
            fit_inverse_transform=True,
            eigen_solver="randomized",
            n_jobs=-1,
        )
        kpca.fit(data_train)

        cumulative_variance = np.cumsum(kpca.eigenvalues_) / np.sum(kpca.eigenvalues_)
        hyperdimension = np.searchsorted(cumulative_variance, 0.99) + 1

        # Cross-validate to find optimal number of components
        errors = []
        n_components = np.linspace(start=1, stop=hyperdimension, num=100, endpoint=True, dtype=int)
        for n in tqdm(n_components):
            kpca = KernelPCA(
                n_components=n,
                kernel="rbf",
                gamma=gamma,
                fit_inverse_transform=True,
                eigen_solver="randomized",
                n_jobs=-1,
            )
            kpca.fit(data_train)

            data_val_transformed = kpca.transform(data_val)
            data_val_reconstructed = kpca.inverse_transform(data_val_transformed)

            error = np.mean(np.square(data_val_reconstructed - data_val))
            errors.append(error)

        errors = np.asarray(errors)

        elbow_index = kneedle(n_components, errors)

        errors_gamma.append(errors[elbow_index])
        components.append(n_components)
        errors_components.append(errors)
        elbows.append(elbow_index)

    best_gamma_index = np.argmin(errors_gamma)

    best_gamma = gamma_values[best_gamma_index]
    best_components = components[best_gamma_index][elbows[best_gamma_index]]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(
        gamma_values,
        errors_gamma,
        marker="o",
    )
    ax[0].set_xlabel("Gamma")
    ax[0].set_xscale("log")
    ax[0].set_ylabel("MSRE")
    ax[0].set_title("Error vs gamma")
    ax[1].plot(components[best_gamma_index], errors_components[best_gamma_index], marker="o", label="MSRE")
    ax[1].axvline(x=best_components, linestyle="--", color="k", label=f"{best_components} components")
    ax[1].set_xlabel("Number of principal components")
    ax[1].set_ylabel("MSRE")
    ax[1].set_title(f"Error vs number of components for optimal gamma = {best_gamma:.4f}")
    ax[1].legend()
    plt.show()

    return best_gamma, best_components


def kneedle(x: np.ndarray, y: np.ndarray) -> int:
    # If the values are decreasing, flip them
    if y[-1] < y[0]:
        y = -y

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Find the index of the point furthest from the diagonal (elbow point)
    return np.argmax(y_norm - x_norm)
