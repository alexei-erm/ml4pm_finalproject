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
        subsampling=cfg.subsampling,
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
            mean=training_dataset.mean,
            std=training_dataset.std,
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

    x_healthy = training_dataset.measurements.cpu().numpy()
    indices = np.arange(x_healthy.shape[0])
    np.random.shuffle(indices)
    x_healthy = x_healthy[indices[:: cfg.subsampling], :]
    print(x_healthy.shape)

    fig, axes = plt.subplots(1, 2)
    fig2, axes2 = plt.subplots(2, 3)
    axes2 = axes2.flatten()

    for i, name in enumerate(["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]):
        print(f"Testing on synthetic anomalies {name}")
        dataset = SlidingDataset(
            parquet_file=os.path.join(dataset_root, "synthetic_anomalies", f"{cfg.unit}_anomaly_{name}.parquet"),
            operating_mode=cfg.operating_mode,
            transient=cfg.transient,
            window_size=cfg.window_size,
            features=cfg.features,
            downsampling=cfg.measurement_downsampling,
            device=device,
            mean=training_dataset.mean,
            std=training_dataset.std,
        )

        x_test = dataset.measurements.cpu().numpy()

        t2 = hotelling_t2(train=x_healthy, test=x_test)

        labels = dataset.ground_truth.cpu().numpy()
        RocCurveDisplay.from_predictions(y_true=labels, y_pred=t2, ax=axes[0], name=name)

        axes2[i].plot(labels, label="label")
        t2 = t2.clip(max=t2.mean() + 3 * t2.std())
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())
        axes2[i].plot(t2, label="T2")
        axes2[i].legend()
        axes2[i].set_title(name)

    axes[0].plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.legend()
    plt.show()


def fit_kpca(cfg: Config, dataset_root: str, log_dir: str) -> None:
    dump_yaml(os.path.join(log_dir, "config.yaml"), cfg)
    dump_pickle(os.path.join(log_dir, "config.pkl"), cfg)

    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    training_dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        device="cpu",
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
    )

    x_healthy = training_dataset.measurements.cpu().numpy()
    indices = np.arange(x_healthy.shape[0])
    np.random.shuffle(indices)
    x_healthy = x_healthy[indices[:: cfg.subsampling], :]

    gamma, n_components = optimize_kpca(
        input=x_healthy,
        num_gamma=9,
        num_components=40,
        gamma_log_min=-1.0,
        gamma_log_max=1.0,
        name="test",
    )
    kpca_params = {"gamma": gamma, "n_components": n_components}
    dump_yaml(os.path.join(log_dir, "kpca_params.yaml"), kpca_params)


def test_kpca(cfg: Config, dataset_root: str, log_dir: str) -> None:
    parquet_file = os.path.abspath(
        os.path.join(dataset_root, f"{cfg.unit}_generator_data_training_measurements.parquet")
    )
    training_dataset = SlidingDataset(
        parquet_file=parquet_file,
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        device="cpu",
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
    )

    summer = np.isin(training_dataset.index.astype("datetime64[M]").astype(int) % 12 + 1, [5, 6, 7, 8])
    winter = np.isin(training_dataset.index.astype("datetime64[M]").astype(int) % 12 + 1, [10, 11, 12, 1])

    x_healthy_summer = training_dataset.measurements[summer, :]
    x_healthy_winter = training_dataset.measurements[winter, :]
    x_healthy = training_dataset.measurements

    gamma = 0.021
    n_components = 37

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        eigen_solver="randomized",
        n_jobs=-1,
    )
    x_healthy_pca = kpca.fit_transform(x_healthy)

    fig, axes = plt.subplots(1, 2)
    fig2, axes2 = plt.subplots(2, 3)
    axes2 = axes2.flatten()

    for i, name in enumerate(["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]):
        print(f"Testing on synthetic anomalies {name}")
        dataset = SlidingDataset(
            parquet_file=os.path.join(dataset_root, "synthetic_anomalies", f"{cfg.unit}_anomaly_{name}.parquet"),
            operating_mode=cfg.operating_mode,
            transient=cfg.transient,
            window_size=cfg.window_size,
            features=cfg.features,
            downsampling=cfg.measurement_downsampling,
            device="cpu",
            mean=training_dataset.mean,
            std=training_dataset.std,
        )
        # dataset.measurements = torch.clone(x_healthy)
        dataset.ground_truth = torch.zeros(dataset.measurements.shape[0], dtype=torch.bool, device="cpu")
        dataset.ground_truth[dataset.ground_truth.shape[0] // 3 : dataset.ground_truth.shape[0] // 3 * 2] = 1
        anomalous_cols = torch.from_numpy(dataset.df.columns.to_series().str.match("stat_coil_*").to_numpy())
        # print(dataset.df.columns[anomalous_cols])
        anomalies = np.where(anomalous_cols, 5.0 / dataset.std, 0)
        dataset.measurements[dataset.ground_truth, :] += anomalies

        x_test = dataset.measurements.cpu().numpy()
        x_test_pca = kpca.transform(x_test)
        x_test_reconstructed = kpca.inverse_transform(x_test_pca)
        msre = np.mean(np.square(x_test_reconstructed - x_test), axis=1)

        labels = dataset.ground_truth.cpu().numpy()

        t2 = hotelling_t2(train=x_healthy_pca, test=x_test_pca)

        from sklearn.metrics import roc_auc_score

        """scores = np.zeros((11, 17))
        for i, gamma in enumerate(1.0 / x_healthy_pca.shape[1] * np.logspace(-5, 2, num=11)):
            for j, nu in enumerate(np.logspace(-5, -0.5, num=17)):
                print(f"{i} {j} {gamma:.8f} {nu:.8f}", end="")
                ocsvm = OneClassSVM(kernel="rbf", cache_size=4000, gamma=gamma, nu=nu)
                ocsvm.fit(x_healthy_pca)
                # pred = -ocsvm.score_samples(x_test_pca)
                pred = -ocsvm.decision_function(x_test_pca)
                score = roc_auc_score(y_true=labels, y_score=pred)
                scores[i, j] = score
                print(f" -> {score:.4f}")

        print(scores)
        plt.imshow(scores)
        plt.show()"""
        ocsvm = OneClassSVM(kernel="rbf", cache_size=4000, gamma=0.003, nu=0.31)
        ocsvm.fit(x_healthy_pca)
        t2 = -ocsvm.score_samples(x_test_pca)

        RocCurveDisplay.from_predictions(y_true=labels, y_pred=t2, ax=axes[0], name=name)
        RocCurveDisplay.from_predictions(y_true=labels, y_pred=msre, ax=axes[1], name=name)

        axes2[i].plot(labels, label="label")
        # t2 = t2.clip(max=4 * t2.mean())
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())
        # msre = msre.clip(max=4 * msre.mean())
        msre = (msre - msre.min()) / (msre.max() - msre.min())
        axes2[i].plot(t2, label="T2")
        axes2[i].plot(msre, label="MSRE")
        axes2[i].legend()
        axes2[i].set_title(name)

    axes[0].plot([0, 1], [0, 1], color="k", linestyle="--")
    axes[1].plot([0, 1], [0, 1], color="k", linestyle="--")
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


def optimize_kpca(
    input: np.ndarray, num_gamma: int, gamma_log_min: float, gamma_log_max: float, num_components: int, name: str
) -> tuple[float, int]:

    print(f"Optimizing KPCA model, dataset shape: {input.shape}")

    data_train, data_val = train_test_split(input, test_size=0.2)

    gamma_values = 1.0 / input.shape[1] * np.logspace(gamma_log_min, gamma_log_max, num=num_gamma)
    errors_gamma = []
    components = []
    errors_components = []
    elbows = []

    for gamma in gamma_values:
        print(f"Optimizing number of components for gamma = {gamma:.4f}")

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
        n_components = np.linspace(start=1, stop=hyperdimension, num=num_components, endpoint=True, dtype=int)
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

    best_gamma = float(gamma_values[best_gamma_index])
    best_components = int(components[best_gamma_index][elbows[best_gamma_index]])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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
    ax[1].axvline(x=best_components, linestyle="--", color="r", label=f"{best_components} components")
    ax[1].set_xlabel("Number of principal components")
    ax[1].set_ylabel("MSRE")
    ax[1].set_title(f"Error vs n_components for optimal gamma = {best_gamma:.4f}")
    ax[1].legend()
    fig.savefig(f"plots/kpca_{name}.png")

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


def hotelling_t2(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    # Compute mean and covariance of the training data
    mean_vector = np.mean(train, axis=0)
    covariance_matrix = np.cov(train, rowvar=False)

    # Add a small regularization term to ensure numerical stability
    reg_covariance_matrix = covariance_matrix + 1e-7 * np.eye(covariance_matrix.shape[0])

    # Invert the regularized covariance matrix
    covariance_matrix_inv = np.linalg.inv(reg_covariance_matrix)

    # Center the test data by subtracting the mean of the training data
    centered_test = test - mean_vector

    # Compute the T2 statistic for each test sample
    t2_scores = np.sum((centered_test @ covariance_matrix_inv) * centered_test, axis=1)

    return t2_scores
