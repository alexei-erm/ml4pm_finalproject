from config import Config
from dataloader import *
from model import *  # noqa F401
from utils import *

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
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

    # Load training data
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

    # Load model
    model_type = eval(cfg.model.value)
    model = model_type(
        input_channels=training_dataset.measurements.shape[-1], window_size=cfg.window_size, cfg=cfg.model_cfg
    ).to(device)

    model_state_path = os.path.join(log_dir, "best_model.pt" if load_best else "model.pt")
    model.load_state_dict(torch.load(model_state_path, weights_only=True, map_location=device))

    model.eval()

    # Compute statistics of the training set
    train_loader = create_dataloader(training_dataset, batch_size=cfg.batch_size)
    train_x, train_pred, train_latent = get_all_data(model, train_loader)
    latent_mean, latent_covariance, inv_latent_covariance = get_statistics(train_latent)

    # Compute scores on training set and set thresholds accordingly
    train_msre = torch.mean(torch.square(train_pred - train_x), dim=(1, 2))

    latent_diff = train_latent - latent_mean.unsqueeze(0)
    train_t2 = torch.einsum("bi,ij,bj->b", latent_diff, inv_latent_covariance, latent_diff)

    threshold_t2 = np.percentile(train_t2.cpu().numpy(), 99.9)
    threshold_msre = np.percentile(train_msre.cpu().numpy(), 99.9)

    if cfg.unit != "VG4":
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:

            # Load synthetic anomalies
            testing_dataset = SlidingDataset(
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

            loader = create_dataloader(testing_dataset, batch_size=cfg.batch_size)

            indices = []
            preds = []
            xs = []
            labels = []
            msres = []
            t2s = []

            # Evaluate model
            with torch.no_grad():
                for x, y, index in tqdm(loader):
                    x[y==1] += 0.5
                    xs.append(x)
                    labels.append(y)
                    indices.append(index)

                    reconstruction, latent = model(x)
                    preds.append(reconstruction)

                    msre = torch.mean(torch.square(reconstruction - x), dim=(1, 2))
                    msres.append(msre)

                    latent_diff = latent - latent_mean.unsqueeze(0)
                    t2 = torch.einsum("bi,ij,bj->b", latent_diff, inv_latent_covariance, latent_diff)
                    t2s.append(t2)

            xs = torch.concatenate(xs).cpu().numpy()
            preds = torch.concatenate(preds).cpu().numpy()
            indices = np.concatenate(indices)
            labels = torch.concatenate(labels).cpu().numpy()
            msres = torch.concatenate(msres).cpu().numpy()
            t2s = torch.concatenate(t2s).cpu().numpy()

            fault_t2 = t2s >= threshold_t2
            fault_msre = msres >= threshold_msre

            indices = indices[:, -1]
            labels = labels[:, -1]
            feature_index = 0
            feature_name = testing_dataset.df.columns[feature_index]
            xs = xs[:, -1, feature_index]
            preds = preds[:, -1, feature_index]

            t2s = clip_positive_outliers(t2s, threshold=10)
            msres = clip_positive_outliers(msres, threshold=10)

            plot_scores(
                t2=t2s,
                msre=msres,
                fault_t2=fault_t2,
                fault_msre=fault_msre,
                threshold_t2=threshold_t2,
                threshold_msre=threshold_msre,
                labels=labels,
                x_true=xs,
                x_pred=preds,
                title=f"{cfg.model.value}, operating mode {cfg.operating_mode}: anomaly scores\n"
                f"(T2 and MSRE) and signal reconstruction for {feature_name}",
                file_name=f"plots/{cfg.model.value}_{cfg.unit}_{cfg.operating_mode}_{name}.png",
            )


def fit_spc(cfg: Config, dataset_root: str) -> None:
    """Fits and evaluates a Statistical Process Control model based on the Hotelling T2 score"""

    # Load training data
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
    x_healthy_train, x_healthy_val = train_test_split(training_dataset.measurements.cpu().numpy(), test_size=0.2)
    x_healthy_train = x_healthy_train[:: cfg.subsampling]
    x_healthy_val = x_healthy_val[:: cfg.subsampling]

    # Compute T2 for validation set and select threshold accordingly
    val_t2, _ = hotelling_t2(train=x_healthy_train, test=x_healthy_val)
    threshold = np.percentile(val_t2, 99.95)

    if cfg.unit != "VG4":
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            # Load synthetic anomalies
            testing_dataset = SlidingDataset(
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
            x_test = testing_dataset.measurements.cpu().numpy()

            # Compute T2 and individual feature contributions
            t2, contributions = hotelling_t2(train=x_healthy_train, test=x_test)

            # Fault detection
            pred = t2 >= threshold

            # Compute metrics
            labels = testing_dataset.ground_truth.cpu().numpy()
            tpr, fpr = compute_tpr_fpr(pred=pred, labels=labels)
            ttd = compute_time_to_detection(pred=pred, labels=labels, index=testing_dataset.index)
            mean_ttd = np.mean([t for t in ttd if t is not None])
            print(f"Synthetic anomalies {name}: TPR={tpr:.4f}, FPR={fpr:.4f}, mean TTD={mean_ttd:.2f}h")

            t2 = clip_positive_outliers(t2, threshold=100.0)

            if cfg.transient:
                file_name = f"plots/spc_{cfg.unit}_{name}_transient.png"
            else:
                file_name = f"plots/spc_{cfg.unit}_{name}.png"

            plot_score_and_contributions(
                score=t2,
                pred=pred,
                threshold=threshold,
                contributions=contributions,
                labels=labels,
                index=testing_dataset.index,
                features=testing_dataset.df.columns,
                score_type="T2",
                title=f"SPC: anomaly T2 score and fault detection for {cfg.unit} synthetic anomalies {name}\n"
                f"TPR={tpr:.4f}, FPR={fpr:.4f}, mean TTD={mean_ttd:.2f}h",
                file_name=file_name,
            )

    # Load real testing dataset
    testing_dataset = SlidingDataset(
        parquet_file=os.path.join(dataset_root, f"{cfg.unit}_generator_data_testing_real_measurements.parquet"),
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
        device="cpu",
        mean=training_dataset.mean,
        std=training_dataset.std,
    )
    x_test = testing_dataset.measurements.cpu().numpy()

    # Compute T2 and individual feature contributions
    t2, contributions = hotelling_t2(train=x_healthy_train, test=x_test)

    # Fault detection
    pred = t2 >= threshold

    t2 = clip_positive_outliers(t2, threshold=100.0)
    contributions = clip_positive_outliers(np.abs(contributions), threshold=20)

    if cfg.transient:
        file_name = f"plots/spc_{cfg.unit}_testing_real_transient.png"
    else:
        file_name = f"plots/spc_{cfg.unit}_testing_real.png"

    plot_score_and_contributions(
        score=t2,
        pred=pred,
        threshold=threshold,
        contributions=contributions,
        labels=None,
        index=testing_dataset.index,
        features=testing_dataset.df.columns,
        score_type="T2",
        title=f"SPC: anomaly T2 score and fault detection for {cfg.unit},\nfor the real testing dataset",
        file_name=file_name,
    )


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
        file_name=f"plots/kpca_{cfg.unit}_{cfg.operating_mode}_optimization.png",
    )
    kpca_params = {"gamma": gamma, "n_components": n_components}
    dump_yaml(os.path.join(log_dir, "kpca_params.yaml"), kpca_params)


def test_kpca(cfg: Config, dataset_root: str, log_dir: str) -> None:

    # Load training data
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
    x_healthy_train, x_healthy_val = train_test_split(training_dataset.measurements.cpu().numpy(), test_size=0.2)
    x_healthy_train = x_healthy_train[:: cfg.subsampling]
    x_healthy_val = x_healthy_val[:: cfg.subsampling]

    # Fit KernelPCA model
    params = load_yaml(os.path.join(log_dir, "kpca_params.yaml"))
    gamma = params["gamma"]
    n_components = params["n_components"]

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        eigen_solver="randomized",
        n_jobs=-1,
    )
    kpca.fit(x_healthy_train)

    # Compute reconstruction error on validation samples and select threshold accordingly
    val_msre, _ = compute_msre(kpca, x=x_healthy_val)
    threshold = np.percentile(val_msre, 99.5)

    if cfg.unit != "VG4":
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            # Load synthetic anomalies
            testing_dataset = SlidingDataset(
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
            x_test = testing_dataset.measurements.cpu().numpy()

            # Compute MSRE and individual feature contributions
            msre, contributions = compute_msre(kpca, x=x_test)

            # Fault detection
            pred = msre >= threshold

            # Compute metrics
            labels = testing_dataset.ground_truth.cpu().numpy()
            tpr, fpr = compute_tpr_fpr(pred=pred, labels=labels)
            ttd = compute_time_to_detection(pred=pred, labels=labels, index=testing_dataset.index)
            filtered_ttd = [t for t in ttd if t is not None]
            mean_ttd = np.mean(filtered_ttd) if len(filtered_ttd) > 0 else np.nan
            print(f"Synthetic anomalies {name}: TPR={tpr:.4f}, FPR={fpr:.4f}, mean TTD={mean_ttd:.2f}h")

            msre = clip_positive_outliers(msre, threshold=50.0)

            plot_score_and_contributions(
                score=msre,
                pred=pred,
                threshold=threshold,
                contributions=contributions,
                labels=labels,
                index=testing_dataset.index,
                features=testing_dataset.df.columns,
                score_type="MSRE",
                title=f"KPCA: anomaly MSRE score and fault detection for {cfg.unit} synthetic anomalies {name}\n"
                f"TPR={tpr:.4f}, FPR={fpr:.4f}, mean TTD={mean_ttd:.2f}h",
                file_name=f"plots/kpca_{cfg.unit}_{cfg.operating_mode}_{name}.png",
            )

    # Load real testing dataset
    testing_dataset = SlidingDataset(
        parquet_file=os.path.join(dataset_root, f"{cfg.unit}_generator_data_testing_real_measurements.parquet"),
        operating_mode=cfg.operating_mode,
        transient=cfg.transient,
        window_size=cfg.window_size,
        features=cfg.features,
        downsampling=cfg.measurement_downsampling,
        device="cpu",
        mean=training_dataset.mean,
        std=training_dataset.std,
    )
    x_test = testing_dataset.measurements.cpu().numpy()

    # Compute MSRE and individual feature contributions
    msre, contributions = compute_msre(kpca, x=x_test)

    # Fault detection
    pred = msre >= threshold

    msre = clip_positive_outliers(msre, threshold=50.0)
    contributions = clip_positive_outliers(np.abs(contributions), threshold=10)

    plot_score_and_contributions(
        score=msre,
        pred=pred,
        threshold=threshold,
        contributions=contributions,
        labels=None,
        index=testing_dataset.index,
        features=testing_dataset.df.columns,
        score_type="MSRE",
        title=f"KPCA: anomaly MSRE score and fault detection for {cfg.unit},\nfor the real testing dataset",
        file_name=f"plots/kpca_{cfg.unit}_{cfg.operating_mode}_testing_real.png",
    )


def get_all_data(model: nn.Module, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a tensor of all the input signals and reconstructions,
    as well as all the latent features for the given model and data."""

    model.eval()

    all_x = []
    all_reconstructions = []
    all_latent = []
    with torch.no_grad():
        for x, _, _ in tqdm(dataloader):
            x_reconstructed, latent = model(x)
            all_x.append(x)
            all_reconstructions.append(x_reconstructed)
            all_latent.append(latent)

    return torch.concatenate(all_x), torch.concatenate(all_reconstructions), torch.concatenate(all_latent)


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
    input: np.ndarray, num_gamma: int, gamma_log_min: float, gamma_log_max: float, num_components: int, file_name: str
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
    fig.savefig(file_name)

    return best_gamma, best_components


def kneedle(x: np.ndarray, y: np.ndarray) -> int:
    """Finds the index of the elbow using the Kneedle algorithm."""

    # If the values are decreasing, flip them
    if y[-1] < y[0]:
        y = -y

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Find the index of the point furthest from the diagonal (elbow point)
    return np.argmax(y_norm - x_norm)


def hotelling_t2(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Hotelling's T2 statistic and feature contributions for each test sample."""

    # Compute mean and covariance of the training data
    mean_vector = np.mean(train, axis=0)
    covariance_matrix = np.cov(train, rowvar=False)

    # Add a small regularization term to ensure numerical stability
    reg_covariance_matrix = covariance_matrix + 1e-7 * np.eye(covariance_matrix.shape[0])

    # Invert the regularized covariance matrix
    covariance_matrix_inv = np.linalg.inv(reg_covariance_matrix)

    # Center the test data by subtracting the mean of the training data
    centered_test = test - mean_vector

    # Compute feature contributions
    feature_contributions = (centered_test @ covariance_matrix_inv) * centered_test

    # Compute the T2 statistic for each test sample
    t2_scores = np.sum(feature_contributions, axis=1)

    return t2_scores, feature_contributions


def compute_msre(kpca: KernelPCA, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Mean Square Reconstruction Error and individual feature contributions."""

    x_pca = kpca.transform(x)
    x_reconstructed = kpca.inverse_transform(x_pca)
    sre = np.square(x_reconstructed - x)
    msre = np.mean(sre, axis=1)
    contributions = sre

    return msre, contributions


def compute_tpr_fpr(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Computes the True Positive Rate (TPR) and False Positive Rate (FPR)"""

    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))
    tn = np.sum((pred == 0) & (labels == 0))

    # Compute TPR and FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


def clip_positive_outliers(input: np.ndarray, threshold=1.0) -> np.ndarray:
    """Clips outliers from an array based on the interquartile range"""

    # Compute Q1 and Q3
    q1 = np.percentile(input, 25)
    q3 = np.percentile(input, 75)
    iqr = q3 - q1

    # Compute bound and clip
    upper_bound = q3 + threshold * iqr
    return input.clip(max=upper_bound)


def compute_time_to_detection(pred: np.ndarray, labels: np.ndarray, index: np.ndarray) -> list[float | None]:
    """
    Compute the time to detection (TTD) for each labeled fault.
    Returns a list of time to detection for each labeled fault, in hours.
    If no detection occurs, returns None for that fault.
    """

    ttds = []
    fault_indices = np.where(labels == 1)[0]  # Indices where faults are labeled

    # Iterate through each fault index
    for fault_start in fault_indices:
        if fault_start > 0 and labels[fault_start - 1] == 1:
            # Skip if this fault index is part of an ongoing fault already accounted for
            continue

        # Find the first prediction after the fault starts
        detection_indices = np.where(pred[fault_start:] == 1)[0]
        if len(detection_indices) > 0:
            # Time to detection is the first detection relative to fault start
            ttd = (
                float(
                    (index[fault_start + detection_indices[0]] - index[fault_start])
                    .astype("timedelta64[s]")
                    .astype(float)
                )
                / 3600.0
            )

            ttds.append(ttd)
        else:
            ttds.append(None)  # No detection for this fault

    return ttds


def plot_scores(
    t2: np.ndarray,
    msre: np.ndarray,
    fault_t2: np.ndarray,
    fault_msre: np.ndarray,
    threshold_t2: float,
    threshold_msre: float,
    labels: np.ndarray | None,
    x_true: np.ndarray,
    x_pred: np.ndarray,
    title: str,
    file_name: str,
) -> None:
    """Plots the ground truth label if not None, anomaly score and threshold,
    and saves the figure."""

    if labels is not None:
        fig, (ax_label, ax_t2, ax_msre, ax_signal) = plt.subplots(
            4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 3, 3, 3]}, figsize=(12, 22)
        )
        ax_label.plot(labels, label="Ground truth")
        ax_label.set_xlabel("Sample")
        ax_label.set_yticks([0, 1])
        ax_label.set_yticklabels(["Healthy", "Faulty"])
        ax_label.legend()
    else:
        fig, (ax_t2, ax_msre, ax_signal) = plt.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}, figsize=(12, 17)
        )

    ax_t2.plot(t2, label="T2")
    for i, (start, end) in enumerate(
        zip(np.where(np.diff(fault_t2, prepend=0) == 1)[0], np.where(np.diff(fault_t2, append=0) == -1)[0])
    ):
        ax_t2.axvspan(start, end, color="red", alpha=0.3, label="Detected fault" if i == 0 else None)
    ax_t2.axhline(y=threshold_t2, linestyle="--", color="k", label="T2 threshold")
    ax_t2.set_xlabel("Sample")
    ax_t2.set_ylabel("T2")
    ax_t2.legend()

    ax_msre.plot(msre, label="MSRE")
    for i, (start, end) in enumerate(
        zip(np.where(np.diff(fault_msre, prepend=0) == 1)[0], np.where(np.diff(fault_msre, append=0) == -1)[0])
    ):
        ax_msre.axvspan(start, end, color="red", alpha=0.3, label="Detected fault" if i == 0 else None)
    ax_msre.axhline(y=threshold_msre, linestyle="--", color="k", label="MSRE threshold")
    ax_msre.set_xlabel("Sample")
    ax_msre.set_ylabel("MSRE")
    ax_msre.legend()

    ax_signal.plot(x_true, label="Original signal")
    ax_signal.plot(x_pred, label="Reconstruction")
    ax_signal.set_xlabel("Sample")
    ax_signal.legend()

    fig.tight_layout(pad=1.5)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.95)
    fig.savefig(file_name)


def plot_score_and_contributions(
    score: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    contributions: np.ndarray,
    labels: np.ndarray | None,
    index: np.ndarray,
    features: list[str],
    score_type: str,
    title: str,
    file_name: str,
) -> None:
    """Plots the ground truth label if not None, anomaly score, threshold and individual feature contributions,
    and saves the figure."""

    if labels is not None:
        fig, (ax_label, ax_score, ax_contrib) = plt.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2, 8]}, figsize=(12, 22)
        )
        ax_label.plot(labels, label="Ground truth")
        ax_label.set_xlabel("Time")
        ax_label.set_yticks([0, 1])
        ax_label.set_yticklabels(["Healthy", "Faulty"])
        ax_label.legend()
    else:
        fig, (ax_score, ax_contrib) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 4]}, figsize=(12, 18)
        )

    ax_score.plot(score, label=score_type)
    for i, (start, end) in enumerate(
        zip(np.where(np.diff(pred, prepend=0) == 1)[0], np.where(np.diff(pred, append=0) == -1)[0])
    ):
        ax_score.axvspan(start, end, color="red", alpha=0.3, label="Detected fault" if i == 0 else None)
    ax_score.axhline(y=threshold, linestyle="--", color="k", label="Threshold")
    ax_score.set_xlabel("Time")
    ax_score.set_ylabel(score_type)
    ax_score.legend()

    ax_contrib.imshow(np.abs(contributions).T, aspect="auto", cmap="inferno", interpolation="none")
    ax_contrib.set_xlabel("Time")
    ax_contrib.set_ylabel("Features")
    ax_contrib.set_yticks(ticks=np.arange(len(features)), labels=features)
    x_ticks = np.linspace(start=0, stop=len(index) - 1, endpoint=True, num=20, dtype=int)
    ax_contrib.set_xticks(x_ticks)
    ax_contrib.set_xticklabels(np.datetime_as_string(index[x_ticks], unit="D"), rotation=45, ha="right")

    fig.tight_layout(pad=1.5)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.95)
    fig.savefig(file_name)
