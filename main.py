from dataloader import DataLoader, SlidingDataset, SlidingLabeledDataset, create_train_dataloaders
from model import SimpleAE, ConvAE
from train import train_autoencoder
from utils import seed_all, select_device
import config  # noqa F401

import os
import yaml
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import RocCurveDisplay
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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


def main(args: argparse.Namespace) -> None:
    cfg = eval(f"config.{args.model}Config")()
    model_name = f"{args.model}_{args.unit}_{args.operating_condition}"
    print(model_name)
    exit()

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO]: Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    device = select_device()
    print(f"Using device: {device}")

    seed_all(42)

    window_size = 1
    dataset = SlidingDataset(
        parquet_file="Dataset/VG5_generator_data_training_measurements.parquet",
        # parquet_file="Dataset/synthetic_anomalies/VG5_anomaly_01_type_a.parquet",
        operating_mode="turbine",
        window_size=window_size,
        device=device,
    )

    train_loader, val_loader = create_train_dataloaders(dataset, batch_size=256, validation_split=0.1)

    # model = ConvAE(input_channels=dataset[0].size(0), input_length=window_size).to(device)
    model = SimpleAE(input_features=dataset[0].size(0)).to(device)

    if False:
        train_autoencoder(model, train_loader, val_loader, n_epochs=200, learning_rate=1e-3, experiment="simple")
    else:
        model.load_state_dict(torch.load("models/best_model.pt", weights_only=True))
        model.eval()

        """with torch.no_grad():
            x = dataset[0].unsqueeze(0)
            reconstruction = model(x)
            x = x.cpu().numpy()
            reconstruction = reconstruction.cpu().numpy()
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(x[0, 0, :], label="x")
            axes[0].plot(reconstruction[0, 0, :], label="pred")
            axes[1].plot(x[0, 17, :], label="x")
            axes[1].plot(reconstruction[0, 17, :], label="pred")
            axes[0].legend()
            axes[1].legend()
            plt.show()"""

        """fig, ax = plt.subplots()

        def evaluate(dataset_type, parquet_file):
            dataset = SlidingDataset(
                parquet_file=parquet_file,
                operating_mode="turbine",
                window_size=window_size,
                device=device,
            )

            test_loader = DataLoader(dataset, batch_size=256)
            spes = compute_spes(test_loader, model)
            print(f"{dataset_type}: Mean = {spes.mean()}, Std = {spes.std()}, min = {spes.min()}, max = {spes.max()}")

            ax.hist(spes[spes < 100.0], density=True, bins=200, alpha=0.5, label=dataset_type, log=False)

        evaluate("training", "Dataset/VG5_generator_data_training_measurements.parquet")
        evaluate("synthetic_01_a", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_a.parquet")
        evaluate("synthetic_01_b", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_b.parquet")
        evaluate("synthetic_01_c", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_c.parquet")
        evaluate("synthetic_02_a", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_a.parquet")
        evaluate("synthetic_02_b", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_b.parquet")
        evaluate("synthetic_02_c", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_c.parquet")
        evaluate("testing", "Dataset/VG5_generator_data_testing_real_measurements.parquet")
        plt.legend()"""

        fig, ax = plt.subplots()
        for name in ["01_type_a", "01_type_b", "01_type_c", "02_type_a", "02_type_b", "02_type_c"]:
            spes, labels = compute_spes_and_labels(
                model, f"Dataset/synthetic_anomalies/VG5_anomaly_{name}.parquet", window_size, device
            )
            RocCurveDisplay.from_predictions(y_true=labels, y_pred=spes, ax=ax, name=name)

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["SimpleAE", "ConvAE"], default="SimpleAE")
    parser.add_argument("--unit", type=str, choices=["VG4", "VG5", "VG6"], default="VG5")
    parser.add_argument(
        "--operating_condition", type=str, choices=["pump", "turbine", "short_circuit"], default="turbine"
    )
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    main(args)
