from dataloader import DataLoader, SlidingDataset, create_train_dataloaders
from model import ConvolutionalAutoencoder
from train import train_autoencoder
from utils import seed_all, select_device

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
            reconstruction = model(x)
            spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
            spes.append(spe.cpu().numpy())

    return np.concatenate(spes)


def main() -> None:
    seed_all(42)

    device = select_device()
    print(f"Using device: {device}")

    window_size = 50
    dataset = SlidingDataset(
        parquet_file="Dataset/VG5_generator_data_training_measurements.parquet",
        # parquet_file="Dataset/synthetic_anomalies/VG5_anomaly_01_type_a.parquet",
        operating_mode="turbine",
        window_size=window_size,
        device=device,
    )

    train_loader, val_loader = create_train_dataloaders(dataset, batch_size=256, validation_split=0.2)

    model = ConvolutionalAutoencoder(input_channels=dataset[0].size(0), input_length=window_size).to(device)

    if False:
        train_autoencoder(model, train_loader, val_loader, n_epochs=100)
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

        fig, ax = plt.subplots()

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

            spes = spes[spes < 5000.0]
            ax.hist(spes, density=True, bins=200, alpha=0.5, label=dataset_type, log=False)

        evaluate("training", "Dataset/VG5_generator_data_training_measurements.parquet")
        evaluate("synthetic_01_a", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_a.parquet")
        evaluate("synthetic_01_b", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_b.parquet")
        evaluate("synthetic_01_c", "Dataset/synthetic_anomalies/VG5_anomaly_01_type_c.parquet")
        evaluate("synthetic_02_a", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_a.parquet")
        evaluate("synthetic_02_b", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_b.parquet")
        evaluate("synthetic_02_c", "Dataset/synthetic_anomalies/VG5_anomaly_02_type_c.parquet")
        evaluate("testing", "Dataset/VG5_generator_data_testing_real_measurements.parquet")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
