from dataloader import DataLoader, SlidingDataset, SlidingLabeledDataset, create_dataloaders
from train import train
from utils import seed_all, select_device, dump_yaml
from model import *
from config import *

import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import RocCurveDisplay
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def load_config(args: argparse.Namespace) -> Config:
    # Get defaults from config dataclass
    try:
        # Use specialized config if it exists
        cfg: Config = eval(f"{args.model}Config")()
    except NameError:
        cfg = Config()

    # Override with CLI args
    if args.seed is not None:
        cfg.seed = args.seed
    if args.unit is not None:
        cfg.unit = args.unit
    if args.operating_mode is not None:
        cfg.operating_mode = args.operating_mode

    return cfg


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args)
    model_name = f"{cfg.model}_{cfg.unit}_{cfg.operating_mode}"

    log_dir = os.path.join("logs", model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.abspath(log_dir)

    print("=" * os.get_terminal_size()[0])
    print("")
    device = select_device()
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")
    print(f"Config: {cfg}")

    dump_yaml(os.path.join(log_dir, "config.yaml"), cfg)

    seed_all(cfg.seed)

    exit()

    # model = ConvAE(input_channels=dataset[0].size(0), input_length=window_size).to(device)
    model = SimpleAE(input_features=dataset[0].size(0)).to(device)

    if False:
        train(model, train_loader, val_loader, n_epochs=200, learning_rate=1e-3, experiment="simple")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["SimpleAE", "ConvAE"], default="SimpleAE")
    parser.add_argument("--unit", type=str, choices=["VG4", "VG5", "VG6"], default=None)
    parser.add_argument("--operating_mode", type=str, choices=["pump", "turbine", "short_circuit"], default=None)
    parser.add_argument("--dataset_root", type=str, default="Dataset")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    main(args)
