from dataloader import SlidingDataset, create_dataloaders
from model import ConvolutionalAutoencoder
from train import train_autoencoder
from utils import seed_all, select_device

import torch
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    seed_all(42)

    device = select_device()
    print(f"Using device: {device}")

    window_size = 50
    dataset = SlidingDataset(
        unit="VG5", dataset_type="training", operating_mode="turbine", window_size=window_size, device=device
    )

    train_loader, val_loader = create_dataloaders(dataset, batch_size=256, validation_split=0.2)

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

        dataset = SlidingDataset(
            unit="VG5",
            dataset_type="testing_synthetic_01",
            operating_mode="turbine",
            window_size=window_size,
            device=device,
        )
        _, val_loader = create_dataloaders(dataset, batch_size=256, validation_split=0.9)
        with torch.no_grad():
            spes = []
            for x in val_loader:
                reconstruction = model(x)
                spe = torch.sum(torch.square(reconstruction - x), dim=(1, 2))
                spes.append(spe.cpu().numpy())
                print((x.abs() > 5).sum())
            spes = np.concatenate(spes)
            print((spes > 5000.0).nonzero()[0])
            x = dataset[9154].unsqueeze(0)
            reconstruction = model(x)
            x = x.cpu().numpy()
            reconstruction = reconstruction.cpu().numpy()
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(x[0, 34, :], label="x")
            axes[0].plot(reconstruction[0, 0, :], label="pred")
            axes[1].plot(x[0, 17, :], label="x")
            axes[1].plot(reconstruction[0, 17, :], label="pred")
            axes[0].legend()
            axes[1].legend()
            plt.show()
            # fig, ax = plt.subplots()
            # ax.hist(spes, bins=100)
            # plt.show()


if __name__ == "__main__":
    main()
