from train import *
from utils import *
from model import *
from config import *

import os
from datetime import datetime
import argparse


def override_config(cfg: Config, args: argparse.Namespace) -> Config:
    for arg, value in args.__dict__.items():
        if arg in cfg.__dict__ and value is not None:
            cfg.__dict__[arg] = value
    return cfg


def main(args: argparse.Namespace) -> None:
    print("=" * os.get_terminal_size()[0])
    print("")
    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Device: {device}")

    log_root_dir = os.path.abspath(os.path.join("logs", args.config))

    if CFG[args.config].model == ModelType.SPC:
        cfg = CFG[args.config]
        seed_all(cfg.seed)
        fit_spc(cfg, dataset_root=args.dataset_root, device=device)

    elif CFG[args.config].model == ModelType.KPCA:
        cfg = CFG[args.config]
        seed_all(cfg.seed)
        fit_kpca(cfg, dataset_root=args.dataset_root, device=device)

    elif args.train:
        run_name = args.run_name if args.run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_dir, run_name)
        print(f"Saving model and logs to: {log_dir}")

        cfg = CFG[args.config]
        cfg = override_config(cfg, args)

        seed_all(cfg.seed)
        train_autoencoder(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)

    elif args.eval:
        run_name = args.run_name if args.run_name is not None else sorted(os.listdir(log_root_dir))[-1]
        log_dir = os.path.join(log_root_dir, run_name)
        print(f"Loading model from: {log_dir}")

        # cfg = Config(**load_yaml(os.path.join(log_dir, "config.yaml")))
        cfg = load_pickle(os.path.join(log_dir, "config.pkl"))
        cfg = override_config(cfg, args)

        seed_all(cfg.seed)
        test_autoencoder(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, load_best=args.best, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed to use for all RNGs.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for all Torch operations.")
    parser.add_argument(
        "--config", type=str, choices=CFG.keys(), default=list(CFG.keys())[0], help="Configuration to use."
    )
    parser.add_argument(
        "--run_name", type=str, help="Explicit run name to use. By default, the run name is a timestamp."
    )
    parser.add_argument("--unit", type=str, choices=["VG4", "VG5", "VG6"], help="Plant unit to load data for.")
    parser.add_argument(
        "--operating_mode", type=str, choices=["pump", "turbine", "short_circuit"], help="Generator operating mode."
    )
    parser.add_argument(
        "--transient", action="store_true", default=None, help="Include transient (non equilibrium) samples."
    )
    parser.add_argument(
        "--features", nargs="+", help="Feature(s) to use for model input. By default, uses all features."
    )
    parser.add_argument(
        "--dataset_root", type=str, default="Dataset", help="Root path of the folder to load the datasets from."
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs.")
    parser.add_argument("--subsampling", type=int, help="Subsampling for training samples.")
    parser.add_argument(
        "--best",
        action="store_true",
        help="Load the model with the lowest validation loss. By default, loads the last model saved while training.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train the model.")
    group.add_argument("--eval", action="store_true", help="Evaluate the model.")
    args = parser.parse_args()

    main(args)
