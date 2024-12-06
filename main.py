from runner import Runner
from utils import seed_all, select_device
from model import *
from config import *

import os
from datetime import datetime
import argparse


def get_latest(log_root_dir: str) -> str:
    latest = sorted(os.listdir(log_root_dir))[-1]
    return os.path.join(log_root_dir, latest)


def override_config(cfg: Config, args: argparse.Namespace) -> Config:
    for arg, value in args.__dict__.items():
        if arg in cfg.__dict__ and value is not None:
            cfg.__dict__[arg] = value
    print(cfg)

    if args.features is not None:
        cfg.features = args.features

    print(cfg)
    exit()
    return cfg


def main(args: argparse.Namespace) -> None:
    # FIXME: handle case where the timestamp dir is already provided
    log_root_dir = os.path.abspath(os.path.join("logs", args.config))

    cfg = CFG[args.config]
    cfg = override_config(cfg, args)

    cfg.model = args.model

    seed_all(cfg.seed)

    print("=" * os.get_terminal_size()[0])
    print("")
    device = torch.device("cpu") if args.cpu else select_device()
    print(f"Device: {device}")

    if args.train:
        log_dir = os.path.join(log_root_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Saving model and logs to: {log_dir}")

        runner = Runner(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)
        runner.train_autoencoder()

    elif args.eval:
        log_dir = get_latest(log_root_dir)
        print(f"Loading model from: {log_dir}")

        runner = Runner(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)
        runner.test_autoencoder()

    elif args.spc:
        log_dir = os.path.join(log_root_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        runner = Runner(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)
        runner.fit_spc()

    else:
        log_dir = os.path.join(log_root_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        runner = Runner(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)
        runner.fit_if()


if __name__ == "__main__":
    """
    What's the best interface here?
    --train <config> [<overrides>]
        Read <config> from config.py, override args if necessary
    --eval <config> [<overrides>]
        Load <config> from saved YAML, <config> could also contain the run (timestamp) directory.
        Override args from saved config if necessary.

    Save config.yaml, and load model from model.pkl. Should be flexible enough and even allow for
    small changes to the model definition if necessary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed to use for all RNGs.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for all Torch operations.")
    parser.add_argument("--config", type=str, choices=CFG.keys(), default=CFG.keys()[0], help="Configuration to use.")
    parser.add_argument("--unit", type=str, choices=["VG4", "VG5", "VG6"], help="Plant unit to load data for.")
    parser.add_argument(
        "--operating_mode", type=str, choices=["pump", "turbine", "short_circuit"], help="Generator operating mode."
    )
    parser.add_argument("--transient", action="store_true", help="Include transient (non equilibrium) samples.")
    parser.add_argument(
        "--features", nargs="+", help="Feature(s) to use for model input. By default, uses all features."
    )
    parser.add_argument(
        "--dataset_root", type=str, default="Dataset", help="Root path of the folder to load the datasets from."
    )
    parser.add_argument("--subsampling", type=int, help="Subsampling for training samples.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train the model.")
    group.add_argument("--eval", action="store_true", help="Evaluate the model.")
    group.add_argument("--spc", action="store_true", help="Fit simple SPC model.")
    args = parser.parse_args()

    main(args)
