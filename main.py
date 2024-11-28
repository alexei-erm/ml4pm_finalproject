from runner import Runner
from utils import seed_all, select_device
from model import *
from config import *

import os
from datetime import datetime
import argparse


def load_config(args: argparse.Namespace) -> Config:
    # Get defaults from config dataclass
    try:
        # Use specialized config if it exists
        cfg: Config = eval(f"{args.model}Config")()
    except NameError:
        cfg = Config()

    # Override with CLI args
    for arg, value in args.__dict__.items():
        if value is not None:
            cfg.__dict__[arg] = value

    if args.transient:
        cfg.equilibrium = False

    return cfg


def get_latest(log_root_dir: str) -> str:
    latest = sorted(os.listdir(log_root_dir))[-1]
    return os.path.join(log_root_dir, latest)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args)

    seed_all(cfg.seed)

    model_name = f"{cfg.model}_{cfg.unit}_{cfg.operating_mode}"
    if cfg.transient:
        model_name += "_transient"
    else:
        model_name += "_equilibrium"
    log_root_dir = os.path.abspath(os.path.join("logs", model_name))

    print("=" * os.get_terminal_size()[0])
    print("")
    device = select_device()
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

    else:
        log_dir = os.path.join(log_root_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        runner = Runner(cfg=cfg, dataset_root=args.dataset_root, log_dir=log_dir, device=device)
        runner.fit_spc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["SimpleAE", "ConvAE"], default="SimpleAE")
    parser.add_argument("--unit", type=str, choices=["VG4", "VG5", "VG6"], default=None)
    parser.add_argument("--operating_mode", type=str, choices=["pump", "turbine", "short_circuit"], default=None)
    parser.add_argument("--transient", action="store_true")
    parser.add_argument("--dataset_root", type=str, default="Dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true")
    group.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    main(args)
