import argparse
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.data import TDLDataset
from src.model import CHAST
from src.train import CheckpointConfig, EarlyStoppingConfig, Trainer


DEFAULTS: Dict[str, Any] = {
    "seed": 1337,
    "paths": {
        "out_dir": "runs",
    },
    "split": {
        "val_split": 0.1,
    },
    "train": {
        "epochs": 300,
        "batch_size": 64,
        "num_workers": 4,
        # Optional: load only model weights before training (fresh optimizer/scheduler).
        # Path is absolute or relative to `paths.out_dir`.
        "load_checkpoint": None,
        # Global epoch index already completed before this run; TB/logging start at offset+1.
        "epoch_offset": 0,
    },
    "dataset": {
        "num_subcarriers": 120,
        "num_symbols": 14,
        "pilot_symbols": [2, 7, 11],
        "pilot_every_n": 2,
        "snrs": [0, 5, 10, 15, 20, 25, 30],
    },
    "model": {
        "cnn_filters_1": 32,
        "cnn_filters_2": 2,
        "patch_h": 12,
        "patch_w": 14,
        "embed_dim": 64,
        "num_heads": 4,
        "dropout": 0.1,
        "kernel_size": 3,
    },
    "optim": {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "scheduler": {
            "t_max_epochs": 300,
            "eta_min": 1e-4,
        },
    },
    "early_stopping": {
        "patience": 50,
        "min_delta": 1e-5,
    },
}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CHAST from a YAML config.")
    p.add_argument(
        "config",
        type=str,
        help="Path to YAML config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config.",
    )
    # optional device override (cpu / cuda)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., cpu or cuda:0). If not set, config or auto is used.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override `paths.out_dir` from config (e.g., runs/exp2/tdla_2_seed1).",
    )
    return p


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping/object at the top level.")
    return data


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cfg_get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using a dot-separated key path.
    If the key is not found, return the default value.
        i.e. _cfg_get(cfg, "paths.data_path", "data/") returns the value of the key "data_path" in the dictionary "paths" in the config file.
        if the key is not found, return the default value "data/".
        if the key is found, return the value of the key.
    """
    cur: Any = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    args = _build_argparser().parse_args()

    cfg = _load_yaml(args.config)

    # seed selection: CLI > config > default
    seed = int(args.seed) if args.seed is not None else int(_cfg_get(cfg, "seed", DEFAULTS["seed"]))
    _set_seed(seed)

    # device selection: CLI > config > auto
    cfg_device = _cfg_get(cfg, "device", None)
    dev_name = args.device or cfg_device
    if dev_name is None:
        dev_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_name_str = str(dev_name).lower()
    if dev_name_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Requested device '{dev_name}' but no CUDA available; falling back to 'cpu'.")
            device = torch.device("cpu")
        else:
            device = torch.device(dev_name_str)  # supports 'cuda', 'cuda:0', 'cuda:1', ...
    elif dev_name_str == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device '{dev_name}'; use 'cpu' or 'cuda[:index]'.")

    data_path = _cfg_get(cfg, "paths.data_path")
    if not data_path:
        raise ValueError("Missing `paths.data_path` in config.")
    # output dir selection: CLI > config > default
    out_dir = str(args.out_dir) if args.out_dir is not None else str(_cfg_get(cfg, "paths.out_dir", DEFAULTS["paths"]["out_dir"]))
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Save the (possibly user-edited) config once per run for reproducibility.
    resolved_cfg: Dict[str, Any] = dict(cfg)
    resolved_cfg["seed"] = seed
    resolved_cfg["device"] = str(device)
    if not isinstance(resolved_cfg.get("paths"), dict):
        resolved_cfg["paths"] = {}
    resolved_cfg["paths"]["data_path"] = str(data_path)
    resolved_cfg["paths"]["out_dir"] = str(out_dir_path)
    resolved_cfg["paths"]["config_path"] = str(Path(args.config).resolve())
    config_dump_path = out_dir_path / "config.yaml"
    with open(config_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)

    dataset = TDLDataset(
        data_path,
        normalization_stats=None,  # my TDL dataset is already normalized
        return_pilots_only=False,  # CHAST needs sparse LS estimate
        num_subcarriers=int(_cfg_get(cfg, "dataset.num_subcarriers", DEFAULTS["dataset"]["num_subcarriers"])),
        num_symbols=int(_cfg_get(cfg, "dataset.num_symbols", DEFAULTS["dataset"]["num_symbols"])),
        SNRs=list(_cfg_get(cfg, "dataset.snrs", DEFAULTS["dataset"]["snrs"])),
        pilot_symbols=list(_cfg_get(cfg, "dataset.pilot_symbols", DEFAULTS["dataset"]["pilot_symbols"])),
        pilot_every_n=int(_cfg_get(cfg, "dataset.pilot_every_n", DEFAULTS["dataset"]["pilot_every_n"])),
    )

    n_total = len(dataset)
    val_split = float(_cfg_get(cfg, "split.val_split", DEFAULTS["split"]["val_split"]))
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = int(_cfg_get(cfg, "train.batch_size", DEFAULTS["train"]["batch_size"]))
    num_workers = int(_cfg_get(cfg, "train.num_workers", DEFAULTS["train"]["num_workers"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    num_subcarriers = int(_cfg_get(cfg, "dataset.num_subcarriers", DEFAULTS["dataset"]["num_subcarriers"]))
    num_symbols = int(_cfg_get(cfg, "dataset.num_symbols", DEFAULTS["dataset"]["num_symbols"]))

    model = CHAST(
        num_subcarriers=num_subcarriers,
        num_symbols=num_symbols,
        in_channels=2,
        cnn_filters_1=int(_cfg_get(cfg, "model.cnn_filters_1", DEFAULTS["model"]["cnn_filters_1"])),
        cnn_filters_2=int(_cfg_get(cfg, "model.cnn_filters_2", DEFAULTS["model"]["cnn_filters_2"])),
        patch_h=int(_cfg_get(cfg, "model.patch_h", DEFAULTS["model"]["patch_h"])),
        patch_w=int(_cfg_get(cfg, "model.patch_w", DEFAULTS["model"]["patch_w"])),
        embed_dim=int(_cfg_get(cfg, "model.embed_dim", DEFAULTS["model"]["embed_dim"])),
        num_heads=int(_cfg_get(cfg, "model.num_heads", DEFAULTS["model"]["num_heads"])),
        dropout=float(_cfg_get(cfg, "model.dropout", DEFAULTS["model"]["dropout"])),
        kernel_size=int(_cfg_get(cfg, "model.kernel_size", DEFAULTS["model"]["kernel_size"])),
    ).to(device)

    epochs = int(_cfg_get(cfg, "train.epochs", DEFAULTS["train"]["epochs"]))
    lr = float(_cfg_get(cfg, "optim.lr", DEFAULTS["optim"]["lr"]))
    weight_decay = float(_cfg_get(cfg, "optim.weight_decay", DEFAULTS["optim"]["weight_decay"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    opt_cfg = cfg.get("optim")
    if isinstance(opt_cfg, dict) and "scheduler" in opt_cfg:
        sched_cfg = opt_cfg["scheduler"]
    else:
        sched_cfg = DEFAULTS["optim"]["scheduler"]

    if sched_cfg is None:
        scheduler = None
    else:
        if not isinstance(sched_cfg, dict):
            raise ValueError("optim.scheduler must be a mapping or null (constant learning rate).")
        t_max = int(
            sched_cfg.get(
                "t_max_epochs",
                DEFAULTS["optim"]["scheduler"]["t_max_epochs"],
            )
        )
        eta_min = float(
            sched_cfg.get("eta_min", DEFAULTS["optim"]["scheduler"]["eta_min"])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    tb_writer = SummaryWriter(log_dir=str(out_dir_path / "tb"))

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=CheckpointConfig(out_dir=out_dir_path, filename="best.pt"),
        early_stopping=EarlyStoppingConfig(
            patience=int(_cfg_get(cfg, "early_stopping.patience", DEFAULTS["early_stopping"]["patience"])),
            min_delta=float(_cfg_get(cfg, "early_stopping.min_delta", DEFAULTS["early_stopping"]["min_delta"])),
        ),
        run_config=resolved_cfg,
        tb_writer=tb_writer,
    )

    load_ckpt = _cfg_get(cfg, "train.load_checkpoint", DEFAULTS["train"]["load_checkpoint"])
    if load_ckpt:
        ckpt_path = Path(load_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = out_dir_path / ckpt_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"train.load_checkpoint does not exist: {ckpt_path} "
                f"(resolved from config value {load_ckpt!r}; out_dir={out_dir_path})"
            )
        Trainer.load_model_weights(ckpt_path, model=model, map_location=device)
        print(f"Loaded model weights only from {ckpt_path} (new optimizer/scheduler from config).")

    epoch_offset = int(_cfg_get(cfg, "train.epoch_offset", DEFAULTS["train"]["epoch_offset"]))
    if epoch_offset < 0:
        raise ValueError(f"train.epoch_offset must be >= 0, got {epoch_offset}")

    summary = trainer.train(epochs=epochs, epoch_offset=epoch_offset)
    print("done:", summary)
    tb_writer.close()


if __name__ == "__main__":
    main()

