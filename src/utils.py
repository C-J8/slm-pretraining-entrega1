import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_lr(step: int, training_cfg: dict) -> float:
    lr = float(training_cfg["learning_rate"])
    min_lr = float(training_cfg["min_learning_rate"])
    warmup_steps = int(training_cfg["warmup_steps"])
    max_steps = int(training_cfg["max_steps"])

    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def init_train_log(path: str | Path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tokens_seen", "train_loss", "val_loss", "learning_rate"])


def append_train_log(
    path: str | Path,
    step: int,
    tokens_seen: int,
    train_loss: float,
    val_loss: float | None,
    learning_rate: float,
):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            tokens_seen,
            f"{train_loss:.6f}",
            "" if val_loss is None else f"{val_loss:.6f}",
            f"{learning_rate:.8f}",
        ])


def worker_count_from_env(default: int = 1) -> int:
    return int(os.environ.get("NUM_WORKERS", default))

