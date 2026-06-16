import argparse
import csv
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import BinaryTokenDataset
from model import LlamaConfig, LlamaModel
from utils import (
    append_train_log,
    count_parameters,
    ensure_dir,
    get_device,
    get_lr,
    init_train_log,
    load_config,
    set_seed,
)


def configure_optimizer(model: torch.nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": float(training_cfg["weight_decay"])},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups,
        lr=float(training_cfg["learning_rate"]),
        betas=(float(training_cfg["beta1"]), float(training_cfg["beta2"])),
    )


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def distributed_context() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = get_device()

    return is_distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(is_distributed: bool):
    if is_distributed:
        dist.barrier()


@torch.no_grad()
def estimate_loss(
    model: LlamaModel,
    dataset: BinaryTokenDataset,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
    dtype_name: str,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x, y = dataset.get_batch(batch_size, block_size, device)
        with autocast_context(device, dtype_name):
            _, loss = model(x, y)
        losses[k] = loss.detach()
    mean_loss = losses.mean()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= dist.get_world_size()
    model.train()
    return float(mean_loss.item())


def save_checkpoint(
    path: Path,
    model: LlamaModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    train_loss: float,
    val_loss: float | None,
    best_val_loss: float | None = None,
    tokens_seen: int | None = None,
):
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def read_best_val_loss(log_path: str | Path) -> float:
    path = Path(log_path)
    if not path.exists():
        return float("inf")

    best = float("inf")
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("val_loss"):
                best = min(best, float(row["val_loss"]))
    return best


def run_training(config: dict, resume_path: str | None = None):
    seed = int(config.get("project", {}).get("seed", 1337))
    is_distributed, rank, local_rank, world_size, device = distributed_context()
    set_seed(seed + rank)

    model_cfg = LlamaConfig.from_dict(config["model"])
    training_cfg = config["training"]
    data_cfg = config["data"]
    paths_cfg = config["paths"]

    train_data = BinaryTokenDataset(data_cfg["train_bin"])
    val_data = BinaryTokenDataset(data_cfg["val_bin"])

    raw_model = LlamaModel(model_cfg).to(device)
    optimizer = configure_optimizer(raw_model, training_cfg)
    model = raw_model
    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None)

    checkpoint_dir = Path(paths_cfg["checkpoint_dir"])
    if is_main_process(rank):
        checkpoint_dir = ensure_dir(checkpoint_dir)
        ensure_dir(paths_cfg["output_dir"])
        init_train_log(paths_cfg["train_log"])
    barrier(is_distributed)

    batch_size = int(training_cfg["batch_size"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    block_size = int(model_cfg.block_size)
    max_steps = int(training_cfg["max_steps"])
    log_interval = int(training_cfg["log_interval"])
    eval_interval = int(training_cfg["eval_interval"])
    eval_iters = int(training_cfg["eval_iters"])
    save_interval = int(training_cfg["save_interval"])
    grad_clip = float(training_cfg["grad_clip"])
    dtype_name = str(training_cfg.get("dtype", "float32"))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype_name == "float16"))
    tokens_per_step = batch_size * block_size * grad_accum * world_size
    tokens_seen = int(training_cfg.get("initial_tokens_seen", 0))
    start_step = 1
    last_train_loss = float("nan")
    last_val_loss = None
    best_val_loss = read_best_val_loss(paths_cfg["train_log"])

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        tokens_seen = int(checkpoint.get("tokens_seen") or training_cfg.get("initial_tokens_seen", 0))
        if tokens_seen == 0:
            tokens_seen = int(checkpoint["step"]) * batch_size * block_size * grad_accum
        last_train_loss = float(checkpoint.get("train_loss", float("nan")))
        val_loss = checkpoint.get("val_loss")
        last_val_loss = None if val_loss is None else float(val_loss)
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        if bool(training_cfg.get("reset_best_val_on_resume", False)):
            best_val_loss = float("inf")
        if is_main_process(rank):
            print(f"resuming from {resume_path} at step {start_step}")

    if is_main_process(rank):
        print(f"distributed: {is_distributed} | world_size: {world_size}")
        print(f"device: {device}")
        print(f"parameters: {count_parameters(raw_model):,}")
        print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")
        print(f"tokens per step: {tokens_per_step:,} | initial tokens seen: {tokens_seen:,}")

    model.train()
    for step in range(start_step, max_steps + 1):
        lr = get_lr(step, training_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = train_data.get_batch(batch_size, block_size, device)
            sync_context = (
                model.no_sync()
                if is_distributed and micro_step < grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                with autocast_context(device, dtype_name):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                accumulated_loss += loss.item()
                scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step
        last_train_loss = accumulated_loss

        should_eval = step == 1 or step % eval_interval == 0
        if should_eval:
            last_train_loss = estimate_loss(model, train_data, batch_size, block_size, device, eval_iters, dtype_name)
            last_val_loss = estimate_loss(model, val_data, batch_size, block_size, device, eval_iters, dtype_name)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                if is_main_process(rank):
                    save_checkpoint(
                        checkpoint_dir / "ckpt_best.pt",
                        raw_model,
                        optimizer,
                        step,
                        config,
                        last_train_loss,
                        last_val_loss,
                        best_val_loss,
                        tokens_seen,
                    )

        if is_main_process(rank) and (step % log_interval == 0 or should_eval):
            append_train_log(paths_cfg["train_log"], step, tokens_seen, last_train_loss, last_val_loss, lr)
            val_text = "nan" if last_val_loss is None else f"{last_val_loss:.4f}"
            print(f"step {step:06d} | train {last_train_loss:.4f} | val {val_text} | lr {lr:.6g}")

        if is_main_process(rank) and (step % save_interval == 0 or step == max_steps):
            ckpt_path = checkpoint_dir / f"ckpt_step_{step:07d}.pt"
            save_checkpoint(ckpt_path, raw_model, optimizer, step, config, last_train_loss, last_val_loss, best_val_loss, tokens_seen)
            save_checkpoint(checkpoint_dir / "ckpt_last.pt", raw_model, optimizer, step, config, last_train_loss, last_val_loss, best_val_loss, tokens_seen)

    barrier(is_distributed)
    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_llama_100m.yaml")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to continue training from.")
    args = parser.parse_args()
    run_training(load_config(args.config), resume_path=args.resume)


if __name__ == "__main__":
    main()
