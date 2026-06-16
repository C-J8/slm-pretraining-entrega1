import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from dataset import BinaryTokenDataset
from model import LlamaConfig, LlamaModel
from supervised_dataset import SupervisedTokenDataset
from train import configure_optimizer, estimate_loss, save_checkpoint
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


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def build_dataset(data_cfg: dict, split: str):
    if f"{split}_labels_bin" in data_cfg:
        return SupervisedTokenDataset(data_cfg[f"{split}_tokens_bin"], data_cfg[f"{split}_labels_bin"])
    return BinaryTokenDataset(data_cfg[f"{split}_bin"])


def load_model_weights(model: LlamaModel, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_finetune(
    config: dict,
    init_checkpoint: str | None = None,
    resume_path: str | None = None,
):
    seed = int(config.get("project", {}).get("seed", 1337))
    set_seed(seed)

    model_cfg = LlamaConfig.from_dict(config["model"])
    training_cfg = config["training"]
    data_cfg = config["data"]
    paths_cfg = config["paths"]

    device = get_device()
    train_data = build_dataset(data_cfg, "train")
    val_data = build_dataset(data_cfg, "val")

    model = LlamaModel(model_cfg).to(device)
    optimizer = configure_optimizer(model, training_cfg)

    checkpoint_dir = ensure_dir(paths_cfg["checkpoint_dir"])
    ensure_dir(paths_cfg["output_dir"])
    init_train_log(paths_cfg["train_log"])

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
    checkpoint_prefix = paths_cfg.get("checkpoint_prefix", "ckpt")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype_name == "float16"))
    tokens_seen = 0
    start_step = 1
    last_train_loss = float("nan")
    last_val_loss = None
    best_val_loss = float("inf")

    if init_checkpoint:
        load_model_weights(model, init_checkpoint, device)
        print(f"initialized weights from {init_checkpoint}")

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        tokens_seen = int(checkpoint["step"]) * batch_size * block_size * grad_accum
        last_train_loss = float(checkpoint.get("train_loss", float("nan")))
        val_loss = checkpoint.get("val_loss")
        last_val_loss = None if val_loss is None else float(val_loss)
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resuming from {resume_path} at step {start_step}")

    print(f"device: {device}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    model.train()
    for step in range(start_step, max_steps + 1):
        lr = get_lr(step, training_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(grad_accum):
            x, y = train_data.get_batch(batch_size, block_size, device)
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

        tokens_seen += batch_size * block_size * grad_accum
        last_train_loss = accumulated_loss

        should_eval = step == 1 or step % eval_interval == 0
        if should_eval:
            last_train_loss = estimate_loss(model, train_data, batch_size, block_size, device, eval_iters, dtype_name)
            last_val_loss = estimate_loss(model, val_data, batch_size, block_size, device, eval_iters, dtype_name)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                save_checkpoint(
                    checkpoint_dir / f"{checkpoint_prefix}_best.pt",
                    model,
                    optimizer,
                    step,
                    config,
                    last_train_loss,
                    last_val_loss,
                    best_val_loss,
                )

        if step % log_interval == 0 or should_eval:
            append_train_log(paths_cfg["train_log"], step, tokens_seen, last_train_loss, last_val_loss, lr)
            val_text = "nan" if last_val_loss is None else f"{last_val_loss:.4f}"
            print(f"step {step:06d} | train {last_train_loss:.4f} | val {val_text} | lr {lr:.6g}")

        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(
                checkpoint_dir / f"{checkpoint_prefix}_step_{step:07d}.pt",
                model,
                optimizer,
                step,
                config,
                last_train_loss,
                last_val_loss,
                best_val_loss,
            )
            save_checkpoint(
                checkpoint_dir / f"{checkpoint_prefix}_last.pt",
                model,
                optimizer,
                step,
                config,
                last_train_loss,
                last_val_loss,
                best_val_loss,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights only from a previous stage.")
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state from this stage.")
    args = parser.parse_args()
    run_finetune(load_config(args.config), init_checkpoint=args.init_checkpoint, resume_path=args.resume)


if __name__ == "__main__":
    main()
