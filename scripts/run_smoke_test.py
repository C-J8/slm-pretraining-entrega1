import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import BinaryTokenDataset
from model import LlamaConfig, LlamaModel
from tokenizer import GPT2Tokenizer
from train import configure_optimizer, save_checkpoint
from utils import ensure_dir, set_seed


def main():
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(ROOT / "data")
    ensure_dir(ROOT / "outputs")
    ensure_dir(ROOT / "checkpoints")

    tokenizer = GPT2Tokenizer()
    text = (
        "Machine learning is a field of study that gives computers the ability to learn from data. "
        "Education is important because it helps people understand science and society. "
    )
    tokens = tokenizer.encode(text * 600, add_eot=True)
    split = int(0.9 * len(tokens))

    np.asarray(tokens[:split], dtype=np.uint16).tofile(ROOT / "data" / "smoke_train.bin")
    np.asarray(tokens[split:], dtype=np.uint16).tofile(ROOT / "data" / "smoke_val.bin")

    config = {
        "model": {
            "name": "smoke_llama",
            "vocab_size": 50257,
            "block_size": 64,
            "n_layer": 2,
            "n_embd": 128,
            "n_head": 4,
            "head_dim": 32,
            "n_kv_head": 4,
            "ffn_hidden_size": 384,
            "dropout": 0.0,
            "bias": False,
            "tie_weights": True,
        },
        "training": {
            "learning_rate": 1.0e-3,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
        },
    }

    model_cfg = LlamaConfig.from_dict(config["model"])
    model = LlamaModel(model_cfg).to(device)
    optimizer = configure_optimizer(model, config["training"])
    train_data = BinaryTokenDataset(ROOT / "data" / "smoke_train.bin")

    losses = []
    model.train()
    for step in range(1, 31):
        x, y = train_data.get_batch(batch_size=8, block_size=model_cfg.block_size, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"smoke step {step:02d} | loss {loss.item():.4f}")

    save_checkpoint(
        ROOT / "checkpoints" / "smoke_ckpt.pt",
        model,
        optimizer,
        30,
        config,
        losses[-1],
        None,
    )

    prompt = "Machine learning is"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=50)
    sample = tokenizer.decode(out[0].tolist())
    with open(ROOT / "outputs" / "smoke_samples.txt", "w", encoding="utf-8") as f:
        f.write(sample + "\n")

    print(f"initial loss: {losses[0]:.4f}")
    print(f"final loss: {losses[-1]:.4f}")
    print("saved checkpoints/smoke_ckpt.pt and outputs/smoke_samples.txt")


if __name__ == "__main__":
    main()

