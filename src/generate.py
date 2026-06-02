import argparse
from pathlib import Path

import torch

from model import LlamaConfig, LlamaModel
from tokenizer import GPT2Tokenizer
from utils import ensure_dir, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model_cfg = LlamaConfig.from_dict(config["model"])

    model = LlamaModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = GPT2Tokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tokenizer.decode(out[0].tolist())
    print(text)

    output_path = args.output or config.get("paths", {}).get("samples_file")
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"PROMPT: {args.prompt}\n")
            f.write(text)
            f.write("\n\n")


if __name__ == "__main__":
    main()

