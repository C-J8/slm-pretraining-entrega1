import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from dataset import BinaryTokenDataset
from model import LlamaConfig, LlamaModel
from tokenizer import GPT2Tokenizer
from train import estimate_loss
from utils import ensure_dir, get_device, load_config


def load_model(checkpoint_path: str, device: torch.device) -> LlamaModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = LlamaConfig.from_dict(checkpoint["config"]["model"])
    model = LlamaModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_perplexity(args):
    device = get_device()
    model = load_model(args.checkpoint, device)
    data_path = args.data
    if args.config and not data_path:
        data_path = load_config(args.config)["data"]["val_bin"]
    if not data_path:
        raise ValueError("--data or --config is required for perplexity evaluation")

    dataset = BinaryTokenDataset(data_path)
    loss = estimate_loss(
        model,
        dataset,
        batch_size=args.batch_size,
        block_size=model.config.block_size,
        device=device,
        eval_iters=args.eval_iters,
        dtype_name=args.dtype,
    )
    result = {
        "checkpoint": args.checkpoint,
        "data": data_path,
        "loss": loss,
        "perplexity": math.exp(loss),
        "eval_iters": args.eval_iters,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)


def continuation_score(
    model: LlamaModel,
    tokenizer: GPT2Tokenizer,
    context: str,
    continuation: str,
    device: torch.device,
) -> float:
    context_tokens = tokenizer.encode(context)
    continuation_tokens = tokenizer.encode(continuation)
    if not continuation_tokens:
        return -float("inf")

    tokens = context_tokens + continuation_tokens
    if len(tokens) < 2:
        return -float("inf")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    continuation_start = len(context_tokens)
    target_positions = list(range(1, len(tokens)))
    mask = [pos >= continuation_start for pos in target_positions]

    if len(input_tokens) > model.config.block_size:
        overflow = len(input_tokens) - model.config.block_size
        input_tokens = input_tokens[overflow:]
        target_tokens = target_tokens[overflow:]
        mask = mask[overflow:]

    x = torch.tensor([input_tokens], dtype=torch.long, device=device)
    y = torch.tensor([target_tokens], dtype=torch.long, device=device)
    active = torch.tensor(mask, dtype=torch.bool, device=device)
    if not bool(active.any()):
        return -float("inf")

    with torch.no_grad():
        logits, _ = model(x)
        log_probs = F.log_softmax(logits[0], dim=-1)
        token_log_probs = log_probs.gather(1, y[0].unsqueeze(1)).squeeze(1)
    return float(token_log_probs[active].sum().item())


def load_benchmark(name: str, split: str):
    if name == "hellaswag":
        return load_dataset("Rowan/hellaswag", split=split)
    if name == "piqa":
        return load_dataset("piqa", split=split)
    if name == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split=split)
    if name == "arc_easy":
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if name == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    raise ValueError(f"Unknown benchmark: {name}")


def benchmark_choices(name: str, example: dict) -> tuple[str, list[str], int]:
    if name == "hellaswag":
        context = f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
        endings = [str(ending) for ending in example["endings"]]
        return context, endings, int(example["label"])

    if name == "piqa":
        return str(example["goal"]), [str(example["sol1"]), str(example["sol2"])], int(example["label"])

    if name == "winogrande":
        sentence = str(example["sentence"])
        before, after = sentence.split("_", 1)
        choices = [str(example["option1"]) + after, str(example["option2"]) + after]
        return before, choices, int(example["answer"]) - 1

    if name in {"arc_easy", "arc_challenge"}:
        question = str(example["question"])
        choices = [str(text) for text in example["choices"]["text"]]
        labels = [str(label) for label in example["choices"]["label"]]
        answer_key = str(example["answerKey"])
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        if answer_key not in label_to_index and answer_key.isdigit():
            label_to_index[answer_key] = int(answer_key) - 1
        return question + "\nAnswer:", choices, label_to_index[answer_key]

    raise ValueError(f"Unknown benchmark: {name}")


def run_multiple_choice(args):
    device = get_device()
    model = load_model(args.checkpoint, device)
    tokenizer = GPT2Tokenizer()
    dataset = load_benchmark(args.benchmark, args.split)

    rows = []
    correct = 0
    total = 0
    limit = args.max_examples or len(dataset)

    for example in tqdm(dataset, total=min(limit, len(dataset)), desc=args.benchmark):
        context, choices, gold = benchmark_choices(args.benchmark, example)
        scores = [
            continuation_score(model, tokenizer, context, " " + choice, device)
            for choice in choices
        ]
        pred = max(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(pred == gold)
        rows.append({"index": total, "gold": gold, "pred": pred, "scores": scores})
        total += 1
        if args.max_examples is not None and total >= args.max_examples:
            break

    result = {
        "checkpoint": args.checkpoint,
        "benchmark": args.benchmark,
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(1, total),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, {"summary": result, "examples": rows})


def write_json(path: str, payload: dict):
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_val_curve(args):
    rows = []
    with open(args.log, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                loss = float(row["val_loss"])
                rows.append({
                    "step": int(row["step"]),
                    "tokens_seen": int(row["tokens_seen"]),
                    "val_loss": loss,
                    "perplexity": math.exp(loss),
                })
    if not rows:
        raise ValueError(f"No validation rows found in {args.log}")
    write_json(args.output, {"source_log": args.log, "points": rows, "best": min(rows, key=lambda r: r["val_loss"])})
    print(json.dumps({"points": len(rows), "best": min(rows, key=lambda r: r["val_loss"])}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    ppl = subparsers.add_parser("perplexity")
    ppl.add_argument("--checkpoint", required=True)
    ppl.add_argument("--config", default=None)
    ppl.add_argument("--data", default=None)
    ppl.add_argument("--eval-iters", type=int, default=100)
    ppl.add_argument("--batch-size", type=int, default=1)
    ppl.add_argument("--dtype", default="float32")
    ppl.add_argument("--output", default=None)
    ppl.set_defaults(func=run_perplexity)

    mc = subparsers.add_parser("multiple_choice")
    mc.add_argument("--checkpoint", required=True)
    mc.add_argument("--benchmark", choices=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"], required=True)
    mc.add_argument("--split", default="validation")
    mc.add_argument("--max-examples", type=int, default=None)
    mc.add_argument("--output", default=None)
    mc.set_defaults(func=run_multiple_choice)

    curve = subparsers.add_parser("val_curve")
    curve.add_argument("--log", required=True)
    curve.add_argument("--output", required=True)
    curve.set_defaults(func=export_val_curve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
