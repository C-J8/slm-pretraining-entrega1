import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer import GPT2Tokenizer
from utils import ensure_dir, load_config


ROLE_NAMES = {
    "human": "User",
    "user": "User",
    "prompter": "User",
    "assistant": "Assistant",
    "gpt": "Assistant",
    "bot": "Assistant",
    "system": "System",
}


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text_value(item) for item in value).strip()
    if isinstance(value, dict):
        if "text" in value:
            return text_value(value["text"])
        if "content" in value:
            return text_value(value["content"])
    return str(value).strip()


def normalize_role(role: Any) -> str:
    return ROLE_NAMES.get(str(role).lower(), str(role).title() or "User")


def messages_from_example(example: dict) -> list[tuple[str, str]]:
    if isinstance(example.get("messages"), list):
        messages = []
        for message in example["messages"]:
            role = normalize_role(message.get("role", message.get("from", "user")))
            content = text_value(message.get("content", message.get("value", message.get("text", ""))))
            if content:
                messages.append((role, content))
        return messages

    if isinstance(example.get("conversations"), list):
        messages = []
        for message in example["conversations"]:
            role = normalize_role(message.get("from", message.get("role", "user")))
            content = text_value(message.get("value", message.get("content", "")))
            if content:
                messages.append((role, content))
        return messages

    prompt = text_value(
        example.get("prompt")
        or example.get("instruction")
        or example.get("question")
        or example.get("input")
    )
    response = text_value(
        example.get("response")
        or example.get("output")
        or example.get("answer")
        or example.get("completion")
    )
    extra_input = text_value(example.get("input")) if example.get("instruction") else ""
    if prompt and extra_input and extra_input != prompt:
        prompt = f"{prompt}\n{extra_input}"
    if prompt and response:
        return [("User", prompt), ("Assistant", response)]
    return []


def encode_mid(messages: list[tuple[str, str]], tokenizer: GPT2Tokenizer) -> list[int]:
    text = "".join(f"{role}:\n{content}\n" for role, content in messages)
    tokens = tokenizer.encode(text)
    tokens.append(tokenizer.eot_token)
    return tokens


def encode_sft(messages: list[tuple[str, str]], tokenizer: GPT2Tokenizer) -> tuple[list[int], list[int]]:
    tokens: list[int] = []
    labels: list[int] = []
    has_assistant = False

    for role, content in messages:
        if role == "Assistant":
            prefix_tokens = tokenizer.encode("Assistant:\n")
            response_tokens = tokenizer.encode(content)
            response_tokens.append(tokenizer.eot_token)

            tokens.extend(prefix_tokens)
            labels.extend([-100] * len(prefix_tokens))
            tokens.extend(response_tokens)
            labels.extend(response_tokens)
            has_assistant = True
        else:
            segment_tokens = tokenizer.encode(f"{role}:\n{content}\n")
            tokens.extend(segment_tokens)
            labels.extend([-100] * len(segment_tokens))

    if not has_assistant:
        return [], []
    return tokens, labels


def append_uint16(path: Path, values: list[int]):
    with open(path, "ab") as f:
        np.asarray(values, dtype=np.uint16).tofile(f)


def append_int32(path: Path, values: list[int]):
    with open(path, "ab") as f:
        np.asarray(values, dtype=np.int32).tofile(f)


def dataset_iterator(data_cfg: dict):
    dataset_config = data_cfg.get("dataset_config")
    kwargs = {
        "split": data_cfg.get("split", "train"),
        "streaming": bool(data_cfg.get("streaming", True)),
    }
    if dataset_config:
        return load_dataset(data_cfg["dataset_name"], dataset_config, **kwargs)
    return load_dataset(data_cfg["dataset_name"], **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["mid", "sft"], required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    tokenizer = GPT2Tokenizer()
    random.seed(args.seed)

    val_fraction = float(data_cfg.get("val_fraction", 0.02))
    dataset = dataset_iterator(data_cfg)

    if args.phase == "mid":
        train_path = Path(data_cfg["train_bin"])
        val_path = Path(data_cfg["val_bin"])
        paths = [train_path, val_path]
    else:
        train_path = Path(data_cfg["train_tokens_bin"])
        val_path = Path(data_cfg["val_tokens_bin"])
        train_labels_path = Path(data_cfg["train_labels_bin"])
        val_labels_path = Path(data_cfg["val_labels_bin"])
        paths = [train_path, val_path, train_labels_path, val_labels_path]

    tmp_paths = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    for path in paths + tmp_paths:
        ensure_dir(path.parent)
        path.unlink(missing_ok=True)
    for tmp_path in tmp_paths:
        tmp_path.touch()

    train_examples = val_examples = train_tokens = val_tokens = 0
    with tqdm(total=args.max_examples, desc=f"preparing {args.phase} data") as pbar:
        for example_idx, example in enumerate(dataset, start=1):
            messages = messages_from_example(example)
            if not messages:
                continue

            is_val = random.random() < val_fraction
            if args.phase == "mid":
                tokens = encode_mid(messages, tokenizer)
                if len(tokens) < 2:
                    continue
                append_uint16(val_path.with_suffix(val_path.suffix + ".tmp") if is_val else train_path.with_suffix(train_path.suffix + ".tmp"), tokens)
                token_count = len(tokens)
            else:
                tokens, labels = encode_sft(messages, tokenizer)
                if len(tokens) < 2:
                    continue
                token_path = val_path.with_suffix(val_path.suffix + ".tmp") if is_val else train_path.with_suffix(train_path.suffix + ".tmp")
                label_path = val_labels_path.with_suffix(val_labels_path.suffix + ".tmp") if is_val else train_labels_path.with_suffix(train_labels_path.suffix + ".tmp")
                append_uint16(token_path, tokens)
                append_int32(label_path, labels)
                token_count = len(tokens)

            if is_val:
                val_examples += 1
                val_tokens += token_count
            else:
                train_examples += 1
                train_tokens += token_count

            pbar.update(1)
            if args.max_examples is not None and example_idx >= args.max_examples:
                break

    for tmp_path, final_path in zip(tmp_paths, paths):
        tmp_path.replace(final_path)

    print(f"train examples: {train_examples:,} | train tokens: {train_tokens:,}")
    print(f"val examples: {val_examples:,} | val tokens: {val_tokens:,}")


if __name__ == "__main__":
    main()
