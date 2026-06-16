import argparse
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer import GPT2Tokenizer
from utils import ensure_dir, load_config


def write_tokens(path: Path, tokens: list[int]):
    arr = np.asarray(tokens, dtype=np.uint16)
    with open(path, "ab") as f:
        arr.tofile(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_llama_100m.yaml")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--skip-documents", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    random.seed(args.seed)

    train_path = Path(data_cfg["train_bin"])
    val_path = Path(data_cfg["val_bin"])
    train_tmp_path = train_path.with_suffix(train_path.suffix + ".tmp")
    val_tmp_path = val_path.with_suffix(val_path.suffix + ".tmp")
    ensure_dir(train_path.parent)
    train_tmp_path.unlink(missing_ok=True)
    val_tmp_path.unlink(missing_ok=True)

    tokenizer = GPT2Tokenizer()
    dataset = load_dataset(
        data_cfg["dataset_name"],
        data_cfg["dataset_config"],
        split=data_cfg["split"],
        streaming=bool(data_cfg.get("streaming", True)),
    )

    text_field = data_cfg.get("text_field", "text")
    val_fraction = float(data_cfg.get("val_fraction", 0.01))
    iterator = iter(dataset)

    train_docs = 0
    val_docs = 0
    train_tokens = 0
    val_tokens = 0

    processed_docs = 0
    with tqdm(total=args.max_documents, desc="tokenizing documents") as pbar:
        for doc_idx, example in enumerate(iterator, start=1):
            if doc_idx <= args.skip_documents:
                continue

            text = example.get(text_field, "")
            if not text:
                continue

            tokens = tokenizer.encode(text, add_eot=True)
            if random.random() < val_fraction:
                write_tokens(val_tmp_path, tokens)
                val_docs += 1
                val_tokens += len(tokens)
            else:
                write_tokens(train_tmp_path, tokens)
                train_docs += 1
                train_tokens += len(tokens)

            processed_docs += 1
            pbar.update(1)
            if args.max_documents is not None and processed_docs >= args.max_documents:
                break

    train_tmp_path.replace(train_path)
    val_tmp_path.replace(val_path)

    print(f"train docs: {train_docs:,} | train tokens: {train_tokens:,}")
    print(f"val docs: {val_docs:,} | val tokens: {val_tokens:,}")
    print(f"skipped source docs: {args.skip_documents:,} | processed docs: {processed_docs:,}")
    print(f"wrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
