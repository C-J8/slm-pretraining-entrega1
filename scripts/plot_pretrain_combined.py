import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_points(log_paths: list[str]) -> list[dict]:
    points = []
    for path in log_paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("val_loss"):
                    continue
                points.append(
                    {
                        "source_log": path,
                        "step": int(row["step"]),
                        "tokens_seen": int(row["tokens_seen"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "perplexity": math.exp(float(row["val_loss"])),
                    }
                )

    # Keep the most recent measurement for repeated token/step pairs after resumes.
    by_key = {(p["step"], p["tokens_seen"]): p for p in points}
    return sorted(by_key.values(), key=lambda p: (p["tokens_seen"], p["step"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--plot-output", default="outputs/pretrain_loss_curve_0_2b.png")
    parser.add_argument("--json-output", default="outputs/pretrain_val_curve_0_2b.json")
    args = parser.parse_args()

    points = read_points(args.logs)
    if not points:
        raise ValueError("No validation points found.")

    best = min(points, key=lambda p: p["val_loss"])

    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(
        json.dumps({"source_logs": args.logs, "points": points, "best": best}, indent=2),
        encoding="utf-8",
    )

    x = [p["tokens_seen"] / 1e9 for p in points]
    train_loss = [p["train_loss"] for p in points]
    val_loss = [p["val_loss"] for p in points]

    plt.figure(figsize=(9, 5))
    plt.plot(x, train_loss, label="train loss", alpha=0.75)
    plt.plot(x, val_loss, label="val loss", alpha=0.9)
    plt.scatter([best["tokens_seen"] / 1e9], [best["val_loss"]], s=36, label=f"best val {best['val_loss']:.3f}")
    plt.xlabel("tokens seen (billions)")
    plt.ylabel("loss")
    plt.title("Pre-training: 0.5B to 2B tokens")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=160)

    print(json.dumps({"points": len(points), "best": best}, indent=2))


if __name__ == "__main__":
    main()
