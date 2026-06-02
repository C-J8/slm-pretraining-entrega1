import argparse
import csv

import matplotlib.pyplot as plt


def read_loss_log(path: str):
    steps = []
    train_loss = []
    val_steps = []
    val_loss = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))
    return steps, train_loss, val_steps, val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/train_log_0_60k.csv")
    parser.add_argument("--output", default="outputs/loss_curve_0_60k.png")
    args = parser.parse_args()

    steps, train_loss, val_steps, val_loss = read_loss_log(args.log)

    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_loss, label="train loss")
    if val_loss:
        plt.plot(val_steps, val_loss, label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
